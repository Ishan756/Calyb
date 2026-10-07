"""Fetch RAG papers from the Semantic Scholar Graph API into data/raw_papers.json.

Endpoints used (all under /graph/v1):
  GET  /paper/search              -> seed the corpus from the seed queries
  POST /paper/batch               -> full metadata for a known list of paperIds
  GET  /paper/{id}/citations      -> who cites a paper (+ isInfluential/contexts/intents)
  GET  /paper/{id}/references     -> what a paper cites    (+ isInfluential/contexts/intents)

Everything is cached to data/raw_papers.json, so re-running only fetches what is
missing and each paper is fetched at most once.

Auth is optional: set S2_API_KEY to use a key, otherwise run anonymously (much
lower rate limit). Every request is spaced out and 429/5xx responses are retried
with exponential backoff that honours the Retry-After header.

Usage:
    python src/fetch_papers.py                    # top up the cache from config
    python src/fetch_papers.py --limit 20         # fetch at most 20 new papers
    python src/fetch_papers.py --refresh          # refetch metadata, keep edges
    python src/fetch_papers.py --reset            # start the cache from scratch
    python src/fetch_papers.py --dry-run          # print the plan, hit no API

Environment:
    S2_API_KEY    optional Semantic Scholar API key
    S2_BASE_URL   optional API root override (default https://api.semanticscholar.org)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "schema.yaml"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "raw_papers.json"

# Fields requested for every paper. Kept in one place so the cache is uniform.
PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "citationCount",
    "externalIds",
]

# Citation/references endpoints return the same paper fields plus per-edge
# metadata. `contexts` and `intents` are requested together: the API expects
# `intents` to be accompanied by `contexts`.
EDGE_EXTRA_FIELDS = ["contexts", "intents", "isInfluential"]

# S2 caps a single /paper/search page at 1000 and a batch at 500 ids.
MAX_SEARCH_PAGE = 100
MAX_BATCH_IDS = 500

# Citations/references are paged too. The API serves at most 1000 rows per call
# but a highly cited paper can have tens of thousands of citation edges, so we
# page until we hit fetch.max_edges_per_paper (default 100).
EDGE_PAGE_SIZE = 100
DEFAULT_MAX_EDGES_PER_PAPER = 100


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def reject_unfilled_placeholders(path: Path) -> None:
    """Refuse to run against a config template that still contains `TODO:`.

    A half-filled config would otherwise produce a quietly wrong graph, so this
    fails loudly and names the offending lines instead.
    """
    offenders = [(i, line.strip().lstrip("-# ").strip())
                 for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                 if "TODO" in line and not line.lstrip().startswith("#")]
    if not offenders:
        return
    shown = "\n".join(f"    line {i}: {text[:70]}" for i, text in offenders[:10])
    more = f"\n    ... and {len(offenders) - 10} more" if len(offenders) > 10 else ""
    sys.exit(
        f"error: {path} still contains {len(offenders)} unfilled placeholder(s):\n"
        f"{shown}{more}\n"
        f"       Replace every TODO with your own value, or delete the block."
    )


def load_fetch_config(config_path: Path) -> dict:
    """Read the `fetch:` section of config/schema.yaml.

    Expected shape (all keys optional except seed_queries / num_papers):

        fetch:
          seed_queries: ["retrieval augmented generation", ...]
          num_papers: 60
          year_from: 2018
          year_to: 2025
          request_delay_seconds: 1.0
          max_retries: 6
          search_page_size: 100
          fetch_edges: true
    """
    if not config_path.exists():
        sys.exit(
            f"error: config not found: {config_path}\n"
            f"       fetch_papers.py reads the `fetch:` section of that file "
            f"(seed_queries, num_papers, year_from, year_to)."
        )
    reject_unfilled_placeholders(config_path)
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = raw.get("fetch")
    if not cfg:
        sys.exit(f"error: no `fetch:` section found in {config_path}")

    queries = cfg.get("seed_queries") or []
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        sys.exit("error: fetch.seed_queries must be a list of strings")

    missing = [k for k in ("num_papers",) if k not in cfg]
    if missing:
        sys.exit(f"error: fetch.{', fetch.'.join(missing)} missing from {config_path}")

    return {
        "seed_queries": queries,
        "num_papers": int(cfg["num_papers"]),
        "year_from": cfg.get("year_from"),
        "year_to": cfg.get("year_to"),
        "request_delay_seconds": float(cfg.get("request_delay_seconds", 1.0)),
        "max_retries": int(cfg.get("max_retries", 6)),
        "search_page_size": int(cfg.get("search_page_size", MAX_SEARCH_PAGE)),
        "max_edges_per_paper": int(cfg.get("max_edges_per_paper",
                                           DEFAULT_MAX_EDGES_PER_PAPER)),
        "fetch_edges": bool(cfg.get("fetch_edges", True)),
    }


# --------------------------------------------------------------------------
# HTTP client with retry / backoff
# --------------------------------------------------------------------------
class S2Client:
    """Thin Semantic Scholar client.

    Responsibilities: auth header, polite spacing between calls, and retrying
    429 / 5xx with exponential backoff + jitter. Not much more.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        delay: float = 1.0,
        max_retries: int = 6,
        verbose: bool = True,
    ) -> None:
        self.base_url = (base_url or os.environ.get("S2_BASE_URL")
                         or "https://api.semanticscholar.org").rstrip("/")
        self.api_key = api_key
        self.delay = max(0.0, delay)
        self.max_retries = max_retries
        self.verbose = verbose

        self._session = requests.Session()
        self._last_call = 0.0
        self.call_count = 0

    # -- internals ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _wait_turn(self) -> None:
        """Sleep so that consecutive calls are at least `delay` seconds apart."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.monotonic()

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        """Exponential backoff with jitter, deferring to Retry-After when given."""
        if retry_after:
            try:
                return min(120.0, float(retry_after))
            except ValueError:
                pass  # Retry-After may be an HTTP-date; ignore and use our own backoff
        return min(120.0, (2 ** attempt)) + random.uniform(0, 1.0)

    # -- public ------------------------------------------------------------
    def request(self, method: str, path: str, **kwargs) -> dict:
        """Perform one API call, retrying rate limits and transient server errors."""
        url = f"{self.base_url}/graph/v1{path}"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        last_error: str = "unknown"
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                self.call_count += 1
                resp = self._session.request(
                    method, url, headers=headers, timeout=30, **kwargs
                )
            except requests.RequestException as exc:
                # Connection reset / timeout: treat like a transient 5xx.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    break
                wait = self._backoff(attempt, None)
                self._log(f"  network error, retrying in {wait:.1f}s ({last_error})")
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}"
                if attempt == self.max_retries:
                    break
                wait = self._backoff(attempt, resp.headers.get("Retry-After"))
                self._log(
                    f"  {last_error}, backing off {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                # A missing paper is a legitimate answer, not a failure.
                return {}
            if resp.status_code == 400:
                # Usually a malformed query; retrying will not help.
                sys.exit(f"error: S2 rejected the request (HTTP 400)\n"
                         f"       {method} {url}\n       {resp.text[:300]}")
            sys.exit(f"error: unexpected HTTP {resp.status_code} from {method} {url}\n"
                     f"       {resp.text[:300]}")

        sys.exit(f"error: giving up on {method} {url} after "
                 f"{self.max_retries + 1} attempts ({last_error})")

    def search_papers(self, query: str, limit: int, offset: int,
                      year_from: int | None, year_to: int | None) -> list[dict]:
        """One page of /paper/search. S2 caps `limit` at 1000 per request."""
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, MAX_SEARCH_PAGE),
            "offset": offset,
            "fields": ",".join(PAPER_FIELDS),
        }
        if year_from is not None or year_to is not None:
            # S2 accepts "2018-2025", "2018-" and "-2025".
            params["year"] = f"{year_from or ''}-{year_to or ''}"

        data = self.request("GET", "/paper/search", params=params)
        return data.get("data") or []

    def get_batch(self, paper_ids: list[str]) -> list[dict]:
        """POST /paper/batch. Returns partial records; unknown ids come back as None."""
        if not paper_ids:
            return []
        results: list[dict] = []
        for start in range(0, len(paper_ids), MAX_BATCH_IDS):
            chunk = paper_ids[start:start + MAX_BATCH_IDS]
            data = self.request(
                "POST", "/paper/batch",
                params={"fields": ",".join(PAPER_FIELDS)},
                json={"ids": chunk},
            )
            # S2 may return a list or a dict with 'data'
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.extend(data.get("data") or data.get("papers") or [])
            else:
                results.extend([])
        return [r for r in results if r]

    def _edge_fields(self) -> str:
        return ",".join(PAPER_FIELDS + EDGE_EXTRA_FIELDS)

    def _paged_edges(self, paper_id: str, kind: str, limit: int
                     ) -> tuple[list[dict], bool]:
        """Page through /paper/{id}/citations or /references.

        `kind` is "citations" or "references". Returns (edges, truncated), where
        `truncated` is True when we stopped because we hit `limit` rather than
        because the API ran out of rows. The API does not report a total, so a
        truncated result means "at least `limit` edges exist", not an exact count.
        """
        collected: list[dict] = []
        offset = 0
        while len(collected) < limit:
            page = min(EDGE_PAGE_SIZE, limit - len(collected))
            data = self.request(
                "GET", f"/paper/{paper_id}/{kind}",
                params={
                    "fields": self._edge_fields(),
                    "limit": page,
                    "offset": offset,
                },
            )
            batch = data.get("data") or []
            collected.extend(batch)
            if len(batch) < page:
                # Short page: the API had nothing more to give.
                return collected, False
            offset += page
        # We filled the cap without ever seeing a short page.
        return collected, True

    def get_citations(self, paper_id: str, limit: int = DEFAULT_MAX_EDGES_PER_PAPER
                      ) -> tuple[list[dict], bool]:
        """Who cites `paper_id`, with per-edge metadata. See _paged_edges."""
        return self._paged_edges(paper_id, "citations", limit)

    def get_references(self, paper_id: str, limit: int = DEFAULT_MAX_EDGES_PER_PAPER
                       ) -> tuple[list[dict], bool]:
        """What `paper_id` cites, with per-edge metadata. See _paged_edges."""
        return self._paged_edges(paper_id, "references", limit)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def empty_cache() -> dict:
    return {"meta": {}, "papers": {}, "neighbors": {}}


def load_cache(path: Path) -> dict:
    """Read the cache, tolerating a missing or empty file."""
    if not path.exists():
        return empty_cache()
    with path.open(encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return empty_cache()
    data = json.loads(text)
    for key in ("papers", "neighbors"):
        data.setdefault(key, {})
    data.setdefault("meta", {})
    return data


def save_cache(path: Path, cache: dict) -> None:
    """Write the cache atomically so an interrupted run cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def has_metadata(rec: dict) -> bool:
    """A cached paper has metadata once the batch endpoint gave us a title."""
    return bool(rec.get("title"))


def has_edges(rec: dict) -> bool:
    """Citation edges are done only once we have actually asked for them.

    This is an explicit flag rather than "the edge maps exist" because a paper
    fetched with --no-edges also has (empty) edge maps, and must still be
    eligible for edge fetching on a later run.
    """
    return rec.get("edges_fetched") is True


def paper_is_complete(rec: dict) -> bool:
    """A cached paper is complete once it has metadata *and* both edge maps."""
    return has_metadata(rec) and has_edges(rec)


# --------------------------------------------------------------------------
# fetch steps
# --------------------------------------------------------------------------
def collect_candidates(client: S2Client, cfg: dict, cache: dict,
                       want: int, log) -> list[str]:
    """Page through the seed queries and return up to `want` new paperIds.

    Papers already in the cache are skipped so that re-runs are cheap, and the
    queries are rotated round-robin so no single seed dominates the corpus.
    """
    known = set(cache["papers"])
    candidates: list[str] = []
    seen: set[str] = set()

    per_query_target = max(1, -(-want // max(1, len(cfg["seed_queries"]))))
    page = cfg["search_page_size"]
    offset = 0

    while len(candidates) < want:
        progressed = False
        for query in cfg["seed_queries"]:
            if len(candidates) >= want:
                break
            batch = client.search_papers(
                query,
                limit=page,
                offset=offset,
                year_from=cfg["year_from"],
                year_to=cfg["year_to"],
            )
            if not batch:
                log(f"  no more results for {query!r} at offset {offset}")
                continue
            progressed = True
            added_here = 0
            for rec in batch:
                pid = rec.get("paperId")
                if not pid or pid in seen or pid in known:
                    continue
                seen.add(pid)
                candidates.append(pid)
                added_here += 1
                if added_here >= per_query_target or len(candidates) >= want:
                    break
            log(f"  query {query!r} offset {offset}: +{added_here} "
                f"(total {len(candidates)}/{want})")

        if not progressed:
            break
        offset += page
        if offset >= 1000:  # S2 will not page past 1000 results
            log("  reached the S2 search paging limit (offset 1000)")
            break

    return candidates[:want]


def fetch_metadata(client: S2Client, paper_ids: list[str], cache: dict,
                   refresh: bool, log) -> int:
    """Fill in title/abstract/year/... for the given ids via /paper/batch.

    Deliberately does not create empty edge maps for brand-new papers: those
    must stay absent so that fetch_edges() knows they still need fetching.
    """
    fetched = 0
    for start in range(0, len(paper_ids), MAX_BATCH_IDS):
        chunk = paper_ids[start:start + MAX_BATCH_IDS]
        log(f"  batch metadata for {len(chunk)} paper(s)")
        for rec in client.get_batch(chunk):
            pid = rec.get("paperId")
            if not pid:
                continue
            existing = cache["papers"].get(pid, {})
            if refresh or not has_metadata(existing):
                if existing:
                    # Preserve any edges already collected for this paper.
                    rec["references"] = existing.get("references", {})
                    rec["citations"] = existing.get("citations", {})
                    if has_edges(existing):
                        rec["edges_fetched"] = True
                cache["papers"][pid] = rec
                fetched += 1
    return fetched


def _absorb_edge(cache: dict, edge: dict, other_key: str) -> dict | None:
    """Record one citation edge.

    In-corpus papers live in `papers`; papers seen only at the far end of an edge
    are kept in `neighbors` as partial records so no API data is thrown away.
    Returns the normalised edge metadata, or None if the edge has no paperId.
    """
    other = edge.get(other_key) or {}
    pid = other.get("paperId")
    if not pid:
        return None
    if pid not in cache["papers"] and pid not in cache["neighbors"]:
        cache["neighbors"][pid] = {k: v for k, v in other.items() if k in PAPER_FIELDS}
    return {
        "isInfluential": bool(edge.get("isInfluential")),
        "contexts": edge.get("contexts") or [],
        "intents": edge.get("intents") or [],
    }


def fetch_edges(client: S2Client, paper_ids: list[str], cache: dict, log,
                max_edges_per_paper: int = DEFAULT_MAX_EDGES_PER_PAPER
                ) -> tuple[int, list[dict]]:
    """Fetch citations + references for every paper that has not got them yet.

    Returns (edges_collected, truncations) where truncations is a list of
    {"paperId", "title", "kind", "kept"} records for every endpoint that hit
    max_edges_per_paper, so the caller can record the limitation in meta.
    """
    todo = [pid for pid in paper_ids if not has_edges(cache["papers"].get(pid, {}))]
    if not todo:
        log("  all edge sets already cached")
        return 0, []

    log(f"  fetching citations/references for {len(todo)} paper(s) "
        f"(cap {max_edges_per_paper} edges per paper per direction)")
    collected = 0
    truncations: list[dict] = []

    for i, pid in enumerate(todo, start=1):
        rec = cache["papers"].setdefault(pid, {"paperId": pid})

        for kind, wrapper in (("citations", "citingPaper"), ("references", "citedPaper")):
            edges, truncated = getattr(client, f"get_{kind}")(pid, max_edges_per_paper)
            for edge in edges:
                meta = _absorb_edge(cache, edge, wrapper)
                if meta is None:
                    continue
                target = edge[wrapper]["paperId"]
                rec.setdefault(kind, {})[target] = meta
                collected += 1
            if truncated:
                truncations.append({
                    "paperId": pid,
                    "title": rec.get("title"),
                    "kind": kind,
                    "kept": len(edges),
                })

        # An empty edge map is a real answer ("this paper cites nothing"), so the
        # keys must exist and the paper must be marked done.
        rec.setdefault("citations", {})
        rec.setdefault("references", {})
        rec["edges_fetched"] = True

        if i % 10 == 0 or i == len(todo):
            log(f"    edges {i}/{len(todo)}")

    if truncations:
        log(f"  WARNING: {len(truncations)} endpoint(s) hit the "
            f"{max_edges_per_paper}-edge cap and were truncated; "
            f"recorded in meta.edges_truncated")

    return collected, truncations


# --------------------------------------------------------------------------
# cache validation
# --------------------------------------------------------------------------
def validate_cache(path: Path) -> dict:
    """Inspect the cache and report its shape. Reads only the cache, no config.

    Returns a dict of findings; the caller prints it.
    """
    cache = load_cache(path)
    papers = cache.get("papers", {})
    neighbors = cache.get("neighbors", {})
    meta = cache.get("meta", {})

    years = [p["year"] for p in papers.values() if isinstance(p.get("year"), int)]
    missing_abstract = [pid for pid, p in papers.items()
                        if not (p.get("abstract") or "").strip()]
    zero_edges = [pid for pid, p in papers.items()
                  if not p.get("citations") and not p.get("references")]

    # Unique citation edges, as (citer_id, cited_id). Every edge appears twice in
    # the cache -- once as P.references[R], once as R.citations[P] -- so both
    # sides are read and deduplicated. Reading only `references` would miss
    # edges whose citer is outside the corpus.
    edges: set[tuple[str, str]] = set()
    for pid, p in papers.items():
        for cited_id in (p.get("references") or {}):
            edges.add((pid, cited_id))
        for citer_id in (p.get("citations") or {}):
            edges.add((citer_id, pid))

    internal = sum(1 for a, b in edges if a in papers and b in papers)
    external = len(edges) - internal

    truncations = meta.get("edges_truncated") or []
    return {
        "path": path,
        "is_fixture": meta.get("is_fixture"),
        "source": meta.get("source"),
        "fetched_at": meta.get("fetched_at"),
        "papers": len(papers),
        "neighbors": len(neighbors),
        "missing_abstract": len(missing_abstract),
        "missing_abstract_ids": missing_abstract,
        "zero_edges": len(zero_edges),
        "zero_edges_ids": zero_edges,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "years_missing": sum(1 for p in papers.values()
                             if not isinstance(p.get("year"), int)),
        "internal_edges": internal,
        "external_edges": external,
        "total_edges": len(edges),
        "edges_truncated": len(truncations),
    }


def print_validation(report: dict, stream=sys.stdout) -> None:
    """Print a cache validation report in a scannable form."""
    def line(label, value):
        stream.write(f"  {label:<34} {value}\n")

    fixture = report["is_fixture"]
    fixture_txt = ("yes -- OFFLINE TEST FIXTURE, not real API output"
                   if fixture else "no") if fixture is not None else "unknown"

    stream.write("=" * 68 + "\n")
    stream.write("Cache validation: " + str(report["path"]) + "\n")
    stream.write("=" * 68 + "\n")
    line("is_fixture", fixture_txt)
    line("source", report["source"] or "unknown")
    line("fetched_at", report["fetched_at"] or "unknown")
    line("papers", report["papers"])
    line("neighbors (edge endpoints only)", report["neighbors"])
    line("papers missing an abstract", report["missing_abstract"])
    line("papers with zero edges", report["zero_edges"])
    if report["year_min"] is None:
        line("year range", "no papers with an int year")
    else:
        line("year range", f"{report['year_min']} - {report['year_max']}")
    line("papers with no year", report["years_missing"])
    line("citation edges, both ends in corpus", report["internal_edges"])
    line("citation edges, one end outside", report["external_edges"])
    line("citation edges, total (deduplicated)", report["total_edges"])
    line("endpoints truncated by the cap", report["edges_truncated"])
    stream.write("\n")

    def show_ids(label, ids):
        if ids:
            shown = ", ".join(ids[:5]) + (" ..." if len(ids) > 5 else "")
            stream.write(f"  {label}: {shown}\n")

    show_ids("papers without an abstract", report["missing_abstract_ids"])
    show_ids("papers without any edge", report["zero_edges_ids"])
    stream.write("\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch RAG papers from Semantic Scholar into data/raw_papers.json",
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="config file containing the `fetch:` section")
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE,
                    help="cache file to read and write")
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch at most this many NEW papers (overrides num_papers)")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch metadata for papers already in the cache")
    ap.add_argument("--reset", action="store_true",
                    help="discard the existing cache first")
    ap.add_argument("--no-edges", action="store_true",
                    help="skip citations/references (much faster, no graph edges)")
    ap.add_argument("--validate", action="store_true",
                    help="report on the cache and exit; makes no API calls and "
                         "does not read the config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and make no API calls")
    args = ap.parse_args(list(argv) if argv is not None else None)

    def log(msg: str = "") -> None:
        print(msg, file=sys.stderr, flush=True)

    # --validate inspects the cache only: no config, no API. It must work even
    # when the config template is still full of TODO placeholders.
    if args.validate:
        if not args.out.exists():
            print(f"error: no cache at {args.out}", file=sys.stderr)
            return 1
        print_validation(validate_cache(args.out))
        return 0

    cfg = load_fetch_config(args.config)
    api_key = os.environ.get("S2_API_KEY") or None

    cache = empty_cache() if args.reset else load_cache(args.out)
    existing = len(cache["papers"])
    target_total = cfg["num_papers"]
    want = target_total - existing
    if args.limit is not None:
        want = min(want, max(0, args.limit)) if existing else max(0, args.limit)

    log("=" * 68)
    log("Semantic Scholar fetch")
    log("=" * 68)
    log(f"config        : {args.config}")
    log(f"cache         : {args.out} ({existing} paper(s) already cached)")
    log(f"seed queries  : {len(cfg['seed_queries'])} -> {cfg['seed_queries']}")
    log(f"year range    : {cfg['year_from'] or 'any'} - {cfg['year_to'] or 'any'}")
    log(f"target papers : {target_total} (fetching {max(0, want)} new)")
    log(f"api key       : {'yes (S2_API_KEY)' if api_key else 'no (anonymous pool)'}")
    log(f"base url      : {os.environ.get('S2_BASE_URL', 'https://api.semanticscholar.org')}")
    log(f"fetch edges   : {cfg['fetch_edges'] and not args.no_edges}")
    log()

    edges_wanted = cfg["fetch_edges"] and not args.no_edges
    edges_pending = [pid for pid in cache["papers"]
                     if not has_edges(cache["papers"][pid])]

    # Three reasons to keep going: new papers are wanted, --refresh was asked
    # for, or an earlier --no-edges run left citation edges unfetched.
    if want <= 0 and not args.refresh and not edges_pending:
        log("Nothing to do: the cache already meets the target paper count "
            "and every paper has its citation edges.")
        log("Use --refresh to refetch metadata, or --reset to start over.")
        return 0

    if args.dry_run:
        log("--dry-run: no API calls made.")
        return 0

    client = S2Client(
        api_key=api_key,
        delay=cfg["request_delay_seconds"],
        max_retries=cfg["max_retries"],
    )

    n_meta = 0
    if want <= 0:
        # Cache is full: no search, no metadata fetch, just the outstanding edges.
        log("[1/3] cache already at target paper count; skipping search")
        log(f"[2/3] skipping metadata ({existing} paper(s) cached)")
        candidates = []
    else:
        log("[1/3] searching seed queries")
        candidates = collect_candidates(client, cfg, cache, want, log)
        if not candidates:
            log("No new papers found for the seed queries. "
                "Widen fetch.year_from / fetch.year_to or add seed queries.")
            return 1
        log(f"[2/3] fetching metadata for {len(candidates)} paper(s)")
        n_meta = fetch_metadata(client, candidates, cache, args.refresh, log)

    log("[3/3] fetching citation edges")
    n_edges = 0
    truncations: list[dict] = []
    if edges_wanted:
        n_edges, truncations = fetch_edges(
            client, list(cache["papers"]), cache, log,
            max_edges_per_paper=cfg["max_edges_per_paper"],
        )
    else:
        # Leave `edges_fetched` unset so a later run without --no-edges fills these in.
        for pid in cache["papers"]:
            cache["papers"][pid].setdefault("citations", {})
            cache["papers"][pid].setdefault("references", {})

    cache["meta"] = {
        "source": "semantic-scholar-graph-api",
        "is_fixture": False,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_key_used": bool(api_key),
        "seed_queries": cfg["seed_queries"],
        "year_from": cfg["year_from"],
        "year_to": cfg["year_to"],
        "api_calls": client.call_count,
        "max_edges_per_paper": cfg["max_edges_per_paper"],
        # Known limitation: endpoints that hit the cap. The API gives no total,
        # so these papers have AT LEAST this many edges; the rest are not fetched.
        "edges_truncated": truncations,
        "edges_truncated_count": len(truncations),
        "note": "Cached Semantic Scholar Graph API output. Consumed by build_graph.py.",
    }

    save_cache(args.out, cache)

    log()
    log(f"papers cached : {len(cache['papers'])}")
    log(f"neighbours    : {len(cache['neighbors'])} (papers seen only at the end of an edge)")
    log(f"new metadata  : {n_meta}")
    log(f"new edges     : {n_edges}")
    log(f"api calls     : {client.call_count}")
    if truncations:
        log(f"TRUNCATED     : {len(truncations)} endpoint(s) hit the "
            f"{cfg['max_edges_per_paper']}-edge cap "
            f"(recorded in meta.edges_truncated)")
    log(f"wrote         : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
