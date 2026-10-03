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
            results.extend(data.get("data") or [])
        return [r for r in results if r]

    def _edge_fields(self) -> str:
        return ",".join(PAPER_FIELDS + EDGE_EXTRA_FIELDS)

    def get_citations(self, paper_id: str) -> list[dict]:
        """Who cites `paper_id`, with per-edge metadata."""
        data = self.request(
            "GET", f"/paper/{paper_id}/citations",
            params={"fields": self._edge_fields(), "limit": MAX_SEARCH_PAGE},
        )
        return data.get("data") or []

    def get_references(self, paper_id: str) -> list[dict]:
        """What `paper_id` cites, with per-edge metadata."""
        data = self.request(
            "GET", f"/paper/{paper_id}/references",
            params={"fields": self._edge_fields(), "limit": MAX_SEARCH_PAGE},
        )
        return data.get("data") or []


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


def fetch_edges(client: S2Client, paper_ids: list[str], cache: dict, log) -> int:
    """Fetch citations + references for every paper that has not got them yet."""
    todo = [pid for pid in paper_ids if not has_edges(cache["papers"].get(pid, {}))]
    if not todo:
        log("  all edge sets already cached")
        return 0

    log(f"  fetching citations/references for {len(todo)} paper(s)")
    collected = 0
    for i, pid in enumerate(todo, start=1):
        rec = cache["papers"].setdefault(pid, {"paperId": pid})

        for edge in client.get_citations(pid):
            meta = _absorb_edge(cache, edge, "citingPaper")
            if meta is None:
                continue
            target = edge["citingPaper"]["paperId"]
            rec.setdefault("citations", {})[target] = meta
            collected += 1

        for edge in client.get_references(pid):
            meta = _absorb_edge(cache, edge, "citedPaper")
            if meta is None:
                continue
            target = edge["citedPaper"]["paperId"]
            rec.setdefault("references", {})[target] = meta
            collected += 1

        # An empty edge map is a real answer ("this paper cites nothing"), so the
        # keys must exist and the paper must be marked done.
        rec.setdefault("citations", {})
        rec.setdefault("references", {})
        rec["edges_fetched"] = True

        if i % 10 == 0 or i == len(todo):
            log(f"    edges {i}/{len(todo)}")

    return collected


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
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and make no API calls")
    args = ap.parse_args(list(argv) if argv is not None else None)

    def log(msg: str = "") -> None:
        print(msg, file=sys.stderr, flush=True)

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

    if want <= 0 and not (args.refresh and existing):
        log("Nothing to do: the cache already meets the target paper count.")
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

    if want <= 0:
        # --refresh on an already-full cache: refetch metadata, skip the search.
        log("[1/3] --refresh: reusing the cached paper ids, no search needed")
        candidates = list(cache["papers"])
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
    if cfg["fetch_edges"] and not args.no_edges:
        n_edges = fetch_edges(client, list(cache["papers"]), cache, log)
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
        "note": "Cached Semantic Scholar Graph API output. Consumed by build_graph.py.",
    }

    save_cache(args.out, cache)

    log()
    log(f"papers cached : {len(cache['papers'])}")
    log(f"neighbours    : {len(cache['neighbors'])} (papers seen only at the end of an edge)")
    log(f"new metadata  : {n_meta}")
    log(f"new edges     : {n_edges}")
    log(f"api calls     : {client.call_count}")
    log(f"wrote         : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
