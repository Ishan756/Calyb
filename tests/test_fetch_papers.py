"""Offline tests for src/fetch_papers.py.

The real Semantic Scholar API cannot be hit from a test (anonymous rate limit),
so this spins up a tiny local mock of the four endpoints the fetcher uses, then
drives the real fetch_papers code against it via S2_BASE_URL.

The mock honours limit/offset on every endpoint, which is what makes the
pagination tests meaningful.

Run:  python -m pytest tests/test_fetch_papers.py -v
  or: python tests/test_fetch_papers.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fetch_papers as fp  # noqa: E402

# --- a fake corpus: 5 papers with a p1<-p3 / p2<-p1,p4 / p4<-p3 chain, plus
# --- p6, a "hub" paper cited by 250 synthetic papers so paging is exercised.
MOCK_PAPERS = {
    "p1": {"paperId": "p1", "title": "Retrieval Augmented Generation", "abstract": "RAG",
           "year": 2020, "venue": "NeurIPS", "authors": [{"authorId": "a1", "name": "A"}],
           "citationCount": 10, "externalIds": {"DOI": "10.1/x"}},
    "p2": {"paperId": "p2", "title": "Dense Passage Retrieval", "abstract": "DPR",
           "year": 2020, "venue": "EMNLP", "authors": [{"authorId": "a2", "name": "B"}],
           "citationCount": 20, "externalIds": {}},
    "p3": {"paperId": "p3", "title": "Self RAG", "abstract": "critique tokens",
           "year": 2024, "venue": "ICLR", "authors": [{"authorId": "a3", "name": "C"}],
           "citationCount": 5, "externalIds": {}},
    "p4": {"paperId": "p4", "title": "Active RAG", "abstract": "FLARE",
           "year": 2023, "venue": "EMNLP", "authors": [{"authorId": "a4", "name": "D"}],
           "citationCount": 3, "externalIds": {}},
    "p5": {"paperId": "p5", "title": "Unrelated Paper", "abstract": "nothing to do with rag",
           "year": 2019, "venue": "ICML", "authors": [{"authorId": "a5", "name": "E"}],
           "citationCount": 1, "externalIds": {}},
    "p6": {"paperId": "p6", "title": "Hub Survey", "abstract": "a survey of surveys",
           "year": 2022, "venue": "TACL", "authors": [{"authorId": "a6", "name": "F"}],
           "citationCount": 250, "externalIds": {}},
}

HUB = "p6"
HUB_CITERS = 250
SYNTH = {f"c{i}": {"paperId": f"c{i}", "title": f"Synthetic citer {i}", "year": 2021}
         for i in range(1, HUB_CITERS + 1)}

# Every paper the mock can serve as an edge endpoint.
ALL_PAPERS = {**MOCK_PAPERS, **SYNTH}

# Which paper cites which, as citer_id -> [(cited_id, isInfluential)].
CITES = {
    "p1": [("p2", False)],
    "p3": [("p1", True), ("p4", False)],
    "p4": [("p2", True)],
}
# The hub's 250 citers live outside the corpus and are only ever edge endpoints.
for _i in range(1, HUB_CITERS + 1):
    CITES[f"c{_i}"] = [(HUB, False)]


def references_of(pid: str):
    """(cited_id, isInfluential) pairs for papers that `pid` cites."""
    return list(CITES.get(pid, []))


def citations_of(pid: str):
    """(citing_id, isInfluential) pairs for papers that cite `pid`."""
    return [(citer, infl)
            for citer, lst in CITES.items()
            for cited, infl in lst if cited == pid]


TOTAL_REFS = sum(len(references_of(p)) for p in MOCK_PAPERS)
TOTAL_CITS = sum(len(citations_of(p)) for p in MOCK_PAPERS)
CORPUS_SIZE = len(MOCK_PAPERS)


class MockS2Handler(BaseHTTPRequestHandler):
    calls: list[str] = []

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _page(qs, rows):
        """Apply the request's limit/offset to a row list."""
        limit = int((qs.get("limit") or ["100"])[0])
        offset = int((qs.get("offset") or ["0"])[0])
        return rows[offset:offset + limit], offset, limit

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        fields = (qs.get("fields") or [""])[0].split(",")
        MockS2Handler.calls.append(f"GET {url.path}")

        # /paper/search
        if url.path == "/graph/v1/paper/search":
            page, offset, _ = self._page(qs, list(MOCK_PAPERS.values()))
            return self._send({
                "total": len(MOCK_PAPERS), "offset": offset, "next": offset + len(page),
                "data": [{k: p[k] for k in fields if k in p} for p in page],
            })

        # /paper/{id}/citations and /paper/{id}/references -- both paginated
        if url.path.startswith("/graph/v1/paper/") and (
                url.path.endswith("/citations") or url.path.endswith("/references")):
            pid = url.path.split("/paper/")[1].split("/")[0]
            if pid not in MOCK_PAPERS:
                return self._send({"error": "not found"}, status=404)

            kind = "citations" if url.path.endswith("/citations") else "references"
            pairs = citations_of(pid) if kind == "citations" else references_of(pid)
            wrapper = "citingPaper" if kind == "citations" else "citedPaper"

            rows = []
            for other_id, is_influential in pairs:
                other = ALL_PAPERS[other_id]
                rows.append({
                    wrapper: {k: other[k] for k in fields if k in other},
                    "isInfluential": is_influential,
                    "contexts": [f"we build on {other['title']}"],
                    "intents": ["methodology"],
                })

            page, offset, _ = self._page(qs, rows)
            return self._send({"data": page, "offset": offset, "next": offset + len(page)})

        return self._send({"error": "not found"}, status=404)

    def do_POST(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        fields = (qs.get("fields") or [""])[0].split(",")
        MockS2Handler.calls.append(f"POST {url.path}")
        if url.path != "/graph/v1/paper/batch":
            return self._send({"error": "not found"}, status=404)
        length = int(self.headers.get("Content-Length", 0))
        ids = json.loads(self.rfile.read(length) or b"{}").get("ids", [])
        data = []
        for pid in ids:
            paper = MOCK_PAPERS.get(pid)
            data.append({k: paper[k] for k in fields if k in paper} if paper else None)
        return self._send({"data": data})


def write_config(tmp_path: Path, **overrides) -> Path:
    """Write a minimal, fully-filled `fetch:` config for the mock run."""
    cfg = {
        "seed_queries": ["retrieval augmented generation"],
        "num_papers": CORPUS_SIZE,
        "year_from": 2018,
        "year_to": 2025,
        "request_delay_seconds": 0,
        "max_retries": 2,
    }
    cfg.update(overrides)
    lines = ["fetch:"]
    lines.append("  seed_queries:")
    lines += [f"    - {q!r}" for q in cfg["seed_queries"]]
    for key in ("num_papers", "year_from", "year_to",
                "request_delay_seconds", "max_retries", "max_edges_per_paper"):
        if key in cfg:
            lines.append(f"  {key}: {cfg[key]}")
    path = tmp_path / "schema.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_fetch(tmp_path: Path, *cli_args, **config_overrides):
    """Invoke the real fetch_papers.main() against the mock server."""
    config = write_config(tmp_path, **config_overrides)
    out = tmp_path / "raw_papers.json"
    rc = fp.main(["--config", str(config), "--out", str(out), *cli_args])
    cache = json.loads(out.read_text()) if out.exists() else {}
    return rc, cache, out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_fetch_cold_start():
    MockS2Handler.calls.clear()
    with TempDir() as td:
        rc, cache, _ = run_fetch(Path(td), max_edges_per_paper=500)
        assert rc == 0
        assert len(cache["papers"]) == CORPUS_SIZE
        assert cache["meta"]["source"] == "semantic-scholar-graph-api"
        assert cache["meta"]["is_fixture"] is False
        assert all("references" in p and "citations" in p
                   for p in cache["papers"].values())
        assert sum(len(p["references"]) for p in cache["papers"].values()) == TOTAL_REFS
        assert sum(len(p["citations"]) for p in cache["papers"].values()) == TOTAL_CITS
        assert any(e["isInfluential"] for p in cache["papers"].values()
                   for e in p["references"].values())
        assert all(e["contexts"] and e["intents"] for p in cache["papers"].values()
                   for e in list(p["references"].values()) + list(p["citations"].values()))
        # p3 cites p1 (influential) and p4; nothing cites p3.
        assert sorted(cache["papers"]["p3"]["references"]) == ["p1", "p4"]
        assert cache["papers"]["p3"]["references"]["p1"]["isInfluential"] is True
        assert cache["papers"]["p3"]["citations"] == {}
        # Out-of-corpus edge endpoints land in `neighbors`, not `papers`.
        assert set(cache["neighbors"]) == set(SYNTH)


def test_fetch_is_idempotent():
    with TempDir() as td:
        run_fetch(Path(td))
        before = len(MockS2Handler.calls)
        rc, cache, _ = run_fetch(Path(td))
        assert rc == 0
        assert len(cache["papers"]) == CORPUS_SIZE
        assert len(MockS2Handler.calls) == before, "second run made API calls"


def test_fetch_refresh_keeps_edges():
    with TempDir() as td:
        run_fetch(Path(td), max_edges_per_paper=500)
        rc, cache, _ = run_fetch(Path(td), "--refresh", max_edges_per_paper=500)
        assert rc == 0
        assert len(cache["papers"]) == CORPUS_SIZE
        assert sum(len(p["references"]) for p in cache["papers"].values()) == TOTAL_REFS
        assert sum(len(p["citations"]) for p in cache["papers"].values()) == TOTAL_CITS


def test_fetch_no_edges():
    with TempDir() as td:
        MockS2Handler.calls.clear()
        rc, cache, _ = run_fetch(Path(td), "--reset", "--no-edges")
        assert rc == 0
        assert len(cache["papers"]) == CORPUS_SIZE
        assert all(not p["references"] and not p["citations"]
                   for p in cache["papers"].values())
        assert not any("/citations" in c for c in MockS2Handler.calls)
        # ...and a later run without --no-edges must still fetch them.
        rc, cache, _ = run_fetch(Path(td))
        assert sum(len(p["references"]) for p in cache["papers"].values()) == TOTAL_REFS


def test_fetch_reset_and_limit():
    with TempDir() as td:
        rc, cache, _ = run_fetch(Path(td), "--reset", "--limit", "2")
        assert rc == 0
        assert len(cache["papers"]) == 2


# --- pagination ------------------------------------------------------------
def test_citations_are_paginated():
    """250 citation edges must come back complete, across several pages."""
    MockS2Handler.calls.clear()
    with TempDir() as td:
        rc, cache, _ = run_fetch(Path(td), "--reset", max_edges_per_paper=500)
        assert rc == 0
        hub = cache["papers"][HUB]
        assert len(hub["citations"]) == HUB_CITERS
        # 250 edges at 100 per page = 3 GETs against the hub.
        hub_calls = [c for c in MockS2Handler.calls if f"/paper/{HUB}/citations" in c]
        assert len(hub_calls) == 3, hub_calls
        # The default cap was not reached, so nothing is flagged truncated.
        assert cache["meta"]["edges_truncated_count"] == 0
        assert cache["meta"]["edges_truncated"] == []


def test_edge_cap_truncates_and_records():
    with TempDir() as td:
        rc, cache, _ = run_fetch(Path(td), "--reset", max_edges_per_paper=120)
        assert rc == 0
        hub = cache["papers"][HUB]
        assert len(hub["citations"]) == 120, "cap not respected"
        trunc = cache["meta"]["edges_truncated"]
        assert len(trunc) == 1, trunc
        assert trunc[0]["paperId"] == HUB
        assert trunc[0]["kind"] == "citations"
        assert trunc[0]["kept"] == 120
        assert cache["meta"]["max_edges_per_paper"] == 120
        assert cache["meta"]["edges_truncated_count"] == 1


def test_cap_larger_than_available_is_not_truncated():
    """A cap above the true edge count must NOT be reported as truncated."""
    with TempDir() as td:
        rc, cache, _ = run_fetch(Path(td), "--reset", max_edges_per_paper=10_000)
        assert rc == 0
        assert len(cache["papers"][HUB]["citations"]) == HUB_CITERS
        assert cache["meta"]["edges_truncated_count"] == 0


def test_shipped_default_caps_at_100_and_records_it():
    """With no max_edges_per_paper in the config, the default cap applies."""
    with TempDir() as td:
        rc, cache, _ = run_fetch(Path(td), "--reset")
        assert rc == 0
        hub = cache["papers"][HUB]
        assert len(hub["citations"]) == fp.DEFAULT_MAX_EDGES_PER_PAPER == 100
        assert cache["meta"]["max_edges_per_paper"] == 100
        assert cache["meta"]["edges_truncated_count"] == 1
        assert cache["meta"]["edges_truncated"][0]["paperId"] == HUB


def test_default_cap_is_100():
    assert fp.DEFAULT_MAX_EDGES_PER_PAPER == 100


# --- --validate ------------------------------------------------------------
def test_validate_reports_cache(tmp_path=None):
    with TempDir() as td:
        rc, cache, out = run_fetch(Path(td), "--reset", max_edges_per_paper=500)
        assert rc == 0

        rc = fp.main(["--out", str(out), "--validate"])
        assert rc == 0

        report = fp.validate_cache(out)
        assert report["is_fixture"] is False
        assert report["papers"] == CORPUS_SIZE
        assert report["missing_abstract"] == 0
        assert report["year_min"] == 2019
        assert report["year_max"] == 2024
        # Unique edges, deduplicated across the references/citations mirror.
        assert report["internal_edges"] == TOTAL_REFS
        assert report["external_edges"] == HUB_CITERS
        assert report["total_edges"] == TOTAL_REFS + HUB_CITERS
        # p5 is deliberately isolated.
        assert report["zero_edges"] == 1
        assert report["zero_edges_ids"] == ["p5"]
        assert report["edges_truncated"] == 0


def test_validate_needs_no_config():
    """--validate must work even when the config is full of TODO placeholders."""
    with TempDir() as td:
        td = Path(td)
        run_fetch(td, "--reset")
        out = td / "raw_papers.json"
        broken = td / "schema.yaml"
        broken.write_text("fetch:\n  num_papers: TODO\n")
        rc = fp.main(["--config", str(broken), "--out", str(out), "--validate"])
        assert rc == 0
        assert fp.validate_cache(out)["papers"] == CORPUS_SIZE


def test_validate_missing_cache(tmp_path=None):
    with TempDir() as td:
        rc = fp.main(["--out", str(Path(td) / "nope.json"), "--validate"])
        assert rc == 1


# --- runner ----------------------------------------------------------------
class TempDir:
    """Minimal stand-in for tempfile.TemporaryDirectory as a context manager."""

    def __enter__(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        return self._td.name

    def __exit__(self, *exc):
        self._td.cleanup()
        return False


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), MockS2Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["S2_BASE_URL"] = f"http://127.0.0.1:{port}"
    print(f"mock S2 API on 127.0.0.1:{port}\n")

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for test in tests:
        print(f"{test.__name__} ... ", end="", flush=True)
        try:
            test()
            print("PASS")
        except AssertionError as exc:
            print(f"FAIL -- {exc}")
            failed.append(test.__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR -- {type(exc).__name__}: {exc}")
            failed.append(test.__name__)

    server.shutdown()
    print("\n" + "=" * 60)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
