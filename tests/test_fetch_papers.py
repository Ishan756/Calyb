"""Offline test for src/fetch_papers.py.

The real Semantic Scholar API cannot be hit from a test (anonymous rate limit),
so this spins up a tiny local mock that mimics the four endpoints the fetcher
uses, then drives the real fetch_papers code against it via S2_BASE_URL.

Run:  python -m pytest tests/test_fetch_papers.py -v
  or: python tests/test_fetch_papers.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fetch_papers as fp  # noqa: E402

# --- a fake corpus: 5 papers, A<->B<->C citation chain plus a lone paper ------
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
}

# Which papers cite which, as cited_id -> [(citing_id, isInfluential)].
CITES = {"p1": [("p3", True)], "p2": [("p1", False), ("p4", True)], "p4": [("p3", False)]}


def citations_of(pid: str):
    """(citing_id, isInfluential) pairs for papers that cite `pid`."""
    return [(citing, infl)
            for cited, lst in CITES.items()
            for citing, infl in lst if cited == pid]


def references_of(pid: str):
    """(cited_id, isInfluential) pairs for papers that `pid` cites."""
    return list(CITES.get(pid, []))


TOTAL_REFS = sum(len(references_of(p)) for p in MOCK_PAPERS)
TOTAL_CITS = sum(len(citations_of(p)) for p in MOCK_PAPERS)


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

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        fields = (qs.get("fields") or [""])[0].split(",")
        MockS2Handler.calls.append(f"GET {url.path}")

        # /paper/search -- honour limit/offset so pagination can be tested
        if url.path == "/graph/v1/paper/search":
            limit = int((qs.get("limit") or ["100"])[0])
            offset = int((qs.get("offset") or ["0"])[0])
            everything = list(MOCK_PAPERS.values())
            page = everything[offset:offset + limit]
            return self._send({
                "total": len(everything), "offset": offset, "next": offset + len(page),
                "data": [{k: p[k] for k in fields if k in p} for p in page],
            })

        # /paper/{id}/citations and /paper/{id}/references
        if url.path.startswith("/graph/v1/paper/") and (
                url.path.endswith("/citations") or url.path.endswith("/references")):
            pid = url.path.split("/paper/")[1].split("/")[0]
            if pid not in MOCK_PAPERS:
                return self._send({"error": "not found"}, status=404)

            kind = "citations" if url.path.endswith("/citations") else "references"
            pairs = citations_of(pid) if kind == "citations" else references_of(pid)
            wrapper = "citingPaper" if kind == "citations" else "citedPaper"

            data = []
            for other_id, is_influential in pairs:
                other = MOCK_PAPERS[other_id]
                data.append({
                    wrapper: {k: other[k] for k in fields if k in other},
                    "isInfluential": is_influential,
                    "contexts": [f"we build on {other['title']}"],
                    "intents": ["methodology"],
                })
            return self._send({"data": data})

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


def run_fetch(tmp_path: Path, *cli_args):
    """Invoke the real fetch_papers.main() against the mock server."""
    config = tmp_path / "schema.yaml"
    config.write_text(
        "fetch:\n"
        "  seed_queries: ['retrieval augmented generation']\n"
        "  num_papers: 5\n"
        "  year_from: 2018\n"
        "  year_to: 2025\n"
        "  request_delay_seconds: 0\n"
        "  max_retries: 2\n"
    )
    out = tmp_path / "raw_papers.json"
    argv = ["--config", str(config), "--out", str(out), *cli_args]
    rc = fp.main(argv)
    cache = json.loads(out.read_text()) if out.exists() else {}
    return rc, cache, out


def main() -> int:
    import tempfile

    server = HTTPServer(("127.0.0.1", 0), MockS2Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import os
    os.environ["S2_BASE_URL"] = f"http://127.0.0.1:{port}"
    print(f"mock S2 API on 127.0.0.1:{port}\n")

    failures = []

    def check(label, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f" -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- 1. cold fetch --------------------------------------------------
        print("test_fetch_cold_start")
        MockS2Handler.calls.clear()
        rc, cache, out = run_fetch(tmp)
        check("exit code 0", rc == 0, f"rc={rc}")
        check("all 5 papers cached", len(cache.get("papers", {})) == 5,
              f"got {len(cache.get('papers', {}))}")
        check("meta records real source", cache["meta"]["source"] == "semantic-scholar-graph-api")
        check("meta not marked fixture", cache["meta"]["is_fixture"] is False)
        check("every paper has both edge maps",
              all("references" in p and "citations" in p for p in cache["papers"].values()))
        check("some edges were collected",
              sum(len(p["references"]) for p in cache["papers"].values()) == TOTAL_REFS,
              f"refs={sum(len(p['references']) for p in cache['papers'].values())} "
              f"expected {TOTAL_REFS}")
        check("isInfluential survived the round trip",
              any(e["isInfluential"] for p in cache["papers"].values()
                  for e in p["references"].values()))
        check("contexts/intents survived the round trip",
              all(e["contexts"] and e["intents"] for p in cache["papers"].values()
                  for e in list(p["references"].values()) + list(p["citations"].values())))
        check("citation-side edges also collected",
              sum(len(p["citations"]) for p in cache["papers"].values()) == TOTAL_CITS,
              f"cits={sum(len(p['citations']) for p in cache['papers'].values())} "
              f"expected {TOTAL_CITS}")
        check("p1 is cited by p3 exactly once",
              list(cache["papers"]["p1"]["citations"]) == ["p3"])

        # ---- 2. re-run is a no-op -----------------------------------------
        print("\ntest_fetch_is_idempotent")
        calls_before = len(MockS2Handler.calls)
        rc, cache2, _ = run_fetch(tmp)
        check("exit code 0", rc == 0)
        check("paper count unchanged", len(cache2["papers"]) == 5)
        check("no new API calls", len(MockS2Handler.calls) == calls_before,
              f"{len(MockS2Handler.calls) - calls_before} extra calls")

        # ---- 3. --refresh re-fetches metadata but reuses edges ------------
        print("\ntest_fetch_refresh")
        MockS2Handler.calls.clear()
        rc, cache3, _ = run_fetch(tmp, "--refresh")
        check("exit code 0", rc == 0)
        check("still 5 papers", len(cache3["papers"]) == 5)
        check("edges preserved across refresh",
              sum(len(p["references"]) for p in cache3["papers"].values()) == TOTAL_REFS)

        # ---- 4. --no-edges -------------------------------------------------
        print("\ntest_fetch_no_edges")
        MockS2Handler.calls.clear()
        rc, cache4, _ = run_fetch(tmp, "--reset", "--no-edges")
        check("exit code 0", rc == 0)
        check("5 papers", len(cache4["papers"]) == 5)
        check("no edges fetched",
              all(not p["references"] and not p["citations"]
                  for p in cache4["papers"].values()))
        check("no citation endpoints called",
              not any("/citations" in c for c in MockS2Handler.calls))

        # ---- 5. --reset --limit -------------------------------------------
        print("\ntest_fetch_reset_and_limit")
        rc, cache5, _ = run_fetch(tmp, "--reset", "--limit", "2")
        check("exit code 0", rc == 0)
        check("exactly 2 papers fetched", len(cache5["papers"]) == 2,
              f"got {len(cache5['papers'])}")

    server.shutdown()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all fetch_papers tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
