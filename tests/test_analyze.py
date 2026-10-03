"""Tests for the input -> reading-order analysis.

Runs standalone (`python tests/test_analyze.py`) and under pytest. Every test
builds a real knowledge state from tests/fixtures/ through build_graph, so the
tests exercise the same code path a user would run.
"""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
FIXTURES = HERE / "fixtures"

for extra in (str(HERE), str(PROJECT_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import analyze as az  # noqa: E402
import build_graph as bg  # noqa: E402


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
@contextlib.contextmanager
def built_state(schema_mutator=None):
    """Build a knowledge state from the fixtures into a temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="calyb-analyze-"))
    try:
        schema = bg.load_yaml(FIXTURES / "schema.yaml")
        if schema_mutator is not None:
            schema_mutator(schema)
        schema_path = tmp / "schema.yaml"
        schema_path.write_text(bg.yaml.safe_dump(schema, sort_keys=False),
                               encoding="utf-8")

        state_path = tmp / "knowledge_state.json"
        rc = bg.main([
            "--schema", str(schema_path),
            "--vocab", str(FIXTURES / "vocab.yaml"),
            "--cache", str(FIXTURES / "raw_papers.json"),
            "--out", str(state_path),
            "--quiet",
        ])
        assert rc == 0, f"fixture build failed with rc={rc}"
        yield state_path, tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# A single shared state, built once at import, for tests that only read it.
# Tests that mutate the schema use built_state() instead, for isolation.
_SHARED_DIR = tempfile.mkdtemp(prefix="calyb-analyze-shared-")
atexit.register(shutil.rmtree, _SHARED_DIR, True)
BUILT_STATE = Path(_SHARED_DIR) / "knowledge_state.json"
_rc = bg.main([
    "--schema", str(FIXTURES / "schema.yaml"),
    "--vocab", str(FIXTURES / "vocab.yaml"),
    "--cache", str(FIXTURES / "raw_papers.json"),
    "--out", str(BUILT_STATE),
    "--quiet",
])
assert _rc == 0, f"shared fixture build failed with rc={_rc}"


def analyse(state_path: Path, text: str, extra_argv: list[str] | None = None):
    """Run analyze's main() and return (rc, report, stdout, stderr)."""
    argv = ["--state", str(state_path), "--text", text, "--json"]
    argv += extra_argv or []
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = az.main(argv)
    stdout = out.getvalue()
    return rc, (json.loads(stdout) if stdout.strip() else None), stdout, err.getvalue()


def signal(report: dict, paper_id: str, name: str) -> dict:
    rec = next(r for r in report["recommendations"] if r["paper_id"] == paper_id)
    return next(s for s in rec["signals"] if s["signal"] == name)


def expect_analyze_error(fn, needle: str) -> None:
    try:
        fn()
    except az.AnalyzeError as exc:
        if needle.lower() not in str(exc).lower():
            raise AssertionError(f"expected {needle!r} in error, got:\n{exc}") from None
        return
    raise AssertionError(f"expected AnalyzeError containing {needle!r}")


# --------------------------------------------------------------------------
# input interpretation
# --------------------------------------------------------------------------
def test_input_matching_reports_configured_vocabulary_terms():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(
            state_path, "I need papers that grow kelp and use net-based farming")
        interp = report["input_interpretation"]
        assert "kelp" in interp["matched_terms"]
        assert "net-based farming" in interp["matched_terms"]
        assert report["input_understood"] is True


def test_input_matching_records_evidence_and_the_condition_that_matched():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "kelp cultivation")
        matches = report["input_interpretation"]["matches"]
        kelp = [m for m in matches if m.get("term") == "kelp"]
        assert kelp, f"expected kelp evidence, got {matches}"
        assert kelp[0]["field"] == "input"
        assert kelp[0]["kind"] == "vocab_term"
        # provenance: which schema condition produced this reading
        assert kelp[0]["condition"].startswith("edge_types[") or \
               kelp[0]["condition"].startswith("reading_order.signals[")


def test_input_matching_covers_regex_conditions_too():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "does vertical farming work")
        assert "vertical" in report["input_interpretation"]["regex_hits"]


def test_input_matching_is_case_insensitive():
    with built_state() as (state_path, _tmp):
        lower = analyse(state_path, "kelp")[1]["input_interpretation"]["matched_terms"]
        upper = analyse(state_path, "KELP Nori")[1]["input_interpretation"]["matched_terms"]
        assert "kelp" in lower and "kelp" in upper
        assert "nori" in upper


def test_off_topic_input_is_reported_as_not_understood():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(
            state_path, "quantum chromodynamics lattice gauge theory")
        interp = report["input_interpretation"]
        assert interp["matched_terms"] == []
        assert interp["regex_hits"] == []
        assert report["input_understood"] is False


# --------------------------------------------------------------------------
# candidate shortlisting
# --------------------------------------------------------------------------
def test_tfidf_only_shortlists_at_most_candidate_pool():
    with built_state() as (state_path, _tmp):
        state = az.load_state(state_path)
        pool = state["schema"]["reading_order"]["candidate_pool"]
        _rc, report, _o, _e = analyse(state_path, "kelp hydroculture")
        assert report["candidate_pool"] == pool
        assert len(report["recommendations"]) <= pool


def test_tfidf_shortlist_puts_lexically_similar_papers_first():
    """The shortlist itself is similarity-ordered; scoring happens afterwards."""
    state = az.load_state(BUILT_STATE)
    paper_type = next(name for name, info in state["node_types"].items()
                      if info["source"] == "Paper")
    papers = [n for n in state["nodes"] if n["type"] == paper_type]
    ranked = az.shortlist("net-based farming of nori in closed systems", papers, 4)

    assert len(ranked) == 4
    sims = [s for _p, s in ranked]
    assert sims == sorted(sims, reverse=True)
    # the lexically closest paper is the one actually about closed-system farming
    assert ranked[0][0]["properties"]["title"] == "Nutrient collapse in closed systems"
    assert ranked[0][1] > 0


def test_tfidf_is_not_used_as_a_score():
    """A high-similarity paper with no signal match must not outrank a scored one."""
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(
            state_path, "net-based farming of nori in closed systems")
        collapse = next(r for r in report["recommendations"]
                        if r["title"] == "Nutrient collapse in closed systems")
        # The signals score the paper, not its lexical similarity to the input.
        if collapse["score"] == 0:
            assert collapse["position"] > max(
                r["position"] for r in report["recommendations"]
                if r["score"] > 0)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def test_score_is_the_weighted_sum_of_signal_contributions():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "kelp hydroculture")
        for rec in report["recommendations"]:
            total = sum(s["contribution"] for s in rec["signals"])
            assert abs(total - rec["score"]) < 1e-6
            for s in rec["signals"]:
                assert s["contribution"] == round(s["weight"] * s["matches"], 4)


def test_signal_breakdown_names_the_signal_and_its_weight():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "kelp hydroculture")
        rec = report["recommendations"][0]
        by_name = {s["signal"]: s for s in rec["signals"]}
        assert set(by_name) == {"crop_match", "method_match", "influential_cites"}
        assert by_name["crop_match"]["weight"] == 2.0
        assert by_name["method_match"]["weight"] == 1.5
        assert by_name["influential_cites"]["weight"] == 1.0
        assert by_name["crop_match"]["description"]


def test_zero_score_papers_are_kept_but_reported_as_unmatched():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "kelp")
        zero = [r for r in report["recommendations"] if r["score"] == 0]
        for rec in zero:
            assert all(s["contribution"] == 0 for s in rec["signals"])


def test_edge_conditions_in_signals_fire_against_graph_edges():
    """influential_cites must see the citation edges built by build_graph."""
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "scaling vertical farming seaweed")
        rec = next((r for r in report["recommendations"]
                    if r["title"] == "Scaling vertical farming of seaweed"), None)
        assert rec is not None
        inc = signal(report, rec["paper_id"], "influential_cites")
        assert inc["matches"] == 1
        edge_ev = [e for e in inc["evidence"] if e["kind"] == "citation_edge"]
        assert edge_ev, f"expected citation_edge evidence, got {inc['evidence']}"
        assert edge_ev[0]["isInfluential"] is True
        assert edge_ev[0]["direction"] == "references"


def test_no_match_case_explains_itself():
    """A state whose signals cannot match anything must say so, not invent order."""
    def break_signals(schema):
        schema["reading_order"]["signals"] = [
            {"name": "nothing_matches", "weight": 1.0,
             "rule": {"all": [
                 {"vocab": {"name": "problems", "where": ["title"]}},
                 {"regex": {"pattern": "zzz_never_present", "where": ["title"]}},
             ]}}
        ]

    with built_state(break_signals) as (state_path, _tmp):
        rc, report, _o, _e = analyse(state_path, "kelp")
        assert rc == 0
        assert report["status"] == "no_match"
        assert report["matched_count"] == 0
        for rec in report["recommendations"]:
            assert rec["score"] == 0


# --------------------------------------------------------------------------
# graph evidence / provenance
# --------------------------------------------------------------------------
def test_graph_evidence_cites_the_edge_type_and_the_other_paper():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "scaling vertical farming seaweed")
        rec = next(r for r in report["recommendations"]
                   if r["title"] == "Scaling vertical farming of seaweed")
        kinds = {e["edge_type"] for e in rec["graph_evidence"]}
        assert "EXTENDS" in kinds
        edge = next(e for e in rec["graph_evidence"] if e["edge_type"] == "EXTENDS")
        assert edge["rule"] == "EXTENDS"
        assert edge["other_paper_title"] == "Hydroculture for kelp"
        assert edge["other_paper_year"] == 2019
        assert edge["isInfluential"] is True


def test_every_recommendation_carries_a_title_and_year():
    with built_state() as (state_path, _tmp):
        _rc, report, _o, _e = analyse(state_path, "kelp nori")
        assert report["recommendations"]
        for rec in report["recommendations"]:
            assert rec["title"]
            assert rec["year"]
            assert rec["paper_id"]


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------
def test_readable_report_lists_position_title_year_score_and_breakdown():
    with built_state() as (state_path, _tmp):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = az.main(["--state", str(state_path), "--text", "kelp hydroculture"])
        assert rc == 0
        text = out.getvalue()
        assert "Reading order" in text
        assert "[1]" in text
        assert "score" in text
        assert "Input interpreted as" in text
        assert "crop_match" in text
        # no raw json leaked into the human-readable view
        assert '"recommendations"' not in text


def test_no_match_report_explains_the_reason():
    def break_signals(schema):
        schema["reading_order"]["signals"] = [
            {"name": "nothing_matches", "weight": 1.0,
             "rule": {"regex": {"pattern": "zzz_never_present", "where": ["title"]}}}
        ]

    with built_state(break_signals) as (state_path, _tmp):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = az.main(["--state", str(state_path), "--text", "kelp"])
        assert rc == 0
        text = out.getvalue()
        assert "NO RECOMMENDATIONS" in text
        assert "vocabularies" in text or "signals" in text


def test_disconnected_warning_appears_when_input_matched_nothing():
    with built_state() as (state_path, _tmp):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            az.main(["--state", str(state_path),
                     "--text", "quantum chromodynamics lattice gauge theory"])
        text = out.getvalue()
        if "recommendation(s) with a non-zero score" in text:
            assert "WARNING" in text
            assert "did not actually select these papers" in text


def test_json_output_is_valid_and_deterministic():
    with built_state() as (state_path, _tmp):
        _rc, first, _o, _e = analyse(state_path, "kelp hydroculture")
        _rc, second, _o, _e = analyse(state_path, "kelp hydroculture")
        assert first == second
        assert first["status"] == "ok"


def test_out_flag_writes_the_same_json():
    with built_state() as (state_path, tmp):
        target = tmp / "nested" / "result.json"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = az.main(["--state", str(state_path), "--text", "kelp",
                          "--json", "--out", str(target)])
        assert rc == 0
        assert json.loads(target.read_text()) == json.loads(out.getvalue())


def test_min_similarity_filters_candidates():
    with built_state() as (state_path, _tmp):
        baseline = analyse(state_path, "kelp hydroculture")[1]
        _rc, strict, _o, _e = analyse(state_path, "kelp hydroculture",
                                       ["--min-similarity", "0.05"])
        assert len(strict["recommendations"]) <= len(baseline["recommendations"])
        for rec in strict["recommendations"]:
            assert rec["tfidf_similarity"] >= 0.05


# --------------------------------------------------------------------------
# input sources and refusals
# --------------------------------------------------------------------------
def test_reads_input_from_a_file():
    with built_state() as (state_path, tmp):
        path = tmp / "input.txt"
        path.write_text("Blight in nori farms", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = az.main(["--state", str(state_path), "--file", str(path), "--json"])
        assert rc == 0
        report = json.loads(out.getvalue())
        assert report["input"] == "Blight in nori farms"
        assert "nori" in report["input_interpretation"]["matched_terms"]


def test_reads_input_from_stdin():
    with built_state() as (state_path, _tmp):
        real_stdin = sys.stdin
        sys.stdin = io.StringIO("kelp via stdin")
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = az.main(["--state", str(state_path), "--json"])
        finally:
            sys.stdin = real_stdin
        assert rc == 0
        report = json.loads(out.getvalue())
        assert report["input"] == "kelp via stdin"


def test_refuses_a_missing_knowledge_state():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "absent.json"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = az.main(["--state", str(missing), "--text", "kelp"])
        assert rc == 1
        assert "no knowledge state" in err.getvalue()
        assert "build_graph.py" in err.getvalue()


def test_refuses_a_missing_input_file():
    with built_state() as (state_path, tmp):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = az.main(["--state", str(state_path), "--file", str(tmp / "nope.txt")])
        assert rc == 1
        assert "not found" in err.getvalue()


def test_refuses_when_reading_order_is_missing():
    with built_state(lambda s: s.__setitem__("reading_order", None)) as (state_path, _tmp):
        expect_analyze_error(
            lambda: az.load_reading_order(az.load_state(state_path)),
            "reading_order is not configured")


def test_refuses_when_reading_order_has_no_signals():
    with built_state(lambda s: s["reading_order"].__setitem__("signals", [])) as (
            state_path, _tmp):
        expect_analyze_error(
            lambda: az.load_reading_order(az.load_state(state_path)),
            "no `signals`")


def test_refuses_an_unreadable_or_incomplete_state():
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        expect_analyze_error(lambda: az.load_state(broken), "not valid json")

        incomplete = Path(tmp) / "incomplete.json"
        incomplete.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
        expect_analyze_error(lambda: az.load_state(incomplete), "missing the 'schema'")

        absent = Path(tmp) / "absent.json"
        expect_analyze_error(lambda: az.load_state(absent), "no knowledge state")


def test_reading_order_is_checked_before_input_is_read():
    """A misconfigured schema should be reported even with no input supplied."""
    with built_state(lambda s: s.__setitem__("reading_order", None)) as (state_path, _tmp):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = az.main(["--state", str(state_path)])
        assert rc == 1
        assert "reading_order is not configured" in err.getvalue()


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS  {name}")
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())