"""Unit tests for src/build_graph.py, using the tiny kelp-farming fixtures.

Covers every condition type the engine implements, both edge-rule kinds, the
config validation refusals, and the shape of the export.

Run:  python -m pytest tests/test_build_graph.py -v
  or: python tests/test_build_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import build_graph as bg  # noqa: E402

FIXTURES = HERE / "fixtures"
SCHEMA_PATH = FIXTURES / "schema.yaml"
VOCAB_PATH = FIXTURES / "vocab.yaml"
CACHE_PATH = FIXTURES / "raw_papers.json"


def load_all():
    schema = bg.load_yaml(SCHEMA_PATH)
    vocab_doc = bg.load_yaml(VOCAB_PATH)
    vocab_index = bg.build_vocab_index(vocab_doc)
    bg.validate_schema(schema, vocab_index)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return schema, vocab_doc, vocab_index, cache


def build(**kwargs):
    schema, vocab_doc, vocab_index, cache = load_all()
    graph, warnings = bg.build_graph(cache, schema, vocab_index, **kwargs)
    return graph, warnings, schema, vocab_doc, cache


def edges_of(graph, edge_type):
    return [(s, t) for s, t, d in graph.edges(data=True) if d.get("type") == edge_type]


def edge_data(graph, src, dst, edge_type):
    for s, t, d in graph.edges(data=True):
        if (s, t, d.get("type")) == (src, dst, edge_type):
            return d
    raise AssertionError(f"no {edge_type} edge {src} -> {dst}")


# ==========================================================================
# vocabulary / regex conditions
# ==========================================================================
def test_vocab_condition_is_case_insensitive_on_word_boundaries():
    engine = bg.RuleEngine({"v": {"terms": ["kelp"], "description": ""}})
    # Lowercase in the vocabulary, mixed case in the text.
    assert engine.evaluate({"vocab": {"name": "v"}}, {"title": "KELP farming"}).matched
    # Word boundary: "kelpweed" must NOT match "kelp".
    assert not engine.evaluate({"vocab": {"name": "v"}},
                               {"title": "kelpweed bloom"}).matched
    # ...but a multi-word phrase matches as written.
    two = bg.RuleEngine({"v": {"terms": ["net-based farming"], "description": ""}})
    assert two.evaluate({"vocab": {"name": "v"}},
                        {"title": "We use Net-Based Farming here"}).matched
    assert not two.evaluate({"vocab": {"name": "v"}},
                            {"title": "net based farming"}).matched, \
        "the hyphen is significant"


def test_vocab_reports_which_term_and_field_matched():
    graph, _, _, _, _ = build()
    d = edge_data(graph, "paper:k1", "term:kelp", "GROWS")
    ev = d["evidence"][0]
    assert ev["kind"] == "vocab_term"
    assert ev["vocabulary"] == "crops"
    assert ev["term"] == "kelp"
    assert ev["field"] == "title"
    assert ev["matched_text"] == "kelp"


def test_vocab_counts_a_term_once_even_if_in_two_fields():
    """k3 mentions 'kelp' in both title-adjacent abstract and elsewhere."""
    engine = bg.RuleEngine({"v": {"terms": ["kelp"], "description": ""}})
    m = engine.evaluate({"vocab": {"name": "v"}},
                        {"title": "kelp", "abstract": "more kelp here"})
    assert m.strength == 1, m.strength


def test_regex_condition():
    engine = bg.RuleEngine({})
    ok = engine.evaluate({"regex": {"pattern": "hydro\\w+", "where": ["abstract"]}},
                         {"abstract": "using HYDROCULTURE here"})
    assert ok.matched and ok.strength == 1
    assert ok.evidence[0]["matched_text"] == "HYDROCULTURE"
    assert not engine.evaluate({"regex": {"pattern": "hydro\\w+"}},
                               {"abstract": "nothing"}).matched
    # `where` is respected.
    assert not engine.evaluate(
        {"regex": {"pattern": "hydro\\w+", "where": ["venue"]}},
        {"abstract": "hydroculture", "venue": "Nature"}).matched


def test_regex_where_contexts_reads_citation_sentences():
    engine = bg.RuleEngine({})
    paper = {"references": {"x": {"contexts": ["we scale this approach"],
                                  "intents": [], "isInfluential": False}}}
    m = engine.evaluate({"regex": {"pattern": "scale", "where": ["contexts"]}}, paper)
    assert m.matched
    assert m.evidence[0]["field"] == "references[x].contexts[0]"


# ==========================================================================
# numeric conditions
# ==========================================================================
def test_year_between_and_comparators():
    engine = bg.RuleEngine({})
    between = {"year": {"between": [2018, 2023]}}
    assert engine.evaluate(between, {"year": 2020}).matched
    assert not engine.evaluate(between, {"year": 2024}).matched
    assert not engine.evaluate(between, {}).matched, "missing year must not match"
    assert engine.evaluate({"year": {"gte": 2020}}, {"year": 2020}).matched
    assert not engine.evaluate({"year": {"gt": 2020}}, {"year": 2020}).matched
    assert engine.evaluate({"year": {"lt": 2021}}, {"year": 2020}).matched
    assert engine.evaluate({"year": {"lte": 2020}}, {"year": 2020}).matched
    assert engine.evaluate({"year": {"eq": 2020}}, {"year": 2020}).matched


def test_citation_count_reads_the_camelcase_cache_field():
    engine = bg.RuleEngine({})
    cond = {"citation_count": {"gte": 100}}
    assert engine.evaluate(cond, {"citationCount": 200}).matched
    assert not engine.evaluate(cond, {"citationCount": 20}).matched
    assert not engine.evaluate(cond, {}).matched


# ==========================================================================
# edge conditions
# ==========================================================================
def _paper():
    return {
        "references": {"old": {"isInfluential": True, "intents": ["methodology"],
                               "contexts": ["we scale this"]}},
        "citations": {"new": {"isInfluential": False, "intents": ["background"],
                              "contexts": ["a mention"]}},
    }


def test_edge_condition_influential():
    engine = bg.RuleEngine({})
    m = engine.evaluate({"edge": {"influential": True}}, _paper())
    assert m.strength == 1
    assert m.evidence[0]["direction"] == "references"
    assert m.evidence[0]["other_paper_id"] == "old"


def test_edge_condition_direction():
    engine = bg.RuleEngine({})
    m = engine.evaluate({"edge": {"influential": True, "direction": "citations"}},
                        _paper())
    assert not m.matched, "the only influential edge is an outgoing reference"
    m = engine.evaluate({"edge": {"influential": True, "direction": "references"}},
                        _paper())
    assert m.matched


def test_edge_condition_intents_any():
    engine = bg.RuleEngine({})
    assert engine.evaluate(
        {"edge": {"intents_any": ["methodology"]}}, _paper()).matched
    assert not engine.evaluate(
        {"edge": {"intents_any": ["result"]}}, _paper()).matched
    # any-of semantics
    assert engine.evaluate(
        {"edge": {"intents_any": ["result", "background"]}}, _paper()).matched


def test_edge_condition_contexts_any_is_a_regex():
    engine = bg.RuleEngine({})
    assert engine.evaluate({"edge": {"contexts_any": "scal"}}, _paper()).matched
    assert not engine.evaluate({"edge": {"contexts_any": "harvest"}}, _paper()).matched


def test_edge_condition_combines_all_criteria():
    engine = bg.RuleEngine({})
    conds = {"edge": {"influential": True, "intents_any": ["methodology"],
                      "contexts_any": "scale", "direction": "references"}}
    assert engine.evaluate(conds, _paper()).matched
    # changing any single criterion must break the match
    assert not engine.evaluate({**conds, "edge": {**conds["edge"],
                                                  "intents_any": ["result"]}},
                               _paper()).matched


# ==========================================================================
# all / any nesting
# ==========================================================================
def test_all_requires_every_child():
    engine = bg.RuleEngine({"v": {"terms": ["kelp"], "description": ""}})
    ok = {"all": [{"vocab": {"name": "v"}}, {"year": {"gte": 2018}}]}
    assert engine.evaluate(ok, {"title": "kelp", "year": 2019}).matched
    assert not engine.evaluate(ok, {"title": "kelp", "year": 2017}).matched
    assert not engine.evaluate(ok, {"title": "other", "year": 2019}).matched


def test_any_requires_one_child_and_sums_strength():
    engine = bg.RuleEngine({"v": {"terms": ["kelp"], "description": ""}})
    cond = {"any": [{"vocab": {"name": "v"}},
                    {"regex": {"pattern": "nori", "where": ["title"]}}]}
    assert engine.evaluate(cond, {"title": "kelp"}).strength == 1
    assert engine.evaluate(cond, {"title": "nori"}).strength == 1
    both = engine.evaluate(cond, {"title": "kelp and nori"})
    assert both.matched and both.strength == 2
    assert not engine.evaluate(cond, {"title": "unrelated"}).matched


def test_nested_all_any():
    engine = bg.RuleEngine({"v": {"terms": ["kelp"], "description": ""}})
    cond = {"all": [
        {"year": {"between": [2018, 2023]}},
        {"any": [{"vocab": {"name": "v"}},
                 {"regex": {"pattern": "nori", "where": ["abstract"]}}]},
    ]}
    assert engine.evaluate(cond, {"title": "kelp", "year": 2020}).matched
    assert engine.evaluate(cond, {"abstract": "nori", "year": 2020}).matched
    assert not engine.evaluate(cond, {"title": "kelp", "year": 2024}).matched
    assert not engine.evaluate(cond, {"title": "other", "year": 2020}).matched


def test_vocabulary_term_rule_requires_a_subject_with_a_vocab_condition():
    """A vocabulary_term rule with no vocab leaf could only ever make zero edges."""
    schema, vocab_doc, vocab_index, _ = load_all()
    spec = next(e for e in schema["edge_types"] if e["name"] == "GROWS")

    saved = spec["rule"].pop("subject")
    try:
        _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                             "rule.subject is required")
    finally:
        spec["rule"]["subject"] = saved

    spec["rule"]["subject"] = {"regex": {"pattern": "kelp", "where": ["title"]}}
    try:
        _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                             "must contain at least one `vocab:` condition")
    finally:
        spec["rule"]["subject"] = saved


# ==========================================================================
# edge rules
# ==========================================================================
def test_vocabulary_term_rule_creates_one_edge_per_term():
    graph, _, _, _, _ = build()
    assert sorted(edges_of(graph, "GROWS")) == [
        ("paper:k1", "term:kelp"),
        ("paper:k2", "term:nori"),
        ("paper:k3", "term:kelp"),
        ("paper:k3", "term:seaweed"),
        ("paper:k5", "term:kelp"),
        ("paper:k5", "term:nori"),
        ("paper:k6", "term:nori"),
    ]
    # Term nodes are created, typed and carry their text.
    node = graph.nodes["term:kelp"]
    assert node["type"] == "Topic"
    assert node["properties"]["term"] == "kelp"
    assert node["derived_by"] == "GROWS"


def test_vocabulary_term_rule_respects_year_filter():
    """k2 (2018) has no technique, and the rule is year-bounded to 2018-2023."""
    graph, _, _, _, _ = build()
    technique_papers = {s for s, _ in edges_of(graph, "USES_TECHNIQUE")}
    # k1 matched the vocab branch in its title; k3 matched 'vertical farming'.
    assert technique_papers == {"paper:k1", "paper:k3"}


def test_citation_rule_direction_references():
    graph, _, _, _, _ = build()
    assert sorted(edges_of(graph, "EXTENDS")) == [
        ("paper:k3", "paper:k1"),
        ("paper:k5", "paper:k1"),
    ]


def test_citation_rule_direction_citations():
    graph, _, _, _, _ = build()
    # Only k1 is cited by a methodology edge whose context mentions "scale".
    assert edges_of(graph, "CITED_BY") == [("paper:k1", "paper:k3")]


def test_citation_rule_edge_carries_provenance():
    graph, _, _, _, _ = build()
    d = edge_data(graph, "paper:k3", "paper:k1", "EXTENDS")
    assert d["rule"] == "EXTENDS"
    fact = d["evidence"][0]
    assert fact["kind"] == "citation"
    assert fact["direction"] == "references"
    assert fact["other_paper_id"] == "k1"
    assert fact["isInfluential"] is True
    assert fact["intents"] == ["methodology"]
    assert fact["context_snippet"] == "we scale this approach to open water"
    # the object condition that let the neighbour through is recorded too
    assert any(e["kind"] == "numeric" and e["test"] == "year lt 2021"
               for e in d["evidence"])


def test_citation_rule_negative_cases_are_excluded():
    """k3->k4 is a real citation but fails both the edge and the object test."""
    graph, _, _, _, _ = build()
    assert ("paper:k3", "paper:k4") not in edges_of(graph, "EXTENDS")
    # k5->k3 is influential but k3 (2022) fails `year lt 2021`.
    assert ("paper:k5", "paper:k3") not in edges_of(graph, "EXTENDS")


def test_citation_rules_skip_out_of_corpus_endpoints():
    schema, vocab_doc, vocab_index, cache = load_all()
    cache["papers"]["k6"]["references"]["outsider"] = {
        "isInfluential": True, "intents": ["methodology"], "contexts": ["x"]}
    cache["neighbors"]["outsider"] = {"paperId": "outsider", "title": "Outsider",
                                      "year": 2015}
    graph, _ = bg.build_graph(cache, schema, vocab_index)
    assert not any(t == "paper:outsider" for _, t, _ in graph.edges(data=True))

    graph, _ = bg.build_graph(cache, schema, vocab_index, include_neighbors=True)
    assert ("paper:k6", "paper:outsider") in edges_of(graph, "EXTENDS")


# ==========================================================================
# graph shape
# ==========================================================================
def test_graph_is_a_multidigraph_with_typed_nodes():
    graph, _, _, _, _ = build()
    assert graph.is_directed() and graph.is_multigraph()
    assert set(graph.nodes) == {
        "paper:k1", "paper:k2", "paper:k3", "paper:k4", "paper:k5", "paper:k6",
        "term:kelp", "term:nori", "term:seaweed", "term:hydroculture",
        "term:vertical farming", "term:blight",
    }
    assert all("type" in d for _, d in graph.nodes(data=True))


def test_paper_nodes_carry_only_configured_properties():
    graph, _, _, _, _ = build()
    props = graph.nodes["paper:k1"]["properties"]
    assert set(props) == {"paperId", "title", "year", "venue", "citationCount"}
    assert props["paperId"] == "k1"
    assert "abstract" not in props


# ==========================================================================
# validation refusals
# ==========================================================================
def _expect_config_error(fn, needle):
    try:
        fn()
    except bg.ConfigError as exc:
        assert needle in str(exc), f"expected {needle!r} in: {exc}"
        return
    raise AssertionError(f"expected ConfigError containing {needle!r}")


def test_refuses_todo_placeholders(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "schema.yaml"
        p.write_text("node_types:\n  - name: TODO\n")
        _expect_config_error(lambda: bg.load_yaml(p), "unfilled placeholder")


def test_refuses_empty_node_types():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["node_types"] = []
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "empty `node_types`")


def test_refuses_empty_edge_types():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"] = []
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "empty `edge_types`")


def test_refuses_unknown_vocabulary_name():
    """An undefined vocabulary must be caught before any graph work happens."""
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["subject"]["all"][0] = {"vocab": {"name": "nope"}}
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "is not defined in vocab.yaml")

    # ...and the same for a reading_order signal.
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["reading_order"]["signals"][0]["rule"]["all"][0] = {"vocab": {"name": "nope"}}
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "is not defined in vocab.yaml")


def test_missing_vocabulary_is_reported_at_evaluation_time():
    engine = bg.RuleEngine({"real": {"terms": ["kelp"], "description": ""}})
    _expect_config_error(
        lambda: engine.evaluate({"vocab": {"name": "ghost"}}, {"title": "x"}),
        "not defined in vocab.yaml")


def test_refuses_unknown_condition_key():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["subject"] = {"sentiment": {"pos": True}}
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "unknown condition")


def test_refuses_condition_with_more_than_one_key():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["subject"] = {
        "vocab": {"name": "crops"}, "year": {"gte": 2000}}
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "exactly one key")


def test_refuses_edge_type_pointing_at_undeclared_node_type():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["target"] = "Ghost"
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "is not a declared node type")


def test_refuses_bad_direction_and_bad_target():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["direction"] = "sideways"
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "rule.direction")

    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["target"] = "vibes"
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "rule.target")


def test_refuses_unknown_where_field():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["subject"]["all"][0] = {
        "vocab": {"name": "crops", "where": ["keywords"]}}
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "unknown field")


def test_refuses_bad_regex():
    schema, vocab_doc, vocab_index, _ = load_all()
    schema["edge_types"][0]["rule"]["subject"]["all"][0] = {
        "regex": {"pattern": "([unclosed"}}
    _expect_config_error(lambda: bg.validate_schema(schema, vocab_index),
                         "bad pattern")


def test_refuses_no_paper_node_type():
    schema, vocab_doc, vocab_index, cache = load_all()
    for spec in schema["node_types"]:
        spec["source"] = "Derived"
    bg.validate_schema(schema, vocab_index)
    _expect_config_error(lambda: bg.build_graph(cache, schema, vocab_index),
                         "nowhere to put the corpus papers")


def test_refuses_empty_vocab_file(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "vocab.yaml"
        p.write_text("vocabularies: []\n")
        doc = bg.load_yaml(p)
        _expect_config_error(lambda: bg.build_vocab_index(doc), "no vocabularies")


# ==========================================================================
# export
# ==========================================================================
def test_export_shape_and_contents():
    graph, warnings, schema, vocab_doc, cache = build()
    state = bg.export_state(graph, schema, vocab_doc, cache, warnings, False)

    assert list(state) == ["metadata", "schema", "vocabularies", "node_types",
                           "edge_types", "nodes", "edges"]
    # schema and vocabularies are embedded verbatim
    assert state["schema"] == schema
    assert state["vocabularies"] == vocab_doc
    # metadata
    assert state["metadata"]["corpus_size"] == 6
    assert state["metadata"]["corpus_years"] == {"min": 2018, "max": 2023}
    assert state["metadata"]["generated_at"].endswith("Z")
    assert state["metadata"]["source_cache_meta"]["is_fixture"] is True
    assert state["metadata"]["graph"] == {"nodes": 12, "edges": 13}
    # stats per type
    assert state["metadata"]["node_type_counts"] == {"Paper": 6, "Topic": 6}
    assert state["metadata"]["edge_type_counts"]["EXTENDS"] == 2
    # descriptions carried through so the file explains itself
    assert "vocabulary_term" in state["node_types"]["Topic"]["description"]
    assert state["edge_types"]["EXTENDS"]["count"] == 2
    assert state["edge_types"]["EXTENDS"]["rule"]["direction"] == "references"
    # every edge has provenance
    assert all(e["rule"] and e["type"] and e["evidence"] for e in state["edges"])
    # vocabulary descriptions are present for a human reader
    assert state["vocabularies"]["vocabularies"][0]["description"]


def test_export_is_valid_json_and_deterministic(tmp_path=None):
    import tempfile
    graph, warnings, schema, vocab_doc, cache = build()
    state = bg.export_state(graph, schema, vocab_doc, cache, warnings, False)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ks.json"
        bg.write_state(state, out)
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded["metadata"]["graph"] == state["metadata"]["graph"]
        assert not list(Path(td).glob("*.tmp")), "temp file left behind"


# ==========================================================================
# CLI
# ==========================================================================
def test_cli_end_to_end(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "knowledge_state.json"
        rc = bg.main([
            "--schema", str(SCHEMA_PATH), "--vocab", str(VOCAB_PATH),
            "--cache", str(CACHE_PATH), "--out", str(out), "--quiet",
        ])
        assert rc == 0
        state = json.loads(out.read_text(encoding="utf-8"))
        assert state["metadata"]["graph"]["nodes"] == 12
        assert state["metadata"]["graph"]["edges"] == 13


def test_cli_refuses_real_unfilled_config():
    """The project's real config still has TODOs, so the CLI must refuse."""
    rc = bg.main(["--quiet"])
    assert rc == 1


# ==========================================================================
# runner
# ==========================================================================
def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except AssertionError as exc:
            print(f"FAIL  {name} -- {exc}")
            failed.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {name} -- {type(exc).__name__}: {exc}")
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
