"""Turn a new input into a structured reading order, grounded in the graph.

    python src/analyze.py --text "I want to learn dense retrieval for RAG"
    python src/analyze.py --file abstract.txt
    cat abstract.txt | python src/analyze.py
    python src/analyze.py --text "..." --json

Reads data/knowledge_state.json, which carries the schema and the vocabularies
verbatim. Using the embedded copies rather than re-reading config/ is deliberate:
it guarantees the rules applied here are byte-for-byte the rules that built the
graph.

How a recommendation is produced
--------------------------------
1. The input text is run through every `vocab` and `regex` condition that
   appears anywhere in the schema. Whatever matches is reported as "what the
   input was understood as". Nothing is inferred beyond those conditions.
2. TF-IDF cosine similarity shortlists `reading_order.candidate_pool` papers.
   This is the only job TF-IDF does: choosing what to look at.
3. Each shortlisted paper is scored by evaluating `reading_order.signals`
   against it. Contribution per signal is `weight * matches`. The list is sorted
   by that total.

No other scoring factor is applied. If a paper scores zero it is reported as
unmatched rather than being nudged up the list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = PROJECT_ROOT / "data" / "knowledge_state.json"


class AnalyzeError(Exception):
    """Raised when the analysis cannot be run as configured."""


# ==========================================================================
# loading
# ==========================================================================
def load_state(path: Path) -> dict:
    if not path.exists():
        raise AnalyzeError(
            f"no knowledge state at {path}\n"
            f"       Run: python src/build_graph.py"
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalyzeError(f"{path} is not valid JSON: {exc}") from exc
    for key in ("schema", "vocabularies", "nodes", "edges"):
        if key not in state:
            raise AnalyzeError(f"{path} is missing the {key!r} section; "
                               f"rebuild it with src/build_graph.py")
    return state


def load_reading_order(state: dict) -> dict:
    """Return reading_order, or refuse with an explanation."""
    reading_order = state["schema"].get("reading_order")
    if not reading_order:
        raise AnalyzeError(
            "reading_order is not configured in schema.yaml.\n"
            "       Set `reading_order:` with `candidate_pool:` and a list of\n"
            "       `signals:` (each with name, weight and rule), then rebuild\n"
            "       the knowledge state: python src/build_graph.py"
        )
    signals = reading_order.get("signals")
    if not signals:
        raise AnalyzeError(
            "reading_order is present but has no `signals`, so there is\n"
            "       nothing to score candidates with.\n"
            "       Add at least one signal with a name, a weight and a rule,\n"
            "       then rebuild."
        )
    return reading_order


def vocab_index(vocab_doc: dict) -> dict[str, dict]:
    return {v["name"]: v for v in (vocab_doc.get("vocabularies") or []) if v.get("name")}


# ==========================================================================
# input interpretation
# ==========================================================================
def collect_input_conditions(schema: dict) -> list[tuple[str, dict]]:
    """Every vocab/regex leaf anywhere in the schema, as (label, condition).

    Used to describe the input in the schema's own vocabulary. Reading order
    signals are included, since they are part of the configured vocabulary of
    concepts.
    """
    found: list[tuple[str, dict]] = []

    def walk(conds: Any, path: str) -> None:
        if not isinstance(conds, dict) or len(conds) != 1:
            return
        key, body = next(iter(conds.items()))
        if key in ("all", "any"):
            for i, child in enumerate(body):
                walk(child, f"{path}.{key}[{i}]")
        elif key in ("vocab", "regex"):
            found.append((path, {key: body}))

    for spec in schema.get("edge_types") or []:
        rule = spec.get("rule") or {}
        for part in ("subject", "object", "edge"):
            walk(rule.get(part), f"edge_types[{spec.get('name')}].rule.{part}")
    reading_order = schema.get("reading_order") or {}
    for signal in reading_order.get("signals") or []:
        walk(signal.get("rule"), f"reading_order.signals[{signal.get('name')}].rule")

    # De-duplicate identical (vocab, where) probes, keeping the first label.
    seen: set[str] = set()
    unique = []
    for path, cond in found:
        key = json.dumps(cond, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, cond))
    return unique


def interpret_input(text: str, schema: dict, vocab: dict[str, dict],
                    engine) -> dict:
    """Apply the schema's vocab/regex conditions to the new input text.

    The input is treated as a pseudo-paper with only a title, so the same
    conditions that read a paper's title can read the input.
    """
    as_paper = {"title": text, "abstract": "", "venue": ""}
    matches: list[dict] = []
    unknown_vocab: list[str] = []

    for path, cond in collect_input_conditions(schema):
        if "vocab" in cond:
            name = cond["vocab"].get("name")
            if name not in vocab:
                unknown_vocab.append(name)
                continue
        # Every condition in the schema was validated at build time, so these
        # probes cannot fail here.
        result = engine.evaluate(cond, as_paper)
        for item in result.evidence:
            matches.append({**item, "field": "input", "condition": path})

    terms = sorted({m["term"] for m in matches if m.get("kind") == "vocab_term"})
    regex_hits = sorted({m["matched_text"].lower() for m in matches
                         if m.get("kind") == "regex"})
    return {
        "matched_terms": terms,
        "regex_hits": regex_hits,
        "matches": matches,
        "unknown_vocabularies": sorted(set(unknown_vocab)),
        "understood_as": terms + regex_hits,
    }


# ==========================================================================
# candidate shortlisting
# ==========================================================================
def shortlist(input_text: str, papers: list[dict], pool: int) -> list[tuple[dict, float]]:
    """Rank papers by TF-IDF cosine similarity. This only picks who to look at."""
    if not papers:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def doc(paper: dict) -> str:
        return " ".join(str(paper["properties"].get(f) or "")
                        for f in ("title", "abstract", "venue"))

    corpus = [doc(p) for p in papers]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                     ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus + [input_text])
    except ValueError:
        # Empty vocabulary (e.g. every document is stop words).
        return [(p, 0.0) for p in papers[:pool]]

    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(papers, scores), key=lambda pair: (-pair[1], pair[0]["id"]))
    return [(p, float(s)) for p, s in ranked[:pool]]


# ==========================================================================
# scoring
# ==========================================================================
def score_paper(paper: dict, signals: list[dict], engine,
                edges_by_source: dict[str, list[dict]]) -> dict:
    """Evaluate every signal against one paper. Total = sum(weight * matches)."""
    # Signals read the same fields the paper node carries, plus its citation
    # edges so that `edge:` conditions work.
    record = dict(paper["properties"])
    record["citations"] = {}
    record["references"] = {}
    for edge in edges_by_source.get(paper["id"], []):
        fact = next((e for e in edge["evidence"] if e.get("kind") == "citation"), None)
        if not fact:
            continue
        other_id = fact.get("other_paper_id")
        direction = fact.get("direction")
        if direction not in ("citations", "references"):
            continue
        bucket = "citations" if direction == "citations" else "references"
        record[bucket][other_id or ""] = {
            "isInfluential": fact.get("isInfluential", False),
            "intents": fact.get("intents", []),
            "contexts": [fact.get("context_snippet")] if fact.get("context_snippet") else [],
        }

    breakdown = []
    total = 0.0
    for signal in signals:
        weight = float(signal.get("weight", 0.0))
        result = engine.evaluate(signal.get("rule"), record)
        contribution = weight * result.strength if result.matched else 0.0
        total += contribution
        breakdown.append({
            "signal": signal.get("name"),
            "description": (signal.get("description") or "").strip(),
            "weight": weight,
            "matches": result.strength,
            "contribution": round(contribution, 4),
            "evidence": result.evidence,
        })

    return {"score": round(total, 4), "signals": breakdown}


def graph_evidence(paper: dict, edges_by_source: dict[str, list[dict]],
                   neighbours: dict[str, dict]) -> list[dict]:
    """The edges attached to this paper, phrased as justification."""
    out = []
    for edge in edges_by_source.get(paper["id"], []):
        fact = next((e for e in edge["evidence"] if e.get("kind") == "citation"), None)
        if fact is None:
            continue
        other_id = fact.get("other_paper_id", "")
        other = neighbours.get(other_id)
        out.append({
            "edge_type": edge.get("type"),
            "rule": edge.get("rule"),
            "direction": fact.get("direction"),
            "other_paper_id": other_id,
            "other_paper_title": (other or {}).get("title")
                                 or fact.get("other_paper_title"),
            "other_paper_year": (other or {}).get("year")
                                or fact.get("other_paper_year"),
            "isInfluential": fact.get("isInfluential"),
            "intents": fact.get("intents"),
            "context_snippet": fact.get("context_snippet"),
        })
    return out


# ==========================================================================
# report
# ==========================================================================
def build_report(input_text: str, state: dict, min_similarity: float) -> dict:
    """Produce the full analysis payload."""
    import build_graph as bg  # reuse the one rule engine, so rules cannot drift

    schema = state["schema"]
    vocab = vocab_index(state["vocabularies"])
    reading_order = load_reading_order(state)
    engine = bg.RuleEngine(vocab)

    paper_type_names = {n for n, info in (state.get("node_types") or {}).items()
                        if (info or {}).get("source") == "Paper"}
    papers = [n for n in state["nodes"] if n.get("type") in paper_type_names]
    if not papers:
        raise AnalyzeError("the knowledge state contains no paper nodes")

    neighbours = {n["id"].split(":", 1)[1]: n["properties"]
                  for n in state["nodes"]
                  if n.get("type") in paper_type_names and ":" in n["id"]}

    edges_by_source: dict[str, list[dict]] = {}
    for edge in state["edges"]:
        edges_by_source.setdefault(edge["source"], []).append(edge)

    interpretation = interpret_input(input_text, schema, vocab, engine)
    pool = int(reading_order.get("candidate_pool") or 25)
    signals = reading_order["signals"]
    ranked = shortlist(input_text, papers, pool)

    recommendations = []
    for paper, similarity in ranked:
        if similarity < min_similarity:
            continue
        scored = score_paper(paper, signals, engine, edges_by_source)
        recommendations.append({
            "paper_id": paper["id"].split(":", 1)[-1],
            "title": paper["properties"].get("title"),
            "year": paper["properties"].get("year"),
            "venue": paper["properties"].get("venue"),
            "tfidf_similarity": round(similarity, 4),
            "score": scored["score"],
            "signals": scored["signals"],
            "graph_evidence": graph_evidence(paper, edges_by_source, neighbours),
            "in_degree": paper.get("in_degree"),
            "out_degree": paper.get("out_degree"),
        })

    # Highest score first; ties broken by TF-IDF, then year, then title so the
    # order is stable between runs.
    recommendations.sort(key=lambda r: (-r["score"], -r["tfidf_similarity"],
                                        -(r["year"] or 0), r["title"] or ""))
    for i, rec in enumerate(recommendations, start=1):
        rec["position"] = i

    matched_papers = [r for r in recommendations if r["score"] > 0]
    understood = bool(interpretation["understood_as"])
    # The input only reaches the papers through TF-IDF shortlisting. If the
    # input matched nothing configured, that bridge is blind, so any ordering
    # below comes from the papers' own signals rather than from the input.
    disconnected = not understood and bool(recommendations)
    max_similarity = max((r["tfidf_similarity"] for r in recommendations), default=0.0)
    return {
        "input": input_text,
        "input_interpretation": interpretation,
        "candidate_pool": pool,
        "reading_order": {
            "signals": [{"name": s.get("name"), "weight": s.get("weight"),
                         "description": (s.get("description") or "").strip()}
                        for s in signals],
        },
        "graph": state.get("metadata", {}).get("graph", {}),
        "corpus_size": state.get("metadata", {}).get("corpus_size"),
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "matched_count": len(matched_papers),
        "input_understood": understood,
        "disconnected": disconnected,
        "max_tfidf_similarity": round(max_similarity, 4),
        "status": "ok" if matched_papers else "no_match",
    }


def _print_evidence(out, evidence: list[dict], indent: str) -> None:
    """Render one signal's evidence, switching on the evidence kind.

    Evidence items are whatever the rule engine produced, so each kind needs its
    own phrasing.
    """
    for ev in evidence:
        kind = ev.get("kind")
        if kind == "vocab_term":
            out.write(f"{indent}{ev['term']!r} in {ev['field']}\n")
        elif kind == "regex":
            out.write(f"{indent}/{ev['pattern']}/ matched {ev['matched_text']!r} "
                      f"in {ev['field']}\n")
        elif kind == "numeric":
            out.write(f"{indent}{ev['field']} satisfies {ev['test']} "
                      f"(actual: {ev['actual']})\n")
        elif kind == "citation_edge":
            other = ev.get("other_paper_id", "?")
            out.write(f"{indent}citation edge {ev['direction']} {other}: "
                      f"isInfluential={ev['isInfluential']}, "
                      f"intents={ev['intents']}\n")
            if ev.get("context_snippet"):
                out.write(f"{indent}  \"{ev['context_snippet']}\"\n")
        else:
            out.write(f"{indent}{ev}\n")


def print_report(report: dict, input_source: str) -> None:
    """Readable report. Deliberately plain text -- no colours, no tables of bars."""
    out = sys.stdout
    interp = report["input_interpretation"]

    out.write("=" * 72 + "\n")
    out.write("Reading order\n")
    out.write("=" * 72 + "\n")
    out.write(f"input source : {input_source}\n")
    out.write(f"corpus       : {report['corpus_size']} papers, "
              f"{report['graph'].get('nodes', '?')} nodes / "
              f"{report['graph'].get('edges', '?')} edges\n")
    out.write(f"shortlist    : top {report['candidate_pool']} by TF-IDF cosine "
              f"similarity\n")
    out.write("\nInput interpreted as:\n")
    if interp["understood_as"]:
        out.write("  terms : " + ", ".join(interp["matched_terms"]) + "\n")
        if interp["regex_hits"]:
            out.write("  regex : " + ", ".join(interp["regex_hits"]) + "\n")
    else:
        out.write("  (nothing -- no configured vocabulary term or regex matched "
                  "this input)\n")
    if interp["unknown_vocabularies"]:
        out.write("  note  : schema refers to undefined vocabularies: "
                  + ", ".join(interp["unknown_vocabularies"]) + "\n")

    out.write("\nScoring signals (from schema.yaml):\n")
    for sig in report["reading_order"]["signals"]:
        out.write(f"  {sig['name']} (weight {sig['weight']}): "
                  f"{sig['description'] or '(no description given)'}\n")

    out.write("\n" + "-" * 72 + "\n")
    if report["status"] != "ok":
        out.write("NO RECOMMENDATIONS\n\n")
        if not interp["understood_as"]:
            out.write(
                "Nothing in this input matched any vocabulary term or regex in\n"
                "schema.yaml, and no candidate paper scored above zero on your\n"
                "reading_order signals.\n\n"
                "This usually means the input is about something outside the\n"
                "topics the graph covers. Try wording it with terms from your\n"
                "vocabularies, or widen the vocabularies in config/vocab.yaml and\n"
                "rebuild the knowledge state.\n")
        else:
            out.write(
                "The input matched these terms -- " +
                ", ".join(interp["understood_as"]) + " -- but no paper in the\n"
                "shortlist scored above zero on your reading_order signals.\n"
                "The graph may not cover this topic, or your signals may be too\n"
                "narrow.\n")
        return

    out.write(f"{report['matched_count']} recommendation(s) with a non-zero score "
              f"(of {report['recommendation_count']} shortlisted)\n\n")

    if report["disconnected"]:
        out.write("!" * 72 + "\n")
        out.write("WARNING: the input matched none of your vocabulary terms or\n"
                  "regexes, so it did not actually select these papers.\n")
        if report["max_tfidf_similarity"] <= 0.0:
            out.write("TF-IDF gave every candidate a similarity of 0.0, and the\n"
                      "order below reflects only each paper's own signals.\n")
        else:
            out.write(f"The best TF-IDF similarity was only "
                      f"{report['max_tfidf_similarity']}, on words your\n"
                      f"vocabularies do not define, so the order below mostly\n"
                      f"reflects each paper's own signals.\n")
        out.write("Treat this as a summary of the corpus, not as an answer to\n"
                  "your question.\n"
                  + "!" * 72 + "\n\n")

    for rec in report["recommendations"]:
        out.write(f"[{rec['position']}] {rec['title']}\n")
        out.write(f"    year {rec['year']}  |  score {rec['score']}  |  "
                  f"tf-idf {rec['tfidf_similarity']}\n")

        fired = [s for s in rec["signals"] if s["contribution"] > 0]
        if fired:
            out.write("    why:\n")
            for sig in fired:
                out.write(f"      + {sig['signal']}  "
                          f"({sig['weight']} x {sig['matches']} "
                          f"= {sig['contribution']})\n")
                if sig["evidence"]:
                    _print_evidence(out, sig["evidence"], "          ")
        else:
            out.write("    why: no configured signal fired on this paper\n")

        if rec["graph_evidence"]:
            out.write("    graph edges:\n")
            for edge in rec["graph_evidence"][:5]:
                arrow = "references" if edge["direction"] == "references" else "cited by"
                out.write(f"      {edge['edge_type']}: {arrow} "
                          f"{edge['other_paper_title']} ({edge['other_paper_year']})\n")
                out.write(f"        rule {edge['rule']}, "
                          f"isInfluential={edge['isInfluential']}, "
                          f"intents={edge['intents']}\n")
                if edge["context_snippet"]:
                    out.write(f"        \"{edge['context_snippet']}\"\n")
        out.write("\n")


# ==========================================================================
# main
# ==========================================================================
def read_input(args) -> tuple[str, str]:
    """Return (text, description-of-where-it-came-from)."""
    if args.text is not None:
        return args.text, "--text argument"
    if args.file is not None:
        path = Path(args.file)
        if not path.exists():
            raise AnalyzeError(f"input file not found: {path}")
        return path.read_text(encoding="utf-8"), f"--file {path}"
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data, "stdin"
    raise AnalyzeError(
        "no input given.\n"
        "       Use --text \"...\", --file <path>, or pipe text on stdin."
    )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Map a new input into the knowledge graph and print a "
                    "grounded reading order",
    )
    ap.add_argument("--text", help="the new input, as an argument")
    ap.add_argument("--file", type=Path, help="read the new input from a file")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE,
                    help="knowledge state to analyse against")
    ap.add_argument("--json", action="store_true",
                    help="print raw JSON instead of the readable report")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the JSON result to this path")
    ap.add_argument("--min-similarity", type=float, default=0.0,
                    help="drop candidates whose TF-IDF similarity is below this "
                         "(default 0.0, i.e. keep every shortlisted paper and "
                         "let the signals decide)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        state = load_state(args.state)
        load_reading_order(state)          # refuse early, before reading input
        text, source = read_input(args)
        report = build_report(text, state, args.min_similarity)
    except AnalyzeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report, source)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
