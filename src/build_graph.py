"""Build a knowledge graph from cached papers, driven entirely by config/schema.yaml.

    python src/build_graph.py

Reads
    data/raw_papers.json   cached Semantic Scholar output
    config/schema.yaml     node types, edge types, rules, reading order
    config/vocab.yaml      named keyword vocabularies

Writes
    data/knowledge_state.json   the whole graph plus provenance, human-readable

Design rules this file obeys
---------------------------
* No entity type, relationship type or mapping rule is defined here. The engine
  executes only what the YAML files declare. If a config section is empty, the
  corresponding part of the graph is empty.
* No automatic entity or relation extraction. A node or edge exists only where a
  configured rule fires on a configured condition, and every edge records which
  rule created it and the text that triggered it.
* Unknown keys in a condition are an error, not a silently ignored branch. A
  typo in a rule must not quietly produce a smaller graph.

Condition reference (the exact set the engine implements)
-------------------------------------------------------
A condition is a single-key mapping. Groups combine with `all` (AND) and `any`
(OR) and nest arbitrarily.

    all:    [<condition>, ...]        every child must match
    any:    [<condition>, ...]        at least one child must match
    vocab:  {name: <str>, where: [title, abstract, venue, contexts]}
                                    any term of vocabulary <name> occurs in one
                                    of the fields, case-insensitively, on word
                                    boundaries. Default where: title+abstract.
    regex:  {pattern: <str>, where: [...]}
                                    Python regex, case-insensitive.
    year:   {gte|lte|gt|lt|eq: <int>} or {between: [a, b]}
    citation_count: {gte|lte: <int>}
    edge:   {influential: <bool>, intents_any: [<str>, ...],
             contexts_any: <regex>, direction: references|citations|both}
                                    Conditions on the paper's citation edges.
                                    `direction` defaults to `both` here; inside
                                    an edge_type rule it defaults to that rule's
                                    own `direction`. `contexts_any` is a regex
                                    tested against the edge's context sentences.

Edge rules
----------
    rule.target: vocabulary_term | citation
    rule.direction: references | citations      (required when target: citation)
    rule.subject: <conditions>                  which papers the rule starts from
    rule.object:  <conditions>                  citation only: the neighbour
    rule.edge:    <conditions>                  citation only: the edge itself

`target: vocabulary_term` creates one edge per distinct *vocabulary* term that
matched the subject, in the order the subject's `all:`/`any:` tree produced them.
Regex hits in the subject filter and are recorded in the edge's evidence, but do
not themselves create term nodes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = PROJECT_ROOT / "config" / "schema.yaml"
DEFAULT_VOCAB = PROJECT_ROOT / "config" / "vocab.yaml"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "raw_papers.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "knowledge_state.json"

# Fields a condition's `where:` may name.
TEXT_FIELDS = ("title", "abstract", "venue", "contexts")

# Condition key -> key in the cached paper record. The cache is camelCase
# (Semantic Scholar's own shape) while conditions read snake_case.
CONDITION_FIELD = {"year": "year", "citation_count": "citationCount"}

# Numeric comparators accepted by `year:` and `citation_count:`.
NUMERIC_OPS = ("gte", "gt", "lte", "lt", "eq")

# Leaf condition keys the engine understands. Anything else is a typo.
LEAF_CONDITIONS = ("vocab", "regex", "year", "citation_count", "edge")


class ConfigError(Exception):
    """Raised when the config cannot be executed as written."""


# ==========================================================================
# loading and validation
# ==========================================================================
def reject_todo_placeholders(path: Path) -> None:
    """Refuse to run against a config template that still contains `TODO`.

    A half-filled config would quietly produce a smaller, wrong graph, so this
    fails loudly and names every offending line.
    """
    offenders = [
        (i, line.strip())
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "TODO" in line and not line.lstrip().startswith("#")
    ]
    if not offenders:
        return
    shown = "\n".join(f"    line {i}: {text[:72]}" for i, text in offenders[:12])
    more = f"\n    ... and {len(offenders) - 12} more" if len(offenders) > 12 else ""
    raise ConfigError(
        f"{path.name} still contains {len(offenders)} unfilled placeholder(s):\n"
        f"{shown}{more}\n"
        f"       Replace every TODO with your own value, or delete the block."
    )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    reject_todo_placeholders(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise ConfigError(f"{path.name} is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping at the top level")
    return data


def build_vocab_index(vocab: dict) -> dict[str, dict]:
    """vocabularies: [...] -> {name: {description, terms: [...]}}."""
    out: dict[str, dict] = {}
    for entry in vocab.get("vocabularies") or []:
        name = (entry or {}).get("name")
        if not name:
            raise ConfigError("every vocabulary needs a `name`")
        if name in out:
            raise ConfigError(f"vocabulary {name!r} is defined twice")
        terms = entry.get("terms") or []
        if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
            raise ConfigError(f"vocabulary {name!r}: `terms` must be a list of strings")
        out[name] = {
            "description": (entry.get("description") or "").strip(),
            "terms": terms,
        }
    if not out:
        raise ConfigError("vocab.yaml defines no vocabularies")
    return out


def validate_schema(schema: dict, vocab_index: dict[str, dict]) -> None:
    """Check the schema is executable, before any graph work happens."""
    node_types = schema.get("node_types")
    edge_types = schema.get("edge_types")

    if not node_types:
        raise ConfigError(
            "schema.yaml has an empty `node_types` list.\n"
            "       Declare at least one entity type, e.g.\n"
            "         node_types:\n"
            "           - name: Paper\n"
            "             description: ...\n"
            "             source: Paper\n"
            "             properties: [paperId, title]"
        )
    if not edge_types:
        raise ConfigError(
            "schema.yaml has an empty `edge_types` list.\n"
            "       Declare at least one relationship type and the rule that "
            "creates it."
        )

    for required in ("name", "description"):
        for i, spec in enumerate(node_types):
            value = spec.get(required)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"node_types[{i}] needs a non-empty `{required}`")
    names = [n["name"] for n in node_types]
    for name in names:
        if names.count(name) > 1:
            raise ConfigError(f"node type {name!r} is declared twice")
    for n in node_types:
        unknown = set(n) - {"name", "description", "source", "properties"}
        if unknown:
            raise ConfigError(f"node type {n.get('name')!r}: unknown key(s) "
                              f"{sorted(unknown)}")

    for e in edge_types:
        for required in ("name", "description", "source", "target", "rule"):
            if not e.get(required):
                raise ConfigError(f"edge type {e.get('name', '<unnamed>')!r} is missing "
                                  f"`{required}`")
        if e["source"] not in names:
            raise ConfigError(f"edge type {e['name']!r}: source {e['source']!r} is not a "
                              f"declared node type")
        if e["target"] not in names:
            raise ConfigError(f"edge type {e['name']!r}: target {e['target']!r} is not a "
                              f"declared node type")
        rule = e["rule"]
        target_kind = rule.get("target")
        if target_kind not in ("vocabulary_term", "citation"):
            raise ConfigError(f"edge type {e['name']!r}: rule.target must be "
                              f"'vocabulary_term' or 'citation', got {target_kind!r}")
        if target_kind == "citation":
            direction = rule.get("direction")
            if direction not in ("references", "citations"):
                raise ConfigError(f"edge type {e['name']!r}: rule.direction must be "
                                  f"'references' or 'citations' for target: citation")
        else:
            if "direction" in rule:
                raise ConfigError(f"edge type {e['name']!r}: rule.direction is only "
                                  f"meaningful for target: citation")

    # Vocabulary references must resolve, anywhere they appear.
    for e in edge_types:
        _check_condition_keys(e["rule"].get("subject"), f"edge_types[{e['name']}].subject",
                              vocab_index)
        _check_condition_keys(e["rule"].get("object"), f"edge_types[{e['name']}].object",
                              vocab_index)
        _check_condition_keys(e["rule"].get("edge"), f"edge_types[{e['name']}].edge",
                              vocab_index)
        if e["rule"]["target"] == "vocabulary_term":
            # Without a vocab condition somewhere in the subject there is no term
            # to point at, and the rule would silently create zero edges.
            if e["rule"].get("subject") is None:
                raise ConfigError(
                    f"edge type {e['name']!r}: rule.target is vocabulary_term, so "
                    f"rule.subject is required (it decides which terms can match)")
            if not _contains_vocab(e["rule"]["subject"]):
                raise ConfigError(
                    f"edge type {e['name']!r}: rule.target is vocabulary_term, so "
                    f"rule.subject must contain at least one `vocab:` condition.\n"
                    f"       Only a vocab hit can name a target node; a regex hit "
                    f"alone would produce no edges.")

    reading_order = schema.get("reading_order")
    if reading_order:
        for i, signal in enumerate(reading_order.get("signals") or []):
            if not signal.get("name"):
                raise ConfigError(f"reading_order.signals[{i}] needs a `name`")
            if signal.get("weight") is None:
                raise ConfigError(f"reading_order signal {signal['name']!r} needs a "
                                  f"`weight`")
            _check_condition_keys(signal.get("rule"),
                                  f"reading_order.signals[{signal['name']}].rule",
                                  vocab_index)


def _contains_vocab(conds: Any) -> bool:
    """True if a condition tree contains a `vocab:` leaf anywhere."""
    if not isinstance(conds, dict) or len(conds) != 1:
        return False
    key, body = next(iter(conds.items()))
    if key == "vocab":
        return True
    if key in ("all", "any"):
        return any(_contains_vocab(child) for child in body)
    return False


def _check_condition_keys(conds: Any, where: str,
                          vocab_index: dict[str, dict] | None = None) -> None:
    """Recursively reject unknown condition keys and unresolvable vocab names."""
    if conds is None:
        return
    if not isinstance(conds, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(conds).__name__}")
    if len(conds) != 1:
        raise ConfigError(f"{where}: a condition must have exactly one key, got "
                          f"{sorted(conds)}")
    key, body = next(iter(conds.items()))
    if key in ("all", "any"):
        if not isinstance(body, list) or not body:
            raise ConfigError(f"{where}.{key}: must be a non-empty list of conditions")
        for i, child in enumerate(body):
            _check_condition_keys(child, f"{where}.{key}[{i}]", vocab_index)
        return
    if key not in LEAF_CONDITIONS:
        raise ConfigError(f"{where}: unknown condition {key!r}. Supported: "
                          f"{sorted(LEAF_CONDITIONS)} plus all/any.")
    if key == "vocab":
        if not isinstance(body, dict) or not body.get("name"):
            raise ConfigError(f"{where}.vocab: needs a `name`")
        if vocab_index is not None and body["name"] not in vocab_index:
            known = ", ".join(sorted(vocab_index)) or "none"
            raise ConfigError(
                f"{where}.vocab: vocabulary {body['name']!r} is not defined in "
                f"vocab.yaml (defined: {known})")
    if key == "regex":
        if not isinstance(body, dict) or not body.get("pattern"):
            raise ConfigError(f"{where}.regex: needs a `pattern`")
        try:
            re.compile(body["pattern"])
        except re.error as exc:
            raise ConfigError(f"{where}.regex: bad pattern {body['pattern']!r}: {exc}")
    if key in ("year", "citation_count"):
        if not isinstance(body, dict) or not body:
            raise ConfigError(f"{where}.{key}: needs one of {NUMERIC_OPS} or `between`")
        if "between" in body:
            pair = body["between"]
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ConfigError(f"{where}.{key}.between: must be [min, max]")
        else:
            ops = [o for o in body if o in NUMERIC_OPS]
            if not ops:
                raise ConfigError(f"{where}.{key}: needs one of {NUMERIC_OPS} or "
                                  f"`between`, got {sorted(body)}")
    if key == "edge":
        if not isinstance(body, dict) or not body:
            raise ConfigError(f"{where}.edge: needs at least one of influential, "
                              f"intents_any, contexts_any")
        if "direction" in body and body["direction"] not in ("references", "citations",
                                                             "both"):
            raise ConfigError(f"{where}.edge.direction must be references, citations "
                              f"or both")
        if "contexts_any" in body:
            try:
                re.compile(body["contexts_any"])
            except re.error as exc:
                raise ConfigError(f"{where}.edge.contexts_any: bad regex: {exc}")
    if key == "vocab" and body.get("where"):
        _check_where(body["where"], where)
    if key == "regex" and body.get("where"):
        _check_where(body["where"], where)


def _check_where(where: Any, where_path: str) -> None:
    if not isinstance(where, list) or not where:
        raise ConfigError(f"{where_path}: `where` must be a non-empty list")
    unknown = [f for f in where if f not in TEXT_FIELDS]
    if unknown:
        raise ConfigError(f"{where_path}: unknown field(s) {unknown}. Supported: "
                          f"{list(TEXT_FIELDS)}")


# ==========================================================================
# condition evaluation
# ==========================================================================
@dataclass
class Match:
    """The result of evaluating a condition tree against one paper."""

    matched: bool = False
    strength: int = 0
    evidence: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.matched


def _term_pattern(term: str) -> re.Pattern:
    """Word-boundary, case-insensitive matcher for one vocabulary term.

    Lookarounds rather than \\b, because \\b misbehaves for terms that begin or
    end with a non-word character.
    """
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def paper_fields(paper: dict) -> dict[str, list[tuple[str, str]]]:
    """Field name -> [(source_text, label)] for the fields conditions may read.

    `contexts` is special: it is a list of citation sentences, so each sentence
    is its own searchable unit and the label records where it came from.
    """
    fields: dict[str, list[tuple[str, str]]] = {
        "title": [(paper.get("title") or "", "title")],
        "abstract": [(paper.get("abstract") or "", "abstract")],
        "venue": [(paper.get("venue") or "", "venue")],
        "contexts": [],
    }
    for direction in ("references", "citations"):
        for other_id, meta in (paper.get(direction) or {}).items():
            for i, sentence in enumerate((meta or {}).get("contexts") or []):
                fields["contexts"].append(
                    (sentence, f"{direction}[{other_id}].contexts[{i}]"))
    return fields


def compare(value: Any, op: str, target: Any) -> bool:
    if value is None:
        return False
    try:
        if op == "gte":
            return value >= target
        if op == "gt":
            return value > target
        if op == "lte":
            return value <= target
        if op == "lt":
            return value < target
        if op == "eq":
            return value == target
    except TypeError:
        return False
    raise ConfigError(f"unknown comparator {op!r}")


class RuleEngine:
    """Evaluates conditions and applies edge rules. Holds no schema of its own."""

    def __init__(self, vocab_index: dict[str, dict]) -> None:
        self.vocab = vocab_index
        self._term_cache: dict[str, re.Pattern] = {}

    # -- helpers -----------------------------------------------------------
    def term_pattern(self, term: str) -> re.Pattern:
        if term not in self._term_cache:
            self._term_cache[term] = _term_pattern(term)
        return self._term_cache[term]

    @staticmethod
    def _fields_for(paper: dict, where: list[str] | None) -> dict[str, list[tuple[str, str]]]:
        fields = paper_fields(paper)
        if not where:
            where = ["title", "abstract"]
        return {f: fields.get(f, []) for f in where}

    # -- evaluation --------------------------------------------------------
    def evaluate(self, conds: Any, paper: dict,
                 default_direction: str | None = None) -> Match:
        """Evaluate a condition tree against one paper record."""
        if conds is None:
            return Match(matched=True, strength=0)
        key, body = next(iter(conds.items()))

        if key == "all":
            total_strength = 0
            evidence: list[dict] = []
            for child in body:
                result = self.evaluate(child, paper, default_direction)
                if not result.matched:
                    return Match(matched=False)
                total_strength += result.strength
                evidence.extend(result.evidence)
            return Match(matched=True, strength=total_strength, evidence=evidence)

        if key == "any":
            total_strength = 0
            evidence: list[dict] = []
            matched_any = False
            for child in body:
                result = self.evaluate(child, paper, default_direction)
                if result.matched:
                    matched_any = True
                    total_strength += result.strength
                    evidence.extend(result.evidence)
            if not matched_any:
                return Match(matched=False)
            return Match(matched=True, strength=total_strength, evidence=evidence)

        if key == "vocab":
            return self._eval_vocab(body, paper)
        if key == "regex":
            return self._eval_regex(body, paper)
        if key == "year":
            return self._eval_numeric("year", body, paper)
        if key == "citation_count":
            return self._eval_numeric("citation_count", body, paper)
        if key == "edge":
            return self._eval_edge(body, paper, default_direction)

        raise ConfigError(f"unknown condition {key!r}")

    def _eval_vocab(self, body: dict, paper: dict) -> Match:
        """One hit per vocabulary term, in the first field listed that contains it.

        Deduplicating per term (rather than per term+field) keeps `strength`
        meaningful for reading_order scoring and stops a term appearing in both
        the title and the abstract from counting twice.
        """
        name = body["name"]
        if name not in self.vocab:
            raise ConfigError(f"vocabulary {name!r} is not defined in vocab.yaml")
        terms = self.vocab[name]["terms"]
        where = self._fields_for(paper, body.get("where"))

        found: dict[str, dict] = {}
        for entries in where.values():
            for text, label in entries:
                if not text:
                    continue
                for term in terms:
                    if term in found:
                        continue
                    m = self.term_pattern(term).search(text)
                    if m:
                        found[term] = {
                            "kind": "vocab_term",
                            "vocabulary": name,
                            "term": term,
                            "field": label,
                            "matched_text": m.group(0),
                        }
            if len(found) == len(terms):
                break

        evidence = list(found.values())
        return Match(matched=bool(evidence), strength=len(evidence), evidence=evidence)

    def _eval_regex(self, body: dict, paper: dict) -> Match:
        pattern = re.compile(body["pattern"], re.IGNORECASE)
        where = self._fields_for(paper, body.get("where"))
        evidence: list[dict] = []
        for field_name, entries in where.items():
            for text, label in entries:
                for m in pattern.finditer(text or ""):
                    evidence.append({
                        "kind": "regex",
                        "pattern": body["pattern"],
                        "field": label,
                        "matched_text": m.group(0),
                    })
        return Match(matched=bool(evidence), strength=len(evidence), evidence=evidence)

    def _eval_numeric(self, key: str, body: dict, paper: dict) -> Match:
        field = CONDITION_FIELD[key]
        value = paper.get(field)
        if "between" in body:
            low, high = body["between"]
            try:
                ok = value is not None and low <= value <= high
            except TypeError:
                # Mixed types (e.g. a string year) are a non-match, exactly as
                # `compare()` treats them, not a crash mid-build.
                ok = False
            detail = f"{low} <= {key} <= {high}"
        else:
            op = next(o for o in body if o in NUMERIC_OPS)
            ok = compare(value, op, body[op])
            detail = f"{key} {op} {body[op]}"
        evidence = ([{"kind": "numeric", "field": field, "test": detail,
                      "actual": value}] if ok else [])
        return Match(matched=ok, strength=1 if ok else 0, evidence=evidence)

    def _eval_edge(self, body: dict, paper: dict, default_direction: str | None) -> Match:
        direction = body.get("direction") or default_direction or "both"
        directions = (["references", "citations"] if direction == "both"
                      else [direction])

        influential = body.get("influential")
        intents_any = [i.lower() for i in (body.get("intents_any") or [])]
        contexts_any = (re.compile(body["contexts_any"], re.IGNORECASE)
                        if body.get("contexts_any") else None)

        evidence: list[dict] = []
        for d in directions:
            for other_id, meta in (paper.get(d) or {}).items():
                meta = meta or {}
                if influential is not None and bool(meta.get("isInfluential")) != influential:
                    continue
                if intents_any:
                    got = [str(i).lower() for i in (meta.get("intents") or [])]
                    if not any(i in got for i in intents_any):
                        continue
                if contexts_any is not None:
                    sentences = meta.get("contexts") or []
                    hit = next((s for s in sentences
                                if s and contexts_any.search(s)), None)
                    if hit is None:
                        continue
                else:
                    hit = None
                evidence.append({
                    "kind": "citation_edge",
                    "direction": d,
                    "other_paper_id": other_id,
                    "isInfluential": bool(meta.get("isInfluential")),
                    "intents": meta.get("intents") or [],
                    "context_snippet": hit or (
                        (meta.get("contexts") or [None])[0]),
                })
        return Match(matched=bool(evidence), strength=len(evidence), evidence=evidence)


# ==========================================================================
# graph construction
# ==========================================================================
def paper_node_id(paper_id: str) -> str:
    return f"paper:{paper_id}"


def term_node_id(term: str) -> str:
    return f"term:{term.strip().lower()}"


def build_graph(cache: dict, schema: dict, vocab_index: dict[str, dict],
                include_neighbors: bool = False,
                log=lambda m: None) -> tuple[nx.MultiDiGraph, list[str]]:
    """Build the MultiDiGraph. Returns (graph, warnings)."""
    engine = RuleEngine(vocab_index)
    graph = nx.MultiDiGraph()
    warnings: list[str] = []

    papers = cache.get("papers") or {}
    neighbors = cache.get("neighbors") or {}

    # The node type that corpus papers are materialised as. Declared via
    # `source: Paper`; if several qualify, the first wins and we warn.
    paper_node_types = [n["name"] for n in schema["node_types"]
                        if (n.get("source") or "Paper") == "Paper"]
    if not paper_node_types:
        raise ConfigError(
            "no node type has `source: Paper`, so there is nowhere to put the "
            "corpus papers.\n"
            "       Add e.g. `source: Paper` to the type that represents a paper."
        )
    if len(paper_node_types) > 1:
        warnings.append(
            f"several node types declare source: Paper ({', '.join(paper_node_types)}); "
            f"using {paper_node_types[0]!r}"
        )
    paper_type = paper_node_types[0]

    type_by_name = {n["name"]: n for n in schema["node_types"]}

    def add_paper_node(pid: str, record: dict, stub: bool) -> None:
        node_id = paper_node_id(pid)
        spec = type_by_name[paper_type]
        props = {}
        for field in (spec.get("properties") or []):
            if field in record:
                props[field] = record[field]
        graph.add_node(node_id, type=paper_type, properties=props)
        if stub:
            graph.nodes[node_id]["stub"] = True

    for pid, record in papers.items():
        add_paper_node(pid, record, stub=False)

    # Optionally give out-of-corpus edge endpoints their own (stub) nodes.
    if include_neighbors:
        for pid, record in neighbors.items():
            add_paper_node(pid, record, stub=True)

    def lookup(pid: str) -> dict | None:
        if pid in papers:
            return papers[pid]
        if include_neighbors and pid in neighbors:
            return neighbors[pid]
        return None

    # ---- apply each edge rule --------------------------------------------
    for spec in schema["edge_types"]:
        rule = spec["rule"]
        rule_name = spec["name"]
        target_type = spec["target"]

        for pid, record in papers.items():
            subject = engine.evaluate(rule.get("subject"), record,
                                      default_direction=rule.get("direction"))
            if not subject.matched:
                continue

            if rule["target"] == "vocabulary_term":
                # The documented semantics: the object is the *vocabulary* term
                # that matched. One edge per distinct term, so a term present in
                # both the title and the abstract still yields a single edge.
                # Regex conditions in the subject act as filters, not as a source
                # of term nodes (a regex like `vertical` would otherwise create a
                # meaningless `term:vertical` node).
                targets: dict[str, dict] = {}
                for item in subject.evidence:
                    if item.get("kind") != "vocab_term" or not item.get("term"):
                        continue
                    targets.setdefault(term_node_id(item["term"]), item)
                for node_id, item in targets.items():
                    term = item["term"]
                    if node_id not in graph:
                        graph.add_node(node_id, type=target_type,
                                       properties={"term": term.strip()})
                        graph.nodes[node_id]["derived_by"] = rule_name
                    elif graph.nodes[node_id]["type"] != target_type:
                        warnings.append(
                            f"term {term!r} already exists as "
                            f"{graph.nodes[node_id]['type']!r}; {rule_name} wanted "
                            f"{target_type!r}"
                        )
                        continue
                    graph.add_edge(
                        paper_node_id(pid), node_id,
                        type=rule_name,
                        rule=rule_name,
                        evidence=[item],
                    )
                continue

            # target: citation
            direction = rule["direction"]
            for other_id, meta in (record.get(direction) or {}).items():
                other = lookup(other_id)
                if other is None:
                    continue
                obj = engine.evaluate(rule.get("object"), other,
                                      default_direction=rule.get("direction"))
                if not obj.matched:
                    continue
                edge_match = engine.evaluate(rule.get("edge"), record,
                                             default_direction=direction)
                if not edge_match.matched:
                    continue

                citation_fact = {
                    "kind": "citation",
                    "direction": direction,
                    "other_paper_id": other_id,
                    "other_paper_title": other.get("title"),
                    "other_paper_year": other.get("year"),
                    "isInfluential": bool((meta or {}).get("isInfluential")),
                    "intents": (meta or {}).get("intents") or [],
                    "context_snippet": ((meta or {}).get("contexts") or [None])[0],
                }
                evidence = [citation_fact]
                evidence += subject.evidence + obj.evidence + edge_match.evidence
                graph.add_edge(
                    paper_node_id(pid), paper_node_id(other_id),
                    type=rule_name,
                    rule=rule_name,
                    evidence=evidence,
                )

    return graph, warnings


# ==========================================================================
# export
# ==========================================================================
def export_state(graph: nx.MultiDiGraph, schema: dict, vocab: dict,
                 cache: dict, warnings: list[str],
                 include_neighbors: bool) -> dict:
    """Assemble the knowledge_state.json payload, human-readable by design."""
    papers = cache.get("papers") or {}
    neighbors = cache.get("neighbors") or {}
    years = sorted({p["year"] for p in papers.values()
                    if isinstance(p.get("year"), int)})

    nodes = []
    for node_id, data in sorted(graph.nodes(data=True)):
        nodes.append({
            "id": node_id,
            "type": data["type"],
            "properties": data.get("properties", {}),
            "stub": bool(data.get("stub")),
            "derived_by": data.get("derived_by"),
            "out_degree": graph.out_degree(node_id),
            "in_degree": graph.in_degree(node_id),
        })

    edges = []
    for src, dst, key, data in sorted(
            graph.edges(keys=True, data=True),
            key=lambda e: (e[3].get("type", ""), e[0], e[1])):
        edges.append({
            "source": src,
            "target": dst,
            "type": data.get("type"),
            "rule": data.get("rule"),
            "evidence": data.get("evidence", []),
        })

    node_type_counts: dict[str, int] = {}
    for n in nodes:
        node_type_counts[n["type"]] = node_type_counts.get(n["type"], 0) + 1

    edge_type_counts: dict[str, int] = {}
    for e in edges:
        edge_type_counts[e["type"]] = edge_type_counts.get(e["type"], 0) + 1

    described_nodes = {
        spec["name"]: {
            "description": spec.get("description", ""),
            # The engine treats an omitted `source` as `Paper` (see build_graph),
            # so the export must record the effective value: analyze.py reads it
            # back to find the paper nodes.
            "source": spec.get("source") or "Paper",
            "properties": spec.get("properties", []),
            "count": node_type_counts.get(spec["name"], 0),
        }
        for spec in schema["node_types"]
    }
    described_edges = {
        spec["name"]: {
            "description": spec.get("description", ""),
            "source": spec["source"],
            "target": spec["target"],
            "rule": spec["rule"],
            "count": edge_type_counts.get(spec["name"], 0),
        }
        for spec in schema["edge_types"]
    }

    return {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": "src/build_graph.py",
            "corpus_size": len(papers),
            "corpus_years": {"min": years[0] if years else None,
                             "max": years[-1] if years else None},
            "neighbors_in_cache": len(neighbors),
            "include_neighbors": include_neighbors,
            "graph": {"nodes": len(nodes), "edges": len(edges)},
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
            "source_cache_meta": cache.get("meta", {}),
            "warnings": warnings,
        },
        "schema": schema,
        "vocabularies": vocab,
        "node_types": described_nodes,
        "edge_types": described_edges,
        "nodes": nodes,
        "edges": edges,
    }


def write_state(state: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(out_path)


# ==========================================================================
# main
# ==========================================================================
def load_cache(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(
            f"no paper cache at {path}\n"
            f"       Run: python src/fetch_papers.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build data/knowledge_state.json from the paper cache and config",
    )
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-neighbors", action="store_true",
                    help="also create stub nodes for citation endpoints that are "
                         "outside the corpus (off by default)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    def log(msg: str = "") -> None:
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    try:
        schema = load_yaml(args.schema)
        vocab_doc = load_yaml(args.vocab)
        vocab_index = build_vocab_index(vocab_doc)
        validate_schema(schema, vocab_index)
        cache = load_cache(args.cache)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        graph, warnings = build_graph(
            cache, schema, vocab_index,
            include_neighbors=args.include_neighbors, log=log)
        state = export_state(graph, schema, vocab_doc, cache, warnings,
                             args.include_neighbors)
    except ConfigError as exc:
        # Refuse with the message, not a traceback: the config is the problem.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_state(state, args.out)

    meta = state["metadata"]
    log("=" * 68)
    log("Knowledge graph build")
    log("=" * 68)
    log(f"schema         : {args.schema}")
    log(f"vocab          : {args.vocab} "
        f"({len(vocab_index)} vocabularies)")
    log(f"cache          : {args.cache} ({meta['corpus_size']} papers, "
        f"{meta['corpus_years']['min']}-{meta['corpus_years']['max']})")
    log(f"include_neighbors: {args.include_neighbors}")
    log()
    log("node types:")
    for name, info in sorted(state["node_types"].items()):
        log(f"  {name:<24} {info['count']:>5}   {info['description'][:60]}")
    log("edge types:")
    for name, info in sorted(state["edge_types"].items()):
        log(f"  {name:<24} {info['count']:>5}   {info['description'][:60]}")
    if warnings:
        log()
        log("warnings:")
        for w in warnings:
            log(f"  - {w}")
    log()
    log(f"nodes          : {meta['graph']['nodes']}")
    log(f"edges          : {meta['graph']['edges']}")
    log(f"wrote          : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
