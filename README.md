# Calyb — RAG paper knowledge graph

A small, configuration-driven pipeline that turns a cache of Semantic Scholar
papers into a knowledge graph, and then maps a *new* input onto that graph as a
grounded reading order.

Everything runs offline once the cache exists. There are no LLM calls, no NER
and no relation extraction at build or run time: the graph contains exactly what
the YAML rules declare, and every edge records the rule that created it plus the
text that triggered it.

The pipeline is three commands, in order:

1. `python src/fetch_papers.py` — fetch papers and citation edges into
   `data/raw_papers.json`.
2. `python src/build_graph.py` — execute `config/schema.yaml` and
   `config/vocab.yaml` against the cache, write `data/knowledge_state.json`.
3. `python src/analyze.py --text "..."` — rank the papers for a new input.

The design rationale (what the types and rules mean, and why) lives in
`approach.md`. This README only covers how to run things.

## Installation

Python 3.12 is what this repo is built and tested on (the test suite also
passes on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\Activate.ps1     # PowerShell (Windows)

pip install -r requirements.txt
pip install pytest               # tests only; deliberately not pinned in requirements.txt
```

Dependencies: `requests`, `pandas`, `networkx`, `scikit-learn`, `PyYAML`
(see `requirements.txt` for why PyYAML is there).

## Optional: Semantic Scholar API key

`S2_API_KEY` is optional. Without it the fetcher runs against the anonymous
rate limit, which is much slower but works for a small corpus.

PowerShell (Windows):

```powershell
$env:S2_API_KEY = "your-key-here"     # current session only
python src/fetch_papers.py
Remove-Item Env:S2_API_KEY            # drop it again when done
```

Mac / Linux:

```bash
export S2_API_KEY="your-key-here"     # current session only
python src/fetch_papers.py
unset S2_API_KEY                      # drop it again when done
```

**Never commit the key.** Keep it out of files you commit, out of commit
messages, and out of issues. `.gitignore` already excludes `.env`, so a
`.env` file is safe locally — but the shell examples above are the recommended
way. The fetcher only ever prints `api key : yes (S2_API_KEY)`, never the key
itself.

`S2_BASE_URL` is also honoured if you need to point at a mirror; nothing else
in the pipeline uses environment variables.

## Fetching the papers

```bash
# print the fetch plan and make no API calls at all
python src/fetch_papers.py --dry-run

# inspect the existing cache without touching the config or the API
python src/fetch_papers.py --validate

# normal run: top the cache up from config/schema.yaml's fetch section
python src/fetch_papers.py

# discard the cache and refetch everything from scratch
python src/fetch_papers.py --reset
```

The fetch is incremental: re-running only fetches what is missing, and each
paper is fetched at most once. Other useful flags: `--limit N` (fetch at most
N new papers), `--refresh` (refetch metadata for papers already cached),
`--no-edges` (skip citation edges; much faster). Run with no arguments to see
the full flag list.

Note that `--dry-run` reports "nothing to do" when the cache already satisfies
`fetch.num_papers` and every paper has its edges — combine it with `--reset`
or `--refresh` if you want to see the plan for a real run.

## Regenerating the knowledge state

```bash
python src/build_graph.py
```

Reads `data/raw_papers.json`, `config/schema.yaml` and `config/vocab.yaml`,
and writes `data/knowledge_state.json` (the graph plus provenance, in
human-readable JSON). The command refuses to run and explains itself if the
config still contains unfilled `TODO` placeholders or declares no node/edge
types.

## Running the CLI on a new input

```bash
# 1. make sure the knowledge state is current
python src/build_graph.py

# 2. ask the question
python src/analyze.py --text "how do dense retrievers improve question answering?"
```

Other ways to supply the input:

```bash
python src/analyze.py --file my_question.txt      # from a file
echo "my question" | python src/analyze.py        # from stdin
python src/analyze.py --text "..." --json         # machine-readable report
python src/analyze.py --text "..." --out result.json   # also write the JSON
```

## Running the tests

```bash
python -m pytest            # whole suite
python -m pytest tests/test_build_graph.py -v     # one file
```

Each test file is also a standalone runner:

```bash
python tests/test_build_graph.py
python tests/test_analyze.py
python tests/test_fetch_papers.py
```

The tests are fully offline: `tests/test_fetch_papers.py` spins up a local mock
of the four Semantic Scholar endpoints (pytest starts it automatically via
`tests/conftest.py`), and `tests/fixtures/` holds a tiny hand-made schema,
vocab and paper cache that has nothing to do with the real config.

## File map

```
README.md                 this file
approach.md               design document (owner-written)
requirements.txt          runtime dependencies + install note

config/schema.yaml        fetch settings and the graph's declared types/rules
config/vocab.yaml         named keyword vocabularies referenced by the rules

src/fetch_papers.py       Semantic Scholar fetcher -> data/raw_papers.json
src/build_graph.py        YAML-driven graph builder -> data/knowledge_state.json
src/analyze.py            CLI: new input -> grounded reading order

data/raw_papers.json      fetched cache (committed)
data/knowledge_state.json generated by build_graph.py (not committed until built)
check_corpus.py           ad-hoc script for eyeballing the cache; not part of the pipeline

tests/conftest.py         starts the mock API for pytest
tests/fixtures/           tiny hand-made schema.yaml, vocab.yaml, raw_papers.json
tests/test_fetch_papers.py
tests/test_build_graph.py
tests/test_analyze.py
```

## Known limitations (data)

- **Citation edges are capped at 100 per paper, per direction.** From the cache
  meta (`data/raw_papers.json` → `meta`): `max_edges_per_paper` is `100` and
  `edges_truncated_count` is **32** — 32 paper/direction endpoints hit the cap,
  so those papers have *at least* 100 edges and the remainder were never
  fetched. `meta.edges_truncated` lists each one.
- **Papers were selected by keyword search relevance.** The corpus is whatever
  the seed queries returned from Semantic Scholar's search endpoint in the
  configured year range — not a curated reading list, so coverage follows the
  queries.
- **4 papers have no abstract** (of 70). Anything that reads abstracts matches
  on their title and venue only. `python src/fetch_papers.py --validate`
  prints their ids.
