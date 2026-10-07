import json, collections

d = json.load(open("data/raw_papers.json", encoding="utf-8"))
papers = d["papers"]
ids = set(papers)

def edge_ids(x):
    if not x:
        return []
    items = x.keys() if isinstance(x, dict) else x
    return [i.get("paperId") if isinstance(i, dict) else i for i in items]

internal = 0
degree = collections.Counter()
external_refs = collections.Counter()
for pid, p in papers.items():
    for r in edge_ids(p.get("references")):
        if r in ids:
            internal += 1
            degree[pid] += 1
            degree[r] += 1
        elif r:
            external_refs[r] += 1

print("papers:", len(papers))
print("citation links between corpus papers:", internal)
isolated = [p for p in papers if degree[p] == 0]
print("papers with zero links inside the corpus:", len(isolated))
for pid in isolated:
    p = papers[pid]
    print("  ", p.get("year"), p.get("citationCount"), (p.get("title") or "")[:70])

print("\nno abstract:")
for p in papers.values():
    if not p.get("abstract"):
        print("  ", p.get("year"), (p.get("title") or "")[:70])

print("\nyears:", dict(sorted(collections.Counter(p.get("year") for p in papers.values()).items())))

neighbors = d.get("neighbors", {})
print("\nPapers cited by many of your corpus papers but NOT in the corpus:")
for rid, n in external_refs.most_common(15):
    meta = neighbors.get(rid, {}) if isinstance(neighbors, dict) else {}
    print("  cited by", n, "|", meta.get("year"), "|", (meta.get("title") or rid)[:70])