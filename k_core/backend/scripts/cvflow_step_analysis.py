"""Why does hit rate drop with more steps? Split by step position vs. episode length, 5-fold CV."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cvflow_assist_cv as cv  # noqa: E402
from kcore.assist import AssistService  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.encoding import Vocabulary, event_token  # noqa: E402
from kcore.evaluation import FrequencyBaselines  # noqa: E402
from kcore.hmm.hybrid import MixturePredictor  # noqa: E402
from kcore.hmm.training import train_with_selection  # noqa: E402
from kcore.service import KCoreService  # noqa: E402

service = KCoreService(make_session_factory(f"sqlite:///{ROOT / 'kcore.db'}"))
assist = AssistService(service)
rows = cv.load(service, "CVFLOW", {})
order = list(range(len(rows)))
random.Random(0).shuffle(order)
records = []
for fold in range(5):
    test_idx = set(order[fold::5])
    train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]
    library, _ = cv.build_library(train)
    tokens = [[s["token"] for s in r["steps"]] for r in train]
    vocab = Vocabulary.build(tokens, 1)
    norm = [[vocab.normalize(t) for t in seq] for seq in tokens]
    trained = train_with_selection([vocab.encode(s) for s in norm], vocab.symbols, cv.CONFIG)
    mix = MixturePredictor.fit(trained.model, norm, cv.CONFIG.smoothing).predict
    bigram = FrequencyBaselines(norm, vocab.symbols, cv.CONFIG.smoothing).predict_bigram
    for row in test:
        prefix = []
        n = len(row["steps"])
        for t, step in enumerate(row["steps"]):
            nodes, edges = step["before"]
            candidates, _ = assist._candidates(nodes, edges, library)
            for c in candidates:
                c["token"] = vocab.normalize(event_token(c["operation_type"], {"activity_type": c["activity_type"]}, cv.SCHEMA))
            key = (step["op"], step["type"])
            p_mix = dict(zip(vocab.symbols, (float(p) for p in mix(prefix)), strict=True))
            p_bi = dict(zip(vocab.symbols, (float(p) for p in bigram(prefix)), strict=True))
            overall = lambda k: library.keys[k]  # noqa: E731
            ranked = cv.rank(candidates, p_mix, overall)
            ranked_bi = cv.rank(candidates, p_bi, overall)
            ranked_freq = cv.rank(candidates, None, overall)
            r = ranked.index(key) + 1 if key in ranked else None
            records.append({
                "teacher": row["teacher"], "length": n, "pos": t, "rel": t / max(1, n - 1), "op": step["op"],
                "nodes": len(nodes), "candidates": len(candidates), "covered": r is not None,
                "hit3": r is not None and r <= 3,
                "hit3_bigram": key in ranked_bi[:3], "hit3_freq": key in ranked_freq[:3],
                "same_op_as_prev": t > 0 and row["steps"][t - 1]["op"] == step["op"],
            })
            prefix.append(vocab.normalize(step["token"]))
    print("fold", fold + 1, flush=True)

(ROOT / "cvflow_step_analysis.json").write_text(json.dumps(records), encoding="utf-8")


def table(title, key_fn, keys):
    print(f"\n== {title}")
    print(f"{'group':>10s} {'steps':>6s} {'hit3':>6s} {'bigram':>7s} {'freq':>6s} {'cover':>6s} {'cands':>6s} {'nodes':>6s}  op mix")
    for k in keys:
        g = [r for r in records if key_fn(r) == k]
        if not g:
            continue
        m = lambda f: sum(f(r) for r in g) / len(g)  # noqa: E731
        ops = Counter(r["op"] for r in g)
        mix = " ".join(f"{op[:6]}={c / len(g):.0%}" for op, c in ops.most_common(4))
        print(f"{str(k):>10s} {len(g):6d} {m(lambda r: r['hit3']):6.2f} {m(lambda r: r['hit3_bigram']):7.2f} "
              f"{m(lambda r: r['hit3_freq']):6.2f} {m(lambda r: r['covered']):6.2f} {m(lambda r: r['candidates']):6.1f} "
              f"{m(lambda r: r['nodes']):6.1f}  {mix}")


def pos_bucket(r):
    p = r["pos"]
    return "1" if p == 0 else "2" if p == 1 else "3-4" if p < 4 else "5-7" if p < 7 else "8+"


def len_bucket(r):
    n = r["length"]
    return "1-2" if n <= 2 else "3-4" if n <= 4 else "5-7" if n <= 7 else "8-10" if n <= 10 else "11+"


B = ["1", "2", "3-4", "5-7", "8+"]
L = ["1-2", "3-4", "5-7", "8-10", "11+"]
table("by step position (all episodes)", pos_bucket, B)
table("by episode length", len_bucket, L)
for lb in ("5-7", "8-10", "11+"):
    table(f"by step position, episodes of length {lb} only", lambda r, lb=lb: pos_bucket(r) if len_bucket(r) == lb else None, B)
table("by teacher", lambda r: r["teacher"], ["codex", "gemini", "opus", "sonnet", "haiku"])
print("\nmean length by teacher:", {t: round(sum(r["length"] for r in records if r["teacher"] == t and r["pos"] == 0) /
                                         max(1, sum(1 for r in records if r["teacher"] == t and r["pos"] == 0)), 1)
                                     for t in ["codex", "gemini", "opus", "sonnet", "haiku"]})
