"""加入 2026-11-17 更新包的資料，AoV 協助的命中率會不會變好？

A. 測試集固定為原本 10/13 那包的序列（5 折，與 cvflow_learning_curve.py 相同的切法），
   訓練資料 = 原本序列的其餘 4 折 + 新資料（0%、25%、50%、100%），另外比較「只用新資料訓練」。
B. 新資料本身的命中率：新資料切 5 折，比較三種訓練資料：只用原本序列、原本序列 + 新資料的其餘 4 折、
   只用新資料的其餘 4 折。

新資料以序列代號 CVF2- 開頭辨識（kcore.adapters.cvflow_aov）。
用法（在 k_core/backend 執行）：.venv\\Scripts\\python.exe scripts\\cvflow_more_data.py
"""
import argparse
import json
import random
import sys
from collections import Counter
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

parser = argparse.ArgumentParser()
parser.add_argument("--parts", default="AB")
PARTS = parser.parse_args().parts.upper()
OUTPUT = ROOT / "cvflow_more_data.json"

service = KCoreService(make_session_factory(f"sqlite:///{ROOT / 'kcore.db'}"))
assist = AssistService(service)
rows = cv.load(service, "CVFLOW", {})
original = [r for r in rows if not r["episode_id"].startswith("CVF2-")]
new = [r for r in rows if r["episode_id"].startswith("CVF2-")]
print(f"original {len(original)} episodes / {sum(len(r['steps']) for r in original)} steps; "
      f"new {len(new)} / {sum(len(r['steps']) for r in new)}", flush=True)


def evaluate(train, test, label):
    library, _ = cv.build_library(train)
    tokens = [[s["token"] for s in r["steps"]] for r in train]
    vocab = Vocabulary.build(tokens, 1)
    norm = [[vocab.normalize(t) for t in seq] for seq in tokens]
    trained = train_with_selection([vocab.encode(s) for s in norm], vocab.symbols, cv.CONFIG)
    mix = MixturePredictor.fit(trained.model, norm, cv.CONFIG.smoothing).predict
    bigram = FrequencyBaselines(norm, vocab.symbols, cv.CONFIG.smoothing).predict_bigram
    out = []
    for row in test:
        prefix = []
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
            out.append({"label": label, "teacher": row["teacher"], "op": step["op"], "pos": t,
                        "length": len(row["steps"]), "covered": key in ranked,
                        "hit1": key in ranked[:1], "hit3": key in ranked[:3], "hit5": key in ranked[:5],
                        "hit3_bigram": key in cv.rank(candidates, p_bi, overall)[:3],
                        "hit3_freq": key in cv.rank(candidates, None, overall)[:3],
                        "states": int(trained.model.n_states) if hasattr(trained.model, "n_states") else None})
            prefix.append(vocab.normalize(step["token"]))
    return out


records = []
if OUTPUT.exists():
    records = [r for r in json.loads(OUTPUT.read_text(encoding="utf-8")) if r["part"] not in PARTS]
order = list(range(len(original)))
random.Random(0).shuffle(order)
CONDITIONS = [("original only", 0.0), ("+ new 25%", 0.25), ("+ new 50%", 0.5), ("+ new 100%", 1.0), ("new only", None)]
for fold in range(5 if "A" in PARTS else 0):
    test_idx = set(order[fold::5])
    orig_train = [r for i, r in enumerate(original) if i not in test_idx]
    test = [r for i, r in enumerate(original) if i in test_idx]
    for label, frac in CONDITIONS:
        if frac is None:
            train = new
        else:
            train = orig_train + random.Random(fold).sample(new, round(len(new) * frac))
        records += [{**r, "part": "A"} for r in evaluate(train, test, label)]
        print(f"A fold {fold + 1} {label}: train {len(train)} episodes", flush=True)

new_order = list(range(len(new)))
random.Random(0).shuffle(new_order)
for fold in range(5 if "B" in PARTS else 0):
    test_idx = set(new_order[fold::5])
    test = [r for i, r in enumerate(new) if i in test_idx]
    new_train = [r for i, r in enumerate(new) if i not in test_idx]
    for label, train in (("original only", original), ("original + new", original + new_train),
                         ("new only", new_train)):
        records += [{**r, "part": "B"} for r in evaluate(train, test, label)]
        print(f"B fold {fold + 1} {label}: train {len(train)} episodes", flush=True)

OUTPUT.write_text(json.dumps(records), encoding="utf-8")


def summary(part, labels, title):
    print(f"\n== {title}")
    print(f"{'condition':16s} {'steps':>6s} {'hit1':>6s} {'hit3':>6s} {'hit5':>6s} {'bigram3':>8s} {'freq3':>6s} {'cover':>6s} {'long8+':>7s}  hit3 by op")
    for label in labels:
        g = [r for r in records if r["part"] == part and r["label"] == label]
        m = lambda f, g=g: sum(f(r) for r in g) / len(g)  # noqa: E731
        long = [r for r in g if r["length"] >= 8]
        by_op = {op: f"{sum(r['hit3'] for r in g if r['op'] == op) / n:.2f}"
                 for op, n in Counter(r["op"] for r in g).most_common()}
        print(f"{label:16s} {len(g):6d} {m(lambda r: r['hit1']):6.3f} {m(lambda r: r['hit3']):6.3f} {m(lambda r: r['hit5']):6.3f} "
              f"{m(lambda r: r['hit3_bigram']):8.3f} {m(lambda r: r['hit3_freq']):6.3f} {m(lambda r: r['covered']):6.3f} "
              f"{(sum(r['hit3'] for r in long) / len(long)) if long else 0:7.3f}  {by_op}")


summary("A", [c for c, _ in CONDITIONS], "A. test = original 10/13 episodes (5-fold)")
summary("B", ["original only", "original + new", "new only"], "B. test = new 11/17 episodes (5-fold)")
