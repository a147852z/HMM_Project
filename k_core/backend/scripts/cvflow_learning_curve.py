"""Learning curve: train on 20..100% of each fold's training episodes, test on the same held-out fold."""
import json
import random
import sys
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

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [1, 2, 3]

service = KCoreService(make_session_factory(f"sqlite:///{ROOT / 'kcore.db'}"))
assist = AssistService(service)
rows = cv.load(service, "CVFLOW", {})
order = list(range(len(rows)))
random.Random(0).shuffle(order)
results = []  # one record per (fraction, seed, fold, step)
for fold in range(5):
    test_idx = set(order[fold::5])
    full_train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]
    for frac in FRACTIONS:
        for seed in (SEEDS if frac < 1.0 else [0]):
            train = random.Random(seed * 100 + fold).sample(full_train, max(2, round(len(full_train) * frac)))
            library, _ = cv.build_library(train)
            tokens = [[s["token"] for s in r["steps"]] for r in train]
            vocab = Vocabulary.build(tokens, 1)
            norm = [[vocab.normalize(t) for t in seq] for seq in tokens]
            trained = train_with_selection([vocab.encode(s) for s in norm], vocab.symbols, cv.CONFIG)
            mix = MixturePredictor.fit(trained.model, norm, cv.CONFIG.smoothing).predict
            bigram = FrequencyBaselines(norm, vocab.symbols, cv.CONFIG.smoothing).predict_bigram
            for row in test:
                prefix = []
                for step in row["steps"]:
                    nodes, edges = step["before"]
                    candidates, _ = assist._candidates(nodes, edges, library)
                    for c in candidates:
                        c["token"] = vocab.normalize(event_token(c["operation_type"], {"activity_type": c["activity_type"]}, cv.SCHEMA))
                    key = (step["op"], step["type"])
                    p_mix = dict(zip(vocab.symbols, (float(p) for p in mix(prefix)), strict=True))
                    p_bi = dict(zip(vocab.symbols, (float(p) for p in bigram(prefix)), strict=True))
                    overall = lambda k: library.keys[k]  # noqa: E731
                    ranked = cv.rank(candidates, p_mix, overall)
                    results.append({
                        "frac": frac, "seed": seed, "fold": fold, "length": len(row["steps"]),
                        "covered": key in ranked,
                        "hit1": key in ranked[:1], "hit3": key in ranked[:3],
                        "hit3_bigram": key in cv.rank(candidates, p_bi, overall)[:3],
                        "hit3_freq": key in cv.rank(candidates, None, overall)[:3],
                        "train_eps": len(train), "train_steps": sum(len(r["steps"]) for r in train),
                    })
                    prefix.append(vocab.normalize(step["token"]))
            print(f"fold {fold + 1} frac {frac} seed {seed} train_eps {len(train)}", flush=True)

(ROOT / "cvflow_learning_curve.json").write_text(json.dumps(results), encoding="utf-8")

print(f"\n{'frac':>5s} {'eps':>5s} {'steps':>6s} {'hit1':>6s} {'hit3':>6s} {'bigram':>7s} {'freq':>6s} {'cover':>6s} {'long8+':>7s}")
for frac in FRACTIONS:
    g = [r for r in results if r["frac"] == frac]
    m = lambda f, g=g: sum(f(r) for r in g) / len(g)  # noqa: E731
    runs = {(r["seed"], r["fold"]): r for r in g}.values()
    eps = sum(r["train_eps"] for r in runs) / len(runs)
    steps = sum(r["train_steps"] for r in runs) / len(runs)
    long = [r for r in g if r["length"] >= 8]
    # spread across seeds (mean over folds per seed)
    per_seed = []
    for seed in {r["seed"] for r in g}:
        s = [r for r in g if r["seed"] == seed]
        per_seed.append(sum(r["hit3"] for r in s) / len(s))
    spread = f"±{(max(per_seed) - min(per_seed)) / 2:.3f}" if len(per_seed) > 1 else ""
    print(f"{frac:5.1f} {eps:5.0f} {steps:6.0f} {m(lambda r: r['hit1']):6.3f} {m(lambda r: r['hit3']):6.3f} "
          f"{m(lambda r: r['hit3_bigram']):7.3f} {m(lambda r: r['hit3_freq']):6.3f} {m(lambda r: r['covered']):6.3f} "
          f"{sum(r['hit3'] for r in long) / len(long):7.3f}  {spread}")
