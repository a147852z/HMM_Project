"""CVFLOW 合併案例的 5 折交叉驗證：比較 HMM、bigram、unigram、HMM＋bigram 平均、HMM＋上一次修改（route bigram）。

依序列切分；每一折只用訓練折估計所有方法，K 以 BIC 在 1～4 中選擇。只讀取資料庫，不寫入模型。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_hybrid_cv.py --out ..\\..\\docs\\cvflow_hybrid_cv_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_DEFAULT, CVFLOW_ACTIVITY_MAP  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.encoding import FeatureSchema, Vocabulary, event_token  # noqa: E402
from kcore.evaluation import FrequencyBaselines, sequence_metrics  # noqa: E402
from kcore.hmm.hybrid import MixturePredictor, RouteBigramPredictor  # noqa: E402
from kcore.hmm.training import TrainingConfig, train_with_selection  # noqa: E402
from kcore.service import HYBRID_BACKOFF, KCoreService  # noqa: E402

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
CONFIG = TrainingConfig(k_min=1, k_max=4, restarts=3, n_iter=300, smoothing=0.1, seed=0, criterion="bic")
SCHEMAS = {
    "op_category": FeatureSchema(version="cvflow-fs-v1", fields=("operation_type", "activity_type"),
                                 activity_map=CVFLOW_ACTIVITY_MAP, activity_default=CVFLOW_ACTIVITY_DEFAULT),
    "op_only": FeatureSchema(version="cvflow-fs-v1-op", fields=("operation_type",)),
}
METHODS = ("hmm", "bigram", "unigram", "hmm_bigram_mix", "hmm_route_bigram")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--case", default="CVFLOW")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("cvflow_hybrid_cv_results.json"))
    args = parser.parse_args()
    service = KCoreService(make_session_factory(args.database_url))
    with service.session_factory() as session:
        data = service._training_sequences(session, args.case)
    teachers = [(ep.initial_context or {}).get("teacher") for ep, _ in data]
    order = list(range(len(data)))
    random.Random(0).shuffle(order)
    results = {"case": args.case, "folds": args.folds, "config": CONFIG.to_dict(), "backoff": HYBRID_BACKOFF,
               "episodes": len(data), "encodings": {}}
    for name, schema in SCHEMAS.items():
        sequences = [[event_token(e.operation_type, e.features or {}, schema) for e in events] for _, events in data]
        totals = {m: defaultdict(float) for m in METHODS}
        by_teacher = {m: defaultdict(lambda: defaultdict(float)) for m in ("bigram", "hmm_route_bigram")}
        chosen_k = []
        for f in range(args.folds):
            test_idx = set(order[f::args.folds])
            train = [s for i, s in enumerate(sequences) if i not in test_idx]
            vocab = Vocabulary.build(train, 1)
            train_n = [[vocab.normalize(t) for t in s] for s in train]
            result = train_with_selection([vocab.encode(s) for s in train_n], vocab.symbols, CONFIG)
            chosen_k.append(result.selected.k)
            base = FrequencyBaselines(train_n, vocab.symbols, CONFIG.smoothing)
            predictors = {
                "hmm": result.model.next_observation_distribution, "bigram": base.predict_bigram,
                "unigram": base.predict_unigram,
                "hmm_bigram_mix": MixturePredictor.fit(result.model, train_n, CONFIG.smoothing).predict,
                "hmm_route_bigram": RouteBigramPredictor.fit(result.model, train_n, HYBRID_BACKOFF).predict,
            }
            for i in sorted(test_idx):
                seq = [vocab.normalize(t) for t in sequences[i]]
                for method, predict in predictors.items():
                    m = sequence_metrics(predict, [seq], vocab.symbols)
                    t = totals[method]
                    t["log"] += m["log_likelihood_per_event"] * m["events"]
                    t["events"] += m["events"]
                    if m["ranked_steps"]:
                        for key in ("top1", "top3", "mrr"):
                            t[key] += m[f"next_{key}"] * m["ranked_steps"]
                        t["ranked"] += m["ranked_steps"]
                        if method in by_teacher:
                            bt = by_teacher[method][teachers[i]]
                            bt["top1"] += m["next_top1"] * m["ranked_steps"]
                            bt["ranked"] += m["ranked_steps"]
        summary = {"chosen_k": chosen_k}
        for method, t in totals.items():
            summary[method] = {"perplexity": math.exp(-t["log"] / t["events"]), "top1": t["top1"] / t["ranked"],
                               "top3": t["top3"] / t["ranked"], "mrr": t["mrr"] / t["ranked"],
                               "events": int(t["events"]), "ranked": int(t["ranked"])}
        summary["top1_by_teacher"] = {m: {tch: round(v["top1"] / v["ranked"], 3) for tch, v in d.items()}
                                      for m, d in by_teacher.items()}
        results["encodings"][name] = summary
        print(f"\n[{name}] K per fold={chosen_k}")
        for method in METHODS:
            s = summary[method]
            print(f"  {method:17s} ppl={s['perplexity']:.2f} top1={s['top1']:.3f} top3={s['top3']:.3f} "
                  f"mrr={s['mrr']:.3f}", flush=True)
        print("  top1 by teacher:", summary["top1_by_teacher"], flush=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
