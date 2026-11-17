"""CVflow 探索分析：每個模型以 5 折交叉驗證（依序列切分）比較不同觀測編碼下的 HMM 與基準方法。

與 cvflow_experiment.py 不同，這裡不寫入新模型版本，只讀取已匯入的序列；K 的範圍包含 1（等同沒有隱藏結構）。
編碼「op_result」使用每個版本的執行結果（通過／未通過／無法執行）。執行結果在下一次修改之前就已知，
所以用於預測下一步是合法的；但它目前不是 K 正式支援的特徵，屬於探索性實驗。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_cv.py --out ..\\..\\docs\\cvflow_cv_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_DEFAULT, CVFLOW_ACTIVITY_MAP  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.encoding import Vocabulary  # noqa: E402
from kcore.evaluation import FrequencyBaselines, sequence_metrics  # noqa: E402
from kcore.hmm.training import TrainingConfig, train_with_selection  # noqa: E402
from kcore.service import KCoreService  # noqa: E402

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
TEACHERS = ["codex", "gemini", "opus", "sonnet", "haiku"]
CONFIG = TrainingConfig(k_min=1, k_max=4, restarts=4, n_iter=300, smoothing=0.1, seed=0, criterion="bic")


def result_label(event) -> str:
    result = (event.attributes or {}).get("result") or {}
    if result.get("executable") is False:
        return "error"
    if result.get("passed") is True:
        return "pass"
    if result.get("passed") is False:
        return "fail"
    return "unknown"


ENCODINGS = {
    "op": lambda e: e.operation_type,
    "op_category": lambda e: f"{e.operation_type}|"
                             f"{CVFLOW_ACTIVITY_MAP.get((e.features or {}).get('activity_type'), CVFLOW_ACTIVITY_DEFAULT)}",
    "op_result": lambda e: f"{e.operation_type}|{result_label(e)}",
}


def cross_validate(sequences: list[list[str]], folds: int, seed: int) -> dict:
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    totals = {m: {"log": 0.0, "events": 0, "hit1": 0.0, "hit3": 0.0, "rr": 0.0, "ranked": 0} for m in ("hmm", "bigram", "unigram")}
    chosen_k = []
    for f in range(folds):
        test_idx = set(order[f::folds])
        train = [s for i, s in enumerate(sequences) if i not in test_idx]
        test = [s for i, s in enumerate(sequences) if i in test_idx]
        vocab = Vocabulary.build(train, 1)
        train_n = [[vocab.normalize(t) for t in s] for s in train]
        test_n = [[vocab.normalize(t) for t in s] for s in test]
        result = train_with_selection([vocab.encode(s) for s in train_n], vocab.symbols, CONFIG)
        chosen_k.append(result.selected.k)
        base = FrequencyBaselines(train_n, vocab.symbols, CONFIG.smoothing)
        for name, predict in (("hmm", result.model.next_observation_distribution),
                              ("bigram", base.predict_bigram), ("unigram", base.predict_unigram)):
            m = sequence_metrics(predict, test_n, vocab.symbols)
            t = totals[name]
            t["log"] += m["log_likelihood_per_event"] * m["events"]
            t["events"] += m["events"]
            if m["ranked_steps"]:
                t["hit1"] += m["next_top1"] * m["ranked_steps"]
                t["hit3"] += m["next_top3"] * m["ranked_steps"]
                t["rr"] += m["next_mrr"] * m["ranked_steps"]
                t["ranked"] += m["ranked_steps"]
    out = {"chosen_k": chosen_k}
    for name, t in totals.items():
        out[name] = {"perplexity": math.exp(-t["log"] / t["events"]), "top1": t["hit1"] / t["ranked"],
                     "top3": t["hit3"] / t["ranked"], "mrr": t["rr"] / t["ranked"], "events": t["events"],
                     "ranked": t["ranked"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--out", type=Path, default=Path("cvflow_cv_results.json"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    service = KCoreService(make_session_factory(args.database_url))
    results: dict = {"config": CONFIG.to_dict(), "folds": args.folds, "cases": {}}
    pooled: dict[str, list[list[str]]] = {name: [] for name in ENCODINGS}
    for teacher in TEACHERS:
        case_id = f"CVFLOW-{teacher.upper()}"
        with service.session_factory() as session:
            data = service._training_sequences(session, case_id)
        results["cases"][case_id] = {"episodes": len(data), "events": sum(len(ev) for _, ev in data)}
        for name, encode in ENCODINGS.items():
            seqs = [[encode(e) for e in events] for _, events in data]
            pooled[name].extend(seqs)
            r = cross_validate(seqs, args.folds, seed=0)
            results["cases"][case_id][name] = r
            print(f"{case_id:14s} {name:12s} K={r['chosen_k']} "
                  + " | ".join(f"{m} ppl={r[m]['perplexity']:.2f} top1={r[m]['top1']:.3f} top3={r[m]['top3']:.3f}"
                               for m in ("hmm", "bigram", "unigram")), flush=True)
    # 參考：五個模型合併（僅供判斷資料量的影響，不是正式案例）
    results["pooled_reference"] = {}
    for name, seqs in pooled.items():
        r = cross_validate(seqs, args.folds, seed=0)
        results["pooled_reference"][name] = r
        print(f"{'POOLED(ref)':14s} {name:12s} K={r['chosen_k']} "
              + " | ".join(f"{m} ppl={r[m]['perplexity']:.2f} top1={r[m]['top1']:.3f} top3={r[m]['top3']:.3f}"
                           for m in ("hmm", "bigram", "unigram")), flush=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
