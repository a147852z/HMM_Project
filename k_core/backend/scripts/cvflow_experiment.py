"""CVflow 實驗：每個 AI 模型一個案例，匯入佳元實驗材料包、訓練 HMM、分析隱藏狀態。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_experiment.py "E:\\程式碼\\2026-10-13-jiayuan-package" --out ..\\..\\docs\\cvflow_results.json

可重複執行：已匯入的序列會略過；每次執行都會新增一次訓練與模型版本。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_DEFAULT, CVFLOW_ACTIVITY_MAP, ingest_teacher  # noqa: E402
from kcore.database import EpisodeRecord, make_session_factory  # noqa: E402
from kcore.encoding import FeatureSchema, event_token  # noqa: E402
from kcore.hmm.model import HiddenMarkovModel  # noqa: E402
from kcore.hmm.training import TrainingConfig  # noqa: E402
from kcore.service import KCoreError, KCoreService  # noqa: E402

TEACHERS = ["codex", "gemini", "opus", "sonnet", "haiku"]
SCHEMAS = {
    # 主要模型：修改類型 × 受影響的活動類別（8 類）
    "op_category": FeatureSchema(version="cvflow-fs-v1", fields=("operation_type", "activity_type"),
                                 activity_map=CVFLOW_ACTIVITY_MAP, activity_default=CVFLOW_ACTIVITY_DEFAULT),
    # 對照：只看修改類型（6 種）
    "op_only": FeatureSchema(version="cvflow-fs-v1-op", fields=("operation_type",)),
}
CONFIG = TrainingConfig(k_min=1, k_max=6, restarts=5, n_iter=300, tol=1e-4, smoothing=0.1, seed=0, criterion="bic")


def ensure_case(service: KCoreService, case_id: str, teacher: str) -> None:
    try:
        service.create_case({"case_id": case_id, "name": f"CVflow・{teacher}", "domain_kind": "cvflow",
                             "recording_enabled": False})
    except KCoreError as error:
        if error.code != "case_exists":
            raise


def analyse_states(service: KCoreService, case_id: str, model_id: str) -> dict:
    """以 Viterbi 解碼全部序列，描述每個隱藏狀態：主要觀測、位置、後續成敗、與整題成敗的關係。"""
    record = service.get_model(case_id, model_id)
    model = HiddenMarkovModel.from_dict(record["parameters"])
    schema = FeatureSchema.from_dict(record["feature_schema"])
    with service.session_factory() as session:
        data = service._training_sequences(session, case_id)
        episodes = {e.episode_id: e for e in session.query(EpisodeRecord).filter_by(case_id=case_id)}
    per_state = defaultdict(lambda: {"count": 0, "positions": [], "passed_after": [], "executable_after": [],
                                     "has_reason": 0, "strategies": Counter(), "ops": Counter()})
    occupancy_by_outcome = {True: Counter(), False: Counter()}
    for episode, events in data:
        tokens = [event_token(e.operation_type, e.features or {}, schema) for e in events]
        tokens = [t if t in model.observations else "__UNK__" for t in tokens]
        _, path = model.viterbi(tokens)
        final_ok = (episodes[episode.episode_id].provenance or {}).get("outcome", {}).get("final_ok")
        for i, (state, event) in enumerate(zip(path, events, strict=True)):
            s = per_state[state]
            s["count"] += 1
            s["positions"].append(i / max(1, len(events) - 1) if len(events) > 1 else 0.0)
            result = (event.attributes or {}).get("result", {})
            if result.get("passed") is not None:
                s["passed_after"].append(bool(result["passed"]))
            if result.get("executable") is not None:
                s["executable_after"].append(bool(result["executable"]))
            s["has_reason"] += bool(event.reason)
            s["strategies"][(event.attributes or {}).get("strategy_after")] += 1
            s["ops"][event.operation_type] += 1
            if final_ok is not None:
                occupancy_by_outcome[bool(final_ok)][state] += 1
    stationary = model.stationary_distribution()
    states = {}
    for row, state in enumerate(model.states):
        s = per_state[state]
        top = np.argsort(-model.emission[row])[:4]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        states[state] = {
            "events": s["count"],
            "stationary": round(float(stationary[row]), 3),
            "self_transition": round(float(model.transition[row, row]), 3),
            "top_observations": [(model.observations[i], round(float(model.emission[row, i]), 3)) for i in top],
            "mean_relative_position": mean(s["positions"]),
            "passed_after_rate": mean(s["passed_after"]),
            "executable_after_rate": mean(s["executable_after"]),
            "reason_rate": s["has_reason"] / s["count"] if s["count"] else None,
            "top_strategies": s["strategies"].most_common(3),
            "operation_mix": s["ops"].most_common(),
        }
    occupancy = {}
    for ok, counts in occupancy_by_outcome.items():
        total = sum(counts.values())
        occupancy["final_ok" if ok else "final_fail"] = {st: round(counts[st] / total, 3) if total else None
                                                         for st in model.states}
    return {"states": states, "transition": np.round(model.transition, 3).tolist(),
            "initial": np.round(model.initial, 3).tolist(), "occupancy_by_outcome": occupancy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--out", type=Path, default=Path("cvflow_results.json"))
    args = parser.parse_args()

    service = KCoreService(make_session_factory(args.database_url))
    results = {"config": CONFIG.to_dict(), "cases": {}}
    for teacher in TEACHERS:
        case_id = f"CVFLOW-{teacher.upper()}"
        ensure_case(service, case_id, teacher)
        stats = ingest_teacher(service, args.package, teacher, case_id)
        print(f"[{case_id}] ingested episodes={stats.episodes} events={stats.events} "
              f"skipped_single={stats.skipped_single_version} identical={stats.skipped_identical_versions} "
              f"renamed_nodes={stats.renamed_nodes}", flush=True)
        case_result = {"ingest": {**{k: v for k, v in vars(stats).items() if k != "operation_types"},
                                  "operation_types": dict(stats.operation_types)}, "runs": {}}
        for name, schema in SCHEMAS.items():
            run = service.train(case_id, schema, CONFIG, test_fraction=0.2, min_train_episodes=10,
                                activate=(name == "op_category"))
            entry = {"status": run["status"], "message": run["message"], "run_id": run["training_run_id"],
                     "model_version_id": run["model_version_id"], "dataset": {
                         k: (len(v) if isinstance(v, list) else v) for k, v in run["dataset"].items()}}
            if run["status"] == "succeeded":
                m = run["metrics"]
                entry["selected_k"] = m["training"]["selected_k"]
                entry["candidates"] = [{k: c[k] for k in ("k", "log_likelihood", "bic", "converged")}
                                       for c in m["training"]["candidates"]]
                entry["implementation_gap"] = m["training"]["implementation_gap"]
                entry["vocabulary_size"] = m["vocabulary_size"]
                entry["test_unknown_rate"] = m["test_unknown_rate"]
                entry["test"] = m["test"]
                if name == "op_category":
                    entry["analysis"] = analyse_states(service, case_id, run["model_version_id"])
                t = m["test"]
                print(f"  {name:12s} K={entry['selected_k']} vocab={entry['vocabulary_size']} "
                      f"ppl hmm={t['hmm']['perplexity']:.2f} bigram={t['bigram']['perplexity']:.2f} "
                      f"unigram={t['unigram']['perplexity']:.2f} | top1 hmm={t['hmm']['next_top1']:.3f} "
                      f"bigram={t['bigram']['next_top1']:.3f} | test events={t['hmm']['events']}", flush=True)
            else:
                print(f"  {name}: {run['status']} {run['message']}", flush=True)
            case_result["runs"][name] = entry
        results["cases"][case_id] = case_result
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda o: None
                                   if isinstance(o, float) and math.isnan(o) else str(o)), encoding="utf-8")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
