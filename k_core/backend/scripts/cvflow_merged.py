"""CVflow 合併案例：五個 AI 模型的解題紀錄匯入同一個案例（CVFLOW），訓練一組 HMM 並分析。

AI 模型名稱記在每條序列的 initial_context["teacher"]，用於分析時比較各模型，不進入觀測編碼。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_merged.py "E:\\程式碼\\2026-10-13-jiayuan-package" --out ..\\..\\docs\\cvflow_merged_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_DEFAULT, CVFLOW_ACTIVITY_MAP, ingest_teacher  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.encoding import UNKNOWN, FeatureSchema, event_token  # noqa: E402
from kcore.hmm.model import HiddenMarkovModel  # noqa: E402
from kcore.hmm.training import TrainingConfig  # noqa: E402
from kcore.service import KCoreError, KCoreService  # noqa: E402

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
CASE_ID = "CVFLOW"
TEACHERS = ["codex", "gemini", "opus", "sonnet", "haiku"]
CONFIG = TrainingConfig(k_min=1, k_max=6, restarts=5, n_iter=300, tol=1e-4, smoothing=0.1, seed=0, criterion="bic")
SCHEMAS = {
    "op_category": FeatureSchema(version="cvflow-fs-v1", fields=("operation_type", "activity_type"),
                                 activity_map=CVFLOW_ACTIVITY_MAP, activity_default=CVFLOW_ACTIVITY_DEFAULT),
    "op_only": FeatureSchema(version="cvflow-fs-v1-op", fields=("operation_type",)),
}


def analyse(service: KCoreService, model_id: str) -> dict:
    record = service.get_model(CASE_ID, model_id)
    model = HiddenMarkovModel.from_dict(record["parameters"])
    schema = FeatureSchema.from_dict(record["feature_schema"])
    with service.session_factory() as session:
        data = service._training_sequences(session, CASE_ID)
    per_state = defaultdict(lambda: {"n": 0, "pos": [], "passed": [], "ops": Counter(), "strategies": Counter(),
                                     "teachers": Counter()})
    by_teacher = defaultdict(Counter)
    by_outcome = {"final_ok": Counter(), "final_fail": Counter()}
    switches_per_episode = []
    paths = {}
    for episode, events in data:
        teacher = (episode.initial_context or {}).get("teacher")
        tokens = [event_token(e.operation_type, e.features or {}, schema) for e in events]
        tokens = [t if t in model.observations else UNKNOWN for t in tokens]
        _, path = model.viterbi(tokens)
        paths[episode.episode_id] = path
        switches_per_episode.append(sum(a != b for a, b in zip(path, path[1:])))
        outcome = (episode.provenance or {}).get("outcome", {}).get("final_ok")
        for i, (state, event) in enumerate(zip(path, events, strict=True)):
            s = per_state[state]
            s["n"] += 1
            s["pos"].append(i / (len(events) - 1) if len(events) > 1 else 0.0)
            result = (event.attributes or {}).get("result") or {}
            if result.get("passed") is not None:
                s["passed"].append(bool(result["passed"]))
            s["ops"][event.operation_type] += 1
            s["strategies"][(event.attributes or {}).get("strategy_after")] += 1
            s["teachers"][teacher] += 1
            by_teacher[teacher][state] += 1
            if outcome is not None:
                by_outcome["final_ok" if outcome else "final_fail"][state] += 1
    stationary = model.stationary_distribution()
    share = lambda c: {k: round(v / sum(c.values()), 3) for k, v in c.items()} if c else {}  # noqa: E731
    states = {}
    for row, state in enumerate(model.states):
        s = per_state[state]
        top = np.argsort(-model.emission[row])[:5]
        states[state] = {
            "events": s["n"], "stationary": round(float(stationary[row]), 3),
            "self_transition": round(float(model.transition[row, row]), 3),
            "top_observations": [(model.observations[i], round(float(model.emission[row, i]), 3)) for i in top],
            "mean_relative_position": round(sum(s["pos"]) / len(s["pos"]), 3) if s["pos"] else None,
            "passed_after_rate": round(sum(s["passed"]) / len(s["passed"]), 3) if s["passed"] else None,
            "operation_share": share(s["ops"]), "strategy_share": share(s["strategies"]),
            "teacher_share": share(s["teachers"]),
        }
    return {
        "states": states,
        "initial": np.round(model.initial, 3).tolist(),
        "transition": np.round(model.transition, 3).tolist(),
        "state_share_by_teacher": {t: share(c) for t, c in by_teacher.items()},
        "state_share_by_outcome": {k: share(c) for k, c in by_outcome.items()},
        "episodes_with_state_switch": round(sum(x > 0 for x in switches_per_episode) / len(switches_per_episode), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--out", type=Path, default=Path("cvflow_merged_results.json"))
    args = parser.parse_args()
    service = KCoreService(make_session_factory(args.database_url))
    try:
        service.create_case({"case_id": CASE_ID, "name": "CVflow（五個 AI 模型的流程圖設計紀錄）",
                             "domain_kind": "cvflow", "recording_enabled": False})
    except KCoreError as error:
        if error.code != "case_exists":
            raise
    ingest = {}
    for teacher in TEACHERS:
        stats = ingest_teacher(service, args.package, teacher, CASE_ID, id_prefix="CVFLOW")
        ingest[teacher] = {"episodes": stats.episodes, "events": stats.events}
        print(f"ingested {teacher}: episodes={stats.episodes} events={stats.events}", flush=True)
    results = {"case_id": CASE_ID, "config": CONFIG.to_dict(), "ingest": ingest, "runs": {}}
    for name, schema in SCHEMAS.items():
        run = service.train(CASE_ID, schema, CONFIG, test_fraction=0.2, min_train_episodes=10,
                            activate=(name == "op_category"))
        entry = {"status": run["status"], "model_version_id": run["model_version_id"],
                 "run_id": run["training_run_id"],
                 "dataset": {k: (len(v) if isinstance(v, list) else v) for k, v in run["dataset"].items()}}
        if run["status"] == "succeeded":
            m = run["metrics"]
            entry.update(selected_k=m["training"]["selected_k"], vocabulary_size=m["vocabulary_size"],
                         test_unknown_rate=m["test_unknown_rate"], test=m["test"],
                         implementation_gap=m["training"]["implementation_gap"],
                         candidates=[{k: c[k] for k in ("k", "log_likelihood", "bic", "converged")}
                                     for c in m["training"]["candidates"]])
            entry["analysis"] = analyse(service, run["model_version_id"])
            t = m["test"]
            print(f"{name}: K={entry['selected_k']} vocab={entry['vocabulary_size']} "
                  f"ppl hmm={t['hmm']['perplexity']:.2f} bigram={t['bigram']['perplexity']:.2f} "
                  f"unigram={t['unigram']['perplexity']:.2f} top1 hmm={t['hmm']['next_top1']:.3f} "
                  f"bigram={t['bigram']['next_top1']:.3f}", flush=True)
        results["runs"][name] = entry
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
