"""生成協助的探索實驗：加入「上一版執行結果」與「題目類型」是否讓候選排得更準？

5 折交叉驗證（依序列切分，155 條序列都輪流當測試）。每一折只用訓練折訓練 HMM、建立修改樣式庫與各種統計，
再在測試折的每一步提出候選並與實際修改比對。只讀資料庫，不寫入任何模型。

比較的排法（候選集合完全相同，只有分數不同）：
- frequency        只看樣式在訓練資料的次數
- bigram           只看上一次修改
- assist           目前的做法：P(符號 | 前綴) × 樣式比例
- +result          再乘上「上一版執行結果」的調整：P(符號 | 結果) / P(符號)，且樣式比例改用相同結果下的次數
- +task            樣式比例改用相同題目類型（metric）的次數
- +result+task     兩者都用
- hmm(result)      把執行結果編進 HMM 的觀測（符號|結果），預測時以已知的結果做條件

上一版的執行結果在提出下一次修改之前就已經知道，所以拿來當條件不算偷看答案。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_assist_cv.py --out ..\\..\\docs\\cvflow_assist_cv_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_DEFAULT, CVFLOW_ACTIVITY_MAP  # noqa: E402
from kcore.assist import REPLACE_OPS, SUPPORTED_OPS, SWITCH_OPS, TUNE_OPS, AssistService, PatternLibrary, _types  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.encoding import FeatureSchema, Vocabulary, event_token  # noqa: E402
from kcore.evaluation import FrequencyBaselines  # noqa: E402
from kcore.hmm.hybrid import MixturePredictor  # noqa: E402
from kcore.hmm.training import TrainingConfig, train_with_selection  # noqa: E402
from kcore.service import KCoreService  # noqa: E402

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
CONFIG = TrainingConfig(k_min=1, k_max=4, restarts=3, n_iter=300, smoothing=0.1, seed=0, criterion="bic")
SCHEMA = FeatureSchema(version="cvflow-fs-v1", fields=("operation_type", "activity_type"),
                       activity_map=CVFLOW_ACTIVITY_MAP, activity_default=CVFLOW_ACTIVITY_DEFAULT)
METHODS = ("frequency", "bigram", "assist", "+result", "+task", "+result+task", "hmm(result)")
BACKOFF = 2.0  # 條件次數不足時退回整體次數的強度（固定值，沒有依測試結果調整）
A = 0.5


def result_label(result) -> str:
    result = result or {}
    if result.get("executable") is False:
        return "error"
    return "pass" if result.get("passed") is True else "fail"


def load(service: KCoreService, case_id: str) -> list[dict]:
    assist = AssistService(service)
    rows = []
    with service.session_factory() as session:
        for episode, events in service._training_sequences(session, case_id):
            snapshots = assist._snapshots(session, episode.episode_id)
            prev = result_label((episode.provenance or {}).get("first_version_result"))
            steps = []
            for i, event in enumerate(events):
                features = event.features or {}
                steps.append({
                    "op": event.operation_type, "type": str(features.get("activity_type", "none")),
                    "token": event_token(event.operation_type, features, SCHEMA), "prev_result": prev,
                    "before": (snapshots[i].nodes, snapshots[i].edges), "after": (snapshots[i + 1].nodes, snapshots[i + 1].edges),
                    "change": event.change,
                })
                prev = result_label((event.attributes or {}).get("result"))
            ctx = episode.initial_context or {}
            rows.append({"episode_id": episode.episode_id, "task": ctx.get("metric") or "unknown",
                         "teacher": ctx.get("teacher"), "steps": steps})
    return rows


def build_library(train: list[dict]) -> tuple[PatternLibrary, dict, dict]:
    """與 kcore.assist 相同的樣式庫，另外統計依執行結果與題目類型分開的次數。"""
    library = PatternLibrary()
    by_result: dict[str, Counter] = defaultdict(Counter)
    by_task: dict[str, Counter] = defaultdict(Counter)
    template_counts: Counter = Counter()
    template_graphs: dict = {}
    for row in train:
        for step in row["steps"]:
            for nodes, edges in (step["before"], step["after"]):
                type_of = {n["node_id"]: n["activity_type"] for n in nodes}
                for edge in edges:
                    library.predecessors[type_of[edge["target"]]][type_of[edge["source"]]] += 1
            op, primary = step["op"], step["type"]
            if op not in SUPPORTED_OPS:
                continue
            key = (op, primary)
            library.keys[key] += 1
            by_result[step["prev_result"]][key] += 1
            by_task[row["task"]][key] += 1
            if op in TUNE_OPS:
                for update in step["change"].get("updated_node_attributes", []):
                    library.param_keys[primary].update((update.get("parameters_set") or {}).keys())
            elif op in REPLACE_OPS:
                before_types = {n["node_id"]: n["activity_type"] for n in step["before"][0]}
                for removed in step["change"].get("removed_nodes", []):
                    library.replaced_by[primary][before_types.get(removed, "none")] += 1
            elif op in SWITCH_OPS:
                signature = (primary, *sorted(_types(step["after"][0])))
                template_counts[signature] += 1
                template_graphs.setdefault(signature, {"nodes": step["after"][0], "edges": step["after"][1]})
    seen: set[str] = set()
    for sig, _ in template_counts.most_common():
        if sig[0] not in seen:
            seen.add(sig[0])
            library.templates.append({**template_graphs[sig], "signature": list(sig[1:]), "primary": sig[0],
                                      "count": sum(c for s, c in template_counts.items() if s[0] == sig[0])})
    return library, by_result, by_task


def rank(candidates: list[dict], token_p: dict[str, float] | None, weight) -> list[tuple]:
    """分數 = token_p[符號] × 該候選在同符號候選中的權重比例；token_p 為 None 時只用權重。"""
    weights = {c["key"]: weight(c["key"]) + A for c in candidates}
    by_token: dict[str, float] = defaultdict(float)
    for c in candidates:
        by_token[c["token"]] += weights[c["key"]]
    scored = []
    for c in candidates:
        w = weights[c["key"]]
        score = w if token_p is None else token_p.get(c["token"], 0.0) * w / by_token[c["token"]]
        scored.append((-score, -w, c["key"]))
    scored.sort()
    return [key for _, _, key in scored]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--case", default="CVFLOW")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("cvflow_assist_cv_results.json"))
    args = parser.parse_args()
    service = KCoreService(make_session_factory(args.database_url))
    assist = AssistService(service)
    rows = load(service, args.case)
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)

    totals = {m: Counter() for m in METHODS}
    by_op = {m: defaultdict(Counter) for m in METHODS}
    by_prev = {m: defaultdict(Counter) for m in ("assist", "+result")}
    steps = covered = 0
    chosen_k = {"plain": [], "result": []}

    for fold in range(args.folds):
        test_idx = set(order[fold::args.folds])
        train = [r for i, r in enumerate(rows) if i not in test_idx]
        test = [r for i, r in enumerate(rows) if i in test_idx]
        library, by_result, by_task = build_library(train)

        # 一般的 HMM（符號 = 修改類型|活動類別）
        train_tokens = [[s["token"] for s in r["steps"]] for r in train]
        vocab = Vocabulary.build(train_tokens, 1)
        norm = [[vocab.normalize(t) for t in seq] for seq in train_tokens]
        trained = train_with_selection([vocab.encode(s) for s in norm], vocab.symbols, CONFIG)
        chosen_k["plain"].append(trained.selected.k)
        model = trained.model
        mix = MixturePredictor.fit(model, norm, CONFIG.smoothing).predict
        bigram = FrequencyBaselines(norm, vocab.symbols, CONFIG.smoothing).predict_bigram
        token_count = Counter(t for seq in norm for t in seq)
        token_by_result: dict[str, Counter] = defaultdict(Counter)
        for r in train:
            for s in r["steps"]:
                token_by_result[s["prev_result"]][vocab.normalize(s["token"])] += 1
        n_tokens = sum(token_count.values())

        # 把執行結果編進觀測的 HMM（符號 = 修改類型|活動類別|上一版結果）
        r_tokens = [[f"{s['token']}|{s['prev_result']}" for s in r["steps"]] for r in train]
        r_vocab = Vocabulary.build(r_tokens, 1)
        r_norm = [[r_vocab.normalize(t) for t in seq] for seq in r_tokens]
        r_trained = train_with_selection([r_vocab.encode(s) for s in r_norm], r_vocab.symbols, CONFIG)
        chosen_k["result"].append(r_trained.selected.k)
        r_mix = MixturePredictor.fit(r_trained.model, r_norm, CONFIG.smoothing).predict

        total_keys = max(1, sum(library.keys.values()))

        def conditional(counter_by: dict, context: str, lib=library, total=total_keys):
            """相同情境下的樣式次數；次數少時退回整體比例（相當於加上 BACKOFF 筆整體分布的虛擬資料）。"""
            ctx = counter_by.get(context, Counter())
            return lambda key: ctx[key] + BACKOFF * lib.keys[key] / total

        for row in test:
            prefix: list[str] = []
            r_prefix: list[str] = []
            for step in row["steps"]:
                nodes, edges = step["before"]
                candidates, _ = assist._candidates(nodes, edges, library)
                for c in candidates:
                    c["token"] = vocab.normalize(event_token(c["operation_type"], {"activity_type": c["activity_type"]}, SCHEMA))
                key, prev = (step["op"], step["type"]), step["prev_result"]
                as_map = lambda dist, symbols: dict(zip(symbols, (float(p) for p in dist), strict=True))  # noqa: E731
                p_mix = as_map(mix(prefix), vocab.symbols)
                p_bigram = as_map(bigram(prefix), vocab.symbols)
                # 結果調整：P(符號 | 結果) / P(符號)，兩者都做平滑
                ctx = token_by_result.get(prev, Counter())
                n_ctx = sum(ctx.values())
                p_adjusted = {}
                for t, p in p_mix.items():
                    base = (token_count[t] + A) / (n_tokens + A * len(vocab.symbols))
                    cond = (ctx[t] + BACKOFF * base) / (n_ctx + BACKOFF)
                    p_adjusted[t] = p * cond / base
                # hmm(result)：只保留與已知結果相符的符號，再合併回一般符號
                p_r = as_map(r_mix(r_prefix), r_vocab.symbols)
                p_hmm_result: dict[str, float] = defaultdict(float)
                for t, p in p_r.items():
                    if t.endswith(f"|{prev}"):
                        p_hmm_result[vocab.normalize(t.rsplit("|", 1)[0])] += p

                overall = lambda k: library.keys[k]  # noqa: E731
                w_result, w_task = conditional(by_result, prev), conditional(by_task, row["task"])
                rankings = {
                    "frequency": rank(candidates, None, overall),
                    "bigram": rank(candidates, p_bigram, overall),
                    "assist": rank(candidates, p_mix, overall),
                    "+result": rank(candidates, p_adjusted, w_result),
                    "+task": rank(candidates, p_mix, w_task),
                    "+result+task": rank(candidates, p_adjusted, lambda k: (w_result(k) + w_task(k)) / 2),  # noqa: B023
                    "hmm(result)": rank(candidates, dict(p_hmm_result), w_result),
                }
                steps += 1
                covered += key in rankings["assist"]
                for name, ordered in rankings.items():
                    r = ordered.index(key) + 1 if key in ordered else None
                    hit = {"hit1": r == 1, "hit3": r is not None and r <= 3, "hit5": r is not None and r <= 5}
                    for h, v in hit.items():
                        totals[name][h] += v
                        by_op[name][step["op"]][h] += v
                    totals[name]["rr"] += (1 / r) if r else 0.0
                    by_op[name][step["op"]]["n"] += 1
                    if name in by_prev:
                        by_prev[name][prev]["n"] += 1
                        by_prev[name][prev]["hit3"] += hit["hit3"]
                prefix.append(vocab.normalize(step["token"]))
                r_prefix.append(r_vocab.normalize(f"{step['token']}|{prev}"))
        print(f"fold {fold + 1}/{args.folds} done (K plain={chosen_k['plain'][-1]}, result={chosen_k['result'][-1]})", flush=True)

    summary = {m: {"hit1": totals[m]["hit1"] / steps, "hit3": totals[m]["hit3"] / steps, "hit5": totals[m]["hit5"] / steps,
                   "mrr": totals[m]["rr"] / steps} for m in METHODS}
    results = {
        "case": args.case, "folds": args.folds, "episodes": len(rows), "steps": steps, "coverage": covered / steps,
        "chosen_k": chosen_k, "methods": summary,
        "hit3_by_operation": {m: {op: v["hit3"] / v["n"] for op, v in sorted(by_op[m].items())} for m in METHODS},
        "steps_by_operation": {op: v["n"] for op, v in sorted(by_op["assist"].items())},
        "hit3_by_previous_result": {m: {p: {"steps": v["n"], "hit3": v["hit3"] / v["n"]} for p, v in sorted(by_prev[m].items())}
                                    for m in by_prev},
    }
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsteps={steps} coverage={covered / steps:.3f}")
    for m in METHODS:
        s = summary[m]
        print(f"  {m:14s} hit1={s['hit1']:.3f} hit3={s['hit3']:.3f} hit5={s['hit5']:.3f} mrr={s['mrr']:.3f}")
    print("hit3 by operation:")
    for m in ("assist", "+result", "+task", "+result+task"):
        print(f"  {m:14s}", {op: round(v, 2) for op, v in results["hit3_by_operation"][m].items()})
    print("hit3 by previous result:", results["hit3_by_previous_result"])
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
