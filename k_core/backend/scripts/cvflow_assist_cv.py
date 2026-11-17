"""生成協助的探索實驗：加入額外的已知情境，是否讓候選排得更準？

5 折交叉驗證（依序列切分，155 條序列都輪流當測試）。每一折只用訓練折訓練 HMM、建立修改樣式庫與各種統計，
再在測試折的每一步提出候選並與實際修改比對。只讀資料庫，不寫入任何模型。

比較的排法（候選集合完全相同，只有分數不同）：
- frequency        只看樣式在訓練資料的次數
- bigram           只看上一次修改
- assist           基本做法：P(符號 | 前綴) × 樣式比例
- +result          納入「上一版執行結果」（通過／未通過／無法執行）
- +task            納入題目類型（metric）
- +problem         納入「上一版的問題類型」（本地 LLM 由執行後的觀察文字歸類；需要 --labels）
- +result+problem  兩者都用

納入情境 c 的方式：第一項乘上 P(符號 | c) / P(符號)，第二項改用「相同情境下」的樣式次數（次數少時退回整體比例）。
這些情境都在提出下一次修改之前就已經知道，所以不算偷看答案。沒有情境資料的步驟等同基本做法。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_assist_cv.py --labels ..\\..\\docs\\cvflow_problem_labels.json ^
        --out ..\\..\\docs\\cvflow_assist_cv_results.json
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
from kcore.assist import (  # noqa: E402
    REPLACE_OPS, SUPPORTED_OPS, SWITCH_OPS, TUNE_OPS, AssistService, PatternLibrary, _types, result_label,
)
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
METHODS = ("frequency", "bigram", "assist", "+result", "+task", "+problem", "+result+problem")
CONTEXTS = ("result", "task", "problem")
BACKOFF = 2.0  # 情境次數不足時退回整體比例的強度（固定值，沒有依測試結果調整）
A = 0.5


def load(service: KCoreService, case_id: str, labels: dict) -> list[dict]:
    assist = AssistService(service)
    rows = []
    with service.session_factory() as session:
        for episode, events in service._training_sequences(session, case_id):
            snapshots = assist._snapshots(session, episode.episode_id)
            previous = result_label((episode.provenance or {}).get("first_version_result"))
            ctx = episode.initial_context or {}
            steps = []
            for i, event in enumerate(events):
                features = event.features or {}
                steps.append({
                    "op": event.operation_type, "type": str(features.get("activity_type", "none")),
                    "token": event_token(event.operation_type, features, SCHEMA),
                    "before": (snapshots[i].nodes, snapshots[i].edges), "after": (snapshots[i + 1].nodes, snapshots[i + 1].edges),
                    "change": event.change,
                    "context": {"result": previous, "task": ctx.get("metric"),
                                "problem": (labels.get(event.event_id) or {}).get("primary")},
                })
                previous = result_label((event.attributes or {}).get("result"))
            rows.append({"episode_id": episode.episode_id, "teacher": ctx.get("teacher"), "steps": steps})
    return rows


def build_library(train: list[dict]) -> tuple[PatternLibrary, dict]:
    """與 kcore.assist 相同的樣式庫，另外依各種情境分開統計樣式次數。"""
    library = PatternLibrary()
    keys_by: dict[str, dict[str, Counter]] = {c: defaultdict(Counter) for c in CONTEXTS}
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
            for name in CONTEXTS:
                if step["context"][name] is not None:
                    keys_by[name][step["context"][name]][key] += 1
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
    return library, keys_by


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
    parser.add_argument("--labels", type=Path, default=None, help="cvflow_problem_labels.py 產生的問題類型")
    parser.add_argument("--out", type=Path, default=Path("cvflow_assist_cv_results.json"))
    args = parser.parse_args()
    label_file = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else {}
    service = KCoreService(make_session_factory(args.database_url))
    assist = AssistService(service)
    rows = load(service, args.case, label_file.get("labels", {}))
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)

    # 分三組統計：全部步驟、有問題類型的步驟、沒有問題類型的步驟
    groups = ("all", "labelled", "unlabelled")
    totals = {g: {m: Counter() for m in METHODS} for g in groups}
    by_op = {m: defaultdict(Counter) for m in METHODS}
    by_context = {name: {m: defaultdict(Counter) for m in ("assist", f"+{name}")} for name in CONTEXTS}
    covered = 0
    chosen_k = []

    for fold in range(args.folds):
        test_idx = set(order[fold::args.folds])
        train = [r for i, r in enumerate(rows) if i not in test_idx]
        test = [r for i, r in enumerate(rows) if i in test_idx]
        library, keys_by = build_library(train)
        total_keys = max(1, sum(library.keys.values()))

        train_tokens = [[s["token"] for s in r["steps"]] for r in train]
        vocab = Vocabulary.build(train_tokens, 1)
        norm = [[vocab.normalize(t) for t in seq] for seq in train_tokens]
        trained = train_with_selection([vocab.encode(s) for s in norm], vocab.symbols, CONFIG)
        chosen_k.append(trained.selected.k)
        mix = MixturePredictor.fit(trained.model, norm, CONFIG.smoothing).predict
        bigram = FrequencyBaselines(norm, vocab.symbols, CONFIG.smoothing).predict_bigram
        token_count = Counter(t for seq in norm for t in seq)
        n_tokens = sum(token_count.values())
        tokens_by: dict[str, dict[str, Counter]] = {c: defaultdict(Counter) for c in CONTEXTS}
        for r in train:
            for s in r["steps"]:
                for name in CONTEXTS:
                    if s["context"][name] is not None:
                        tokens_by[name][s["context"][name]][vocab.normalize(s["token"])] += 1

        def adjust(p_token: dict, name: str, value, tokens_by=tokens_by, token_count=token_count, n_tokens=n_tokens,
                   n_symbols=len(vocab.symbols)) -> dict:
            """P(符號 | 前綴) × P(符號 | 情境) / P(符號)；沒有該情境的統計時原樣回傳。"""
            ctx = tokens_by[name].get(value) if value is not None else None
            if not ctx:
                return p_token
            n_ctx = sum(ctx.values())
            out = {}
            for t, p in p_token.items():
                base = (token_count[t] + A) / (n_tokens + A * n_symbols)
                out[t] = p * ((ctx[t] + BACKOFF * base) / (n_ctx + BACKOFF)) / base
            return out

        def weights(name: str, value, keys_by=keys_by, lib=library, total=total_keys):
            ctx = keys_by[name].get(value) if value is not None else None
            if not ctx:
                return lambda key: lib.keys[key]
            return lambda key: ctx[key] + BACKOFF * lib.keys[key] / total

        for row in test:
            prefix: list[str] = []
            for step in row["steps"]:
                nodes, edges = step["before"]
                candidates, _ = assist._candidates(nodes, edges, library)
                for c in candidates:
                    c["token"] = vocab.normalize(event_token(c["operation_type"], {"activity_type": c["activity_type"]}, SCHEMA))
                key, ctx = (step["op"], step["type"]), step["context"]
                p_mix = dict(zip(vocab.symbols, (float(p) for p in mix(prefix)), strict=True))
                p_bigram = dict(zip(vocab.symbols, (float(p) for p in bigram(prefix)), strict=True))
                overall = weights("result", None)
                w_result, w_problem = weights("result", ctx["result"]), weights("problem", ctx["problem"])
                both = lambda k, a=w_result, b=w_problem: (a(k) + b(k)) / 2  # noqa: E731
                rankings = {
                    "frequency": rank(candidates, None, overall),
                    "bigram": rank(candidates, p_bigram, overall),
                    "assist": rank(candidates, p_mix, overall),
                    "+result": rank(candidates, adjust(p_mix, "result", ctx["result"]), w_result),
                    "+task": rank(candidates, p_mix, weights("task", ctx["task"])),
                    "+problem": rank(candidates, adjust(p_mix, "problem", ctx["problem"]), w_problem),
                    "+result+problem": rank(candidates, adjust(adjust(p_mix, "result", ctx["result"]), "problem",
                                                               ctx["problem"]), both),
                }
                covered += key in rankings["assist"]
                group = "labelled" if ctx["problem"] is not None else "unlabelled"
                for name, ordered in rankings.items():
                    r = ordered.index(key) + 1 if key in ordered else None
                    hits = {"n": 1, "hit1": r == 1, "hit3": r is not None and r <= 3, "hit5": r is not None and r <= 5,
                            "rr": (1 / r) if r else 0.0}
                    for g in ("all", group):
                        totals[g][name].update(hits)
                    by_op[name][step["op"]].update({"n": 1, "hit3": hits["hit3"]})
                    for context_name in CONTEXTS:
                        if name in by_context[context_name] and ctx[context_name] is not None:
                            by_context[context_name][name][ctx[context_name]].update({"n": 1, "hit3": hits["hit3"]})
                prefix.append(vocab.normalize(step["token"]))
        print(f"fold {fold + 1}/{args.folds} done (K={chosen_k[-1]})", flush=True)

    def summarize(counter: Counter) -> dict:
        n = counter["n"] or 1
        return {"steps": counter["n"], "hit1": counter["hit1"] / n, "hit3": counter["hit3"] / n,
                "hit5": counter["hit5"] / n, "mrr": counter["rr"] / n}

    steps = totals["all"]["assist"]["n"]
    results = {
        "case": args.case, "folds": args.folds, "episodes": len(rows), "steps": steps, "coverage": covered / steps,
        "chosen_k": chosen_k, "labels": {"file": str(args.labels) if args.labels else None,
                                         "model": label_file.get("model"), "count": len(label_file.get("labels", {}))},
        "methods": {g: {m: summarize(totals[g][m]) for m in METHODS} for g in groups},
        "hit3_by_operation": {m: {op: v["hit3"] / v["n"] for op, v in sorted(by_op[m].items())} for m in METHODS},
        "hit3_by_context": {name: {value: {"steps": by_context[name]["assist"][value]["n"],
                                           "assist": by_context[name]["assist"][value]["hit3"] / by_context[name]["assist"][value]["n"],
                                           f"+{name}": by_context[name][f"+{name}"][value]["hit3"] / by_context[name][f"+{name}"][value]["n"]}
                                   for value in sorted(by_context[name]["assist"])} for name in CONTEXTS},
    }
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsteps={steps} coverage={covered / steps:.3f}")
    for g in groups:
        print(f"[{g}] steps={results['methods'][g]['assist']['steps']}")
        for m in METHODS:
            s = results["methods"][g][m]
            print(f"  {m:16s} hit1={s['hit1']:.3f} hit3={s['hit3']:.3f} hit5={s['hit5']:.3f} mrr={s['mrr']:.3f}")
    print("hit3 by operation:")
    for m in ("assist", "+result", "+problem", "+result+problem"):
        print(f"  {m:16s}", {op: round(v, 2) for op, v in results["hit3_by_operation"][m].items()})
    for name in ("result", "problem"):
        print(f"hit3 by {name}:")
        for value, v in results["hit3_by_context"][name].items():
            print(f"  {value:16s} steps={v['steps']:4d} assist={v['assist']:.2f} +{name}={v['+' + name]:.2f}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
