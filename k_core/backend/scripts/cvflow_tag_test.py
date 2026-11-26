"""探索實驗：知識詞條（詞彙）的生成與預測。

情境：遇到新任務時（AoV 協助生成或其他系統），LLM 只看任務目標就先判斷「可能相關的詞條」，再用詞條找出過去相似任務的知識。

步驟：
1. 正確答案：每一題讓 LLM 讀這題所有解題紀錄（任務目標、觀察、修改、結果），整理出各面向的詞條。
   情況面向：task_type（任務類型）、target_object（目標物）、object_category（上層分類）、
             image_conditions（影像難點）、problems（遇到的問題）
   知識面向：effective_methods（有效的做法）、ineffective_methods（無效的做法）
2. 統一詞彙：每個面向把所有題目的詞條合併同義詞，得到標準詞彙表，再把每題的詞條換成標準詞。
3. 預測：每一題只給任務目標（變體 goal），或任務目標＋第一版流程（變體 goal+flow），
   讓 LLM 從標準詞彙表選出情況面向的詞條。
4. 評估：
   - 詞條：與正確答案比對的精確率、召回率（另外標出只出現在這一題的詞，真正的新任務不可能猜到）。
   - 知識：用預測的詞條找最相似的 3 題，看那 3 題的關鍵節點能涵蓋這一題關鍵節點的比例；
     與隨機 3 題、同題目類型 3 題、用正確詞條找（上限）比較。
     關鍵節點：這一題判分通過的版本用到的節點類型（沒有通過的版本時，用各解題紀錄最後一版），去掉讀圖、計數、疊圖等通用節點。
   判分「通過」用每一版的 v1i（依 CVflow 回覆 docs/REPLY-jiayuan.md；v1 用的是 v1s）。

用法（在 k_core/backend 執行）：.venv\\Scripts\\python.exe scripts\\cvflow_tag_test.py
LLM 回答存在 cvflow_tag_test_cache.json，可以中斷後重跑。結果寫到 cvflow_tag_test.json。
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kcore.conversion.llm_client import LLMError, OpenAICompatibleClient, extract_json  # noqa: E402

PROMPT_VERSION = "tag-v2"  # v2（2026-11-24）：判分「通過」改為 v1i（CVflow 回覆），正確答案與關鍵節點都要重算
CONDITION_FACETS = ["task_type", "target_object", "object_category", "image_conditions", "problems"]
KNOWLEDGE_FACETS = ["effective_methods", "ineffective_methods"]
FACETS = CONDITION_FACETS + KNOWLEDGE_FACETS
FACET_TEXT = {
    "task_type": "任務類型（例如：計數、偵測並畫框、量測）",
    "target_object": "具體的目標物（例如：蘋果、汽車、紅血球）",
    "object_category": "目標物的上層分類或外觀特徵（例如：水果、車輛、細胞、紅色、圓形、小物體）",
    "image_conditions": "影像本身的難點（例如：物體黏連重疊、低光、低對比、背景雜亂、霧霾、遮擋、大小差異大）",
    "problems": "解題過程中實際遇到的問題（例如：漏偵測、多框、黏成一框、碎片框、無法執行）",
    "effective_methods": "有效的做法：讓結果變好或通過的方法，盡量寫節點類型或方法名稱",
    "ineffective_methods": "無效的做法：試過但沒有改善或變差的方法",
}
WEIGHTS = {"task_type": 1.0, "target_object": 1.0, "object_category": 1.0, "image_conditions": 2.0, "problems": 1.0}
GENERIC_NODES = {"io.load_image", "io.blank_image", "flow.count", "flow.overlay_detections", "flow.overlay_keypoints",
                 "cv.overlay_mask", "cv.draw_rectangle", "cv.draw_contours", "cv.put_text", "cv.mask_to_image"}

GOLD_SYSTEM = (
    "你是影像辨識知識整理助理。使用者會提供同一道題目的多筆解題紀錄（任務目標、各版流程用到的節點、修改、"
    "解題者對執行結果的觀察、判分）。請整理出描述這道題目的詞條。\n"
    "規則：每個詞條是 2～8 個字的短詞（節點類型可直接寫英文名稱，例如 cv.watershed），不要寫句子；"
    "只根據紀錄內容；每個面向 0～6 個詞條，沒有證據的面向留空。用繁體中文。只輸出 JSON。\n"
    "面向：\n" + "\n".join(f"- {k}：{v}" for k, v in FACET_TEXT.items())
)
GOLD_SCHEMA = {"type": "object", "properties": {k: {"type": "array", "items": {"type": "string"}} for k in FACETS},
               "required": FACETS}
MERGE_SYSTEM = (
    "你是詞彙整理助理。使用者會提供同一個面向、來自多道題目的詞條清單。請把意思相同或非常接近的詞合併成一個標準詞，"
    "意思不同的不要合併（例如「低光」與「低對比」不同）。每個原始詞條都必須出現在某個標準詞的 members 裡，而且只出現一次。"
    "標準詞用 2～8 個字。用繁體中文。只輸出 JSON。"
)
MERGE_SCHEMA = {"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "object", "properties": {
    "term": {"type": "string"}, "members": {"type": "array", "items": {"type": "string"}}},
    "required": ["term", "members"]}}}, "required": ["terms"]}
PREDICT_SYSTEM = (
    "你是影像辨識流程的規劃助理。使用者會提供一個新任務（以及可能有的第一版流程），還沒有任何執行結果。"
    "請從每個面向的標準詞彙中，選出這個任務可能相關的詞條（每個面向 0～5 個）。"
    "problems 是預期可能遇到的問題。沒有把握的面向可以少選。只輸出 JSON。"
)


# ---------------------------------------------------------------- 資料
def load_tasks(db_path: Path) -> dict[str, dict]:
    db = sqlite3.connect(db_path)
    tasks: dict[str, dict] = {}
    for eid, ctx, prov in db.execute("select episode_id, initial_context, provenance from episodes "
                                     "where case_id='CVFLOW' and episode_id not like 'CVF2-%' order by episode_id"):
        ctx, prov = json.loads(ctx), json.loads(prov)
        task = tasks.setdefault(ctx["task_id"], {"task_id": ctx["task_id"], "request": ctx.get("request"),
                                                 "metric": ctx.get("metric"), "category": ctx.get("category"),
                                                 "episodes": []})
        snaps = [json.loads(n) for (n,) in db.execute(
            "select nodes from workflow_snapshots where episode_id=? order by id", (eid,))]
        events = db.execute("select operation_type, features, reason, attributes from change_events "
                            "where episode_id=? order by sequence_index", (eid,)).fetchall()
        passed = [(prov.get("first_version_result") or {}).get("passed") is True]
        steps = []
        for op, features, reason, attrs in events:
            attrs = json.loads(attrs or "{}")
            passed.append((attrs.get("result") or {}).get("passed") is True)
            steps.append({"op": op, "type": json.loads(features or "{}").get("activity_type"),
                          "reason": (reason or "").strip(), "exec_report": attrs.get("exec_report")})
        task["episodes"].append({"episode_id": eid, "teacher": ctx.get("teacher"), "snapshots": snaps,
                                 "passed": passed, "steps": steps})
    for task in tasks.values():
        task["key_nodes"] = key_nodes(task)
    return tasks


def key_nodes(task: dict) -> set[str]:
    chosen = [snap for ep in task["episodes"] for snap, ok in zip(ep["snapshots"], ep["passed"]) if ok]
    if not chosen:
        chosen = [ep["snapshots"][-1] for ep in task["episodes"]]
    return {n["activity_type"] for snap in chosen for n in snap} - GENERIC_NODES


def gold_prompt(task: dict, budget: int = 9000) -> str:
    parts = [f"## 任務目標\n{task['request']}\n題目類型：{task['metric']}"]
    per_episode = budget // max(1, len(task["episodes"]))
    for ep in task["episodes"]:
        types = " → ".join(n["activity_type"] for n in ep["snapshots"][0])
        lines = [f"## 解題紀錄（{ep['teacher']}）", f"第 0 版節點：{types}",
                 f"第 0 版判分：{'通過' if ep['passed'][0] else '未通過'}"]
        for i, s in enumerate(ep["steps"], start=1):
            line = f"第 {i} 次修改：{s['op']}（{s['type']}），判分：{'通過' if ep['passed'][i] else '未通過'}"
            if s["exec_report"]:
                line += f"\n  修改前的觀察：{s['exec_report'][:220]}"
            lines.append(line)
        last = " → ".join(n["activity_type"] for n in ep["snapshots"][-1])
        lines.append(f"最後一版節點：{last}")
        text = "\n".join(lines)
        parts.append(text if len(text) <= per_episode else text[:per_episode] + "…")
    return "\n\n".join(parts)


def predict_prompt(task: dict, vocab: dict[str, list[str]], with_flow: bool) -> str:
    parts = [f"## 新任務\n{task['request']}\n題目類型：{task['metric']}"]
    if with_flow:
        first = task["episodes"][0]["snapshots"][0]
        parts.append("## 第一版流程（尚未執行）\n" + " → ".join(n["activity_type"] for n in first))
    for facet in CONDITION_FACETS:
        parts.append(f"## {facet}：{FACET_TEXT[facet]}\n標準詞彙：" + "、".join(vocab[facet]))
    return "\n\n".join(parts)


def predict_schema(vocab: dict[str, list[str]]) -> dict:
    return {"type": "object", "properties": {f: {"type": "array", "items": {"type": "string", "enum": vocab[f]}}
                                             for f in CONDITION_FACETS}, "required": CONDITION_FACETS}


# ---------------------------------------------------------------- 評估
def prf(pred: set, gold: set) -> tuple[float, float]:
    precision = len(pred & gold) / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = len(pred & gold) / len(gold) if gold else 1.0
    return precision, recall


def similarity(tags: dict[str, set], other: dict[str, set]) -> float:
    return sum(WEIGHTS[f] * len(tags.get(f, set()) & other.get(f, set())) for f in CONDITION_FACETS)


def coverage(task_id: str, neighbours: list[str], tasks: dict) -> float | None:
    target = tasks[task_id]["key_nodes"]
    if not target:
        return None
    pooled = set().union(*(tasks[t]["key_nodes"] for t in neighbours)) if neighbours else set()
    return len(target & pooled) / len(target)


def top_k(task_id: str, tags: dict[str, set], gold: dict[str, dict[str, set]], k: int = 3) -> list[str]:
    scored = [(similarity(tags, gold[t]), t) for t in gold if t != task_id]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[:k]]


# ---------------------------------------------------------------- 主程式
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.178.212:1234/v1")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b")
    parser.add_argument("--cache", type=Path, default=ROOT / "cvflow_tag_test_cache.json")
    parser.add_argument("--out", type=Path, default=ROOT / "cvflow_tag_test.json")
    args = parser.parse_args()

    tasks = load_tasks(ROOT / "kcore.db")
    print(f"tasks {len(tasks)}; with key nodes {sum(1 for t in tasks.values() if t['key_nodes'])}", flush=True)
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    client = OpenAICompatibleClient(args.base_url, args.model, temperature=0.0, max_tokens=3000, timeout=900)

    def ask(key, system, user, schema):
        if key not in cache:
            start = time.time()
            try:
                cache[key] = {"answer": extract_json(client.complete(system, user, schema)),
                              "seconds": round(time.time() - start, 1)}
            except (LLMError, ValueError, json.JSONDecodeError) as error:
                cache[key] = {"answer": None, "error": str(error)[:300], "seconds": round(time.time() - start, 1)}
            args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return cache[key]

    # 1. 正確答案（原始詞條）
    raw: dict[str, dict[str, list[str]]] = {}
    for i, (tid, task) in enumerate(tasks.items(), start=1):
        reply = ask(f"{PROMPT_VERSION}|gold|{args.model}|{tid}", GOLD_SYSTEM, gold_prompt(task), GOLD_SCHEMA)
        raw[tid] = {f: sorted({x.strip() for x in (reply.get("answer") or {}).get(f, []) if x.strip()}) for f in FACETS}
        print(f"gold {i}/{len(tasks)} {tid} {reply.get('seconds')}s" + (f" ERROR {reply.get('error')}" if reply.get("error") else ""),
              flush=True)

    # 2. 統一詞彙
    canonical: dict[str, dict[str, str]] = {}
    vocab: dict[str, list[str]] = {}
    for facet in FACETS:
        terms = sorted({x for t in raw.values() for x in t[facet]})
        reply = ask(f"{PROMPT_VERSION}|merge|{args.model}|{facet}|{len(terms)}", MERGE_SYSTEM,
                    f"面向：{facet}（{FACET_TEXT[facet]}）\n詞條：\n" + "\n".join(terms), MERGE_SCHEMA)
        mapping = {}
        for group in (reply.get("answer") or {}).get("terms", []):
            for member in group.get("members", []):
                if member in terms and member not in mapping:
                    mapping[member] = group["term"].strip()
        for term in terms:  # LLM 漏掉的詞保留原樣
            mapping.setdefault(term, term)
        canonical[facet] = mapping
        vocab[facet] = sorted(set(mapping.values()))
        print(f"merge {facet}: {len(terms)} raw -> {len(vocab[facet])} terms ({reply.get('seconds')}s)", flush=True)
    gold = {tid: {f: {canonical[f][x] for x in raw[tid][f]} for f in FACETS} for tid in tasks}
    term_tasks = {f: Counter(x for t in gold.values() for x in t[f]) for f in FACETS}

    # 3. 預測
    predictions: dict[str, dict[str, dict[str, set]]] = {"goal": {}, "goal+flow": {}}
    for variant in predictions:
        for i, (tid, task) in enumerate(tasks.items(), start=1):
            reply = ask(f"{PROMPT_VERSION}|predict|{variant}|{args.model}|{tid}", PREDICT_SYSTEM,
                        predict_prompt(task, vocab, variant == "goal+flow"), predict_schema(vocab))
            answer = reply.get("answer") or {}
            predictions[variant][tid] = {f: {x for x in answer.get(f, []) if x in vocab[f]} for f in CONDITION_FACETS}
        print(f"predict {variant} done", flush=True)

    # 4. 評估
    rng = random.Random(0)
    report: dict = {"facets": {}, "retrieval": {}}
    for variant, preds in predictions.items():
        for facet in CONDITION_FACETS:
            ps, rs, rs_shared, n_gold, n_unique = [], [], [], 0, 0
            for tid in tasks:
                g, p = gold[tid][facet], preds[tid][facet]
                shared = {x for x in g if term_tasks[facet][x] > 1}
                n_gold += len(g)
                n_unique += len(g - shared)
                precision, recall = prf(p, g)
                ps.append(precision)
                rs.append(recall)
                rs_shared.append(prf(p, shared)[1])
            report["facets"][f"{variant}:{facet}"] = {
                "precision": sum(ps) / len(ps), "recall": sum(rs) / len(rs),
                "recall_shared_terms": sum(rs_shared) / len(rs_shared),
                "gold_terms": n_gold, "unique_gold_terms": n_unique}
    methods = {
        "random": lambda tid: rng.sample([t for t in tasks if t != tid], 3),
        "same_metric": lambda tid: sorted((t for t in tasks if t != tid and tasks[t]["metric"] == tasks[tid]["metric"]),
                                          key=lambda t: rng.random())[:3],
        "predicted:goal": lambda tid: top_k(tid, predictions["goal"][tid], gold),
        "predicted:goal+flow": lambda tid: top_k(tid, predictions["goal+flow"][tid], gold),
        "gold_tags(upper)": lambda tid: top_k(tid, {f: gold[tid][f] for f in CONDITION_FACETS}, gold),
    }
    neighbours = {}
    for name, pick in methods.items():
        trials = 20 if name in ("random", "same_metric") else 1
        values = []
        for tid in tasks:
            for _ in range(trials):
                chosen = pick(tid)
                value = coverage(tid, chosen, tasks)
                if value is not None:
                    values.append(value)
                if trials == 1:
                    neighbours.setdefault(name, {})[tid] = chosen
        report["retrieval"][name] = sum(values) / len(values)

    output = {"prompt_version": PROMPT_VERSION, "model": args.model, "vocab": vocab, "canonical": canonical,
              "raw": raw, "gold": {t: {f: sorted(v) for f, v in g.items()} for t, g in gold.items()},
              "predictions": {v: {t: {f: sorted(x) for f, x in p.items()} for t, p in ps.items()}
                              for v, ps in predictions.items()},
              "neighbours": neighbours, "key_nodes": {t: sorted(x["key_nodes"]) for t, x in tasks.items()},
              "report": report}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n== vocabulary size:", {f: len(v) for f, v in vocab.items()})
    print("== tag prediction (mean over tasks)")
    for key, v in report["facets"].items():
        print(f"  {key:30s} P {v['precision']:.2f}  R {v['recall']:.2f}  R(shared terms) {v['recall_shared_terms']:.2f}"
              f"  gold terms {v['gold_terms']} (unique {v['unique_gold_terms']})")
    print("== knowledge retrieval: share of the task's key nodes covered by 3 retrieved tasks")
    for name, value in report["retrieval"].items():
        print(f"  {name:22s} {value:.3f}")


if __name__ == "__main__":
    main()
