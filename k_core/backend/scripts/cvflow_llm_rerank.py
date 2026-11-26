"""探索實驗：讓 LLM 依任務目標與目前流程，替 HMM 選出的前 10 名候選修改重新排序，能不能更常猜中下一步？

評估指標是「猜中實際的下一次修改」（修改類型 × 節點類型），另外分開看成功的修改：
- final_ok：最後有改成功的序列裡的步驟
- passed：這次修改後判分通過的步驟
- improved：上一版沒通過、這次修改後通過的步驟

切分與訓練：沿用 5 折交叉驗證（依序列切分，亂數種子 0），每一折重新訓練 HMM 與樣式庫。
- orig：測試原本 10/13 那包，訓練 = 原本序列的其餘 4 折
- new ：測試 11/17 更新包（CVF2-），訓練 = 全部原本序列 + 新資料的其餘 4 折
兩種都是 cvflow_more_data.py 裡成績最好的訓練方式。

給 LLM 的資訊只用「真實場景也拿得到、而且在這次修改之前就知道」的內容：
- basic：任務目標（原本那包有文字需求；新資料只有題目類型）、目前流程（節點、參數、連線）、
         先前的修改與當時寫下的修改理由
- +obs ：再加上上一版的執行結果與觀察文字（原本那包有 exec_report；新資料只有數量與錯誤類型）
不提供：這次修改自己的理由、上一輪寫的修正計畫（planned_fix）——兩者都等於直接透露答案。
候選以隨機順序呈現，不給 HMM 分數，避免 LLM 只是照抄 HMM 的順序。

排序方式（都只重排 HMM 前 10 名，第 11 名以後維持 HMM 順序）：
- hmm   ：原本的 HMM 排序
- llm   ：完全照 LLM 的排序
- fused ：HMM 名次與 LLM 名次平均，同分時以 HMM 為準（規則事先固定，沒有依結果調整）

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_llm_rerank.py --base-url http://192.168.178.212:1234/v1 ^
        --model google/gemma-4-26b-a4b --sample 300
LLM 回答存在 cvflow_llm_rerank_cache.jsonl，可以中斷後重跑；--dry-run 只產生提示詞、不呼叫 LLM；
--cache-only 不呼叫 LLM，只統計快取裡已經有回答的步驟（實驗還在跑時查看部分結果用）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cvflow_assist_cv as cv  # noqa: E402
from kcore.assist import AssistService, result_label  # noqa: E402
from kcore.conversion.llm_client import LLMError, OpenAICompatibleClient, extract_json  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.encoding import Vocabulary, event_token  # noqa: E402
from kcore.hmm.hybrid import MixturePredictor  # noqa: E402
from kcore.hmm.training import train_with_selection  # noqa: E402
from kcore.service import KCoreService  # noqa: E402

PROMPT_VERSION = "rerank-v2"  # v2：先輸出排序，理由限一句話（v1 每次約 33 秒，太慢）
TOP_N = 10
VARIANTS = ("basic", "+obs")
OP_TEXT = {"param_tune": "調整參數", "add_step": "新增節點", "remove_step": "刪除節點",
           "replace_step": "替換節點", "strategy_switch": "改用另一套流程（換策略）", "rewire": "改變連線"}
SYSTEM = (
    "你是影像辨識流程（OpenCV 節點組成的資料流程圖）的修改顧問。一位工程師正在反覆修改流程，"
    "直到結果符合任務目標。使用者會提供任務目標、目前的流程、先前的修改紀錄，以及編號的候選修改。\n"
    "請判斷工程師「下一步最可能」做哪一個修改，把所有候選依可能性由高到低排序。\n"
    "判斷時請考慮：目前流程能不能達成任務目標、先前修改的方向與理由、上一版的問題（如果有提供）。\n"
    "ranking 必須包含每一個候選編號各一次。reason 只寫一句話（50 字以內）。只輸出 JSON。"
)
SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}, "reason": {"type": "string"}},
    "required": ["ranking", "reason"],
}


# ---------------------------------------------------------------- 資料
def load_rows(service: KCoreService) -> list[dict]:
    """cv.load 的資料，另外加上任務目標、修改理由、執行結果與觀察文字。"""
    rows = cv.load(service, "CVFLOW", {})
    by_id = {r["episode_id"]: r for r in rows}
    with service.session_factory() as session:
        for episode, events in service._training_sequences(session, "CVFLOW"):
            row = by_id[episode.episode_id]
            ctx, prov = episode.initial_context or {}, episode.provenance or {}
            row.update(new=episode.episode_id.startswith("CVF2-"), request=ctx.get("request"),
                       metric=ctx.get("metric"), category=ctx.get("category"),
                       final_ok=bool((prov.get("outcome") or {}).get("final_ok")))
            previous = prov.get("first_version_result")
            for step, event in zip(row["steps"], events, strict=True):
                attrs = event.attributes or {}
                result = attrs.get("result") or {}
                step.update(event_id=event.event_id, reason=(event.reason or "").strip(),
                            prev_result=previous, exec_report=attrs.get("exec_report"),
                            passed=result.get("passed") is True,
                            improved=result.get("passed") is True and result_label(previous) != "pass")
                previous = result
    return rows


def describe_result(result: dict | None) -> str:
    if not result:
        return "（沒有紀錄）"
    if result.get("executable") is False or result.get("exec_error"):
        return "無法執行" + (f"：{result.get('exec_error')}" if result.get("exec_error") else "")
    parts = [{True: "判分通過", False: "判分未通過"}.get(result.get("passed"), "沒有判分")]
    # 只有 11/17 包（有 error_types 欄位）附上數量。10/13 包從 2026-11-24 起也存了解題當下的 count，
    # 但為了和原本的實驗比較，提示詞維持原樣，只讓「通過」的判定改變。
    if result.get("count") is not None and "error_types" in result:
        parts.append(f"數到 {result['count']} 個")
    errors = [e for e in result.get("error_types") or [] if e != "ok"]
    if errors:
        parts.append("錯誤類型：" + "、".join(errors))
    return "，".join(parts)


def describe_flow(nodes: list[dict], edges: list[dict]) -> str:
    lines = []
    for n in nodes:
        params = json.dumps(n.get("parameters") or {}, ensure_ascii=False)
        lines.append(f"- {n['node_id']}：{n['activity_type']} {params if len(params) <= 160 else params[:160] + '…'}")
    lines.append("連線：" + "、".join(f"{e['source']}→{e['target']}" for e in edges))
    return "\n".join(lines)


def describe_candidate(c: dict, nodes: list[dict]) -> str:
    type_of = {n["node_id"]: n["activity_type"] for n in nodes}
    op, node_type, d = c["operation_type"], c["key"][1], c["detail"]
    if op == "param_tune":
        hint = f"（常調的參數：{'、'.join(d['suggested_parameters'])}）" if d.get("suggested_parameters") else ""
        return f"調整 {d['target_node']}（{node_type}）的參數{hint}"
    if op == "remove_step":
        return f"刪除 {d['target_node']}（{type_of.get(d['target_node'], node_type)}）"
    if op == "add_step":
        return f"在 {d['after_node']}（{type_of.get(d['after_node'], '?')}）之後新增 {node_type}"
    if op == "replace_step":
        return f"把 {d['replaced_node']}（{d['replaced_type']}）換成 {node_type}"
    if op == "strategy_switch":
        return "改用另一套流程：" + " → ".join(n["activity_type"] for n in c["nodes"])
    return f"{OP_TEXT.get(op, op)}：{node_type}"


def build_prompt(row: dict, t: int, options: list[tuple[int, str]], with_obs: bool) -> str:
    step = row["steps"][t]
    goal = row["request"] or f"（沒有文字需求）題目類型：{row['metric']}，類別：{row['category']}"
    parts = [f"## 任務目標\n{goal}", f"## 目前的流程\n{describe_flow(*step['before'])}"]
    history = []
    for k, prev in enumerate(row["steps"][:t], start=1):
        line = f"{k}. {OP_TEXT.get(prev['op'], prev['op'])}（{prev['type']}）"
        if k > t - 3 and prev["reason"]:  # 最近 3 次附上當時的修改理由
            line += f"\n   理由：{prev['reason'][:300]}"
        history.append(line)
    parts.append("## 先前的修改\n" + ("\n".join(history) if history else "（這是第一次修改）"))
    if with_obs:
        obs = f"執行結果：{describe_result(step['prev_result'])}"
        if step["exec_report"]:
            obs += f"\n觀察：{step['exec_report'][:600]}"
        parts.append(f"## 上一版的執行情形\n{obs}")
    parts.append("## 候選修改\n" + "\n".join(f"[{i}] {text}" for i, text in options))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- HMM 候選
def hmm_candidates(train: list[dict], test_steps: list[tuple[dict, int]], assist: AssistService) -> dict:
    library, _ = cv.build_library(train)
    tokens = [[s["token"] for s in r["steps"]] for r in train]
    vocab = Vocabulary.build(tokens, 1)
    norm = [[vocab.normalize(t) for t in seq] for seq in tokens]
    trained = train_with_selection([vocab.encode(s) for s in norm], vocab.symbols, cv.CONFIG)
    mix = MixturePredictor.fit(trained.model, norm, cv.CONFIG.smoothing).predict
    out = {}
    for row, t in test_steps:
        step = row["steps"][t]
        prefix = [vocab.normalize(s["token"]) for s in row["steps"][:t]]
        nodes, edges = step["before"]
        candidates, _ = assist._candidates(nodes, edges, library)
        for c in candidates:
            c["token"] = vocab.normalize(event_token(c["operation_type"], {"activity_type": c["activity_type"]}, cv.SCHEMA))
        p_mix = dict(zip(vocab.symbols, (float(p) for p in mix(prefix)), strict=True))
        ranked = cv.rank(candidates, p_mix, lambda k: library.keys[k])
        by_key = {c["key"]: c for c in candidates}
        out[step["event_id"]] = {"ranked": ranked, "top": [by_key[k] for k in ranked[:TOP_N]]}
    return out


def sample_steps(rows: list[dict], n: int, seed: int) -> set[str]:
    ids = [s["event_id"] for r in rows for s in r["steps"]]
    return set(random.Random(seed).sample(ids, min(n, len(ids))))


# ---------------------------------------------------------------- 主程式
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.178.212:1234/v1")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b")
    parser.add_argument("--sample", type=int, default=300, help="每一部分抽多少步驟")
    parser.add_argument("--parts", default="orig,new")
    parser.add_argument("--cache", type=Path, default=ROOT / "cvflow_llm_rerank_cache.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "cvflow_llm_rerank.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--refresh-from-db", type=Path, default=None,
                        help="和舊資料庫比較：「上一版執行情形」文字不同的步驟，用新的快取索引重問 LLM")
    args = parser.parse_args()

    service = KCoreService(make_session_factory(f"sqlite:///{ROOT / 'kcore.db'}"))
    assist = AssistService(service)
    rows = load_rows(service)
    # 2026-11-24 起「通過」改用 v1i／error_types ok（CVflow 回覆）。只有 +obs 提示詞會用到上一版結果；
    # 文字和舊資料庫不同的步驟換一個快取索引重問，其他沿用原本的回答。
    refresh: set[str] = set()
    if args.refresh_from_db:
        old_rows = load_rows(KCoreService(make_session_factory(f"sqlite:///{args.refresh_from_db}")))
        old_text = {s["event_id"]: describe_result(s["prev_result"]) for r in old_rows for s in r["steps"]}
        refresh = {s["event_id"] for r in rows for s in r["steps"]
                   if old_text.get(s["event_id"]) != describe_result(s["prev_result"])}
        print(f"steps whose previous-result text changed: {len(refresh)}", flush=True)
    original = [r for r in rows if not r["new"]]
    new = [r for r in rows if r["new"]]
    cache: dict[str, dict] = {}
    if args.cache.exists():
        for line in args.cache.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            cache[item["cache_key"]] = item
    client = None if args.dry_run or args.cache_only else OpenAICompatibleClient(args.base_url, args.model, temperature=0.0,
                                                              max_tokens=400, timeout=300)
    records = []
    started, calls, failures = time.time(), 0, 0
    for part in args.parts.split(","):
        test_pool, fixed_train = (original, []) if part == "orig" else (new, original)
        chosen = sample_steps(test_pool, args.sample, seed=1)
        order = list(range(len(test_pool)))
        random.Random(0).shuffle(order)
        for fold in range(5):
            test_idx = set(order[fold::5])
            train = fixed_train + [r for i, r in enumerate(test_pool) if i not in test_idx]
            test_steps = [(r, t) for i, r in enumerate(test_pool) if i in test_idx
                          for t, s in enumerate(r["steps"]) if s["event_id"] in chosen]
            if args.cache_only:
                test_steps = [(r, t) for r, t in test_steps if all(
                    f"{PROMPT_VERSION}|{args.model}|{v}|{r['steps'][t]['event_id']}" in cache for v in VARIANTS)]
            if not test_steps:
                continue
            cands = hmm_candidates(train, test_steps, assist)
            print(f"{part} fold {fold + 1}: train {len(train)} episodes, {len(test_steps)} sampled steps", flush=True)
            for row, t in test_steps:
                step = row["steps"][t]
                info = cands[step["event_id"]]
                actual = (step["op"], step["type"])
                top_keys = [c["key"] for c in info["top"]]
                shuffled = list(range(len(top_keys)))
                random.Random(step["event_id"]).shuffle(shuffled)
                options = [(j + 1, describe_candidate(info["top"][i], step["before"][0])) for j, i in enumerate(shuffled)]
                record = {"part": part, "event_id": step["event_id"], "teacher": row["teacher"], "op": step["op"],
                          "pos": t, "length": len(row["steps"]), "final_ok": row["final_ok"], "passed": step["passed"],
                          "improved": step["improved"], "covered": actual in info["ranked"],
                          "in_top": actual in top_keys, "hmm_rank": (info["ranked"].index(actual) + 1
                                                                     if actual in info["ranked"] else None),
                          "n_top": len(top_keys), "hmm_top1_op": top_keys[0][0] if top_keys else None,
                          "has_history_reason": any(s["reason"] for s in row["steps"][:t]),
                          "has_exec_report": bool(step["exec_report"]), "has_request": bool(row["request"])}
                for variant in VARIANTS:
                    prompt = build_prompt(row, t, options, with_obs=(variant == "+obs"))
                    key = f"{PROMPT_VERSION}|{args.model}|{variant}|{step['event_id']}"
                    if variant == "+obs" and step["event_id"] in refresh:
                        key += "|results-20261124"
                    if args.dry_run:
                        if len(records) < 2:
                            print("=" * 60, variant, "\n" + prompt)
                        continue
                    if key not in cache and len(top_keys) > 1:
                        calls += 1
                        try:
                            answer = extract_json(client.complete(SYSTEM, prompt, SCHEMA))
                            cache[key] = {"cache_key": key, "ranking": answer.get("ranking"), "reason": answer.get("reason")}
                        except (LLMError, ValueError, json.JSONDecodeError) as error:
                            failures += 1
                            cache[key] = {"cache_key": key, "ranking": None, "error": str(error)[:300]}
                        with args.cache.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
                        if calls % 20 == 0:
                            print(f"  {calls} calls, {failures} failures ({time.time() - started:.0f}s)", flush=True)
                    llm_order = (cache.get(key) or {}).get("ranking") or []
                    # LLM 的編號換回候選；缺的、重複的、超出範圍的忽略，沒排到的接在後面（維持 HMM 順序）
                    picked: list[int] = []
                    for number in llm_order:
                        if isinstance(number, int) and 1 <= number <= len(shuffled) and shuffled[number - 1] not in picked:
                            picked.append(shuffled[number - 1])
                    valid = len(picked) == len(top_keys)
                    picked += [i for i in range(len(top_keys)) if i not in picked]
                    llm_rank = {top_keys[i]: r for r, i in enumerate(picked)}
                    hmm_rank = {k: r for r, k in enumerate(top_keys)}
                    fused = sorted(top_keys, key=lambda k: ((hmm_rank[k] + llm_rank[k]) / 2, hmm_rank[k]))
                    llm_only = [top_keys[i] for i in picked]
                    rest = info["ranked"][TOP_N:]
                    for name, ranking in (("llm", llm_only + rest), ("fused", fused + rest)):
                        record[f"{variant}:{name}"] = ranking.index(actual) + 1 if actual in ranking else None
                    record[f"{variant}:valid"] = valid
                    record[f"{variant}:llm_top1_op"] = llm_only[0][0] if llm_only else None
                    record[f"{variant}:fused_top1_op"] = fused[0][0] if fused else None
                records.append(record)
    if args.dry_run:
        return
    args.out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"\ncalls {calls}, failures {failures}, {time.time() - started:.0f}s")
    report(records)


def report(records: list[dict]) -> None:
    methods = ["hmm_rank"] + [f"{v}:{m}" for v in VARIANTS for m in ("llm", "fused")]
    subsets = {"all": lambda r: True, "final_ok": lambda r: r["final_ok"], "passed": lambda r: r["passed"],
               "improved": lambda r: r["improved"]}
    for part in sorted({r["part"] for r in records}):
        rs = [r for r in records if r["part"] == part]
        print(f"\n== {part}: {len(rs)} steps, covered {sum(r['covered'] for r in rs) / len(rs):.3f}, "
              f"in top{TOP_N} {sum(r['in_top'] for r in rs) / len(rs):.3f}, "
              f"valid LLM answers " + ", ".join(f"{v} {sum(r.get(f'{v}:valid', False) for r in rs)}" for v in VARIANTS))
        for subset, keep in subsets.items():
            g = [r for r in rs if keep(r)]
            if not g:
                continue
            print(f"  [{subset}] n={len(g)}")
            for m in methods:
                hit = lambda k, m=m: sum(1 for r in g if r[m] is not None and r[m] <= k) / len(g)  # noqa: E731
                print(f"    {m:12s} hit1 {hit(1):.3f}  hit3 {hit(3):.3f}  hit5 {hit(5):.3f}")
        print("  hit3 by op (all):")
        for op, n in Counter(r["op"] for r in rs).most_common():
            g = [r for r in rs if r["op"] == op]
            cells = "  ".join(f"{m} {sum(1 for r in g if r[m] is not None and r[m] <= 3) / n:.2f}" for m in methods)
            print(f"    {op:16s} n={n:4d}  {cells}")


if __name__ == "__main__":
    main()
