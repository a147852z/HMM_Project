"""探索實驗：本地 LLM 讀一條完整的 AoV 演變序列，整理成知識；再用程式檢查它引用的證據，最後跨序列彙整。

流程：
1. 挑序列：原本 10/13 那包、有正解框數的題目、3～10 次修改，依解題者分層抽樣（亂數種子固定）。
   只數數量的題目材料包裡沒有正解數量，無法驗證「數量有沒有變接近」，所以不選。
   --episodes-from 可以沿用上一次結果檔裡的序列，方便前後比較。
2. 每一版的數量：用執行當下的 try 檔（解題者當時看到的結果，與觀察文字一致），以流程圖內容對應到版本。
   評分檔裡的數量是事後重播的結果，約 30% 與執行當下不同，所以不用。
   v2（2026-11-24，依 CVflow 回覆 docs/REPLY-jiayuan.md）：
   - 正解框數改用 K ÷ R（provenance 的 ground_truth）；v1 誤把 K（配對成功的框數）當成標準答案。
   - 每一版另外附上正式判分（v1i：重播後 P、R 各 ≥ 0.8，忽略區的框不扣分，不看數量；解題者當時看不到）。
   - 正解框數不含 COCO 有標、但放進忽略區的物體，所以圖上的實際物體可能比正解多。
3. LLM 整理單一序列：時間線（含修改類型）、轉折點、失敗的嘗試、經驗（條件 → 做法 → 預期效果），每條都要標出依據的修改編號。
4. 程式檢查：
   - 引用的修改編號是否存在
   - 時間線的修改類型是否與實際相符
   - 經驗的「做法」提到的節點、節點類型、參數，是否真的出現在引用的修改內容裡
   - 經驗的「預期效果」（數量更接近答案、框變多、框變少）是否與引用的修改前後的實際數量一致
   - 失敗的嘗試：引用的修改之後，數量是否仍不等於答案
   （v2 起：逐步確認每一個引用的修改都碰到相關節點或參數；完整詞比對；「降低／提高某參數」檢查實際方向）
5. 跨序列彙整：只把通過檢查的經驗交給 LLM 合併成通用規則，程式再確認每條規則至少有 2 條序列支持。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_episode_knowledge.py --episodes 20
LLM 回答存在 cvflow_episode_knowledge_cache.json，可以中斷後重跑。結果寫到 cvflow_episode_knowledge.json。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kcore.adapters.cvflow import load_versions  # noqa: E402
from kcore.conversion.llm_client import LLMError, OpenAICompatibleClient, extract_json  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.service import KCoreService  # noqa: E402

PROMPT_VERSION = "episode-knowledge-v2"
PACKAGE = Path(r"E:\程式碼\2026-10-13-jiayuan-package")
OPS = ["param_tune", "add_step", "remove_step", "replace_step", "strategy_switch", "rewire"]
EFFECTS = ["count_closer", "count_further", "more_detections", "fewer_detections", "unclear"]
PER_TEACHER = {"codex": 7, "gemini": 6, "opus": 3, "haiku": 2, "sonnet": 2}

SYSTEM = (
    "你是影像辨識流程的知識整理助理。使用者會提供一條完整的流程演變紀錄：任務目標、初始流程、每次修改的內容、"
    "修改理由、修改前的觀察，以及每一版執行時數到的數量、正解框數與正式判分。請把這段過程整理成之後可以重複使用的知識。\n"
    "說明：正式判分是事後用正解判定的結果（框的精確率與召回率都 ≥ 0.8 才通過，不看數量），解題者當時看不到；"
    "正解只含部分物體，有些圖上可見、但沒有放進正解的物體被框到時不扣分，所以數量多於正解框數不一定是錯。\n"
    "規則：\n"
    "1. 只根據紀錄內容，不要補充紀錄沒有提到的事。\n"
    "2. 每一條都要標出依據的修改編號（第 N 次修改產生第 N 版）。判斷一次修改的效果時，比較第 N-1 版與第 N 版的數量與正式判分。\n"
    "3. 經驗（lessons）的 action 要寫出具體的節點類型或參數名稱；expected_change 只能依數量變化判斷："
    "count_closer（更接近標準答案）、count_further（離答案更遠）、more_detections（框變多）、"
    "fewer_detections（框變少）、unclear（看不出來）。\n"
    "4. 紀錄沒有足夠證據的地方寫進 open_questions，不要猜。用繁體中文。只輸出 JSON。"
)
SCHEMA = {
    "type": "object",
    "properties": {
        "task_summary": {"type": "string"},
        "final_solution": {"type": "string"},
        "timeline": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "integer"}, "operation_type": {"type": "string", "enum": OPS},
            "what": {"type": "string"}, "why": {"type": "string"}, "outcome": {"type": "string"}},
            "required": ["step", "operation_type", "what", "why", "outcome"]}},
        "turning_points": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "integer"}, "description": {"type": "string"}}, "required": ["step", "description"]}},
        "failed_attempts": {"type": "array", "items": {"type": "object", "properties": {
            "approach": {"type": "string"}, "why_failed": {"type": "string"},
            "evidence_steps": {"type": "array", "items": {"type": "integer"}}},
            "required": ["approach", "why_failed", "evidence_steps"]}},
        "lessons": {"type": "array", "items": {"type": "object", "properties": {
            "condition": {"type": "string"}, "action": {"type": "string"},
            "expected_change": {"type": "string", "enum": EFFECTS},
            "evidence_steps": {"type": "array", "items": {"type": "integer"}}},
            "required": ["condition", "action", "expected_change", "evidence_steps"]}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["task_summary", "final_solution", "timeline", "turning_points", "failed_attempts", "lessons",
                 "open_questions"],
}
MERGE_SYSTEM = (
    "你是影像辨識流程的知識整理助理。使用者會提供多條序列整理出、並且已通過檢查的經驗，每條有編號與所屬序列。"
    "請把意思相同或相近的經驗合併成通用規則。規則：只根據提供的經驗；每條規則列出支持它的經驗編號；"
    "只出現在一條序列的經驗也可以保留成規則，但不要把不同意思的經驗硬湊在一起；"
    "如果不同經驗互相矛盾，寫進 conflicts。用繁體中文。只輸出 JSON。"
)
MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {"type": "array", "items": {"type": "object", "properties": {
            "condition": {"type": "string"}, "action": {"type": "string"}, "expected_change": {"type": "string"},
            "lesson_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["condition", "action", "expected_change", "lesson_ids"]}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["rules", "conflicts"],
}


# ---------------------------------------------------------------- 資料
def version_counts(task_dir: Path) -> dict[str, dict]:
    """每個版本檔（g{N}.json）執行當下數到的數量（try 檔，解題者當時看到的結果）。
    v1 曾把評分檔的 K 當成標準答案，但 K 是配對成功的框數（CVflow 回覆 Q5）；v2 改用 provenance 的正解框數（K ÷ R）。"""
    tries = []
    for path in task_dir.glob("try*.json"):
        if re.match(r"^try\d+\.json$", path.name):
            data = json.loads(path.read_text(encoding="utf-8"))
            tries.append(data)
    out = {}
    for version in load_versions(task_dir):
        match = next((t for t in tries if t.get("graph") == version.graph), None)
        if match is None:
            out[version.path.name] = {"count": None, "error": "沒有執行紀錄"}
        elif match.get("exec_error") or match.get("validator_errors"):
            out[version.path.name] = {"count": None, "error": "無法執行"}
        else:
            out[version.path.name] = {"count": match.get("count"), "error": None}
    return out


def load_episodes(service: KCoreService) -> list[dict]:
    episodes = []
    with service.session_factory() as session:
        for episode, events in service._training_sequences(session, "CVFLOW"):
            if episode.episode_id.startswith("CVF2-"):
                continue
            prov, ctx = episode.provenance or {}, episode.initial_context or {}
            task_dir = PACKAGE / prov.get("path", "")
            if not task_dir.is_dir():
                continue
            counts = version_counts(task_dir)
            truth = (prov.get("ground_truth") or {}).get("gt_boxes")
            versions = load_versions(task_dir)
            steps = []
            for event in events:
                attrs = event.attributes or {}
                steps.append({"op": event.operation_type, "change": event.change or {}, "reason": (event.reason or "").strip(),
                              "exec_report": attrs.get("exec_report"), "planned_fix": attrs.get("planned_fix"),
                              "source_file": attrs.get("source_file"),
                              "count": (counts.get(attrs.get("source_file")) or {}).get("count"),
                              "error": (counts.get(attrs.get("source_file")) or {}).get("error"),
                              "passed": (attrs.get("result") or {}).get("passed")})
            first = counts.get(versions[0].path.name, {}) if versions else {}
            episodes.append({"episode_id": episode.episode_id, "teacher": ctx.get("teacher"), "request": ctx.get("request"),
                             "metric": ctx.get("metric"), "truth": truth, "first_count": first.get("count"),
                             "first_error": first.get("error"), "steps": steps,
                             "first_passed": (prov.get("first_version_result") or {}).get("passed"),
                             "flags": ctx.get("status_flags") or [], "outcome": prov.get("outcome") or {}})
    return episodes


def all_flows(service: KCoreService, episode_id: str) -> list[list[dict]]:
    """每一版的節點（第 0 版到最後一版），用來檢查參數調整的方向。"""
    from kcore.assist import AssistService
    with service.session_factory() as session:
        return [snap.nodes for snap in AssistService(service)._snapshots(session, episode_id)]


def first_flow(service: KCoreService, episode_id: str) -> tuple[list, list]:
    from kcore.assist import AssistService
    with service.session_factory() as session:
        snap = AssistService(service)._snapshots(session, episode_id)[0]
        return snap.nodes, snap.edges


def flow_text(nodes: list[dict], edges: list[dict]) -> str:
    lines = [f"- {n['node_id']}：{n['activity_type']} {json.dumps(n.get('parameters') or {}, ensure_ascii=False)}"
             for n in nodes]
    return "\n".join(lines) + "\n連線：" + "、".join(f"{e['source']}→{e['target']}" for e in edges)


def count_text(count, error, truth, passed=None) -> str:
    if error:
        return error
    answer = f"（正解 {truth} 框）" if truth else ""
    text = f"數到 {count} 個{answer}" if count is not None else "沒有數量紀錄"
    if passed is not None:
        text += f"；正式判分：{'通過' if passed else '未通過'}"
    return text


def build_prompt(ep: dict, nodes: list, edges: list) -> str:
    parts = [f"## 任務目標\n{ep['request']}（題目類型：{ep['metric']}；正解 {ep['truth']} 框）",
             f"## 第 0 版（初始流程）\n{flow_text(nodes, edges)}\n"
             f"執行結果：{count_text(ep['first_count'], ep['first_error'], ep['truth'], ep['first_passed'])}"]
    for i, s in enumerate(ep["steps"], start=1):
        block = [f"## 第 {i} 次修改（產生第 {i} 版）"]
        if s["exec_report"]:
            block.append(f"修改前，對上一版執行結果的觀察：{s['exec_report']}")
        if s["planned_fix"]:
            block.append(f"修改前寫下的修正計畫：{s['planned_fix']}")
        block.append(f"修改類型：{s['op']}")
        block.append(f"修改內容：{json.dumps(s['change'], ensure_ascii=False)}")
        if s["reason"]:
            block.append(f"修改者寫的設計理由：{s['reason']}")
        block.append(f"第 {i} 版執行結果：{count_text(s['count'], s['error'], ep['truth'], s['passed'])}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 檢查
def episode_vocabulary(ep: dict, nodes: list) -> set[str]:
    words = {n["node_id"] for n in nodes} | {n["activity_type"] for n in nodes}
    words |= {k for n in nodes for k in (n.get("parameters") or {})}
    for s in ep["steps"]:
        words |= step_terms(s)
    return {w for w in words if len(w) >= 3}


def step_terms(step: dict) -> set[str]:
    """一次修改實際碰到的節點代號、節點類型、參數名稱。"""
    c = step["change"]
    terms = set()
    for n in c.get("added_nodes", []):
        terms |= {n["node_id"], n["activity_type"], *(n.get("parameters") or {})}
    terms |= set(c.get("removed_nodes", []))
    for u in c.get("updated_node_attributes", []):
        terms |= {u["node_id"], *(u.get("parameters_set") or {}), *(u.get("parameters_unset") or [])}
    for e in c.get("added_edges", []) + c.get("removed_edges", []):
        terms |= {e["source"], e["target"]}
    return terms


def counts_by_version(ep: dict) -> list:
    return [ep["first_count"]] + [s["count"] for s in ep["steps"]]


def check_effect(effect: str, steps: list[int], ep: dict) -> str:
    counts, truth = counts_by_version(ep), ep["truth"]
    votes = []
    for i in steps:
        before, after = counts[i - 1], counts[i]
        if before is None or after is None or effect == "unclear":
            continue
        if effect == "count_closer":
            votes.append(abs(after - truth) < abs(before - truth))
        elif effect == "count_further":
            votes.append(abs(after - truth) > abs(before - truth))
        elif effect == "more_detections":
            votes.append(after > before)
        elif effect == "fewer_detections":
            votes.append(after < before)
    if not votes:
        return "unverifiable"
    if all(votes):
        return "supported"
    return "contradicted" if not any(votes) else "partial"


DECREASE = re.compile(r"降低|調低|減少|縮小|調小|放寬下限")
INCREASE = re.compile(r"提高|調高|增加|放大|加大|調大")


def mentioned_terms(text: str, vocab: set[str]) -> set[str]:
    """以完整詞比對（避免 thresh 比對到 conf_threshold）。"""
    return {w for w in vocab if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(w)}(?![A-Za-z0-9_])", text)}


def param_directions(step_index: int, flows: list[list[dict]] | None) -> dict[str, list[str]]:
    """第 step_index 次修改裡，各參數的數值是變大還是變小（同一參數可能出現在多個節點）。"""
    if not flows or step_index >= len(flows):
        return {}
    before = {n["node_id"]: n.get("parameters") or {} for n in flows[step_index - 1]}
    out: dict[str, list[str]] = defaultdict(list)
    for node in flows[step_index]:
        old = before.get(node["node_id"])
        if old is None:
            continue
        for key, value in (node.get("parameters") or {}).items():
            prev = old.get(key)
            if isinstance(value, (int, float)) and isinstance(prev, (int, float)) and value != prev:
                out[key].append("up" if value > prev else "down")
    return out


def verify(ep: dict, nodes: list, knowledge: dict, flows: list[list[dict]] | None = None) -> dict:
    n = len(ep["steps"])
    vocab = episode_vocabulary(ep, nodes)
    result = {"timeline": [], "lessons": [], "failed_attempts": [], "bad_refs": 0}

    def valid(steps):
        ok = [s for s in steps if isinstance(s, int) and 1 <= s <= n]
        result["bad_refs"] += len(steps) - len(ok)
        return ok

    for item in knowledge.get("timeline", []):
        step = item.get("step")
        if isinstance(step, int) and 1 <= step <= n:
            result["timeline"].append(item.get("operation_type") == ep["steps"][step - 1]["op"])
    counts_all, truth_all = counts_by_version(ep), ep["truth"]
    for lesson in knowledge.get("lessons", []):
        steps = valid(lesson.get("evidence_steps") or [])
        text = lesson.get("action", "")
        mentioned = mentioned_terms(text, vocab)
        # 逐步檢查：每一個引用的修改，是否真的碰到 action 提到的節點、節點類型或參數
        per_step = [bool(mentioned & step_terms(ep["steps"][s - 1])) for s in steps]
        if not steps:
            action = "no_evidence"
        elif not mentioned:
            action = "unverifiable"
        elif all(per_step):
            action = "supported"
        else:
            action = "partial" if any(per_step) else "unsupported"
        # 參數調整方向：action 寫「降低／提高 某參數」時，引用的修改裡該參數是否真的變小／變大
        want = "down" if DECREASE.search(text) else "up" if INCREASE.search(text) else None
        params = {w for w in mentioned if any(w in (n.get("parameters") or {}) for f in (flows or []) for n in f)}
        direction = "n/a"
        if want and params and flows:
            seen = [d for s in steps for p in params for d in param_directions(s, flows).get(p, [])]
            if seen:
                direction = "ok" if all(d == want for d in seen) else "wrong" if want not in seen else "mixed"
        effect = check_effect(lesson.get("expected_change", "unclear"), steps, ep) if steps else "no_evidence"
        # 附帶資訊：引用的修改實際上有沒有讓數量更接近答案（不論 LLM 怎麼說）
        closer = [abs(counts_all[s] - truth_all) < abs(counts_all[s - 1] - truth_all) for s in steps
                  if counts_all[s] is not None and counts_all[s - 1] is not None]
        # 附帶資訊（v2）：引用的修改是否讓正式判分由未通過變成通過
        passes = passes_by_version(ep)
        turned = [passes[s] is True and passes[s - 1] is not True for s in steps if passes[s] is not None]
        result["lessons"].append({"action_check": action, "direction_check": direction, "effect_check": effect,
                                  "actually_closer": (sum(closer) / len(closer)) if closer else None,
                                  "led_to_pass": any(turned) if turned else None,
                                  "mentioned": sorted(mentioned)})
    passes = passes_by_version(ep)
    for attempt in knowledge.get("failed_attempts", []):
        steps = valid(attempt.get("evidence_steps") or [])
        known = [passes[s] for s in steps if passes[s] is not None]
        if not steps:
            status = "no_evidence"
        elif not known:
            status = "unverifiable"
        else:  # v2：「失敗」以正式判分為準（v1 用數量是否等於 K，而 K 不是正解數）
            status = "supported" if not any(known) else "contradicted"
        result["failed_attempts"].append(status)
    return result


def passes_by_version(ep: dict) -> list:
    return [ep.get("first_passed")] + [s.get("passed") for s in ep["steps"]]


def lesson_passes(check: dict) -> bool:
    return (check["action_check"] == "supported" and check["direction_check"] in ("ok", "n/a")
            and check["effect_check"] in ("supported", "partial"))


# ---------------------------------------------------------------- 主程式
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.178.212:1234/v1")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cache", type=Path, default=ROOT / "cvflow_episode_knowledge_cache.json")
    parser.add_argument("--out", type=Path, default=ROOT / "cvflow_episode_knowledge.json")
    parser.add_argument("--episodes-from", type=Path, default=None,
                        help="沿用某個結果檔裡的序列（例如 docs/cvflow_episode_knowledge_results.json）")
    args = parser.parse_args()

    service = KCoreService(make_session_factory(f"sqlite:///{ROOT / 'kcore.db'}"))
    everything = load_episodes(service)
    pool = [e for e in everything if e["truth"] and 3 <= len(e["steps"]) <= 10]
    if args.episodes_from:
        wanted = [r["episode_id"] for r in json.loads(args.episodes_from.read_text(encoding="utf-8"))["episodes"]]
        by_id = {e["episode_id"]: e for e in everything}
        chosen = [by_id[eid] for eid in wanted if eid in by_id and by_id[eid]["truth"]]
    else:
        rng = random.Random(0)
        chosen = []
        for teacher, k in PER_TEACHER.items():
            group = [e for e in pool if e["teacher"] == teacher]
            chosen += rng.sample(group, min(k, len(group)))
        chosen = chosen[:args.episodes]
    print(f"pool {len(pool)} episodes; chosen {len(chosen)}: {dict(Counter(e['teacher'] for e in chosen))}", flush=True)

    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    client = OpenAICompatibleClient(args.base_url, args.model, temperature=0.0, max_tokens=6000, timeout=900)

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

    results = []
    for i, ep in enumerate(chosen, start=1):
        nodes, edges = first_flow(service, ep["episode_id"])
        prompt = build_prompt(ep, nodes, edges)
        reply = ask(f"{PROMPT_VERSION}|{args.model}|{ep['episode_id']}", SYSTEM, prompt, SCHEMA)
        knowledge = reply.get("answer")
        check = verify(ep, nodes, knowledge, all_flows(service, ep["episode_id"])) if knowledge else None
        results.append({"episode_id": ep["episode_id"], "teacher": ep["teacher"], "steps": len(ep["steps"]),
                        "truth": ep["truth"], "counts": counts_by_version(ep), "passes": passes_by_version(ep),
                        "flags": ep["flags"], "ops": [s["op"] for s in ep["steps"]],
                        "has_text": any(s["exec_report"] for s in ep["steps"]), "seconds": reply.get("seconds"),
                        "prompt_chars": len(prompt), "error": reply.get("error"), "knowledge": knowledge, "check": check})
        print(f"{i}/{len(chosen)} {ep['episode_id']} {reply.get('seconds')}s"
              + (f" ERROR {reply.get('error')}" if reply.get("error") else ""), flush=True)

    # 跨序列彙整：只用通過檢查的經驗
    passed = []
    for r in results:
        if not r["knowledge"]:
            continue
        for j, (lesson, check) in enumerate(zip(r["knowledge"].get("lessons", []), r["check"]["lessons"])):
            if lesson_passes(check):
                passed.append({"id": f"{r['episode_id'][-6:]}-{j + 1}", "episode": r["episode_id"], **lesson})
    merge_input = "\n".join(f"[{p['id']}]（序列 {p['episode'][-6:]}）條件：{p['condition']}；做法：{p['action']}；"
                            f"預期效果：{p['expected_change']}" for p in passed)
    merged = ask(f"{PROMPT_VERSION}|merge|{args.model}|{len(passed)}|{hash(merge_input)}", MERGE_SYSTEM,
                 merge_input, MERGE_SCHEMA) if passed else {"answer": None}
    by_id = {p["id"]: p for p in passed}
    rules = []
    for rule in (merged.get("answer") or {}).get("rules", []):
        ids = [x for x in rule.get("lesson_ids", []) if x in by_id]
        rules.append({**rule, "valid_lesson_ids": ids, "unknown_ids": [x for x in rule.get("lesson_ids", []) if x not in by_id],
                      "support_episodes": len({by_id[x]["episode"] for x in ids})})
    output = {"prompt_version": PROMPT_VERSION, "model": args.model, "episodes": results, "passed_lessons": passed,
              "merge": merged, "rules": rules}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    report(results, passed, rules, merged)


def report(results, passed, rules, merged) -> None:
    ok = [r for r in results if r["knowledge"]]
    print(f"\n== {len(ok)}/{len(results)} episodes answered; mean {sum(r['seconds'] or 0 for r in results) / len(results):.0f}s")
    timeline = [x for r in ok for x in r["check"]["timeline"]]
    print(f"timeline op correct: {sum(timeline)}/{len(timeline)}")
    print(f"invalid step references: {sum(r['check']['bad_refs'] for r in ok)}")
    lessons = [c for r in ok for c in r["check"]["lessons"]]
    print(f"lessons: {len(lessons)} (per episode {len(lessons) / max(1, len(ok)):.1f})")
    print("  action check:", dict(Counter(c['action_check'] for c in lessons)))
    print("  direction check:", dict(Counter(c['direction_check'] for c in lessons)))
    print("  effect check:", dict(Counter(c['effect_check'] for c in lessons)))
    print(f"  pass all checks: {len(passed)}")
    closer = [c["actually_closer"] for c in lessons if c["actually_closer"] is not None]
    print(f"  cited steps actually closer to the answer: {sum(closer) / max(1, len(closer)):.2f} (mean over lessons)")
    turned = [c["led_to_pass"] for c in lessons if c.get("led_to_pass") is not None]
    print(f"  lessons citing a step that turned the official result into a pass: {sum(turned)}/{len(turned)}")
    print(f"  episodes whose last version passes: {sum(1 for r in ok if r['passes'][-1])}/{len(ok)}; "
          f"last count == GT boxes: {sum(1 for r in ok if r['counts'][-1] == r['truth'])}/{len(ok)}")
    attempts = [s for r in ok for s in r["check"]["failed_attempts"]]
    print("failed attempts:", dict(Counter(attempts)))
    for flag in (True, False):
        g = [r for r in ok if r["has_text"] == flag]
        ls = [c for r in g for c in r["check"]["lessons"]]
        if ls:
            print(f"  has observation text={flag}: episodes {len(g)}, lessons {len(ls)}, pass {sum(lesson_passes(c) for c in ls)}")
    print(f"\nmerged rules: {len(rules)}; conflicts: {len((merged.get('answer') or {}).get('conflicts', []))}")
    for rule in sorted(rules, key=lambda r: -r["support_episodes"]):
        print(f"  [{rule['support_episodes']} episodes] {rule['condition']} → {rule['action']}（{rule['expected_change']}）"
              + (f" unknown ids {rule['unknown_ids']}" if rule["unknown_ids"] else ""))


if __name__ == "__main__":
    main()
