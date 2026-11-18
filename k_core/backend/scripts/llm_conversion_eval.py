"""LLM 資料轉換的準確率評估：走 K 實際的轉換流程（轉換設定 → 上傳文件 → 本地 LLM → 驗證），再與標準答案比對。

測試集在 evaldata/llm_conversion/（人工撰寫的示意文件與 expected.json），不是真實工廠紀錄。
轉換結果會留在指定案例（預設 LLMTEST）中，可以在網頁的「轉換與審核」分頁檢視；不會自動核准匯入。

比對項目（節點以活動類型比較，連線以「活動類型 → 活動類型」比較）：
- 序列數是否正確、草稿是否通過 K 的驗證
- 初始流程、最終流程是否完全正確
- 修改次數是否正確；每一次修改的結構類型（K 依差異推定）是否正確
- 修改原因的來源：原文有寫 → 應標 human；原文沒寫 → null 或 llm 都算正確；把推論標成 human 是最嚴重的錯誤

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\llm_conversion_eval.py --base-url http://192.168.178.212:1234/v1 ^
        --model google/gemma-4-26b-a4b --out ..\\..\\docs\\llm_conversion_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.aov import derive_operation_type  # noqa: E402
from kcore.conversion import ConversionService  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.service import KCoreError, KCoreService  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "evaldata" / "llm_conversion"


def system_prompt(activity_types: dict[str, str]) -> str:
    vocab = "\n".join(f"- {key}：{text}" for key, text in activity_types.items())
    return (
        "你是製造業的流程分析助理。使用者會提供工廠的流程調整紀錄，格式不固定，可能是筆記、表格、各版本流程或會議紀錄。\n"
        "請找出每一張製令（工單）的初始流程，以及之後依序確認採用的每一次修改。\n"
        "活動類型（activity_type）只能使用以下代碼，節點名稱（name）保留原文用詞：\n"
        f"{vocab}\n"
        "注意：\n"
        "- 只記錄確認採用的修改。被否決、只是提議、或與流程無關的內容都不要記錄。\n"
        "- 一張製令是一條 episode；文件中有幾張製令就輸出幾條。work_order 填製令或工單編號。\n"
        "- 只改參數（例如機台、功率、層數、扭力）的修改，用 updated_node_attributes 記錄，不要新增或刪除節點。\n"
        "- 刪除站點時，要把前後兩站直接相連。"
    )


def multiset_edges(nodes: list[dict], edges: list[dict]) -> Counter:
    type_of = {n["node_id"]: n.get("activity_type") for n in nodes}
    return Counter((type_of.get(e["source"]), type_of.get(e["target"])) for e in edges)


def graph_matches(nodes: list[dict], edges: list[dict], expected: dict) -> bool:
    return (Counter(n.get("activity_type") for n in nodes) == Counter(expected["nodes"])
            and multiset_edges(nodes, edges) == Counter(tuple(e) for e in expected["edges"]))


def describe(nodes: list[dict], edges: list[dict]) -> dict:
    return {"nodes": sorted(n.get("activity_type") or "?" for n in nodes),
            "edges": sorted(f"{a}->{b}" for (a, b), c in multiset_edges(nodes, edges).items() for _ in range(c))}


def grade(detail: dict, expected_doc: dict) -> dict:
    draft_episodes = (detail.get("draft") or {}).get("episodes") or []
    previews = detail.get("preview") or []
    report = {"status": detail["status"], "error": detail.get("error"), "valid": bool(detail["validation"].get("ok")),
              "episodes_expected": len(expected_doc["episodes"]), "episodes_found": len(previews), "episodes": []}
    used: set[int] = set()
    for exp in expected_doc["episodes"]:
        index = next((i for i, d in enumerate(draft_episodes)
                      if i not in used and exp["work_order"] in str(d.get("work_order") or "")), None)
        if index is None:
            index = next((i for i in range(len(previews)) if i not in used), None)
        item = {"work_order": exp["work_order"], "found": index is not None and index < len(previews)}
        if item["found"]:
            used.add(index)
            preview = previews[index]
            versions, events = preview["versions"], preview["events"]
            item["initial_ok"] = graph_matches(versions[0]["nodes"], versions[0]["edges"], exp["initial"])
            complete = preview["stopped_at"] is None
            item["final_ok"] = complete and graph_matches(versions[-1]["nodes"], versions[-1]["edges"], exp["final"])
            item["events_expected"], item["events_found"] = len(exp["events"]), len(events)
            ops = [derive_operation_type(e["change"]) for e in events]
            sources = [e["reason_source"] for e in events]
            item["op_correct"] = sum(1 for i, e in enumerate(exp["events"]) if i < len(ops) and ops[i] == e["op"])
            item["reason_correct"] = sum(
                1 for i, e in enumerate(exp["events"]) if i < len(sources)
                and (sources[i] == "human" if e["reason"] == "human" else sources[i] in (None, "llm")))
            item["false_human"] = sum(1 for i, e in enumerate(exp["events"]) if i < len(sources)
                                      and e["reason"] != "human" and sources[i] == "human")
            reports = detail["validation"].get("episodes") or []
            warnings = reports[index]["warnings"] if index < len(reports) else []
            item["structure_warnings"] = [w for w in warnings if "step" in w or "disconnected" in w]
            item["got"] = {"initial": describe(versions[0]["nodes"], versions[0]["edges"]),
                           "final": describe(versions[-1]["nodes"], versions[-1]["edges"]) if complete else None,
                           "ops": ops, "reason_sources": sources, "stopped_at": preview["stopped_at"]}
        report["episodes"].append(item)
    report["all_correct"] = (report["valid"] and report["episodes_found"] == report["episodes_expected"] and all(
        e["found"] and e["initial_ok"] and e["final_ok"] and e["events_found"] == e["events_expected"]
        and e["op_correct"] == e["events_expected"] and e["reason_correct"] == e["events_expected"]
        for e in report["episodes"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--case", default="LLMTEST")
    parser.add_argument("--repeat", type=int, default=1, help="每份文件轉換幾次（檢查穩定性）")
    parser.add_argument("--out", type=Path, default=Path("llm_conversion_eval.json"))
    args = parser.parse_args()

    expected = json.loads((DATA / "expected.json").read_text(encoding="utf-8"))
    service = KCoreService(make_session_factory(args.database_url))
    conversions = ConversionService(service)
    try:
        service.create_case({"case_id": args.case, "name": "LLM 轉換測試（示意文件）", "domain_kind": "other",
                             "recording_enabled": False})
    except KCoreError as error:
        if error.code != "case_exists":
            raise
    profile_id = "eval-" + args.model.split("/")[-1].replace(".", "-")[:40]
    settings = {"name": f"評估用：{args.model}", "base_url": args.base_url, "model": args.model,
                "system_prompt": system_prompt(expected["activity_types"]), "temperature": 0.0,
                "structured_output": True, "timeout_seconds": 600}
    try:
        profile = conversions.create_profile(args.case, {"profile_id": profile_id, **settings})
    except KCoreError as error:
        if error.code != "profile_exists":
            raise
        current = conversions.get_profile(args.case, profile_id)
        changed = {k: v for k, v in settings.items() if current.get(k) != v}
        profile = conversions.update_profile(args.case, profile_id, changed) if changed else current
    reachable = conversions.test_profile(args.case, profile_id)
    print(f"profile {profile_id} v{profile['version']} reachable={reachable.get('reachable')} "
          f"model_available={reachable.get('model_available')}", flush=True)
    if not reachable.get("model_available"):
        raise SystemExit(f"model not available: {reachable}")

    runs = []
    for doc in expected["documents"]:
        path = DATA / doc["file"]
        document = conversions.add_document(args.case, doc["file"], path.read_bytes(), "text/plain")
        for attempt in range(1, args.repeat + 1):
            created = conversions.create_conversion(args.case, document["document_id"], profile_id)
            started = time.time()
            conversions.run_conversion(created["conversion_id"])
            seconds = time.time() - started
            detail = conversions.get_conversion(args.case, created["conversion_id"])
            result = {"file": doc["file"], "style": doc["style"], "attempt": attempt,
                      "conversion_id": created["conversion_id"], "seconds": round(seconds, 1), **grade(detail, doc)}
            runs.append(result)
            eps = result["episodes"]
            print(f"{doc['file']:22s} #{attempt} {seconds:5.1f}s status={result['status']} valid={result['valid']} "
                  f"episodes={result['episodes_found']}/{result['episodes_expected']} "
                  f"initial={sum(e.get('initial_ok', False) for e in eps)}/{len(eps)} "
                  f"final={sum(e.get('final_ok', False) for e in eps)}/{len(eps)} "
                  f"ops={sum(e.get('op_correct', 0) for e in eps)}/{sum(len(x['events']) for x in doc['episodes'])} "
                  f"reasons={sum(e.get('reason_correct', 0) for e in eps)}/{sum(len(x['events']) for x in doc['episodes'])} "
                  f"false_human={sum(e.get('false_human', 0) for e in eps)} ALL={'OK' if result['all_correct'] else 'NO'}",
                  flush=True)

    total_events = sum(len(x["events"]) for d in expected["documents"] for x in d["episodes"]) * args.repeat
    total_eps = sum(len(d["episodes"]) for d in expected["documents"]) * args.repeat
    flat = [e for r in runs for e in r["episodes"]]
    summary = {
        "model": args.model, "profile_version": profile["version"], "documents": len(expected["documents"]),
        "repeat": args.repeat, "runs": len(runs), "all_correct_runs": sum(r["all_correct"] for r in runs),
        "valid_runs": sum(r["valid"] for r in runs),
        "episodes_found": f"{sum(e['found'] for e in flat)}/{total_eps}",
        "initial_ok": f"{sum(e.get('initial_ok', False) for e in flat)}/{total_eps}",
        "final_ok": f"{sum(e.get('final_ok', False) for e in flat)}/{total_eps}",
        "op_correct": f"{sum(e.get('op_correct', 0) for e in flat)}/{total_events}",
        "reason_correct": f"{sum(e.get('reason_correct', 0) for e in flat)}/{total_events}",
        "false_human": sum(e.get("false_human", 0) for e in flat),
        "wrong_final_flagged": f"{sum(1 for e in flat if not e.get('final_ok') and e.get('structure_warnings'))}"
                               f"/{sum(1 for e in flat if not e.get('final_ok'))}",
        "correct_final_flagged": sum(1 for e in flat if e.get("final_ok") and e.get("structure_warnings")),
        "mean_seconds": round(sum(r["seconds"] for r in runs) / len(runs), 1),
    }
    args.out.write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsummary:", json.dumps(summary, ensure_ascii=False))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
