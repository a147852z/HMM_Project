"""LLM 轉換草稿的固定 JSON 格式、提示詞的固定格式說明，以及草稿驗證與正規化。

分工原則：
- LLM 負責「理解」：從原始資料找出流程、修改、原因，並對應到活動類型。
- 程式負責「確定性工作」：產生所有識別（序列、事件、版本）、由完整版本計算差異、推定修改類型、
  檢查 DAG 與原因出處。同一份文件重新轉換時，識別保持不變，避免重複計入訓練資料。
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..aov import AoVError, apply_change, derive_operation_type, diff_workflows, is_empty_change, validate_workflow


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class DraftNode(_Loose):
    node_id: str
    activity_type: str
    name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class DraftEdge(_Loose):
    source: str
    target: str


class DraftWorkflow(_Loose):
    nodes: list[DraftNode]
    edges: list[DraftEdge] = Field(default_factory=list)


class DraftChange(_Loose):
    added_nodes: list[DraftNode] = Field(default_factory=list)
    removed_nodes: list[str] = Field(default_factory=list)
    added_edges: list[DraftEdge] = Field(default_factory=list)
    removed_edges: list[DraftEdge] = Field(default_factory=list)
    updated_node_attributes: list[dict[str, Any]] = Field(default_factory=list)


class DraftEvent(_Loose):
    operation_type: str | None = None
    occurred_at: str | None = None
    change: DraftChange | None = None
    after_workflow: DraftWorkflow | None = None
    reason: str | None = None
    reason_source: str | None = None  # human / llm / None
    reason_quote: str | None = None
    source_excerpt: str | None = None


class DraftEpisode(_Loose):
    work_order: str | None = None
    started_at: str | None = None
    initial_context: dict[str, Any] = Field(default_factory=dict)
    initial_workflow: DraftWorkflow
    events: list[DraftEvent]


class Draft(_Loose):
    episodes: list[DraftEpisode]
    notes: str | None = None


def draft_json_schema() -> dict[str, Any]:
    return Draft.model_json_schema()


FORMAT_INSTRUCTIONS = """\
【輸出格式（K 系統固定要求，請嚴格遵守）】
只輸出一個 JSON 物件，不要輸出任何其他文字。結構如下：
{
  "episodes": [                       // 一次完整演變（從初始流程到定案）為一筆；資料中有幾次就輸出幾筆
    {
      "work_order": "製令編號或 null",
      "started_at": "ISO 8601 時間或 null",
      "initial_context": { },          // 製令需求等當時已知的條件；沒有就給 {}
      "initial_workflow": {
        "nodes": [{"node_id": "n1", "activity_type": "活動類型", "name": "原始名稱", "parameters": {}}],
        "edges": [{"source": "n1", "target": "n2"}]     // source 必須先於 target，整體不可形成循環
      },
      "events": [                     // 依實際發生順序排列，每次修改一筆
        {
          "operation_type": "修改類型或 null",
          "occurred_at": "ISO 8601 時間或 null",
          "change": {                 // 二選一：直接描述差異……
            "added_nodes": [], "removed_nodes": ["node_id"], "added_edges": [], "removed_edges": [],
            "updated_node_attributes": [{"node_id": "n2", "parameters_set": {}, "parameters_unset": []}]
          },
          "after_workflow": null,     // ……或給出修改後的完整流程（K 會自動計算差異）。資料若是各版本的完整流程，請用這個。
          "reason": "修改原因或 null",
          "reason_source": "human | llm | null",
          "reason_quote": "原文中的原因文字或 null",
          "source_excerpt": "這次修改在原文中的依據片段"
        }
      ]
    }
  ],
  "notes": "無法判斷或有疑慮的地方"
}
規則：
1. 不要產生序列、事件或版本的識別碼，K 會自動產生。node_id 只需在同一筆 episode 內唯一，且同一個活動在各版本中必須使用相同的 node_id。
2. 刪除節點時，必須同時在 removed_edges 列出該節點的所有連線；新增節點時要補上它的連線。
   刪除中間的站點時，除非原文另有說明，要把它的前一站與後一站直接相連（added_edges）。
   新增平行的站點或分支時，不要刪除原本的連線，原有的流程仍然保留。
3. 修改原因是「為什麼要改」，不是「改了什麼」。例如「加工改用 CNC-2 機台」是修改內容，不是原因。
   原文有寫原因時，reason 填原因，reason_source 填 "human"，reason_quote 逐字複製原文中說明原因的那段文字；
   原文只寫了改什麼、沒寫為什麼時，reason、reason_source、reason_quote 都填 null；
   原文沒寫但你推論出原因時，reason_source 填 "llm"，reason_quote 填 null。不可把推論寫成原文。
4. change 與 after_workflow 只填其中一個，另一個填 null。
5. 不要捏造原文沒有的活動、修改或時間。原文的時間沒有年份時，occurred_at 填 null。不確定的地方寫在 notes。
"""


def _components(nodes: list[dict], edges: list[dict]) -> int:
    parent = {n["node_id"]: n["node_id"] for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if e["source"] in parent and e["target"] in parent:
            parent[find(e["source"])] = find(e["target"])
    return len({find(n) for n in parent})


def structure_warnings(before_nodes: list[dict], before_edges: list[dict],
                       after_nodes: list[dict], after_edges: list[dict]) -> list[str]:
    """合法但可疑的結構變化：流程斷成多段、某站突然沒有下一站或上一站（常見原因是漏了連線）。"""
    warnings = []
    if _components(after_nodes, after_edges) > _components(before_nodes, before_edges):
        warnings.append("workflow became disconnected")
    out_before = {e["source"] for e in before_edges}
    in_before = {e["target"] for e in before_edges}
    out_after = {e["source"] for e in after_edges}
    in_after = {e["target"] for e in after_edges}
    for node in after_nodes:
        node_id = node["node_id"]
        label = f"{node_id}（{node.get('name') or node.get('activity_type')}）"
        if node_id in out_before and node_id not in out_after:
            warnings.append(f"node {label} no longer has a next step")
        if node_id in in_before and node_id not in in_after:
            warnings.append(f"node {label} no longer has a previous step")
    return warnings


def _same_change(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """比較兩份差異是否描述同一件事（忽略順序與節點的細部屬性）。"""
    def edges(items: list) -> set:
        return {(e["source"], e["target"]) for e in items}

    def node_ids(items: list) -> set:
        return {n["node_id"] if isinstance(n, dict) else n for n in items}

    return (node_ids(a.get("added_nodes", [])) == node_ids(b.get("added_nodes", []))
            and node_ids(a.get("removed_nodes", [])) == node_ids(b.get("removed_nodes", []))
            and edges(a.get("added_edges", [])) == edges(b.get("added_edges", []))
            and edges(a.get("removed_edges", [])) == edges(b.get("removed_edges", []))
            and node_ids(a.get("updated_node_attributes", [])) == node_ids(b.get("updated_node_attributes", [])))


def build_messages(system_prompt: str, user_template: str, document: str, filename: str) -> tuple[str, str]:
    """案例自訂的提示詞在前，K 的固定格式說明附加在系統提示詞最後。"""
    system = f"{system_prompt.strip()}\n\n{FORMAT_INSTRUCTIONS}"
    if "{document}" in user_template:
        user = user_template.replace("{document}", document).replace("{filename}", filename)
    else:
        user = f"{user_template.replace('{filename}', filename).rstrip()}\n\n---\n{document}"
    return system, user


def build_preview(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """供審核畫面使用：逐步套用每次修改，回傳每個版本的完整流程；遇到無法套用的修改即停止。"""
    preview: list[dict[str, Any]] = []
    for payload in payloads:
        workflow = payload["initial_workflow"]
        nodes, edges = workflow["nodes"], workflow["edges"]
        versions = [{"version_id": workflow["version_id"], "event_index": 0, "nodes": nodes, "edges": edges}]
        stopped_at = None
        if not validate_workflow(nodes, edges):
            for event in payload["events"]:
                try:
                    nodes, edges = apply_change(nodes, edges, event["change"])
                except AoVError:
                    stopped_at = event["sequence_index"]
                    break
                versions.append({"version_id": event["after_version_id"], "event_index": event["sequence_index"],
                                 "nodes": nodes, "edges": edges})
        else:
            stopped_at = 0
        preview.append({"episode_id": payload["episode_id"], "versions": versions, "stopped_at": stopped_at,
                        "events": payload["events"]})
    return preview


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def episode_ids(case_id: str, document_sha256: str, index: int) -> str:
    return f"{case_id}-D{document_sha256[:10]}-{index + 1:02d}"


def validate_and_normalize(raw_draft: Any, case_id: str, document_sha256: str,
                           document_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """驗證草稿並轉成 K 的匯入格式。回傳 (可匯入的序列清單, 驗證報告)。

    報告中的 errors 會阻擋核准；warnings 只提示審核者。
    """
    try:
        draft = Draft.model_validate(raw_draft)
    except ValidationError as error:
        problems = [f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}" for item in error.errors()]
        return [], {"ok": False, "schema_errors": problems[:50], "episodes": []}

    normalized_text = _normalize_text(document_text)
    payloads: list[dict[str, Any]] = []
    report: dict[str, Any] = {"ok": True, "schema_errors": [], "episodes": [], "notes": draft.notes}
    if not draft.episodes:
        report["ok"] = False
        report["schema_errors"].append("draft contains no episodes")

    for index, episode in enumerate(draft.episodes):
        episode_id = episode_ids(case_id, document_sha256, index)
        errors: list[str] = []
        warnings: list[str] = []
        nodes = [node.model_dump() for node in episode.initial_workflow.nodes]
        edges = [edge.model_dump() for edge in episode.initial_workflow.edges]
        errors += [f"initial_workflow: {problem}" for problem in validate_workflow(nodes, edges)]
        events: list[dict[str, Any]] = []
        if not episode.events:
            errors.append("episode has no events")

        for position, event in enumerate(episode.events, start=1):
            label = f"event {position}"
            if event.change is None and event.after_workflow is None:
                errors.append(f"{label}: provide change or after_workflow")
                continue
            if event.after_workflow is not None:
                # 兩者都有時以完整流程為準：由程式計算差異，比模型自己描述的差異可靠
                after = event.after_workflow
                change = diff_workflows(nodes, edges, [n.model_dump() for n in after.nodes],
                                        [e.model_dump() for e in after.edges])
                if event.change is not None and not _same_change(change, event.change.model_dump()):
                    warnings.append(f"{label}: change and after_workflow disagree; used after_workflow")
            else:
                change = event.change.model_dump()
            if is_empty_change(change):
                errors.append(f"{label}: change is empty")
                continue
            if not errors:
                try:
                    new_nodes, new_edges = apply_change(nodes, edges, change)
                    warnings += [f"{label}: {w}" for w in structure_warnings(nodes, edges, new_nodes, new_edges)]
                    nodes, edges = new_nodes, new_edges
                except AoVError as error:
                    errors += [f"{label}: {problem}" for problem in error.problems]

            reason = (event.reason or "").strip() or None
            source = event.reason_source if event.reason_source in {"human", "llm"} else None
            if event.reason_source not in (None, "human", "llm"):
                warnings.append(f"{label}: unknown reason_source '{event.reason_source}' treated as missing")
            if reason and source is None:
                source = "llm"
                warnings.append(f"{label}: reason without reason_source is treated as LLM-inferred")
            if source == "human":
                quote = (event.reason_quote or "").strip()
                if not quote or _normalize_text(quote) not in normalized_text:
                    source = "llm"
                    warnings.append(f"{label}: reason marked human but its quote is not found verbatim in the "
                                    "source document; downgraded to llm")
            if not reason:
                source = None

            operation = (event.operation_type or "").strip() or derive_operation_type(change)
            if not event.operation_type:
                warnings.append(f"{label}: operation_type derived by K as '{operation}'")
            events.append({
                "event_id": f"{episode_id}-E{position:02d}",
                "sequence_index": position,
                "occurred_at": event.occurred_at,
                "actor_type": "human",
                "operation_type": operation,
                "before_version_id": f"{episode_id}-V{position - 1}",
                "after_version_id": f"{episode_id}-V{position}",
                "change": change,
                "reason": reason,
                "reason_source": source,
                "attributes": {"source_excerpt": event.source_excerpt, "reason_quote": event.reason_quote},
            })

        report["episodes"].append({"episode_id": episode_id, "event_count": len(events),
                                   "errors": errors, "warnings": warnings})
        if errors:
            report["ok"] = False
        payloads.append({
            "episode_id": episode_id,
            "work_order_id": episode.work_order,
            "started_at": episode.started_at,
            "initial_context": episode.initial_context,
            "initial_workflow": {"workflow_id": f"{episode_id}-WF", "version_id": f"{episode_id}-V0",
                                 "nodes": [node.model_dump() for node in episode.initial_workflow.nodes],
                                 "edges": [edge.model_dump() for edge in episode.initial_workflow.edges]},
            "events": events,
        })
    return payloads, report
