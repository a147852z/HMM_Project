"""AoV（活動節點網路）結構：驗證與套用修改差異。

AoV 已確認為有向無環圖（DAG）：
- 節點代表活動／工序，必須有唯一的 node_id 與 activity_type。
- 連線 source → target 代表先後或依賴關係。
- 不允許自我迴圈、重複連線與任何循環。
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Any

Node = dict[str, Any]
Edge = dict[str, Any]

UPDATABLE_NODE_FIELDS = ("name", "activity_type")


class AoVError(ValueError):
    """修改無法套用或結果不是合法 AoV 時拋出，problems 列出所有具體原因。"""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _edge_key(edge: Edge) -> tuple[str, str]:
    return str(edge.get("source")), str(edge.get("target"))


def validate_workflow(nodes: list[Node], edges: list[Edge]) -> list[str]:
    """回傳所有結構問題；空清單代表是合法的 DAG。"""
    problems: list[str] = []
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        node_id = node.get("node_id")
        if not node_id:
            problems.append(f"node[{index}] is missing node_id")
            continue
        if node_id in node_ids:
            problems.append(f"duplicate node_id: {node_id}")
        node_ids.add(node_id)
        if not node.get("activity_type"):
            problems.append(f"node {node_id} is missing activity_type")

    seen_edges: set[tuple[str, str]] = set()
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source, target = _edge_key(edge)
        if source not in node_ids or target not in node_ids:
            problems.append(f"edge {source}->{target} references a missing node")
            continue
        if source == target:
            problems.append(f"self-loop on node {source}")
            continue
        if (source, target) in seen_edges:
            problems.append(f"duplicate edge {source}->{target}")
            continue
        seen_edges.add((source, target))
        adjacency[source].append(target)
        indegree[target] += 1

    # Kahn 拓撲排序：無法排完的節點代表存在循環。
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited != len(node_ids):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        problems.append(f"cycle detected among nodes: {', '.join(cyclic)}")
    return problems


def apply_change(nodes: list[Node], edges: list[Edge], change: dict[str, Any]) -> tuple[list[Node], list[Edge]]:
    """將一次修改差異套用到流程上，回傳新的 (nodes, edges)；不修改輸入。

    change 欄位：added_nodes、removed_nodes（node_id 清單）、added_edges、removed_edges、
    updated_node_attributes（[{node_id, parameters_set?, parameters_unset?, name?, activity_type?}]）。
    刪除節點時，相關連線必須在 removed_edges 中明確列出，否則視為懸空連線而拒絕。
    """
    problems: list[str] = []
    new_nodes = {node["node_id"]: copy.deepcopy(node) for node in nodes}
    new_edges = {_edge_key(edge): copy.deepcopy(edge) for edge in edges}

    for node_id in change.get("removed_nodes", []):
        node_id = node_id["node_id"] if isinstance(node_id, dict) else node_id
        if new_nodes.pop(node_id, None) is None:
            problems.append(f"cannot remove missing node {node_id}")
    for edge in change.get("removed_edges", []):
        if new_edges.pop(_edge_key(edge), None) is None:
            problems.append(f"cannot remove missing edge {_edge_key(edge)[0]}->{_edge_key(edge)[1]}")
    for node in change.get("added_nodes", []):
        node_id = node.get("node_id")
        if not node_id:
            problems.append("added node is missing node_id")
        elif node_id in new_nodes:
            problems.append(f"cannot add existing node {node_id}")
        else:
            new_nodes[node_id] = {"parameters": {}, **copy.deepcopy(node)}
    for edge in change.get("added_edges", []):
        key = _edge_key(edge)
        if key in new_edges:
            problems.append(f"cannot add existing edge {key[0]}->{key[1]}")
        else:
            new_edges[key] = copy.deepcopy(edge)
    for update in change.get("updated_node_attributes", []):
        node = new_nodes.get(update.get("node_id"))
        if node is None:
            problems.append(f"cannot update missing node {update.get('node_id')}")
            continue
        parameters = dict(node.get("parameters") or {})
        parameters.update(update.get("parameters_set") or {})
        for key in update.get("parameters_unset") or []:
            parameters.pop(key, None)
        node["parameters"] = parameters
        for field in UPDATABLE_NODE_FIELDS:
            if field in update:
                node[field] = update[field]

    if problems:
        raise AoVError(problems)
    result_nodes, result_edges = list(new_nodes.values()), list(new_edges.values())
    structural = validate_workflow(result_nodes, result_edges)
    if structural:
        raise AoVError(structural)
    return result_nodes, result_edges


def diff_workflows(before_nodes: list[Node], before_edges: list[Edge],
                   after_nodes: list[Node], after_edges: list[Edge]) -> dict[str, Any]:
    """由兩個完整版本精確計算修改差異（不需要 LLM 判斷），結果可直接交給 apply_change。

    節點以 node_id 對應，因此同一流程的節點識別必須在版本間保持一致。
    """
    before = {node["node_id"]: node for node in before_nodes}
    after = {node["node_id"]: node for node in after_nodes}
    before_keys = {_edge_key(edge) for edge in before_edges}
    after_keys = {_edge_key(edge) for edge in after_edges}
    updates: list[dict[str, Any]] = []
    for node_id in sorted(set(before) & set(after)):
        old, new = before[node_id], after[node_id]
        update: dict[str, Any] = {"node_id": node_id}
        old_params, new_params = old.get("parameters") or {}, new.get("parameters") or {}
        changed = {key: value for key, value in new_params.items() if old_params.get(key) != value or key not in old_params}
        removed = sorted(key for key in old_params if key not in new_params)
        if changed:
            update["parameters_set"] = changed
        if removed:
            update["parameters_unset"] = removed
        for field in UPDATABLE_NODE_FIELDS:
            if new.get(field) != old.get(field) and field in new:
                update[field] = new[field]
        if len(update) > 1:
            updates.append(update)
    return {
        "added_nodes": [copy.deepcopy(after[node_id]) for node_id in after if node_id not in before],
        "removed_nodes": [node_id for node_id in before if node_id not in after],
        "added_edges": [{"source": s, "target": t} for s, t in sorted(after_keys - before_keys)],
        "removed_edges": [{"source": s, "target": t} for s, t in sorted(before_keys - after_keys)],
        "updated_node_attributes": updates,
    }


def derive_operation_type(change: dict[str, Any]) -> str:
    """原始資料沒有明確修改類型時，依差異內容推定一個通用類型。"""
    added_n, removed_n = bool(change.get("added_nodes")), bool(change.get("removed_nodes"))
    added_e, removed_e = bool(change.get("added_edges")), bool(change.get("removed_edges"))
    updates = change.get("updated_node_attributes") or []
    if added_n and not removed_n and not updates:
        return "insert_activity"
    if removed_n and not added_n and not updates:
        return "remove_activity"
    if updates and not (added_n or removed_n or added_e or removed_e):
        renamed = any(field in update for update in updates for field in UPDATABLE_NODE_FIELDS)
        return "update_activity" if renamed else "update_parameters"
    if not (added_n or removed_n or updates):
        if added_e and removed_e:
            return "rewire_dependency"
        return "add_dependency" if added_e else "remove_dependency"
    return "composite"


def is_empty_change(change: dict[str, Any]) -> bool:
    keys = ("added_nodes", "removed_nodes", "added_edges", "removed_edges", "updated_node_attributes")
    return not any(change.get(key) for key in keys)


def derive_change_features(before_nodes: list[Node], after_nodes: list[Node], change: dict[str, Any]) -> dict[str, Any]:
    """由修改前後的流程計算可供編碼的結構特徵（只使用修改當下已知的資訊）。"""
    before = {node["node_id"]: node for node in before_nodes}
    after = {node["node_id"]: node for node in after_nodes}
    types: list[str] = []

    def add(node: Node | None) -> None:
        if node and node.get("activity_type") and node["activity_type"] not in types:
            types.append(str(node["activity_type"]))

    for node in change.get("added_nodes", []):
        add(node)
    for node_id in change.get("removed_nodes", []):
        add(before.get(node_id["node_id"] if isinstance(node_id, dict) else node_id))
    for update in change.get("updated_node_attributes", []):
        add(after.get(update.get("node_id")) or before.get(update.get("node_id")))
    if not types:
        # 只改連線時，以連線兩端的活動類型描述受影響範圍。
        for edge in [*change.get("added_edges", []), *change.get("removed_edges", [])]:
            source, target = _edge_key(edge)
            add(after.get(source) or before.get(source))
            add(after.get(target) or before.get(target))

    node_delta = len(change.get("added_nodes", [])) - len(change.get("removed_nodes", []))
    edge_delta = len(change.get("added_edges", [])) - len(change.get("removed_edges", []))
    sign = lambda value: "+" if value > 0 else "-" if value < 0 else "0"  # noqa: E731
    return {
        "activity_type": types[0] if types else "none",
        "activity_types": types,
        "node_delta": sign(node_delta),
        "edge_delta": sign(edge_delta),
        "node_count_after": len(after_nodes),
    }
