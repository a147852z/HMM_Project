"""示意資料產生器：產生帶有「已知潛在階段」的 AoV 演變序列，用來驗證資料管線與 HMM 能否找回結構。

所有產生的序列都標記 is_synthetic=True、actor_type="synthetic"。
潛在階段只寫在 attributes["synthetic_latent_phase"]，特徵編碼不會讀取它。
示意資料的訓練結果不代表任何真實工廠成效。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .aov import apply_change

LATENT_PHASE_KEY = "synthetic_latent_phase"

ACTIVITY_NAMES = {
    "prepare": "備料", "cut": "裁切", "process": "加工", "assemble": "組裝", "inspect": "檢驗", "pack": "包裝",
}

# 潛在階段：每個階段偏好不同的（修改類型, 活動類型）組合。
PHASES: dict[str, list[tuple[str, str, float]]] = {
    "expand": [
        ("insert_activity", "process", 0.35), ("insert_activity", "assemble", 0.25),
        ("insert_activity", "cut", 0.20), ("add_dependency", "process", 0.20),
    ],
    "quality": [
        ("insert_activity", "inspect", 0.45), ("update_parameters", "inspect", 0.45),
        ("update_parameters", "process", 0.10),
    ],
    "consolidate": [
        ("remove_activity", "process", 0.30), ("remove_activity", "cut", 0.15),
        ("remove_dependency", "process", 0.20), ("update_parameters", "pack", 0.35),
    ],
}
INITIAL_PHASE = {"expand": 0.6, "quality": 0.3, "consolidate": 0.1}
PHASE_TRANSITION = {
    "expand": {"expand": 0.6, "quality": 0.3, "consolidate": 0.1},
    "quality": {"expand": 0.1, "quality": 0.6, "consolidate": 0.3},
    "consolidate": {"expand": 0.1, "quality": 0.2, "consolidate": 0.7},
}


def _choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


class _Builder:
    def __init__(self, rng: random.Random, episode_tag: str) -> None:
        self.rng = rng
        self.tag = episode_tag
        self.counter = 0

    def new_node(self, activity_type: str) -> dict[str, Any]:
        self.counter += 1
        return {"node_id": f"{self.tag}-n{self.counter}", "activity_type": activity_type,
                "name": ACTIVITY_NAMES[activity_type], "parameters": {}}


def _ancestors(edges: list[dict[str, Any]], node_id: str) -> set[str]:
    parents: dict[str, list[str]] = {}
    for edge in edges:
        parents.setdefault(edge["target"], []).append(edge["source"])
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        for parent in parents.get(stack.pop(), []):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def _make_change(op: str, activity: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                 builder: _Builder) -> dict[str, Any] | None:
    """依修改類型產生一個保持 DAG 合法的差異；條件不足時回傳 None。"""
    rng = builder.rng
    empty = {"added_nodes": [], "removed_nodes": [], "added_edges": [], "removed_edges": [],
             "updated_node_attributes": []}
    typed = [node for node in nodes if node["activity_type"] == activity]

    if op == "insert_activity":
        if not edges:
            return None
        edge = rng.choice(edges)
        node = builder.new_node(activity)
        return {**empty, "added_nodes": [node],
                "added_edges": [{"source": edge["source"], "target": node["node_id"]},
                                {"source": node["node_id"], "target": edge["target"]}],
                "removed_edges": [{"source": edge["source"], "target": edge["target"]}]}
    if op == "remove_activity":
        if not typed or len(nodes) <= 3:
            return None
        node_id = rng.choice(typed)["node_id"]
        incoming = [edge for edge in edges if edge["target"] == node_id]
        outgoing = [edge for edge in edges if edge["source"] == node_id]
        existing = {(edge["source"], edge["target"]) for edge in edges}
        bridges = [{"source": a["source"], "target": b["target"]} for a in incoming for b in outgoing
                   if (a["source"], b["target"]) not in existing]
        return {**empty, "removed_nodes": [node_id], "removed_edges": incoming + outgoing, "added_edges": bridges}
    if op == "update_parameters":
        if not typed:
            return None
        node = rng.choice(typed)
        key = {"inspect": "inspection_mode", "pack": "package_type", "process": "machine"}.get(activity, "setting")
        value = rng.choice({"inspection_mode": ["sampling", "full"], "package_type": ["box", "pallet"],
                            "machine": ["M1", "M2", "M3"]}.get(key, ["A", "B"]))
        return {**empty, "updated_node_attributes": [{"node_id": node["node_id"], "parameters_set": {key: value}}]}
    if op == "add_dependency":
        sources = typed or nodes
        existing = {(edge["source"], edge["target"]) for edge in edges}
        options = [(a["node_id"], b["node_id"]) for a in sources for b in nodes
                   if a["node_id"] != b["node_id"] and (a["node_id"], b["node_id"]) not in existing
                   and b["node_id"] not in _ancestors(edges, a["node_id"])
                   and a["node_id"] not in _ancestors(edges, b["node_id"])]
        if not options:
            return None
        source, target = rng.choice(options)
        return {**empty, "added_edges": [{"source": source, "target": target}]}
    if op == "remove_dependency":
        typed_ids = {node["node_id"] for node in typed}
        # 只移除「另有其他路徑仍可到達」的連線，避免流程被切斷。
        candidates = [edge for edge in edges if edge["source"] in typed_ids or edge["target"] in typed_ids]
        candidates = [edge for edge in candidates if edge["source"] in _ancestors(
            [other for other in edges if other is not edge], edge["target"])]
        if not candidates:
            return None
        edge = rng.choice(candidates)
        return {**empty, "removed_edges": [{"source": edge["source"], "target": edge["target"]}]}
    raise ValueError(f"unknown synthetic operation: {op}")


def generate_episode(rng: random.Random, case_id: str, index: int, start: datetime,
                     min_events: int = 3, max_events: int = 12, stop_probability: float = 0.15) -> dict[str, Any]:
    tag = f"SYN-{case_id}-{index:04d}"
    builder = _Builder(rng, tag)
    nodes = [builder.new_node(kind) for kind in ("prepare", "process", "pack")]
    edges = [{"source": nodes[0]["node_id"], "target": nodes[1]["node_id"]},
             {"source": nodes[1]["node_id"], "target": nodes[2]["node_id"]}]
    initial_workflow = {"workflow_id": f"{tag}-WF", "version_id": f"{tag}-V0", "nodes": nodes, "edges": edges}

    events: list[dict[str, Any]] = []
    phase = _choice(rng, INITIAL_PHASE)
    version = 0
    while len(events) < max_events:
        if len(events) >= min_events and rng.random() < stop_probability:
            break
        weights = {f"{op}:{activity}": weight for op, activity, weight in PHASES[phase]}
        change = None
        for _ in range(10):  # 條件不足時在同一階段內重抽
            op, activity = _choice(rng, weights).split(":")
            change = _make_change(op, activity, nodes, edges, builder)
            if change is not None:
                break
        if change is None:
            phase = _choice(rng, PHASE_TRANSITION[phase])
            continue
        nodes, edges = apply_change(nodes, edges, change)
        version += 1
        events.append({
            "event_id": f"{tag}-E{version:02d}",
            "sequence_index": version,
            "occurred_at": (start + timedelta(minutes=2 * version)).isoformat(),
            "actor_type": "synthetic",
            "operation_type": op,
            "before_version_id": f"{tag}-V{version - 1}",
            "after_version_id": f"{tag}-V{version}",
            "change": change,
            "reason": None,
            "reason_source": None,
            "attributes": {LATENT_PHASE_KEY: phase},
        })
        phase = _choice(rng, PHASE_TRANSITION[phase])

    return {
        "episode_id": tag,
        "work_order_id": f"{tag}-MO",
        "is_synthetic": True,
        "started_at": start.isoformat(),
        "initial_context": {"product_family": "synthetic"},
        "initial_workflow": initial_workflow,
        "events": events,
    }


def generate_episodes(case_id: str, count: int, seed: int, start_index: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    return [generate_episode(rng, case_id, index, base + timedelta(hours=index))
            for index in range(start_index, start_index + count)]
