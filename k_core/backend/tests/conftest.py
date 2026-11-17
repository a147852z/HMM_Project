from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kcore.api import create_app
from kcore.database import make_session_factory
from kcore.service import KCoreService

# 規格書第 9 節的示意資料：備料→加工→包裝，插入檢驗，再改為全檢。
SPEC_EPISODE: dict[str, Any] = {
    "episode_id": "DEMO-EP-001",
    "work_order_id": "DEMO-MO-001",
    "is_synthetic": True,
    "started_at": "2026-11-11T10:00:00+08:00",
    "initial_context": {"product_family": "示例產品"},
    "initial_workflow": {
        "workflow_id": "DEMO-WF-001",
        "version_id": "DEMO-WF-V1",
        "nodes": [
            {"node_id": "n1", "activity_type": "prepare", "name": "備料", "parameters": {}},
            {"node_id": "n2", "activity_type": "process", "name": "加工", "parameters": {}},
            {"node_id": "n3", "activity_type": "pack", "name": "包裝", "parameters": {}},
        ],
        "edges": [{"source": "n1", "target": "n2"}, {"source": "n2", "target": "n3"}],
    },
    "events": [
        {
            "event_id": "DEMO-EVT-001", "sequence_index": 1, "occurred_at": "2026-11-11T10:03:00+08:00",
            "actor_type": "human", "operation_type": "insert_activity",
            "before_version_id": "DEMO-WF-V1", "after_version_id": "DEMO-WF-V2",
            "reason": "新增的客戶驗收要求需要在包裝前完成檢驗。", "reason_source": "human",
            "context_delta": {"inspection_required": True},
            "change": {
                "added_nodes": [{"node_id": "n4", "activity_type": "inspect", "name": "檢驗", "parameters": {}}],
                "removed_nodes": [],
                "added_edges": [{"source": "n2", "target": "n4"}, {"source": "n4", "target": "n3"}],
                "removed_edges": [{"source": "n2", "target": "n3"}],
                "updated_node_attributes": [],
            },
            "related_ui_event_ids": ["DEMO-UI-001", "DEMO-UI-002"],
        },
        {
            "event_id": "DEMO-EVT-002", "sequence_index": 2, "occurred_at": "2026-11-11T10:07:00+08:00",
            "actor_type": "human", "operation_type": "update_parameters",
            "before_version_id": "DEMO-WF-V2", "after_version_id": "DEMO-WF-V3",
            "reason": "進一步確認驗收要求為全檢，補上檢驗範圍。", "reason_source": "human",
            "context_delta": {"inspection_mode": "full"},
            "change": {
                "added_nodes": [], "removed_nodes": [], "added_edges": [], "removed_edges": [],
                "updated_node_attributes": [{"node_id": "n4", "parameters_set": {"inspection_mode": "full"}}],
            },
            "related_ui_event_ids": ["DEMO-UI-003"],
        },
    ],
}


def spec_episode(episode_id: str = "DEMO-EP-001", prefix: str = "") -> dict[str, Any]:
    """回傳一份可修改的範例序列；prefix 用來產生不衝突的事件與版本識別。"""
    data = copy.deepcopy(SPEC_EPISODE)
    data["episode_id"] = episode_id
    for event in data["events"]:
        event["event_id"] = prefix + event["event_id"]
    return data


@pytest.fixture()
def service() -> KCoreService:
    return KCoreService(make_session_factory("sqlite://"))


@pytest.fixture()
def client(tmp_path) -> TestClient:
    return TestClient(create_app(f"sqlite:///{tmp_path / 'kcore.db'}"))
