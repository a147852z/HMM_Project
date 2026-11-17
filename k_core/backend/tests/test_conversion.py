import json

import httpx
import pytest
from fastapi.testclient import TestClient

from kcore.aov import derive_operation_type, diff_workflows
from kcore.api import create_app
from kcore.conversion.llm_client import OpenAICompatibleClient, extract_json

RAW_DOCUMENT = """製令 MO-778 流程調整紀錄
初版：備料 -> 加工 -> 包裝
10:03 王工程師：客戶要求驗收，包裝前要加檢驗站。
10:07 檢驗改成全檢。
"""

NODES_V0 = [{"node_id": "a", "activity_type": "prepare", "name": "備料"},
            {"node_id": "b", "activity_type": "process", "name": "加工"},
            {"node_id": "c", "activity_type": "pack", "name": "包裝"}]
EDGES_V0 = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
NODES_V1 = [*NODES_V0, {"node_id": "d", "activity_type": "inspect", "name": "檢驗"}]
EDGES_V1 = [{"source": "a", "target": "b"}, {"source": "b", "target": "d"}, {"source": "d", "target": "c"}]


def good_draft(reason_quote: str = "客戶要求驗收，包裝前要加檢驗站。") -> dict:
    """LLM 的典型輸出：第一次修改給完整版本（由 K 計算差異），第二次直接給差異。"""
    return {
        "episodes": [{
            "work_order": "MO-778",
            "started_at": "2026-11-12T10:00:00+08:00",
            "initial_context": {},
            "initial_workflow": {"nodes": NODES_V0, "edges": EDGES_V0},
            "events": [
                {"after_workflow": {"nodes": NODES_V1, "edges": EDGES_V1},
                 "reason": "客戶要求驗收", "reason_source": "human", "reason_quote": reason_quote,
                 "source_excerpt": "10:03 王工程師：客戶要求驗收"},
                {"operation_type": "update_parameters",
                 "change": {"updated_node_attributes": [{"node_id": "d", "parameters_set": {"mode": "full"}}]},
                 "reason": "提高品質要求", "reason_source": "llm", "reason_quote": None},
            ],
        }],
        "notes": None,
    }


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def list_models(self) -> list[str]:
        return ["local-model"]

    def complete(self, system: str, user: str, json_schema):
        self.calls.append({"system": system, "user": user, "schema": json_schema})
        return self.response


def make_client(tmp_path, fake: FakeLLM) -> TestClient:
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'conv.db'}", llm_client_factory=lambda profile: fake))
    client.post("/api/v1/cases", json={"case_id": "AP-K-ERP", "domain_kind": "erp"})
    client.post("/api/v1/cases/AP-K-ERP/llm-profiles", json={
        "profile_id": "apk-default", "name": "AP-K 流程紀錄", "model": "local-model",
        "system_prompt": "你是 AP-K 製造流程分析助理。活動類型只能用 prepare/process/inspect/pack。",
    })
    return client


def upload(client, content: bytes = RAW_DOCUMENT.encode("utf-8"), name: str = "mo778.txt") -> dict:
    return client.post("/api/v1/cases/AP-K-ERP/documents", files={"file": (name, content, "text/plain")}).json()


def convert(client, document_id: str) -> dict:
    created = client.post("/api/v1/cases/AP-K-ERP/conversions",
                          json={"document_id": document_id, "profile_id": "apk-default"})
    assert created.status_code == 202
    return client.get(f"/api/v1/cases/AP-K-ERP/conversions/{created.json()['conversion_id']}").json()


def test_full_flow_from_raw_text_to_training_data(tmp_path):
    fake = FakeLLM(json.dumps(good_draft(), ensure_ascii=False))
    client = make_client(tmp_path, fake)
    document = upload(client)
    conversion = convert(client, document["document_id"])

    assert conversion["status"] == "needs_review" and conversion["can_approve"], conversion["validation"]
    # 案例自訂提示詞＋K 固定格式說明；原始內容放入使用者訊息；要求結構化輸出
    call = fake.calls[0]
    assert call["system"].startswith("你是 AP-K") and "【輸出格式" in call["system"]
    assert "王工程師" in call["user"] and call["schema"]["title"] == "Draft"

    approved = client.post(f"/api/v1/cases/AP-K-ERP/conversions/{conversion['conversion_id']}/approve",
                           json={"reviewer": "林研究員"}).json()
    assert approved["status"] == "approved"
    episode_id = approved["ingested_episode_ids"][0]
    episode = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{episode_id}").json()
    assert episode["status"] == "finalized"
    assert episode["provenance"]["method"] == "llm_conversion"
    assert episode["provenance"]["reviewer"] == "林研究員"
    assert episode["provenance"]["profile_version"] == 1
    first, second = episode["events"]
    assert first["operation_type"] == "insert_activity"  # 由 K 依差異推定
    assert first["change"]["removed_edges"] == [{"source": "b", "target": "c"}]
    assert first["reason_source"] == "human" and second["reason_source"] == "llm"
    final = {n["node_id"]: n for n in episode["snapshots"][-1]["nodes"]}
    assert final["d"]["parameters"] == {"mode": "full"}


def test_reconverting_same_document_cannot_double_count(tmp_path):
    client = make_client(tmp_path, FakeLLM(json.dumps(good_draft(), ensure_ascii=False)))
    document = upload(client)
    first = convert(client, document["document_id"])
    approved = client.post(f"/api/v1/cases/AP-K-ERP/conversions/{first['conversion_id']}/approve",
                           json={"reviewer": "r"}).json()
    assert upload(client)["duplicate"] is True
    second = convert(client, document["document_id"])
    assert second["can_approve"] is False
    # 同一份文件重新轉換會得到相同的序列識別，因此被辨識為已匯入
    assert second["validation"]["already_ingested"] == approved["ingested_episode_ids"]
    response = client.post(f"/api/v1/cases/AP-K-ERP/conversions/{second['conversion_id']}/approve",
                           json={"reviewer": "r"})
    assert response.status_code == 422


def test_human_reason_without_verbatim_quote_is_downgraded(tmp_path):
    draft = good_draft(reason_quote="這句話原文裡沒有")
    client = make_client(tmp_path, FakeLLM(json.dumps(draft, ensure_ascii=False)))
    conversion = convert(client, upload(client)["document_id"])
    warnings = conversion["validation"]["episodes"][0]["warnings"]
    assert any("downgraded to llm" in warning for warning in warnings)
    client.post(f"/api/v1/cases/AP-K-ERP/conversions/{conversion['conversion_id']}/approve", json={"reviewer": "r"})
    episode_id = client.get("/api/v1/cases/AP-K-ERP/episodes").json()[0]["episode_id"]
    events = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{episode_id}").json()["events"]
    assert events[0]["reason_source"] == "llm"


def test_invalid_draft_blocks_approval_until_reviewer_fixes_it(tmp_path):
    draft = good_draft()
    draft["episodes"][0]["events"][1]["change"] = {"added_edges": [{"source": "c", "target": "a"}]}
    client = make_client(tmp_path, FakeLLM(json.dumps(draft, ensure_ascii=False)))
    conversion = convert(client, upload(client)["document_id"])
    url = f"/api/v1/cases/AP-K-ERP/conversions/{conversion['conversion_id']}"
    assert conversion["can_approve"] is False
    assert "cycle" in " ".join(conversion["validation"]["episodes"][0]["errors"])
    assert client.post(f"{url}/approve", json={"reviewer": "r"}).status_code == 422

    fixed = client.put(f"{url}/draft", json={"draft": good_draft()}).json()
    assert fixed["can_approve"] and fixed["draft_edited"]
    approved = client.post(f"{url}/approve", json={"reviewer": "r"}).json()
    episode = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{approved['ingested_episode_ids'][0]}").json()
    assert episode["provenance"]["draft_edited"] is True


def test_non_json_response_fails_and_keeps_raw_text(tmp_path):
    client = make_client(tmp_path, FakeLLM("抱歉，我無法處理這份資料。"))
    conversion = convert(client, upload(client)["document_id"])
    assert conversion["status"] == "failed"
    assert conversion["raw_response"] == "抱歉，我無法處理這份資料。"


def test_schema_error_is_reported(tmp_path):
    client = make_client(tmp_path, FakeLLM('{"episodes": [{"events": []}]}'))
    conversion = convert(client, upload(client)["document_id"])
    assert conversion["status"] == "needs_review" and not conversion["can_approve"]
    assert any("initial_workflow" in error for error in conversion["validation"]["schema_errors"])


def test_big5_document_is_decoded(tmp_path):
    client = make_client(tmp_path, FakeLLM("{}"))
    document = upload(client, RAW_DOCUMENT.encode("cp950"), "big5.txt")
    assert document["encoding"] == "cp950"
    content = client.get(f"/api/v1/cases/AP-K-ERP/documents/{document['document_id']}").json()["content"]
    assert "王工程師" in content


def test_binary_document_is_rejected(tmp_path):
    client = make_client(tmp_path, FakeLLM("{}"))
    response = client.post("/api/v1/cases/AP-K-ERP/documents",
                           files={"file": ("x.bin", bytes([0x81, 0xFF, 0x00, 0xFE]) * 5, "application/octet-stream")})
    assert response.status_code == 415


def test_profile_updates_create_versions_and_conversions_pin_the_version(tmp_path):
    client = make_client(tmp_path, FakeLLM(json.dumps(good_draft(), ensure_ascii=False)))
    first = convert(client, upload(client)["document_id"])
    updated = client.put("/api/v1/cases/AP-K-ERP/llm-profiles/apk-default", json={"system_prompt": "新版提示詞"}).json()
    assert updated["version"] == 2 and updated["model"] == "local-model"
    assert first["profile_version"] == 1
    old = client.get("/api/v1/cases/AP-K-ERP/llm-profiles/apk-default", params={"version": 1}).json()
    assert old["system_prompt"].startswith("你是 AP-K")


def test_profile_test_reports_model_availability(tmp_path):
    client = make_client(tmp_path, FakeLLM("{}"))
    result = client.post("/api/v1/cases/AP-K-ERP/llm-profiles/apk-default/test").json()
    assert result == {"reachable": True, "models": ["local-model"], "model_available": True}


def test_unreachable_endpoint_is_reported_not_raised():
    client = OpenAICompatibleClient("http://127.0.0.1:9/v1", "m")
    from kcore.conversion.llm_client import LLMError
    with pytest.raises(LLMError):
        client.list_models()


def test_openai_compatible_request_shape(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, body=json)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    OpenAICompatibleClient("http://localhost:1234/v1/", "qwen", temperature=0.0).complete("s", "u", {"type": "object"})
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["body"]["model"] == "qwen"
    assert captured["body"]["response_format"]["type"] == "json_schema"


def test_extract_json_handles_reasoning_and_fences():
    text = '<think>分析中 {不是答案}</think>\n```json\n{"episodes": []}\n```'
    assert extract_json(text) == {"episodes": []}
    assert extract_json('結果如下：{"a": 1} 謝謝') == {"a": 1}


def test_diff_workflows_and_operation_type():
    change = diff_workflows(NODES_V0, EDGES_V0, NODES_V1, EDGES_V1)
    assert [n["node_id"] for n in change["added_nodes"]] == ["d"]
    assert change["removed_edges"] == [{"source": "b", "target": "c"}]
    assert derive_operation_type(change) == "insert_activity"
    renamed = [dict(n, name="精加工") if n["node_id"] == "b" else n for n in NODES_V0]
    assert derive_operation_type(diff_workflows(NODES_V0, EDGES_V0, renamed, EDGES_V0)) == "update_activity"


def test_conversion_detail_includes_step_by_step_preview(tmp_path):
    client = make_client(tmp_path, FakeLLM(json.dumps(good_draft(), ensure_ascii=False)))
    conversion = convert(client, upload(client)["document_id"])
    preview = conversion["preview"][0]
    assert preview["stopped_at"] is None
    assert [len(v["nodes"]) for v in preview["versions"]] == [3, 4, 4]
    assert preview["events"][0]["operation_type"] == "insert_activity"
