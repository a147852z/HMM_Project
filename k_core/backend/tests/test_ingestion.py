import copy

from .conftest import spec_episode


def _setup(client, case_id="AP-K-ERP"):
    assert client.post("/api/v1/cases", json={"case_id": case_id, "domain_kind": "erp"}).status_code == 201


def _create(client, payload, case_id="AP-K-ERP"):
    body = {key: value for key, value in payload.items() if key != "events"}
    response = client.post(f"/api/v1/cases/{case_id}/episodes", json=body)
    assert response.status_code == 201, response.text


def test_spec_episode_round_trip_rebuilds_final_workflow(client):
    _setup(client)
    payload = spec_episode()
    _create(client, payload)
    result = client.post("/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001/events", json={"events": payload["events"]}).json()
    assert result == {"accepted": ["DEMO-EVT-001", "DEMO-EVT-002"], "duplicates": [], "rejected": []}

    episode = client.post("/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001/finalize").json()
    assert episode["status"] == "finalized"
    assert episode["final_version_id"] == "DEMO-WF-V3"
    assert [s["version_id"] for s in episode["snapshots"]] == ["DEMO-WF-V1", "DEMO-WF-V2", "DEMO-WF-V3"]
    final = {n["node_id"]: n for n in episode["snapshots"][-1]["nodes"]}
    assert final["n4"]["parameters"] == {"inspection_mode": "full"}
    assert [e["features"]["activity_type"] for e in episode["events"]] == ["inspect", "inspect"]
    # 原始紀錄（含原因與來源）完整保留
    assert episode["events"][0]["reason_source"] == "human"


def test_duplicate_events_are_not_counted_twice(client):
    _setup(client)
    payload = spec_episode()
    _create(client, payload)
    url = "/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001/events"
    client.post(url, json={"events": payload["events"]})
    again = client.post(url, json={"events": payload["events"]}).json()
    assert again["accepted"] == [] and again["duplicates"] == ["DEMO-EVT-001", "DEMO-EVT-002"]

    tampered = copy.deepcopy(payload["events"][0])
    tampered["operation_type"] = "remove_activity"
    conflict = client.post(url, json={"events": [tampered]}).json()
    assert conflict["rejected"][0]["problems"] == ["event_id already exists with different content"]
    assert client.get("/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001").json()["event_count"] == 2


def test_events_may_arrive_out_of_order(client):
    _setup(client)
    payload = spec_episode()
    _create(client, payload)
    url = "/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001"
    client.post(f"{url}/events", json={"events": list(reversed(payload["events"]))})
    assert client.post(f"{url}/finalize").json()["final_version_id"] == "DEMO-WF-V3"


def test_missing_sequence_index_blocks_finalize(client):
    _setup(client)
    payload = spec_episode()
    _create(client, payload)
    url = "/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001"
    client.post(f"{url}/events", json={"events": [payload["events"][1]]})
    response = client.post(f"{url}/finalize")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "finalize_failed"
    assert "missing sequence_index 1" in response.json()["error"]["details"][0]["problems"][0]


def test_change_introducing_cycle_blocks_finalize(client):
    _setup(client)
    payload = spec_episode()
    payload["events"][1]["change"] = {"added_edges": [{"source": "n3", "target": "n1"}]}
    _create(client, payload)
    url = "/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001"
    client.post(f"{url}/events", json={"events": payload["events"]})
    response = client.post(f"{url}/finalize")
    assert response.status_code == 422
    assert "cycle" in str(response.json()["error"]["details"])


def test_reason_rules(client):
    _setup(client)
    payload = spec_episode()
    _create(client, payload)
    url = "/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001/events"
    no_source = copy.deepcopy(payload["events"][0])
    no_source["reason_source"] = None
    llm = copy.deepcopy(payload["events"][1])
    llm["reason_source"] = "llm"
    missing = copy.deepcopy(payload["events"][1])
    missing.update(event_id="X-3", sequence_index=3, reason=None, reason_source=None,
                   before_version_id="DEMO-WF-V3", after_version_id="DEMO-WF-V4")
    result = client.post(url, json={"events": [no_source, llm, missing]}).json()
    assert result["rejected"][0]["event_id"] == "DEMO-EVT-001"
    assert set(result["accepted"]) == {"DEMO-EVT-002", "X-3"}  # 原因可缺值；可標示由 LLM 產生


def test_invalid_initial_workflow_is_rejected(client):
    _setup(client)
    payload = spec_episode()
    payload["initial_workflow"]["edges"].append({"source": "n3", "target": "n1"})
    body = {key: value for key, value in payload.items() if key != "events"}
    response = client.post("/api/v1/cases/AP-K-ERP/episodes", json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_workflow"


def test_cases_are_isolated_and_path_must_match(client):
    _setup(client, "AP-K-ERP")
    _setup(client, "AP-D-ERP")
    payload = spec_episode()
    _create(client, payload, "AP-K-ERP")
    assert client.get("/api/v1/cases/AP-D-ERP/episodes").json() == []
    wrong = client.get("/api/v1/cases/AP-D-ERP/episodes/DEMO-EP-001")
    assert wrong.status_code == 409 and wrong.json()["error"]["code"] == "case_mismatch"
    body = {key: value for key, value in spec_episode("EP-2").items() if key != "events"}
    body["case_id"] = "AP-K-ERP"
    assert client.post("/api/v1/cases/AP-D-ERP/episodes", json=body).status_code == 409


def test_finalized_episode_rejects_new_events(client):
    _setup(client)
    payload = spec_episode()
    _create(client, payload)
    url = "/api/v1/cases/AP-K-ERP/episodes/DEMO-EP-001"
    client.post(f"{url}/events", json={"events": payload["events"]})
    client.post(f"{url}/finalize")
    response = client.post(f"{url}/events", json={"events": payload["events"]})
    assert response.status_code == 409
