from kcore.aov import validate_workflow

FAST = {"k_min": 1, "k_max": 3, "restarts": 1, "n_iter": 50}


def setup(client):
    client.post("/api/v1/cases", json={"case_id": "AP-K-ERP", "domain_kind": "erp"})
    client.post("/api/v1/cases/AP-K-ERP/synthetic-episodes", json={"count": 40, "seed": 6})
    return client.post("/api/v1/cases/AP-K-ERP/training-runs", json=FAST).json()


def test_assist_proposes_valid_ranked_candidates_and_compares_with_actual(client):
    run = setup(client)
    episode_id = run["dataset"]["test_episodes"][0]
    result = client.post("/api/v1/cases/AP-K-ERP/aov/assist", json={"episode_id": episode_id, "prefix_length": 1}).json()
    assert result["status"] == "ok" and result["used_hmm"] and result["in_training_set"] is False
    assert result["model_version_id"] == run["model_version_id"]
    candidates = result["candidates"]
    assert candidates and [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True) and sum(scores) <= 1 + 1e-9
    for c in candidates:  # 每個候選都是合法的 DAG，分數 = 符號機率 × 樣式比例
        assert validate_workflow(c["proposed_workflow"]["nodes"], c["proposed_workflow"]["edges"]) == []
        assert abs(c["score"] - c["token_probability"] * c["pattern_prior"]) < 1e-12
    assert set(result["actual"]["ranks"]) == {"assist", "bigram", "frequency"}


def test_assist_does_not_depend_on_future_events(client):
    run = setup(client)
    episode_id = run["dataset"]["test_episodes"][0]
    total = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{episode_id}").json()["event_count"]
    body = {"episode_id": episode_id, "prefix_length": total}  # 序列結尾：沒有實際的下一步可比對
    result = client.post("/api/v1/cases/AP-K-ERP/aov/assist", json=body).json()
    assert result["actual"] is None and result["candidates"]


def test_training_episode_is_flagged(client):
    run = setup(client)
    episodes = [e["episode_id"] for e in client.get("/api/v1/cases/AP-K-ERP/episodes").json()]
    train_id = next(e for e in episodes if e not in run["dataset"]["test_episodes"])
    result = client.post("/api/v1/cases/AP-K-ERP/aov/assist", json={"episode_id": train_id, "prefix_length": 0}).json()
    assert result["in_training_set"] is True


def test_assist_evaluation_and_validate(client):
    run = setup(client)
    report = client.get("/api/v1/cases/AP-K-ERP/aov/assist-evaluation").json()
    assert report["test_episodes"] == len(run["dataset"]["test_episodes"]) and report["steps"] > 0
    assert set(report["methods"]) == {"assist", "bigram", "frequency"}
    for metrics in report["methods"].values():
        assert 0 <= metrics["hit1"] <= metrics["hit3"] <= metrics["hit5"] <= report["coverage"] + 1e-9
    assert report["invalid_candidates_dropped"] >= 0

    bad = {"nodes": [{"node_id": "a", "activity_type": "x"}, {"node_id": "b", "activity_type": "y"}],
           "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    checked = client.post("/api/v1/cases/AP-K-ERP/aov/validate", json=bad).json()
    assert checked["valid"] is False and "cycle" in checked["problems"][0]


def test_assist_without_model_reports_status(client):
    client.post("/api/v1/cases", json={"case_id": "EMPTY", "domain_kind": "erp"})
    response = client.post("/api/v1/cases/EMPTY/aov/assist", json={"episode_id": "x", "prefix_length": 0})
    assert response.status_code == 404 and response.json()["error"]["code"] == "no_active_model"
