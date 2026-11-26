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
    assert set(result["actual"]["ranks"]) == {"assist", "assist_result", "bigram", "frequency"}


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
    assert set(report["methods"]) == {"assist", "assist_result", "bigram", "frequency"}
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


def test_previous_result_option_is_off_by_default_and_noop_without_results(client):
    run = setup(client)  # 示意資料沒有執行結果
    body = {"episode_id": run["dataset"]["test_episodes"][0], "prefix_length": 1}
    plain = client.post("/api/v1/cases/AP-K-ERP/aov/assist", json=body).json()
    option = client.post("/api/v1/cases/AP-K-ERP/aov/assist", json={**body, "use_previous_result": True}).json()
    assert plain["use_previous_result"] is False and option["use_previous_result"] is True
    assert option["previous_result"] is None
    assert [c["score"] for c in plain["candidates"]] == [c["score"] for c in option["candidates"]]


def test_previous_result_changes_ranking_when_results_exist(tmp_path, service):
    import json

    from kcore.adapters.cvflow import ingest_teacher
    from kcore.assist import AssistService
    from kcore.encoding import FeatureSchema
    from kcore.hmm.training import TrainingConfig

    def graph(kinds, conf):
        nodes = [{"id": f"n{i}", "type": k, "params": ({"conf_threshold": conf} if k == "cv.dnn_detect" else {})}
                 for i, k in enumerate(kinds)]
        edges = [{"id": f"e{i}", "source": f"n{i}", "source_port": "o", "target": f"n{i + 1}", "target_port": "i"}
                 for i in range(len(kinds) - 1)]
        return {"nodes": nodes, "edges": edges}

    base = ["io.load_image", "cv.dnn_detect", "flow.count"]
    for k in range(14):  # 無法執行之後總是調參；未通過之後總是加入前處理
        task = tmp_path / "teachers" / "codex" / "tasks" / f"task{k:02d}"
        task.mkdir(parents=True)
        graphs = [graph(base, 0.5), graph(base, 0.4),
                  graph(["io.load_image", "cv.clahe", "cv.dnn_detect", "flow.count"], 0.4)]
        outcomes = [(False, False), (True, False), (True, True)]  # g1 無法執行、g2 未通過、g3 通過
        tries = []
        for i, (g, (executable, passed)) in enumerate(zip(graphs, outcomes, strict=True), start=1):
            (task / f"g{i}.json").write_text(json.dumps(g), encoding="utf-8")
            (task / f"try{i}.json").write_text(json.dumps(
                {"try_index": i, "round": i, "graph": g, "ts": f"2026-10-{k + 1:02d}T05:0{i}:00+00:00"}), encoding="utf-8")
            tries.append({"try_index": i, "round": i, "executable": executable, "v1i": passed, "v1s": passed})
        (task / "score_sealed4.json").write_text(json.dumps({"tries": tries, "final_ok": True}), encoding="utf-8")
    service.create_case({"case_id": "CV", "domain_kind": "cvflow"})
    ingest_teacher(service, tmp_path, "codex", "CV")
    run = service.train("CV", FeatureSchema(), TrainingConfig(k_min=1, k_max=2, restarts=1, n_iter=30))
    assist = AssistService(service)
    episode_id = run["dataset"]["test_episodes"][0]
    first = assist.assist("CV", episode_id, 0, use_previous_result=True)
    assert first["previous_result"] == "error" and first["candidates"][0]["operation_type"] == "param_tune"
    second = assist.assist("CV", episode_id, 1, use_previous_result=True)
    assert second["previous_result"] == "fail" and second["candidates"][0]["operation_type"] == "add_step"
    report = assist.evaluate("CV")
    assert set(report["by_previous_result"]) == {"error", "fail"}
    assert report["methods"]["assist_result"]["hit1"] >= report["methods"]["assist"]["hit1"]
