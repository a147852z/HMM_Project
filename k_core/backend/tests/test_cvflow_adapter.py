import json

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_MAP, AdapterStats, build_episode, classify, ingest_teacher
from kcore.encoding import FeatureSchema, event_token


def graph(nodes, edges):
    return {"schema_version": 1, "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
            "edges": [{"id": f"e{k}", "source": s, "source_port": sp, "target": t, "target_port": tp}
                      for k, (s, sp, t, tp) in enumerate(edges)]}


G1 = graph([("load", "io.load_image", {}), ("mask", "cv.color_mask_hsv", {"s_low": 75}),
            ("count", "flow.count", {})],
           [("load", "image", "mask", "image"), ("mask", "mask", "count", "detections")])
# 第二版：mask 改名為 m2 並調參；load→m2 有兩條不同 port 的連線
G2 = graph([("load", "io.load_image", {}), ("m2", "cv.color_mask_hsv", {"s_low": 100}),
            ("count", "flow.count", {})],
           [("load", "image", "m2", "image"), ("load", "meta", "m2", "meta"), ("m2", "mask", "count", "detections")])
# 第三版：換成 DNN 策略
G3 = graph([("load", "io.load_image", {}), ("det", "cv.dnn_detect", {"conf_threshold": 0.3}),
            ("count", "flow.count", {})],
           [("load", "image", "det", "image"), ("det", "detections", "count", "detections")])


def make_package(root):
    task = root / "teachers" / "codex" / "tasks" / "apples_demo"
    task.mkdir(parents=True)
    for name, g in (("g1", G1), ("g2", G2), ("g3", G3)):
        (task / f"{name}.json").write_text(json.dumps(g), encoding="utf-8")
    (task / "card.json").write_text(json.dumps({"nl_request": "數蘋果", "metric": "count_only", "category": "common"}),
                                    encoding="utf-8")
    tries = []
    for i, (g, ok) in enumerate(((G1, False), (G2, False), (G3, True)), start=1):
        (task / f"try{i}.json").write_text(json.dumps({"try_index": i, "round": i, "graph": g, "ts": f"2026-10-11T05:0{i}:00+00:00"}),
                                           encoding="utf-8")
        tries.append({"try_index": i, "round": i, "executable": True, "v1s": ok})
    (task / "score_sealed4.json").write_text(json.dumps({"tries": tries, "final_ok": True, "first_ok_round": 3}),
                                             encoding="utf-8")
    steps = [{"round": r, "hypothesis": f"第{r}輪設計理由", "fix": f"第{r}輪後的修正計畫", "exec_report": "x"} for r in (1, 2, 3)]
    (task / "traj.json").write_text(json.dumps({"steps": steps, "stop_reason": "satisfied"}, ensure_ascii=False),
                                    encoding="utf-8")
    (root / "teachers" / "codex" / "tasks" / "single_version").mkdir()
    (root / "teachers" / "codex" / "tasks" / "single_version" / "g1.json").write_text(json.dumps(G1), encoding="utf-8")
    return task


def test_build_episode_canonicalizes_renames_and_merges_ports(tmp_path):
    task = make_package(tmp_path)
    stats = AdapterStats()
    episode = build_episode(task, "codex", "tasks", stats)
    first, second = episode["events"]
    # 改名的 m2 沿用舊代號 mask，所以第一次修改只是調參
    assert first["operation_type"] == "param_tune"
    assert first["change"]["updated_node_attributes"] == [{"node_id": "mask", "parameters_set": {"s_low": 100}}]
    assert stats.renamed_nodes == 1 and stats.merged_port_edges == 1
    assert second["operation_type"] == "strategy_switch"
    # 原因是該輪自己的設計理由；上一輪的計畫另存
    assert first["reason"] == "第2輪設計理由" and first["reason_source"] == "actor"
    assert first["attributes"]["planned_fix"] == "第1輪後的修正計畫"
    assert second["attributes"]["result"]["passed"] is True
    assert episode["provenance"]["outcome"]["final_ok"] is True
    assert first["actor_type"] == "ai_agent"


def test_ingest_teacher_is_idempotent_and_skips_single_versions(tmp_path, service):
    make_package(tmp_path)
    service.create_case({"case_id": "CVFLOW-CODEX", "domain_kind": "cvflow"})
    stats = ingest_teacher(service, tmp_path, "codex", "CVFLOW-CODEX")
    assert stats.episodes == 1 and stats.skipped_single_version == 1
    ingest_teacher(service, tmp_path, "codex", "CVFLOW-CODEX")  # 重複執行不會出錯也不會重複
    episodes = service.list_episodes("CVFLOW-CODEX")
    assert len(episodes) == 1 and episodes[0]["status"] == "finalized"
    detail = service.get_episode("CVFLOW-CODEX", episodes[0]["episode_id"])
    # 結果資料不進入衍生特徵
    for event in detail["events"]:
        assert "passed" not in json.dumps(event["features"])


def test_activity_map_coarsens_tokens():
    schema = FeatureSchema(activity_map=CVFLOW_ACTIVITY_MAP, activity_default="other")
    assert event_token("add_step", {"activity_type": "cv.gaussian_blur"}, schema) == "add_step|preprocess"
    assert event_token("add_step", {"activity_type": "cv.brand_new_node"}, schema) == "add_step|other"
    restored = FeatureSchema.from_dict(schema.to_dict())
    assert restored.map_activity("cv.dnn_detect") == "detect"


def test_classify_strategy_switch():
    before = [{"node_id": "a", "activity_type": "cv.color_mask_hsv"}]
    after = [{"node_id": "b", "activity_type": "cv.dnn_detect"}]
    change = {"added_nodes": after, "removed_nodes": ["a"], "added_edges": [], "removed_edges": [],
              "updated_node_attributes": []}
    assert classify(before, after, change) == "strategy_switch"
