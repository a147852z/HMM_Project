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


def make_package(root, metric="count_only", task_name="apples_demo"):
    task = root / "teachers" / "codex" / "tasks" / task_name
    task.mkdir(parents=True)
    for name, g in (("g1", G1), ("g2", G2), ("g3", G3)):
        (task / f"{name}.json").write_text(json.dumps(g), encoding="utf-8")
    (task / "card.json").write_text(json.dumps({"nl_request": "數蘋果", "metric": metric, "category": "common"}),
                                    encoding="utf-8")
    tries = []
    for i, (g, ok) in enumerate(((G1, False), (G2, False), (G3, True)), start=1):
        (task / f"try{i}.json").write_text(json.dumps({"try_index": i, "round": i, "graph": g, "count": 3 + i,
                                                       "ts": f"2026-10-11T05:0{i}:00+00:00"}), encoding="utf-8")
        # 正解 4 框：通過的 try 配到 4 框（R = 1.0），多報 1 框（嚴格的 P = 0.8 但 v1s 不成立，被忽略區吸收後 v1i 成立）
        tries.append({"try_index": i, "round": i, "executable": True, "v0": ok, "v1i": ok, "v1c": ok, "v1s": False,
                      "P": 0.8 if ok else 0.5, "R": 1.0 if ok else 0.5, "K": 4 if ok else 2, "n_pred": 5 if ok else 4})
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


def test_pass_rule_is_v1i_and_ground_truth_is_k_over_r(tmp_path):
    task = make_package(tmp_path, metric="detect_only")
    episode = build_episode(task, "codex", "tasks", AdapterStats())
    result = episode["events"][1]["attributes"]["result"]
    # v1i 成立、v1s 不成立：依 CVflow 的建議算通過
    assert result["passed"] is True and result["judge"] == {"v0": True, "v1i": True, "v1c": True, "v1s": False}
    assert result["count"] == 6 and result["n_pred"] == 5  # 解題當下的框數與重播的框數分開保存
    assert result["precision"] == 0.8 and result["recall"] == 1.0 and result["matched"] == 4
    assert episode["provenance"]["first_version_result"]["passed"] is False
    # K 是配對數，不是正解數；正解框數 = K ÷ R
    assert episode["provenance"]["ground_truth"] == {"gt_boxes": 4, "gt_boxes_source": "K÷R", "gt_count": None}
    assert episode["provenance"]["outcome"]["pass_rule"] == "v1i"


def test_count_only_scores_and_unscored_try(tmp_path):
    task = make_package(tmp_path)  # count_only：評分檔的 P、R、K 是補的 0
    score_file = task / "score_sealed4.json"
    score = json.loads(score_file.read_text(encoding="utf-8"))
    score["tries"] = [t for t in score["tries"] if t["try_index"] != 1]  # 第一版沒有評分紀錄
    score_file.write_text(json.dumps(score), encoding="utf-8")
    episode = build_episode(task, "codex", "tasks", AdapterStats())
    assert episode["provenance"]["first_version_result"]["passed"] is None  # 沒有評分不是失敗
    result = episode["events"][1]["attributes"]["result"]
    assert result["precision"] is None and result["matched"] is None
    assert episode["provenance"]["ground_truth"]["gt_boxes"] is None


def test_status_flags_follow_rulings(tmp_path):
    from kcore.adapters.cvflow import status_flags

    teachers = tmp_path / "teachers" / "gemini" / "tasks"
    sequoia = teachers / "dendro-smithsonian-petrified-sequoia-rings"
    sequoia.mkdir(parents=True)
    (sequoia / "STATUS_超出範圍.md").write_text("過時的標記", encoding="utf-8")
    (sequoia / "STATUS_作廢.md").write_text("作廢", encoding="utf-8")
    assert status_flags(sequoia) == ["作廢"]
    (teachers / "coco_truck_123131").mkdir()
    assert status_flags(teachers / "coco_truck_123131") == ["超出範圍"]
    (teachers / "coco_car_111086").mkdir()
    assert status_flags(teachers / "coco_car_111086") == ["正解事後更正9"]


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
