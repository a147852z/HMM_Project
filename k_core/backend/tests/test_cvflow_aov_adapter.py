import json

from kcore.adapters.cvflow import CVFLOW_ACTIVITY_MAP
from kcore.adapters.cvflow_aov import TrialIndex, aov_to_workflow, build_episode, ingest_batch, iter_task_dirs
from kcore.adapters.cvflow import AdapterStats
from kcore.assist import result_label


def node(node_id, kind, op, inputs, outputs, params=None):
    return {"id": node_id, "kind": kind, "F": op, "I": inputs, "O": outputs, "params": params or {}}


def aov(nodes):
    return {"schemaVersion": "aov-graph/1.0", "graph": {"id": "L0", "nodes": nodes}}


V1 = aov([node("input", "source", "io.load_image", [], ["input.image"]),
          node("mask", "activity", "cv.color_mask_hsv", ["input.image"], ["mask.mask"], {"s_low": 75}),
          node("count", "activity", "flow.count", ["mask.mask"], ["count.number", "count.text"]),
          node("end", "sink", "OutputSink", ["count.number", "count.text"], [])])
# 第二版：mask 改名為 m2 並調參，且多讀一份 input 的資料（同一對節點兩條資料線）
V2 = aov([node("input", "source", "io.load_image", [], ["input.image", "input.meta"]),
          node("m2", "activity", "cv.color_mask_hsv", ["input.image", "input.meta"], ["m2.mask"], {"s_low": 100}),
          node("count", "activity", "flow.count", ["m2.mask"], ["count.number"]),
          node("end", "sink", "OutputSink", ["count.number"], [])])
# 第三版轉換失敗（沒有 L0.json）；第四版換成特徵比對策略
V4 = aov([node("input", "source", "io.load_image", [], ["input.image"]),
          node("sift", "activity", "cv.sift_keypoints", ["input.image"], ["sift.keypoints"]),
          node("count", "activity", "flow.count", ["sift.keypoints"], ["count.number"]),
          node("end", "sink", "OutputSink", ["count.number"], [])])


def make_batch(root):
    task = root / "evolution" / "d171" / "codex" / "20261014T103303" / "d171_C01_demo"
    versions = []
    for n, graph in ((1, V1), (2, V2), (3, None), (4, V4)):
        if graph is not None:
            (task / f"aov_v{n}" / "graphs").mkdir(parents=True)
            (task / f"aov_v{n}" / "graphs" / "L0.json").write_text(json.dumps(graph), encoding="utf-8")
        versions.append({"version": n, "round": n, "try_index": n,
                         "score": {"executable": True, "v1s": n == 4},
                         "reason": {"kind": "fix" if n > 1 else "hypothesis", "fix": f"第{n - 1}輪後的計畫",
                                    "hypothesis": f"第{n}輪設計理由"}})
    (task / "EVOLUTION.json").write_text(json.dumps({"versions": versions}, ensure_ascii=False), encoding="utf-8")
    (root / "evolution" / "d171" / "ollama" / "r1" / "skipped_task").mkdir(parents=True)
    (root / "evolution" / "d171" / "ollama" / "r1" / "skipped_task" / "EVOLUTION.json").write_text('{"versions": []}')
    # try 2 是判分器（judge_v1i，數量有容差）判過、但數量不完全相等的情況
    error_types = {1: ["count_under"], 2: ["count_over"], 4: ["ok"]}
    tries = [{"source": "d171", "solver": "codex", "run": "20261014T103303", "task_id": "d171_C01_demo",
              "metric": "count_only", "try_index": n, "error_types": error_types[n], "judge_v1i": n != 1,
              "gt_count": 5, "pred_count": {1: 3, 2: 6, 4: 5}[n], "count_source": "count", "P": 0.0, "R": 0.0}
             for n in (1, 2, 4)]
    (root / "tries.jsonl").write_text("\n".join(json.dumps(t) for t in tries), encoding="utf-8")
    attempt = {"source": "d171", "solver": "codex", "run": "20261014T103303", "task_id": "d171_C01_demo",
               "ok": True, "stop_reason": "satisfied", "category": "C01", "metric": "count_only",
               "evolution": "evolution/d171/codex/20261014T103303/d171_C01_demo"}
    (root / "attempts.jsonl").write_text(json.dumps(attempt), encoding="utf-8")
    return task


def test_aov_to_workflow_drops_sink_and_merges_data_edges(tmp_path):
    task = make_batch(tmp_path)
    stats = AdapterStats()
    nodes, edges = aov_to_workflow(task / "aov_v2" / "graphs" / "L0.json", stats)
    assert [n["node_id"] for n in nodes] == ["input", "m2", "count"]
    assert edges == [{"source": "input", "target": "m2", "ports": ["input.image", "input.meta"]},
                     {"source": "m2", "target": "count", "ports": ["m2.mask"]}]
    assert stats.merged_port_edges == 1


def test_build_episode_skips_failed_versions_and_keeps_results(tmp_path):
    task = make_batch(tmp_path)
    stats = AdapterStats()
    episode = build_episode(task, "d171", "codex", "20261014T103303", TrialIndex(tmp_path), stats)
    first, second = episode["events"]
    assert first["operation_type"] == "param_tune"  # m2 沿用舊代號 mask
    assert second["operation_type"] == "strategy_switch"
    assert second["attributes"]["skipped_versions_before"] == 1
    assert first["reason"] == "第2輪設計理由" and first["attributes"]["planned_fix"] == "第1輪後的計畫"
    assert second["attributes"]["result"]["passed"] is True
    assert second["attributes"]["result"]["error_types"] == ["ok"]
    # 判分器判過（judge_v1i）但數量不完全相等：依 CVflow 的建議不算通過
    assert first["attributes"]["result"]["passed"] is False and first["attributes"]["result"]["judge_v1i"] is True
    assert first["attributes"]["result"]["precision"] is None  # count_only 題的 P、R 是補的 0
    assert episode["provenance"]["first_version_result"]["error_types"] == ["count_under"]
    assert episode["provenance"]["outcome"] == {"pass_rule": "error_types_ok", "final_ok": True, "any_ok": True,
                                                "package_ok": True, "stop_reason": "satisfied"}
    assert episode["provenance"]["ground_truth"] == {"gt_count": 5}
    assert episode["initial_context"]["category"] == "C01" and episode["initial_context"]["teacher"] == "codex"
    assert episode["started_at"].startswith("2026-10-14T10:33:03")
    assert CVFLOW_ACTIVITY_MAP["cv.sift_keypoints"] == "detect"


def test_excluded_solvers_and_ingest(tmp_path, service):
    make_batch(tmp_path)
    assert [solver for _, solver, _, _ in iter_task_dirs(tmp_path)] == ["codex"]
    service.create_case({"case_id": "CV2", "name": "cv", "domain_kind": "other", "recording_enabled": False})
    stats = ingest_batch(service, tmp_path, "CV2")
    assert stats.episodes == 1 and stats.events == 2
    again = ingest_batch(service, tmp_path, "CV2")  # 重複執行不會重複匯入
    assert again.episodes == 1


def test_photo_tasks_and_missing_error_types_are_not_judged():
    from kcore.adapters.cvflow_aov import _result, passes

    assert passes({"error_types": ["ok"]}) is True
    assert passes({"error_types": ["count_over", "boxes_extra"]}) is False
    assert passes({"error_types": ["unjudged"]}) is None and passes(None) is None
    # 照片題沒有正解：即使不能執行，passed 也是 None（能不能執行另記在 executable）
    photo = _result({"result": {"exec_error": "boom"}}, {"error_types": ["not_executable"]}, judged=False)
    assert photo["passed"] is None and photo["executable"] is False


def test_unjudged_result_is_not_a_failure():
    assert result_label({"executable": True, "passed": None}) is None
    assert result_label({"executable": True, "passed": False}) == "fail"
    assert result_label({"executable": False, "passed": None}) == "error"
