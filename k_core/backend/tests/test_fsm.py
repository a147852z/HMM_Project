import pytest

from kcore.encoding import FeatureSchema
from kcore.fsm import build_state_machine, state_key, state_summary
from kcore.service import KCoreError

from .test_assist import setup


def node(node_id, kind, **params):
    return {"node_id": node_id, "activity_type": kind, "parameters": params}


def chain(*nodes):
    return list(nodes), [{"source": a["node_id"], "target": b["node_id"]} for a, b in zip(nodes, nodes[1:])]


LOAD = node("n1", "io.load_image")
COUNT = node("n9", "flow.count")
V_HSV = chain(LOAD, node("n2", "cv.color_mask_hsv", s_low=75), COUNT)
V_HSV_TUNED = chain(LOAD, node("n2", "cv.color_mask_hsv", s_low=100), COUNT)
V_HSV_RENAMED = chain(LOAD, node("mask", "cv.color_mask_hsv", s_low=75), COUNT)
V_DNN = chain(LOAD, node("n3", "cv.dnn_detect", conf_threshold=0.3), COUNT)
SCHEMA = FeatureSchema(version="t", fields=("operation_type", "activity_type"),
                       activity_map={"cv.color_mask_hsv": "segment", "cv.dnn_detect": "detect",
                                     "io.load_image": "io", "flow.count": "output"}, activity_default="other")


def test_state_key_granularities():
    # 只改參數：結構相同、完整流程圖不同
    assert state_key(*V_HSV, "structure") == state_key(*V_HSV_TUNED, "structure")
    assert state_key(*V_HSV, "full") != state_key(*V_HSV_TUNED, "full")
    # 節點改名不影響狀態
    assert state_key(*V_HSV, "full") == state_key(*V_HSV_RENAMED, "full")
    # 節點類別只看用到哪幾類
    assert state_key(*V_HSV, "category", SCHEMA) != state_key(*V_DNN, "category", SCHEMA)
    assert state_summary(*V_HSV, "category", SCHEMA) == ["io", "segment", "output"]
    assert state_summary(*V_DNN, "structure") == ["cv.dnn_detect"]  # 讀圖、計數這類通用節點省略


def versions(*graphs):
    return [(f"V{i}", nodes, edges) for i, (nodes, edges) in enumerate(graphs)]


EVENTS = [{"event_id": f"E{i}", "operation_type": op, "activity_type": t}
          for i, (op, t) in enumerate((("param_tune", "cv.color_mask_hsv"), ("strategy_switch", "cv.dnn_detect"),
                                       ("strategy_switch", "cv.color_mask_hsv")), start=1)]


def test_self_loops_revisits_and_probabilities():
    # 結構粒度：調參是自迴圈；換成 DNN 後又換回 HSV，回到第一個狀態
    machine = build_state_machine(versions(V_HSV, V_HSV_TUNED, V_DNN, V_HSV_RENAMED), EVENTS, "structure")
    assert machine["path"] == ["S1", "S1", "S2", "S1"]
    assert machine["stats"] == {"versions": 4, "states": 2, "transitions": 3, "self_loop_steps": 1, "revisits": 1}
    by_edge = {(t["source"], t["target"]): t for t in machine["transitions"]}
    assert by_edge[("S1", "S1")]["self_loop"] and by_edge[("S1", "S1")]["probability"] == 0.5
    assert by_edge[("S1", "S2")]["probability"] == 0.5 and by_edge[("S2", "S1")]["probability"] == 1.0
    s1 = machine["states"][0]
    assert s1["visits"] == [0, 1, 3] and s1["is_start"] and s1["is_final"]
    # 完整流程圖粒度：參數不同就是不同狀態，最後一版和第 0 版完全相同（只是改名）所以回到 S1
    full = build_state_machine(versions(V_HSV, V_HSV_TUNED, V_DNN, V_HSV_RENAMED), EVENTS, "full")
    assert full["path"] == ["S1", "S2", "S3", "S1"]


def test_model_steps_are_attached_to_transitions_and_states():
    steps = [{"hidden_state": h, "posterior": {}, "hidden_transition": {}, "model_probability": 0.1, "model_rank": 2}
             for h in ("H0", "H0", "H1")]
    machine = build_state_machine(versions(V_HSV, V_HSV_TUNED, V_DNN, V_HSV_RENAMED), EVENTS, "structure", None, steps)
    s1 = machine["states"][0]
    assert s1["hidden_states"] == {"H0": 1, "H1": 1}  # 第 0 版沒有修改，不計
    back = next(t for t in machine["transitions"] if t["source"] == "S2")
    assert back["hidden_states"] == {"H1": 1} and back["events"][0]["model_rank"] == 2


def test_invalid_granularity():
    with pytest.raises(KCoreError):
        build_state_machine(versions(V_HSV), [], "everything")


def test_state_machine_api_with_and_without_model(client):
    client.post("/api/v1/cases", json={"case_id": "NOMODEL", "domain_kind": "erp"})
    client.post("/api/v1/cases/NOMODEL/synthetic-episodes", json={"count": 1, "seed": 3})
    episode_id = client.get("/api/v1/cases/NOMODEL/episodes").json()[0]["episode_id"]
    plain = client.get(f"/api/v1/cases/NOMODEL/episodes/{episode_id}/state-machine").json()
    assert plain["hmm"] is None and plain["granularity"] == "structure"
    assert len(plain["versions"]) == len(plain["path"]) == plain["stats"]["versions"]

    run = setup(client)
    episode_id = run["dataset"]["test_episodes"][0]
    result = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{episode_id}/state-machine",
                        params={"granularity": "full"}).json()
    assert result["hmm"]["model_version_id"] == run["model_version_id"]
    events = [e for t in result["transitions"] for e in t["events"]]
    assert sorted(e["step"] for e in events) == list(range(1, len(result["path"])))
    for e in events:
        assert e["hidden_state"] in result["hmm"]["states"] and 0 <= e["model_probability"] <= 1
        assert abs(sum(e["posterior"].values()) - 1) < 1e-4
    bad = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{episode_id}/state-machine", params={"granularity": "x"})
    assert bad.status_code == 422


# ------------------------------------------------------------------ 同一題的所有演變過程
def lane(episode_id, *graphs, ops=None):
    events = [{"event_id": f"{episode_id}-E{i}", "operation_type": (ops or ["param_tune"] * len(graphs))[i - 1],
               "activity_type": None} for i in range(1, len(graphs))]
    return {"episode_id": episode_id, "teacher": episode_id, "scope": "tasks", "run": None,
            "versions": versions(*graphs), "events": events, "model_steps": None}


def test_task_merges_lanes_and_aggregates_probabilities():
    from kcore.fsm import build_task_state_machine

    episodes = [
        lane("A", V_HSV, V_DNN, ops=["strategy_switch"]),
        lane("B", V_DNN, V_HSV_TUNED, ops=["strategy_switch"]),            # 從 A 的第 1 版出發，走回 A 的起點（結構相同）
        lane("C", V_HSV, V_HSV_TUNED, V_DNN, ops=["param_tune", "strategy_switch"]),
    ]
    machine = build_task_state_machine(episodes, "structure")
    states = {s["id"]: s for s in machine["states"]}
    # 狀態放在最早走到它的演變過程與版本：HSV 是 A 的第 0 版，DNN 是 B 的第 0 版（比 A 的第 1 版早）
    assert (states["S1"]["lane"], states["S1"]["column"]) == (0, 0) and (states["S2"]["lane"], states["S2"]["column"]) == (1, 0)
    assert [e["path"] for e in machine["episodes"]] == [["S1", "S2"], ["S2", "S1"], ["S1", "S1", "S2"]]
    by_edge = {(t["source"], t["target"]): t for t in machine["transitions"]}
    # S1 出去的修改：A 一次、C 兩次（一次停在原地）→ S1→S2 是 2/3
    assert by_edge[("S1", "S2")]["count"] == 2 and abs(by_edge[("S1", "S2")]["probability"] - 2 / 3) < 1e-9
    assert by_edge[("S1", "S2")]["episodes"] == ["A", "C"] and by_edge[("S1", "S1")]["probability"] == 1 / 3
    assert states["S1"]["starts"] == ["A", "C"] and states["S2"]["finals"] == ["A", "C"]
    assert machine["stats"]["shared_states"] == 2 and machine["stats"]["episodes"] == 3
    cells = [(s["lane"], s["column"]) for s in machine["states"]]
    assert len(cells) == len(set(cells))


def test_task_state_machine_service(service):
    from kcore.fsm import StateMachineService

    from .conftest import spec_episode

    service.create_case({"case_id": "TASK", "domain_kind": "erp"})
    service.ingest_episode("TASK", spec_episode("EP-A", "A-"))
    service.ingest_episode("TASK", spec_episode("EP-B", "B-"))  # 同一個工作單 DEMO-MO-001
    machine = StateMachineService(service).task("TASK", "DEMO-MO-001", "structure")
    assert [e["episode_id"] for e in machine["episodes"]] == ["EP-A", "EP-B"] and machine["hmm"] is None
    # 兩條演變過程完全相同：所有狀態都被兩條共用，轉移次數加倍、機率不變
    assert machine["stats"]["shared_states"] == machine["stats"]["states"]
    assert all(t["count"] == 2 and len(t["episodes"]) == 2 for t in machine["transitions"])
    with pytest.raises(KCoreError):
        StateMachineService(service).task("TASK", "NO-SUCH-TASK")
