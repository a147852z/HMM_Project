import pytest

from kcore.aov import AoVError, apply_change, derive_change_features, validate_workflow

from .conftest import SPEC_EPISODE


def _initial():
    workflow = SPEC_EPISODE["initial_workflow"]
    return workflow["nodes"], workflow["edges"]


def test_initial_spec_workflow_is_valid_dag():
    assert validate_workflow(*_initial()) == []


def test_validator_reports_cycle_dangling_edge_and_duplicates():
    nodes = [{"node_id": "a", "activity_type": "x"}, {"node_id": "b", "activity_type": "x"},
             {"node_id": "b", "activity_type": "x"}]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}, {"source": "a", "target": "zz"},
             {"source": "a", "target": "a"}]
    problems = " | ".join(validate_workflow(nodes, edges))
    assert "duplicate node_id: b" in problems
    assert "missing node" in problems
    assert "self-loop" in problems
    assert "cycle detected" in problems


def test_spec_changes_rebuild_expected_final_workflow():
    nodes, edges = _initial()
    for event in SPEC_EPISODE["events"]:
        nodes, edges = apply_change(nodes, edges, event["change"])
    by_id = {node["node_id"]: node for node in nodes}
    assert set(by_id) == {"n1", "n2", "n3", "n4"}
    assert by_id["n4"]["parameters"] == {"inspection_mode": "full"}
    assert {(e["source"], e["target"]) for e in edges} == {("n1", "n2"), ("n2", "n4"), ("n4", "n3")}


def test_apply_change_does_not_mutate_input():
    nodes, edges = _initial()
    apply_change(nodes, edges, SPEC_EPISODE["events"][0]["change"])
    assert len(nodes) == 3 and len(edges) == 2


def test_change_creating_cycle_is_rejected():
    nodes, edges = _initial()
    with pytest.raises(AoVError, match="cycle"):
        apply_change(nodes, edges, {"added_edges": [{"source": "n3", "target": "n1"}]})


def test_removing_node_without_its_edges_is_rejected():
    nodes, edges = _initial()
    with pytest.raises(AoVError, match="missing node"):
        apply_change(nodes, edges, {"removed_nodes": ["n2"]})


def test_derived_features_describe_affected_activity():
    nodes, edges = _initial()
    change = SPEC_EPISODE["events"][0]["change"]
    after, _ = apply_change(nodes, edges, change)
    features = derive_change_features(nodes, after, change)
    assert features["activity_type"] == "inspect"
    assert features["node_delta"] == "+"
    assert features["edge_delta"] == "+"
