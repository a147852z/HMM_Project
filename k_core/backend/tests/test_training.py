import numpy as np
from fastapi.testclient import TestClient

from kcore.api import create_app
from kcore.encoding import FeatureSchema
from kcore.hmm.training import TrainingConfig
from kcore.synthetic import LATENT_PHASE_KEY, generate_episodes

FAST = {"k_min": 2, "k_max": 4, "restarts": 2, "n_iter": 100, "seed": 3}


def _case(client, case_id="AP-K-ERP"):
    client.post("/api/v1/cases", json={"case_id": case_id, "domain_kind": "erp"})


def test_insufficient_data_returns_explicit_status(client):
    _case(client)
    client.post("/api/v1/cases/AP-K-ERP/synthetic-episodes", json={"count": 3, "seed": 1})
    run = client.post("/api/v1/cases/AP-K-ERP/training-runs", json=FAST).json()
    assert run["status"] == "insufficient_data"
    assert run["model_version_id"] is None
    assert client.get("/api/v1/cases/AP-K-ERP/models").json() == []
    response = client.post("/api/v1/cases/AP-K-ERP/models/active/infer", json={"episode_id": "SYN-AP-K-ERP-0000"})
    assert response.status_code == 404 and response.json()["error"]["code"] == "no_active_model"


def test_synthetic_label_is_never_part_of_observation(service):
    service.create_case({"case_id": "AP-K-ERP", "domain_kind": "erp"})
    payload = generate_episodes("AP-K-ERP", 1, seed=5)[0]
    for event in payload["events"]:
        event["attributes"][LATENT_PHASE_KEY] = "SHOULD-NOT-LEAK"
    episode = service.ingest_episode("AP-K-ERP", payload)
    for event in episode["events"]:
        assert "SHOULD-NOT-LEAK" not in str(event["features"])


def test_training_recovers_planted_structure_and_beats_unigram(service):
    service.create_case({"case_id": "AP-K-ERP", "domain_kind": "erp"})
    service.generate_synthetic("AP-K-ERP", 120, seed=11)
    run = service.train("AP-K-ERP", FeatureSchema(), TrainingConfig(**FAST))
    assert run["status"] == "succeeded", run["message"]
    metrics = run["metrics"]
    # 自寫推論與 hmmlearn 對訓練資料的 log 似然一致（序列邊界一致、參數轉換正確）
    assert metrics["training"]["implementation_gap"] < 1e-6
    # 依時間切分，訓練與測試的序列不重疊
    assert not set(run["dataset"]["train_episodes"]) & set(run["dataset"]["test_episodes"])
    test = metrics["test"]
    assert test["hmm"]["perplexity"] < test["unigram"]["perplexity"]
    assert metrics["synthetic_label_recovery"]["matched_accuracy"] > 0.7

    model = service.get_model("AP-K-ERP", run["model_version_id"])
    emission = np.asarray(model["parameters"]["emission_matrix"])
    np.testing.assert_allclose(emission.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.asarray(model["parameters"]["transition_matrix"]).sum(axis=1), 1.0)
    assert (emission > 0).all()  # 平滑後沒有零機率


def test_training_is_reproducible_with_same_seed(service):
    service.create_case({"case_id": "AP-K-ERP", "domain_kind": "erp"})
    service.generate_synthetic("AP-K-ERP", 40, seed=2)
    config = TrainingConfig(k_min=2, k_max=3, restarts=2, n_iter=50, seed=7)
    first = service.train("AP-K-ERP", FeatureSchema(), config)
    second = service.train("AP-K-ERP", FeatureSchema(), config)
    a = service.get_model("AP-K-ERP", first["model_version_id"])["parameters"]
    b = service.get_model("AP-K-ERP", second["model_version_id"])["parameters"]
    np.testing.assert_allclose(a["emission_matrix"], b["emission_matrix"])
    assert service.list_models("AP-K-ERP")[0]["status"] == "inactive"
    assert service.get_case("AP-K-ERP")["active_model_version_id"] == second["model_version_id"]


def test_saved_model_reloads_and_infers_identically(tmp_path):
    url = f"sqlite:///{tmp_path / 'reload.db'}"
    first = TestClient(create_app(url))
    _case(first)
    first.post("/api/v1/cases/AP-K-ERP/synthetic-episodes", json={"count": 40, "seed": 4})
    run = first.post("/api/v1/cases/AP-K-ERP/training-runs", json={**FAST, "k_max": 3}).json()
    episode_id = run["dataset"]["test_episodes"][0]
    request = {"episode_id": episode_id, "prefix_length": 2, "top_k": 3}
    before = first.post("/api/v1/cases/AP-K-ERP/models/active/infer", json=request).json()

    reloaded = TestClient(create_app(url))  # 模擬服務重啟後重新載入
    after = reloaded.post(f"/api/v1/cases/AP-K-ERP/models/{run['model_version_id']}/infer", json=request).json()
    assert before == after
    assert after["method"] == "hmm_bigram_mix" and after["model_version_id"] == run["model_version_id"]
    assert len(after["steps"]) == 2 and len(after["next_observation"]) == 3


def test_inference_uses_only_the_prefix(service):
    service.create_case({"case_id": "AP-K-ERP", "domain_kind": "erp"})
    payloads = generate_episodes("AP-K-ERP", 30, seed=9)
    for payload in payloads:
        service.ingest_episode("AP-K-ERP", payload)
    service.train("AP-K-ERP", FeatureSchema(), TrainingConfig(k_min=2, k_max=3, restarts=1, n_iter=50))

    # 同一個前綴、不同的後續：把前兩個事件另存成一條尚未定案的序列，推論結果必須相同。
    source = next(p for p in payloads if len(p["events"]) >= 4)
    partial = {**source, "episode_id": "PARTIAL", "events": []}
    service.create_episode("AP-K-ERP", partial)
    renamed = [{**event, "event_id": "P-" + event["event_id"]} for event in source["events"][:2]]
    service.add_events("AP-K-ERP", "PARTIAL", renamed)

    full = service.infer("AP-K-ERP", "active", source["episode_id"], prefix_length=2)
    open_prefix = service.infer("AP-K-ERP", "active", "PARTIAL")
    assert full["next_observation"] == open_prefix["next_observation"]
    assert full["viterbi_path"] == open_prefix["viterbi_path"]
