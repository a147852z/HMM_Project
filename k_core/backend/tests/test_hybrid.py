import numpy as np
import pytest

from kcore.encoding import UNKNOWN, FeatureSchema
from kcore.hmm.hybrid import MixturePredictor, RouteBigramPredictor
from kcore.hmm.model import HiddenMarkovModel
from kcore.hmm.training import TrainingConfig

# 兩條「路線」：路線 0 交替 a→b→a→b，路線 1 交替 c→d→c→d。整體頻率相同，只有「路線＋上一個觀測」才猜得準。
ROUTE0 = ["a", "b"] * 4
ROUTE1 = ["c", "d"] * 4


@pytest.fixture()
def model() -> HiddenMarkovModel:
    return HiddenMarkovModel(
        states=("H0", "H1"), observations=("a", "b", "c", "d", UNKNOWN),
        initial=np.array([0.5, 0.5]),
        transition=np.array([[0.99, 0.01], [0.01, 0.99]]),
        emission=np.array([[0.49, 0.49, 0.005, 0.005, 0.01], [0.005, 0.005, 0.49, 0.49, 0.01]]),
    )


def test_route_bigram_uses_route_and_previous_observation(model):
    hybrid = RouteBigramPredictor.fit(model, [ROUTE0, ROUTE1] * 5, backoff=1.0)
    p = hybrid.predict(["a", "b", "a"])
    index = {o: i for i, o in enumerate(model.observations)}
    assert p.sum() == pytest.approx(1.0)
    assert p[index["b"]] > 0.8  # 路線 0 且上一個是 a → 下一個是 b
    # 只用 HMM 時，a 與 b 機率相近，無法區分
    hmm = model.next_observation_distribution(["a", "b", "a"])
    assert abs(hmm[index["a"]] - hmm[index["b"]]) < 0.05


def test_empty_prefix_falls_back_to_hmm(model):
    hybrid = RouteBigramPredictor.fit(model, [ROUTE0], backoff=1.0)
    np.testing.assert_allclose(hybrid.predict([]), model.next_observation_distribution([]))


def test_unseen_pairs_back_off_to_state_emission(model):
    hybrid = RouteBigramPredictor.fit(model, [ROUTE0], backoff=1.0)
    # 從未在訓練資料出現的上一個觀測 d：轉移表退回各狀態的發射分布
    np.testing.assert_allclose(hybrid.tables[:, 3, :], model.emission)
    assert (hybrid.tables > 0).all()


def test_round_trip(model):
    hybrid = RouteBigramPredictor.fit(model, [ROUTE0, ROUTE1], backoff=2.0)
    restored = RouteBigramPredictor.from_dict(model, hybrid.to_dict())
    np.testing.assert_allclose(restored.predict(["c", "d"]), hybrid.predict(["c", "d"]), atol=1e-6)


def test_mixture_is_normalized(model):
    mix = MixturePredictor.fit(model, [ROUTE0, ROUTE1])
    assert mix.predict(["a"]).sum() == pytest.approx(1.0)


def test_training_saves_hybrid_and_inference_uses_it(service):
    service.create_case({"case_id": "AP-K-ERP", "domain_kind": "erp"})
    service.generate_synthetic("AP-K-ERP", 40, seed=3)
    run = service.train("AP-K-ERP", FeatureSchema(), TrainingConfig(k_min=1, k_max=3, restarts=1, n_iter=50))
    assert run["status"] == "succeeded"
    test = run["metrics"]["test"]
    assert {"hmm", "bigram", "unigram", "hmm_route_bigram", "hmm_bigram_mix"} <= set(test)
    episode_id = run["dataset"]["test_episodes"][0]
    result = service.infer("AP-K-ERP", "active", episode_id, prefix_length=2)
    assert result["method"] == "hmm_bigram_mix"
    assert set(result["available_methods"]) == {"hmm", "hmm_bigram_mix", "hmm_route_bigram"}
    route = service.infer("AP-K-ERP", "active", episode_id, prefix_length=2, method="hmm_route_bigram")
    assert route["method"] == "hmm_route_bigram"
    assert sum(item["probability"] for item in result["next_observation"]) <= 1.0 + 1e-9
    assert result["next_observation_hmm_only"]


def test_replay_and_analysis_endpoints(client):
    client.post("/api/v1/cases", json={"case_id": "AP-K-ERP", "domain_kind": "erp"})
    client.post("/api/v1/cases/AP-K-ERP/synthetic-episodes", json={"count": 30, "seed": 2})
    run = client.post("/api/v1/cases/AP-K-ERP/training-runs",
                      json={"k_min": 1, "k_max": 3, "restarts": 1, "n_iter": 50}).json()
    episode_id = run["dataset"]["test_episodes"][0]
    replay = client.post("/api/v1/cases/AP-K-ERP/models/active/replay", json={"episode_id": episode_id}).json()
    episode = client.get(f"/api/v1/cases/AP-K-ERP/episodes/{episode_id}").json()
    assert len(replay["steps"]) == episode["event_count"]
    assert set(replay["methods"]) == {"hmm", "hmm_bigram_mix", "hmm_route_bigram"}
    first = replay["steps"][0]
    assert sum(first["route_before"].values()) == pytest.approx(1.0, abs=1e-5)
    assert first["predictions"]["hmm"]["actual_rank"] >= 1
    # 第 t 步的預測只用前 t 個修改：與 infer(prefix_length=t) 一致
    infer = client.post("/api/v1/cases/AP-K-ERP/models/active/infer",
                        json={"episode_id": episode_id, "prefix_length": 1, "method": "hmm"}).json()
    assert infer["next_observation"][0]["observation"] == replay["steps"][1]["predictions"]["hmm"]["top"][0]["observation"]

    analysis = client.get("/api/v1/cases/AP-K-ERP/models/active/analysis").json()
    assert analysis["episodes"] == 30
    assert sum(s["events"] for s in analysis["states"]) == analysis["events"]
    assert analysis["test_metrics"]["hmm_bigram_mix"]
