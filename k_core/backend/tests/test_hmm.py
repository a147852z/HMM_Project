import itertools

import numpy as np
import pytest

from kcore.encoding import UNKNOWN, FeatureSchema, Vocabulary, event_token
from kcore.hmm.model import HiddenMarkovModel


@pytest.fixture()
def model() -> HiddenMarkovModel:
    return HiddenMarkovModel(
        states=("H0", "H1"), observations=("a", "b", UNKNOWN),
        initial=np.array([0.6, 0.4]),
        transition=np.array([[0.7, 0.3], [0.2, 0.8]]),
        emission=np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2]]),
    )


def _brute_force_likelihood(model: HiddenMarkovModel, sequence: list[str]) -> float:
    ids = [model.observation_id(o) for o in sequence]
    total = 0.0
    for path in itertools.product(range(len(model.states)), repeat=len(ids)):
        p = model.initial[path[0]] * model.emission[path[0], ids[0]]
        for t in range(1, len(ids)):
            p *= model.transition[path[t - 1], path[t]] * model.emission[path[t], ids[t]]
        total += p
    return total


def test_scaled_forward_matches_brute_force(model):
    sequence = ["a", "b", "b", "a", "b"]
    assert model.log_likelihood(sequence) == pytest.approx(np.log(_brute_force_likelihood(model, sequence)))


def test_posterior_and_filter_rows_sum_to_one(model):
    sequence = ["a", "b", "a", "a", "b", "b"]
    np.testing.assert_allclose(model.posterior(sequence).sum(axis=1), 1.0)
    np.testing.assert_allclose(model.filter(sequence).sum(axis=1), 1.0)


def test_posterior_matches_brute_force_marginal(model):
    sequence = ["a", "b", "b"]
    ids = [model.observation_id(o) for o in sequence]
    marginal = np.zeros(2)
    for path in itertools.product(range(2), repeat=3):
        p = model.initial[path[0]] * model.emission[path[0], ids[0]]
        for t in range(1, 3):
            p *= model.transition[path[t - 1], path[t]] * model.emission[path[t], ids[t]]
        marginal[path[1]] += p
    np.testing.assert_allclose(model.posterior(sequence)[1], marginal / marginal.sum())


def test_long_sequence_does_not_underflow(model):
    assert np.isfinite(model.log_likelihood(["a", "b"] * 2000))


def test_unknown_observation_maps_to_unk(model):
    assert model.log_likelihood(["never-seen"]) == pytest.approx(model.log_likelihood([UNKNOWN]))


def test_next_observation_distribution_is_normalized(model):
    assert model.next_observation_distribution([]).sum() == pytest.approx(1.0)
    assert model.next_observation_distribution(["a", "b"]).sum() == pytest.approx(1.0)


def test_round_trip_preserves_inference(model):
    restored = HiddenMarkovModel.from_dict(model.to_dict())
    assert restored.log_likelihood(["a", "b", "a"]) == pytest.approx(model.log_likelihood(["a", "b", "a"]))


def test_encoder_only_reads_schema_fields():
    schema = FeatureSchema(fields=("operation_type", "activity_type"))
    features = {"activity_type": "inspect", "node_delta": "+"}
    assert event_token("insert_activity", features, schema) == "insert_activity|inspect"


def test_vocabulary_merges_rare_symbols_into_unknown():
    vocabulary = Vocabulary.build([["a", "a", "b"]], min_count=2)
    assert vocabulary.symbols == ("a", UNKNOWN)
    assert vocabulary.encode(["b", "a", "zzz"]) == [1, 0, 1]
