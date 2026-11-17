"""無標籤評估：保留資料的似然、困惑度與下一觀測預測，並與 unigram／bigram 基準比較。

如果 HMM 贏不過一階馬可夫鏈（bigram），代表隱藏狀態沒有帶來額外資訊。
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hmm.hybrid import MixturePredictor, RouteBigramPredictor
from .hmm.model import HiddenMarkovModel

Predictor = Callable[[list[str]], np.ndarray]


class FrequencyBaselines:
    """unigram：只看訓練集符號頻率；bigram：只看上一個符號（拉普拉斯平滑）。"""

    def __init__(self, train: list[list[str]], symbols: tuple[str, ...], smoothing: float = 0.1) -> None:
        self.symbols = symbols
        self.index = {symbol: position for position, symbol in enumerate(symbols)}
        size = len(symbols)
        unigram = np.full(size, smoothing)
        bigram = np.full((size, size), smoothing)
        for sequence in train:
            ids = [self.index[token] for token in sequence]
            for position, current in enumerate(ids):
                unigram[current] += 1
                if position > 0:
                    bigram[ids[position - 1], current] += 1
        self.unigram = unigram / unigram.sum()
        self.bigram = bigram / bigram.sum(axis=1, keepdims=True)

    def predict_unigram(self, prefix: list[str]) -> np.ndarray:
        return self.unigram

    def predict_bigram(self, prefix: list[str]) -> np.ndarray:
        return self.unigram if not prefix else self.bigram[self.index[prefix[-1]]]


def sequence_metrics(predict: Predictor, sequences: list[list[str]], symbols: tuple[str, ...]) -> dict[str, float]:
    """逐步以已知前綴預測下一個觀測。log 似然包含第一步；Top-k 與 MRR 只計算第二步以後。"""
    index = {symbol: position for position, symbol in enumerate(symbols)}
    total_log = 0.0
    events = 0
    hits1 = hits3 = 0
    reciprocal = 0.0
    ranked_steps = 0
    for sequence in sequences:
        for t, token in enumerate(sequence):
            distribution = predict(sequence[:t])
            actual = index[token]
            total_log += math.log(max(float(distribution[actual]), 1e-300))
            events += 1
            if t == 0:
                continue
            rank = int((distribution > distribution[actual]).sum()) + 1
            hits1 += rank == 1
            hits3 += rank <= 3
            reciprocal += 1 / rank
            ranked_steps += 1
    per_event = total_log / events if events else float("nan")
    return {
        "events": events,
        "ranked_steps": ranked_steps,
        "log_likelihood_per_event": per_event,
        "perplexity": math.exp(-per_event) if events else float("nan"),
        "next_top1": hits1 / ranked_steps if ranked_steps else float("nan"),
        "next_top3": hits3 / ranked_steps if ranked_steps else float("nan"),
        "next_mrr": reciprocal / ranked_steps if ranked_steps else float("nan"),
    }


def evaluate_model(model: HiddenMarkovModel, train: list[list[str]], test: list[list[str]],
                   smoothing: float, hybrid: RouteBigramPredictor | None = None) -> dict[str, dict[str, float]]:
    """所有方法都只用訓練資料估計，再在測試資料上以相同方式評分。"""
    baselines = FrequencyBaselines(train, model.observations, smoothing)
    mixture = MixturePredictor.fit(model, train, smoothing)
    results = {
        "hmm": sequence_metrics(model.next_observation_distribution, test, model.observations),
        "bigram": sequence_metrics(baselines.predict_bigram, test, model.observations),
        "unigram": sequence_metrics(baselines.predict_unigram, test, model.observations),
        "hmm_bigram_mix": sequence_metrics(mixture.predict, test, model.observations),
    }
    if hybrid is not None:
        results["hmm_route_bigram"] = sequence_metrics(hybrid.predict, test, model.observations)
    return results


def label_recovery(model: HiddenMarkovModel, sequences: list[list[str]], labels: list[list[str]]) -> dict[str, object]:
    """僅供驗證：以匈牙利演算法把 Viterbi 解碼狀態對應到示意資料埋入的潛在階段，計算配對準確率。

    標籤不參與訓練或推論，只在事後比較。
    """
    label_names = sorted({label for sequence in labels for label in sequence})
    state_index = {state: position for position, state in enumerate(model.states)}
    label_index = {label: position for position, label in enumerate(label_names)}
    confusion = np.zeros((len(model.states), len(label_names)), dtype=int)
    for sequence, truth in zip(sequences, labels, strict=True):
        _, path = model.viterbi(sequence)
        for state, label in zip(path, truth, strict=True):
            confusion[state_index[state], label_index[label]] += 1
    rows, cols = linear_sum_assignment(-confusion)
    matched = int(confusion[rows, cols].sum())
    total = int(confusion.sum())
    return {
        "matched_accuracy": matched / total if total else float("nan"),
        "mapping": {model.states[row]: label_names[col] for row, col in zip(rows, cols, strict=True)},
        "labels": label_names,
        "confusion": confusion.tolist(),
        "note": "verification only: synthetic latent labels are never used for training or inference",
    }
