"""非監督式 HMM 訓練：hmmlearn CategoricalHMM（Baum-Welch／EM）＋隱藏狀態數 K 的選擇。

- 多段序列以「合併觀測陣列＋各序列長度」提交，保留序列邊界，
  不會把上一段的最後一步與下一段的第一步當成相鄰轉移。
- 平滑：Dirichlet 先驗參數 = 1 + smoothing，等同於對 π、A、B 做拉普拉斯平滑，
  因此測試時遇到訓練沒看過的組合，機率不會變成 0。
- 每個 K 以多個亂數種子重跑，取訓練似然最高者；再以 BIC 或 AIC 在不同 K 之間選擇。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from .model import HiddenMarkovModel


@dataclass(frozen=True)
class TrainingConfig:
    k_min: int = 1  # 包含 K=1（等同沒有隱藏結構），才能檢驗隱藏狀態是否真的有幫助
    k_max: int = 5
    restarts: int = 3
    n_iter: int = 200
    tol: float = 1e-4
    smoothing: float = 0.1
    seed: int = 0
    criterion: str = "bic"  # bic / aic

    def __post_init__(self) -> None:
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError("require 1 <= k_min <= k_max")
        if self.restarts < 1 or self.n_iter < 1:
            raise ValueError("restarts and n_iter must be positive")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if self.criterion not in {"bic", "aic"}:
            raise ValueError("criterion must be 'bic' or 'aic'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateResult:
    k: int
    log_likelihood: float
    n_parameters: int
    bic: float
    aic: float
    converged: bool
    iterations: int
    seed: int
    log_likelihood_history: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    model: HiddenMarkovModel
    selected: CandidateResult
    candidates: list[CandidateResult]
    implementation_gap: float  # 自寫前向演算法與 hmmlearn 對同一資料的 log 似然差距

    def summary(self) -> dict[str, Any]:
        return {
            "selected_k": self.selected.k,
            "selected": asdict(self.selected),
            "candidates": [
                {key: value for key, value in asdict(candidate).items() if key != "log_likelihood_history"}
                for candidate in self.candidates
            ],
            "implementation_gap": self.implementation_gap,
        }


def count_parameters(k: int, n_symbols: int) -> int:
    """π、A、B 的自由參數數：(K-1) + K(K-1) + K(M-1)。"""
    return (k - 1) + k * (k - 1) + k * (n_symbols - 1)


def _stack(sequences: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    data = np.concatenate([np.asarray(sequence, dtype=int) for sequence in sequences]).reshape(-1, 1)
    return data, [len(sequence) for sequence in sequences]


def _fit_once(data: np.ndarray, lengths: list[int], k: int, n_symbols: int, seed: int,
              config: TrainingConfig) -> CategoricalHMM:
    prior = 1.0 + config.smoothing
    model = CategoricalHMM(
        n_components=k, n_features=n_symbols, n_iter=config.n_iter, tol=config.tol,
        random_state=seed, startprob_prior=prior, transmat_prior=prior, emissionprob_prior=prior,
        init_params="ste", params="ste",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # hmmlearn 在短序列上可能發出收斂提醒；結果以 converged 欄位回報
        model.fit(data, lengths)
    return model


def to_inference_model(model: CategoricalHMM, symbols: tuple[str, ...]) -> HiddenMarkovModel:
    """轉成自寫推論模型。隱藏狀態使用中性編號 H0..Hk-1，語意需經人工解讀。"""
    return HiddenMarkovModel(
        states=tuple(f"H{index}" for index in range(model.n_components)),
        observations=symbols,
        initial=np.maximum(model.startprob_, 0.0),
        transition=np.maximum(model.transmat_, 0.0),
        emission=np.maximum(model.emissionprob_, 0.0),
    )


def train_with_selection(sequences: list[list[int]], symbols: tuple[str, ...], config: TrainingConfig) -> TrainingResult:
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("training requires non-empty sequences")
    data, lengths = _stack(sequences)
    n_observations, n_symbols = len(data), len(symbols)
    candidates: list[CandidateResult] = []
    best_models: dict[int, CategoricalHMM] = {}

    for k in range(config.k_min, config.k_max + 1):
        best: tuple[float, CategoricalHMM, int] | None = None
        for restart in range(config.restarts):
            seed = config.seed * 1000 + k * 100 + restart
            try:
                model = _fit_once(data, lengths, k, n_symbols, seed, config)
                score = float(model.score(data, lengths))
            except (ValueError, np.linalg.LinAlgError):
                continue
            if math.isfinite(score) and (best is None or score > best[0]):
                best = (score, model, seed)
        if best is None:
            continue
        score, model, seed = best
        n_params = count_parameters(k, n_symbols)
        history = [float(value) for value in model.monitor_.history]
        # hmmlearn 跑滿 n_iter 也會回報 converged；這裡只在似然進步小於 tol 時才算收斂。
        converged = len(history) >= 2 and history[-1] - history[-2] < config.tol
        candidates.append(CandidateResult(
            k=k, log_likelihood=score, n_parameters=n_params,
            bic=-2 * score + n_params * math.log(n_observations), aic=-2 * score + 2 * n_params,
            converged=converged, iterations=int(model.monitor_.iter), seed=seed,
            log_likelihood_history=history,
        ))
        best_models[k] = model

    if not candidates:
        raise RuntimeError("no HMM candidate could be trained")
    selected = min(candidates, key=lambda candidate: getattr(candidate, config.criterion))
    fitted = best_models[selected.k]
    inference_model = to_inference_model(fitted, symbols)
    own_ll = sum(inference_model.log_likelihood([symbols[i] for i in sequence]) for sequence in sequences)
    return TrainingResult(
        model=inference_model, selected=selected, candidates=candidates,
        implementation_gap=abs(own_ll - selected.log_likelihood),
    )
