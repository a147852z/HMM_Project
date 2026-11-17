"""離散觀測 HMM 的推論（由 erp_hmm_simulator 移植並修正縮放方式）。

符號說明：
- initial[i]：序列第一步位於隱藏狀態 i 的機率（π）。
- transition[i][j]：目前位於狀態 i 時，下一步進入狀態 j 的機率（A）。
- emission[i][o]：位於狀態 i 時產生觀測 o 的機率（B）。

前向與後向使用同一組縮放因子 c_t，因此：
- log P(觀測序列) = Σ log c_t
- posterior[t][i] = alpha_hat[t][i] * beta_hat[t][i]（每個時間點總和為 1）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..encoding import UNKNOWN

FloatArray = NDArray[np.float64]
_TINY = np.finfo(float).tiny


def _normalized(values: FloatArray, axis: int | None = None) -> FloatArray:
    values = np.asarray(values, dtype=float)
    total = values.sum(axis=axis, keepdims=axis is not None)
    if np.any(total <= 0):
        raise ValueError("probability distributions must have a positive sum")
    return values / total


@dataclass(slots=True)
class HiddenMarkovModel:
    states: tuple[str, ...]
    observations: tuple[str, ...]
    initial: FloatArray
    transition: FloatArray
    emission: FloatArray

    def __post_init__(self) -> None:
        self.initial = _normalized(np.asarray(self.initial, dtype=float))
        self.transition = _normalized(np.asarray(self.transition, dtype=float), axis=1)
        self.emission = _normalized(np.asarray(self.emission, dtype=float), axis=1)
        n_states, n_obs = len(self.states), len(self.observations)
        if self.initial.shape != (n_states,):
            raise ValueError("initial distribution shape does not match states")
        if self.transition.shape != (n_states, n_states):
            raise ValueError("transition matrix shape does not match states")
        if self.emission.shape != (n_states, n_obs):
            raise ValueError("emission matrix shape does not match observations")

    # ---- 觀測對應 ----
    def observation_id(self, observation: str) -> int:
        """未知觀測對應到 __UNK__；模型沒有 __UNK__ 時才拒絕。"""
        if observation in self.observations:
            return self.observations.index(observation)
        if UNKNOWN in self.observations:
            return self.observations.index(UNKNOWN)
        raise ValueError(f"unknown observation: {observation}")

    def _ids(self, sequence: list[str]) -> list[int]:
        return [self.observation_id(observation) for observation in sequence]

    # ---- 前向／後向 ----
    def _forward_scaled(self, ids: list[int]) -> tuple[FloatArray, FloatArray]:
        """回傳 (alpha_hat, scales)。alpha_hat[t] 是看到 o_1..o_t 後的濾波信念。"""
        alpha = np.zeros((len(ids), len(self.states)))
        scales = np.zeros(len(ids))
        current = self.initial * self.emission[:, ids[0]]
        for t, obs in enumerate(ids):
            if t > 0:
                current = (alpha[t - 1] @ self.transition) * self.emission[:, obs]
            scales[t] = max(float(current.sum()), _TINY)
            alpha[t] = current / scales[t]
        return alpha, scales

    def log_likelihood(self, sequence: list[str]) -> float:
        if not sequence:
            return 0.0
        _, scales = self._forward_scaled(self._ids(sequence))
        return float(np.log(scales).sum())

    def filter(self, sequence: list[str]) -> FloatArray:
        """每一步的濾波信念 P(state_t | o_1..o_t)，只使用到 t 為止的觀測。"""
        if not sequence:
            return self.initial[None, :].copy()
        alpha, _ = self._forward_scaled(self._ids(sequence))
        return alpha

    def posterior(self, sequence: list[str]) -> FloatArray:
        """平滑後驗 P(state_t | 整段序列)。只能用於事後分析，不可用於即時建議。"""
        ids = self._ids(sequence)
        alpha, scales = self._forward_scaled(ids)
        beta = np.ones_like(alpha)
        for t in range(len(ids) - 2, -1, -1):
            beta[t] = self.transition @ (self.emission[:, ids[t + 1]] * beta[t + 1]) / scales[t + 1]
        return _normalized(alpha * beta, axis=1)

    def next_observation_distribution(self, prefix: list[str]) -> FloatArray:
        """P(o_{t+1} | o_1..o_t) = (belief_t · A) · B；prefix 為空時用 π · B。"""
        state_prior = self.initial if not prefix else self.filter(prefix)[-1] @ self.transition
        return _normalized(state_prior @ self.emission)

    def viterbi(self, sequence: list[str]) -> tuple[float, list[str]]:
        """回傳 (最佳路徑的 log 機率, 狀態序列)。"""
        if not sequence:
            return 0.0, []
        ids = self._ids(sequence)
        log_a = np.log(np.maximum(self.transition, _TINY))
        log_b = np.log(np.maximum(self.emission, _TINY))
        score = np.log(np.maximum(self.initial, _TINY)) + log_b[:, ids[0]]
        pointers: list[NDArray[np.int64]] = []
        for obs in ids[1:]:
            candidates = score[:, None] + log_a
            pointers.append(np.argmax(candidates, axis=0))
            score = np.max(candidates, axis=0) + log_b[:, obs]
        last = int(np.argmax(score))
        path = [last]
        for back in reversed(pointers):
            last = int(back[last])
            path.append(last)
        path.reverse()
        return float(np.max(score)), [self.states[index] for index in path]

    # ---- 保存 ----
    def to_dict(self) -> dict[str, object]:
        return {
            "states": list(self.states),
            "observations": list(self.observations),
            "pi": self.initial.tolist(),
            "transition_matrix": self.transition.tolist(),
            "emission_matrix": self.emission.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> HiddenMarkovModel:
        return cls(
            states=tuple(data["states"]),  # type: ignore[arg-type]
            observations=tuple(data["observations"]),  # type: ignore[arg-type]
            initial=np.asarray(data["pi"], dtype=float),
            transition=np.asarray(data["transition_matrix"], dtype=float),
            emission=np.asarray(data["emission_matrix"], dtype=float),
        )

    def stationary_distribution(self) -> FloatArray:
        values, vectors = np.linalg.eig(self.transition.T)
        vector = np.real(vectors[:, np.argmin(np.abs(values - 1))])
        vector = np.abs(vector)
        return vector / vector.sum()
