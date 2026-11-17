"""HMM＋上一次修改的預測（route bigram）。

想法：HMM 的隱藏狀態代表長期的「路線」（例如 CVflow 的 DNN 路線與傳統影像處理路線），
上一次修改則決定短期的下一步。所以每個隱藏狀態各有一張「上一次觀測 → 下一次觀測」的轉移表：

    P(o_{t+1} | o_1..o_t) = Σ_s P(s_{t+1} | o_1..o_t) · T[s][o_t, o_{t+1}]

- P(s_{t+1} | o_1..o_t) = (濾波信念 · A)，只使用已知的前綴。
- T[s] 由訓練資料估計：每一對相鄰觀測 (a, b) 以「時間 t 位於狀態 s 的後驗機率 γ_t(s)」加權計數。
- 某個 a 的資料很少時，T[s][a, :] 會退回狀態 s 的發射分布 B[s, :]（backoff 越大退得越多），
  所以完全沒看過的組合也有合理的機率，不會變成 0。

另提供 MixturePredictor：HMM 與 bigram 以固定權重（0.5）平均。CVFLOW 實測中它的機率估計最好，
因此為推論的預設方法；兩者都會隨模型版本保存，可在推論時指定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import HiddenMarkovModel


@dataclass
class RouteBigramPredictor:
    model: HiddenMarkovModel
    tables: np.ndarray  # shape (K, M, M)
    backoff: float

    method = "hmm_route_bigram"

    @classmethod
    def fit(cls, model: HiddenMarkovModel, sequences: list[list[str]], backoff: float = 2.0) -> RouteBigramPredictor:
        if backoff <= 0:
            raise ValueError("backoff must be positive")
        k, m = len(model.states), len(model.observations)
        counts = np.zeros((k, m, m))
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            gamma = model.posterior(sequence)  # 訓練資料：使用整段序列的平滑後驗
            ids = [model.observation_id(o) for o in sequence]
            for t in range(1, len(ids)):
                counts[:, ids[t - 1], ids[t]] += gamma[t]
        tables = (counts + backoff * model.emission[:, None, :]) / (counts.sum(axis=2, keepdims=True) + backoff)
        return cls(model=model, tables=tables, backoff=backoff)

    def predict(self, prefix: list[str]) -> np.ndarray:
        if not prefix:
            return self.model.next_observation_distribution(prefix)
        state_prior = self.model.filter(prefix)[-1] @ self.model.transition
        last = self.model.observation_id(prefix[-1])
        distribution = state_prior @ self.tables[:, last, :]
        return distribution / distribution.sum()

    def to_dict(self) -> dict[str, object]:
        return {"type": self.method, "backoff": self.backoff, "tables": np.round(self.tables, 8).tolist()}

    @classmethod
    def from_dict(cls, model: HiddenMarkovModel, data: dict[str, object]) -> RouteBigramPredictor:
        return cls(model=model, tables=np.asarray(data["tables"], dtype=float), backoff=float(data["backoff"]))


@dataclass
class MixturePredictor:
    """對照方法：HMM 預測與 bigram 預測以固定權重平均。"""

    model: HiddenMarkovModel
    bigram: np.ndarray  # shape (M, M)，每列已正規化
    weight: float = 0.5

    method = "hmm_bigram_mix"

    @classmethod
    def fit(cls, model: HiddenMarkovModel, sequences: list[list[str]], smoothing: float = 0.1,
            weight: float = 0.5) -> MixturePredictor:
        m = len(model.observations)
        counts = np.full((m, m), smoothing)
        for sequence in sequences:
            ids = [model.observation_id(o) for o in sequence]
            for a, b in zip(ids, ids[1:]):
                counts[a, b] += 1
        return cls(model=model, bigram=counts / counts.sum(axis=1, keepdims=True), weight=weight)

    def predict(self, prefix: list[str]) -> np.ndarray:
        hmm = self.model.next_observation_distribution(prefix)
        if not prefix:
            return hmm
        return self.weight * hmm + (1 - self.weight) * self.bigram[self.model.observation_id(prefix[-1])]

    def to_dict(self) -> dict[str, object]:
        return {"type": self.method, "weight": self.weight, "bigram": np.round(self.bigram, 8).tolist()}

    @classmethod
    def from_dict(cls, model: HiddenMarkovModel, data: dict[str, object]) -> MixturePredictor:
        return cls(model=model, bigram=np.asarray(data["bigram"], dtype=float), weight=float(data["weight"]))


PREDICTOR_TYPES = {RouteBigramPredictor.method: RouteBigramPredictor, MixturePredictor.method: MixturePredictor}
# 預設方法：CVFLOW 5 折交叉驗證中，平均法的機率估計最好（困惑度 18.3，bigram 22.6，HMM 20.1）
DEFAULT_PREDICTOR = MixturePredictor.method
