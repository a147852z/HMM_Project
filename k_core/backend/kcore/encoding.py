"""特徵編碼：把結構化修改事件轉成 HMM 的離散觀測符號。

編碼只讀取事件的 operation_type 與定案時衍生的結構特徵（features）。
修改原因（reason）、示意資料的潛在階段標記（attributes）都不會進入觀測，
前者因為是自由文字且可能缺值，後者因為它是驗證用的標準答案。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

UNKNOWN = "__UNK__"
ALLOWED_FIELDS = ("operation_type", "activity_type", "node_delta", "edge_delta")


@dataclass(frozen=True)
class FeatureSchema:
    version: str = "fs-v1"
    fields: tuple[str, ...] = ("operation_type", "activity_type")
    min_count: int = 1  # 訓練集中出現次數低於此值的符號併入 __UNK__
    # 選用：把細的活動類型歸成粗的類別（例如 77 種 CVflow 節點 → 8 類），避免觀測符號過於稀疏。
    # 沒有列在對照表中的類型使用 activity_default；activity_default 為 None 時保留原類型。
    activity_map: dict[str, str] | None = field(default=None, hash=False, compare=True)
    activity_default: str | None = None

    def __post_init__(self) -> None:
        unknown = [name for name in self.fields if name not in ALLOWED_FIELDS]
        if not self.fields or unknown:
            raise ValueError(f"feature fields must be a non-empty subset of {ALLOWED_FIELDS}; got {unknown}")
        if self.min_count < 1:
            raise ValueError("min_count must be at least 1")

    def map_activity(self, activity_type: str) -> str:
        if self.activity_map is None or activity_type == "none":
            return activity_type
        return self.activity_map.get(activity_type, self.activity_default or activity_type)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fields"] = list(self.fields)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSchema:
        return cls(version=data["version"], fields=tuple(data["fields"]), min_count=int(data["min_count"]),
                   activity_map=data.get("activity_map"), activity_default=data.get("activity_default"))


def event_token(operation_type: str, features: dict[str, Any], schema: FeatureSchema) -> str:
    values = {"operation_type": operation_type, **features}
    if "activity_type" in values:
        values["activity_type"] = schema.map_activity(str(values["activity_type"]))
    return "|".join(str(values.get(name, "none")) for name in schema.fields)


@dataclass
class Vocabulary:
    symbols: tuple[str, ...]
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {symbol: position for position, symbol in enumerate(self.symbols)}

    @classmethod
    def build(cls, sequences: list[list[str]], min_count: int) -> Vocabulary:
        counts = Counter(token for sequence in sequences for token in sequence)
        kept = sorted(token for token, count in counts.items() if count >= min_count)
        return cls((*kept, UNKNOWN))

    def normalize(self, token: str) -> str:
        return token if token in self.index else UNKNOWN

    def encode(self, sequence: list[str]) -> list[int]:
        return [self.index[self.normalize(token)] for token in sequence]
