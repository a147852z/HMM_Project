"""AoV 生成協助（規格書階段 4）的離線版本：候選產生 → HMM 評分 → 結構驗證 → 排序。

尚未接上 AoV 平台，因此以「保留的測試序列」驗證：在序列的每一步，只用當時已知的流程圖與修改前綴
提出候選修改，再與實際發生的下一次修改比對。實際答案只用於比對，不參與候選產生與評分。

候選產生（修改樣式庫）：
- 樣式 = (修改類型, 主要受影響的活動類型)，只從模型的「訓練序列」統計，測試序列不會進入樣式庫。
- 每個樣式在目前流程圖上具體化成一張建議流程圖；做不到的樣式（例如流程中沒有該節點卻要調它的參數）會被排除。

評分：score = P(觀測符號 | 修改前綴) × P(樣式 | 觀測符號，且在目前流程上可行)
- 第一項來自模型版本保存的預測方法（預設 HMM 與上一次修改各半平均），所以 HMM 實際決定排序。
- 第二項是樣式在訓練資料中的相對次數，用來在同一個觀測符號內排序。
- 分數是模型機率，不是成功率。

對照排法（使用相同的候選與驗證）：frequency = 只看樣式在訓練資料的次數；bigram = 只看上一次修改 × 第二項。

實驗選項 assist_result（預設關閉）：納入「上一版的執行結果」（通過／未通過／無法執行）。
- 這個結果在提出下一次修改之前就已經知道，所以不算偷看答案。
- 第一項乘上 P(符號 | 結果) / P(符號)；第二項改用「相同結果下」的樣式次數（次數少時退回整體比例）。
- 序列沒有記錄執行結果時，等同於一般的 assist。
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from .aov import derive_change_features, diff_workflows, is_empty_change, validate_workflow
from .database import EpisodeRecord, TrainingRunRecord, WorkflowSnapshotRecord
from .encoding import UNKNOWN, FeatureSchema, event_token
from .evaluation import FrequencyBaselines
from .hmm.hybrid import DEFAULT_PREDICTOR
from .hmm.model import HiddenMarkovModel
from .service import KCoreError, KCoreService

ADD_OPS = {"add_step", "insert_activity"}
REMOVE_OPS = {"remove_step", "remove_activity"}
TUNE_OPS = {"param_tune", "update_parameters"}
REPLACE_OPS = {"replace_step"}
SWITCH_OPS = {"strategy_switch"}
SUPPORTED_OPS = ADD_OPS | REMOVE_OPS | TUNE_OPS | REPLACE_OPS | SWITCH_OPS
PRIOR_SMOOTHING = 0.5
RESULT_BACKOFF = 2.0  # 相同結果下的次數不足時，退回整體比例的強度（固定值）
Key = tuple[str, str]


@dataclass
class PatternLibrary:
    keys: Counter = field(default_factory=Counter)                       # (修改類型, 活動類型) -> 次數
    param_keys: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    predecessors: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    replaced_by: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))  # 新類型 -> 被取代的舊類型
    templates: list[dict[str, Any]] = field(default_factory=list)        # 換策略後的整張流程（範本）
    train_tokens: list[list[str]] = field(default_factory=list)
    train_episode_ids: set[str] = field(default_factory=set)
    test_episode_ids: list[str] = field(default_factory=list)
    unsupported_events: int = 0
    # 實驗選項（上一版執行結果）用的統計
    keys_by_result: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    tokens_by_result: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    token_count: Counter = field(default_factory=Counter)


def result_label(result: Any) -> str | None:
    """一個流程版本的執行結果：pass／fail／error；沒有記錄時為 None。"""
    if not isinstance(result, dict) or (result.get("executable") is None and result.get("passed") is None):
        return None
    if result.get("executable") is False:
        return "error"
    return "pass" if result.get("passed") is True else "fail"


def previous_results(episode: Any, events: list[Any]) -> list[str | None]:
    """labels[t] = 第 t 次修改之前那個版本的執行結果；長度為 len(events) + 1（最後一個是最終版本的結果）。"""
    labels = [result_label((episode.provenance or {}).get("first_version_result"))]
    labels += [result_label((event.attributes or {}).get("result")) for event in events]
    return labels


def _types(nodes: list[dict]) -> list[str]:
    return [n["activity_type"] for n in nodes]


def _fresh_id(nodes: list[dict], base: str = "proposal") -> str:
    used = {n["node_id"] for n in nodes}
    i = 1
    while f"{base}-{i}" in used:
        i += 1
    return f"{base}-{i}"


class AssistService:
    def __init__(self, core: KCoreService) -> None:
        self.core = core
        self._libraries: dict[str, PatternLibrary] = {}

    # ------------------------------------------------------------------ 樣式庫
    def _snapshots(self, session, episode_id: str) -> list[WorkflowSnapshotRecord]:
        return list(session.scalars(select(WorkflowSnapshotRecord).where(
            WorkflowSnapshotRecord.episode_id == episode_id).order_by(WorkflowSnapshotRecord.id)))

    def _library(self, session, record, model: HiddenMarkovModel, schema: FeatureSchema) -> PatternLibrary:
        cached = self._libraries.get(record.model_version_id)
        if cached is not None:
            return cached
        run = session.get(TrainingRunRecord, record.training_run_id)
        dataset = (run.dataset if run else None) or {}
        library = PatternLibrary(train_episode_ids=set(dataset.get("train_episodes") or []),
                                 test_episode_ids=list(dataset.get("test_episodes") or []))
        if not library.train_episode_ids:
            raise KCoreError("no_training_split", "model has no recorded training episodes", 422)
        template_counts: Counter = Counter()
        template_graphs: dict[tuple, dict[str, Any]] = {}
        for episode, events in self.core._training_sequences(session, record.case_id):
            if episode.episode_id not in library.train_episode_ids:
                continue
            snapshots = self._snapshots(session, episode.episode_id)
            tokens = []
            previous = previous_results(episode, events)
            for snapshot in snapshots:
                type_of = {n["node_id"]: n["activity_type"] for n in snapshot.nodes}
                for edge in snapshot.edges:
                    library.predecessors[type_of[edge["target"]]][type_of[edge["source"]]] += 1
            for index, event in enumerate(events):
                features = event.features or {}
                token = event_token(event.operation_type, features, schema)
                tokens.append(token if token in model.observations else UNKNOWN)
                library.token_count[tokens[-1]] += 1
                if previous[index] is not None:
                    library.tokens_by_result[previous[index]][tokens[-1]] += 1
                op, primary = event.operation_type, str(features.get("activity_type", "none"))
                if op not in SUPPORTED_OPS:
                    library.unsupported_events += 1
                    continue
                library.keys[(op, primary)] += 1
                if previous[index] is not None:
                    library.keys_by_result[previous[index]][(op, primary)] += 1
                before, after = snapshots[index], snapshots[index + 1]
                if op in TUNE_OPS:
                    for update in event.change.get("updated_node_attributes", []):
                        library.param_keys[primary].update((update.get("parameters_set") or {}).keys())
                elif op in REPLACE_OPS:
                    before_types = {n["node_id"]: n["activity_type"] for n in before.nodes}
                    for removed in event.change.get("removed_nodes", []):
                        library.replaced_by[primary][before_types.get(removed, "none")] += 1
                elif op in SWITCH_OPS:
                    signature = (primary, *sorted(_types(after.nodes)))
                    template_counts[signature] += 1
                    template_graphs.setdefault(signature, {"nodes": after.nodes, "edges": after.edges})
            library.train_tokens.append(tokens)
        # 每一種換策略樣式（依主要新增的活動類型）保留最常見的一張流程作為範本
        seen_types: set[str] = set()
        for sig, _ in template_counts.most_common():
            if sig[0] in seen_types:
                continue
            seen_types.add(sig[0])
            library.templates.append({**template_graphs[sig], "signature": list(sig[1:]), "primary": sig[0],
                                      "count": sum(c for s, c in template_counts.items() if s[0] == sig[0])})
        self._libraries[record.model_version_id] = library
        return library

    # ------------------------------------------------------------------ 候選具體化
    def _materialize(self, op: str, node_type: str, nodes: list[dict], edges: list[dict],
                     library: PatternLibrary) -> tuple[list[dict], list[dict], dict[str, Any]] | None:
        """把一個樣式套到目前流程上；回傳 (新節點, 新連線, 說明)，做不到時回傳 None。"""
        present = [n for n in nodes if n["activity_type"] == node_type]
        if op in TUNE_OPS:
            if not present:
                return None
            target = present[-1]
            suggested = [k for k, _ in library.param_keys[node_type].most_common(3)]
            return nodes, edges, {"target_node": target["node_id"], "suggested_parameters": suggested}
        if op in REMOVE_OPS:
            if not present or len(nodes) <= 2:
                return None
            target = present[-1]["node_id"]
            incoming = [e for e in edges if e["target"] == target]
            outgoing = [e for e in edges if e["source"] == target]
            kept = [e for e in edges if target not in (e["source"], e["target"])]
            existing = {(e["source"], e["target"]) for e in kept}
            bridges = [{"source": a["source"], "target": b["target"]} for a in incoming for b in outgoing
                       if (a["source"], b["target"]) not in existing and a["source"] != b["target"]]
            return [n for n in nodes if n["node_id"] != target], kept + bridges, {"target_node": target}
        if op in ADD_OPS:
            new_id = _fresh_id(nodes)
            new_node = {"node_id": new_id, "activity_type": node_type, "parameters": {}}
            by_type = defaultdict(list)
            for n in nodes:
                by_type[n["activity_type"]].append(n["node_id"])
            anchor = next((by_type[t][-1] for t, _ in library.predecessors[node_type].most_common() if by_type.get(t)),
                          nodes[-1]["node_id"] if nodes else None)
            if anchor is None:
                return None
            new_edges = [dict(e) for e in edges]
            out = next((e for e in new_edges if e["source"] == anchor), None)
            if out is not None:  # 插在 anchor 與它的下一個節點之間
                new_edges.remove(out)
                new_edges += [{"source": anchor, "target": new_id}, {"source": new_id, "target": out["target"]}]
            else:
                new_edges.append({"source": anchor, "target": new_id})
            return [*nodes, new_node], new_edges, {"after_node": anchor}
        if op in REPLACE_OPS:
            types_present = set(_types(nodes))
            old_type = next((t for t, _ in library.replaced_by[node_type].most_common() if t in types_present), None)
            if old_type is None or old_type == node_type:
                return None
            old_id = [n for n in nodes if n["activity_type"] == old_type][-1]["node_id"]
            new_id = _fresh_id(nodes)
            rename = lambda x: new_id if x == old_id else x  # noqa: E731
            new_nodes = [{"node_id": new_id, "activity_type": node_type, "parameters": {}} if n["node_id"] == old_id else n
                         for n in nodes]
            return new_nodes, [{**e, "source": rename(e["source"]), "target": rename(e["target"])} for e in edges], \
                {"replaced_node": old_id, "replaced_type": old_type}
        return None

    def _candidates(self, nodes: list[dict], edges: list[dict], library: PatternLibrary) -> tuple[list[dict], dict]:
        """產生在目前流程上可行、且通過 DAG 驗證的候選；同一個樣式只保留一個。"""
        stats = {"patterns": 0, "infeasible": 0, "invalid": 0}
        found: dict[Key, dict[str, Any]] = {}

        def add(op: str, new_nodes, new_edges, detail, weight: float) -> None:
            if validate_workflow(new_nodes, new_edges):
                stats["invalid"] += 1
                return
            change = diff_workflows(nodes, edges, new_nodes, new_edges)
            if op in TUNE_OPS:
                realized = next(n["activity_type"] for n in nodes if n["node_id"] == detail["target_node"])
            elif is_empty_change(change):
                stats["infeasible"] += 1
                return
            else:
                realized = derive_change_features(nodes, new_nodes, change)["activity_type"]
            key = (op, realized)
            if key not in found or weight > found[key]["weight"]:
                found[key] = {"key": key, "operation_type": op, "activity_type": realized, "weight": weight,
                              "nodes": new_nodes, "edges": new_edges, "change": change, "detail": detail}

        for (op, node_type), count in library.keys.items():
            if op in SWITCH_OPS:
                continue
            stats["patterns"] += 1
            result = self._materialize(op, node_type, nodes, edges, library)
            if result is None:
                stats["infeasible"] += 1
                continue
            add(op, result[0], result[1], result[2], float(count))
        switch_op = next((op for op, _ in library.keys if op in SWITCH_OPS), None)
        current = sorted(_types(nodes))
        for template in library.templates:
            stats["patterns"] += 1
            if template["signature"] == current or switch_op is None:
                stats["infeasible"] += 1
                continue
            # 範本中與目前流程類型相同的節點沿用目前的代號（視為同一個節點），其餘才算新增，
            # 與匯入資料時「節點改名對齊」的規則一致。
            available: dict[str, list[str]] = defaultdict(list)
            for n in nodes:
                available[n["activity_type"]].append(n["node_id"])
            mapping = {}
            for n in template["nodes"]:
                pool = available[n["activity_type"]]
                mapping[n["node_id"]] = pool.pop(0) if pool else f"tpl-{n['node_id']}"
            t_nodes = [{**copy.deepcopy(n), "node_id": mapping[n["node_id"]]} for n in template["nodes"]]
            t_edges = [{"source": mapping[e["source"]], "target": mapping[e["target"]]} for e in template["edges"]]
            add(switch_op, t_nodes, t_edges, {"template_types": template["signature"]}, float(template["count"]))
        return list(found.values()), stats

    # ------------------------------------------------------------------ 評分
    @staticmethod
    def _rank(candidates: list[dict], token_probability: dict[str, float] | None,
              weights: dict[Key, float] | None = None) -> list[dict]:
        """token_probability 為 None 時只用樣式次數（frequency 對照）。weights 可取代預設的樣式次數。"""
        weight_of = (lambda c: weights[c["key"]]) if weights is not None else (lambda c: c["weight"])
        by_token: dict[str, float] = defaultdict(float)
        for c in candidates:
            by_token[c["token"]] += weight_of(c) + PRIOR_SMOOTHING
        total = sum(by_token.values()) or 1.0
        ranked = []
        for c in candidates:
            prior = (weight_of(c) + PRIOR_SMOOTHING) / by_token[c["token"]]
            if token_probability is None:
                score, token_p = (weight_of(c) + PRIOR_SMOOTHING) / total, None
            else:
                token_p = token_probability.get(c["token"], 0.0)
                score = token_p * prior
            ranked.append({**c, "score": score, "token_probability": token_p, "pattern_prior": prior})
        ranked.sort(key=lambda c: (-c["score"], -c["weight"], c["key"]))
        return ranked

    def _context(self, session, case_id: str, model_version_id: str):
        record = self.core._model(session, case_id, model_version_id)
        schema = FeatureSchema.from_dict(record.feature_schema)
        model = HiddenMarkovModel.from_dict(record.parameters)
        library = self._library(session, record, model, schema)
        predictors = self.core._predictors(record, model)
        method = DEFAULT_PREDICTOR if DEFAULT_PREDICTOR in predictors else "hmm"
        bigram = FrequencyBaselines(library.train_tokens, model.observations, 0.1).predict_bigram
        return record, schema, model, library, predictors[method], method, bigram

    def _step(self, nodes, edges, prefix_tokens, schema, model, library, predict, bigram,
              previous_result: str | None = None) -> dict[str, Any]:
        candidates, stats = self._candidates(nodes, edges, library)
        for c in candidates:
            token = event_token(c["operation_type"], {"activity_type": c["activity_type"]}, schema)
            c["token"] = token if token in model.observations else UNKNOWN
        as_map = lambda dist: dict(zip(model.observations, (float(p) for p in dist), strict=True))  # noqa: E731
        token_p = as_map(predict(prefix_tokens))
        assist = self._rank(candidates, token_p)
        # 實驗選項：以上一版執行結果調整。沒有該結果的訓練統計時，等同一般的 assist。
        context_tokens = library.tokens_by_result.get(previous_result) if previous_result else None
        if context_tokens:
            n_all, n_ctx = sum(library.token_count.values()), sum(context_tokens.values())
            adjusted = {}
            for token, p in token_p.items():
                base = (library.token_count[token] + PRIOR_SMOOTHING) / (n_all + PRIOR_SMOOTHING * len(token_p))
                adjusted[token] = p * ((context_tokens[token] + RESULT_BACKOFF * base) / (n_ctx + RESULT_BACKOFF)) / base
            norm = sum(adjusted.values()) or 1.0
            adjusted = {token: p / norm for token, p in adjusted.items()}
            context_keys, total_keys = library.keys_by_result[previous_result], max(1, sum(library.keys.values()))
            weights = {c["key"]: context_keys[c["key"]] + RESULT_BACKOFF * library.keys[c["key"]] / total_keys
                       for c in candidates}
            assist_result = self._rank(candidates, adjusted, weights)
        else:
            assist_result = assist
        return {
            "stats": stats,
            "assist": assist,
            "assist_result": assist_result,
            "bigram": self._rank(candidates, as_map(bigram(prefix_tokens))),
            "frequency": self._rank(candidates, None),
        }

    @staticmethod
    def _rank_of(ranked: list[dict], key: Key) -> int | None:
        return next((i + 1 for i, c in enumerate(ranked) if c["key"] == key), None)

    def assist(self, case_id: str, episode_id: str, prefix_length: int, top_k: int = 10,
               model_version_id: str = "active", use_previous_result: bool = False) -> dict[str, Any]:
        """對某條序列的第 prefix_length 步提出候選修改。只使用該步之前的流程與修改。"""
        with self.core.session_factory() as session:
            record, schema, model, library, predict, method, bigram = self._context(session, case_id, model_version_id)
            episode = self.core._episode(session, case_id, episode_id)
            if episode.status != "finalized":
                raise KCoreError("episode_not_finalized", "offline assist needs a finalized episode", 422)
            events = self.core._events(session, episode_id)
            snapshots = self._snapshots(session, episode_id)
            if not 0 <= prefix_length <= len(events):
                raise KCoreError("invalid_prefix", f"prefix_length must be between 0 and {len(events)}", 422)
            current = snapshots[prefix_length]
            tokens = [event_token(e.operation_type, e.features or {}, schema) for e in events[:prefix_length]]
            tokens = [t if t in model.observations else UNKNOWN for t in tokens]
            previous = previous_results(episode, events)[prefix_length]
            step = self._step(current.nodes, current.edges, tokens, schema, model, library, predict, bigram, previous)
            ranked = step["assist_result" if use_previous_result else "assist"]
            result: dict[str, Any] = {
                "case_id": case_id, "episode_id": episode_id, "prefix_length": prefix_length,
                "total_events": len(events), "model_version_id": record.model_version_id,
                "method": f"pattern_library x {method}" + (" x previous_result" if use_previous_result else ""),
                "use_previous_result": use_previous_result, "previous_result": previous,
                "status": "ok" if ranked else "no_valid_candidates",
                "used_hmm": True, "in_training_set": episode_id in library.train_episode_ids,
                "current_workflow": {"version_id": current.version_id, "nodes": current.nodes, "edges": current.edges},
                "candidate_stats": {**step["stats"], "valid_candidates": len(ranked)},
                "candidates": [{
                    "rank": i + 1, "operation_type": c["operation_type"], "activity_type": c["activity_type"],
                    "token": c["token"], "score": c["score"], "token_probability": c["token_probability"],
                    "pattern_prior": c["pattern_prior"], "training_count": int(c["weight"]), "detail": c["detail"],
                    "validation": {"valid": True, "problems": []},
                    "proposed_workflow": {"nodes": c["nodes"], "edges": c["edges"]}, "change": c["change"],
                } for i, c in enumerate(ranked[:top_k])],
                "actual": None,
            }
            if prefix_length < len(events):  # 答案比對：只在這裡讀取實際的下一次修改
                actual = events[prefix_length]
                key = (actual.operation_type, str((actual.features or {}).get("activity_type", "none")))
                result["actual"] = {
                    "operation_type": key[0], "activity_type": key[1], "event_id": actual.event_id,
                    "ranks": {name: self._rank_of(step[name], key)
                              for name in ("assist", "assist_result", "bigram", "frequency")},
                    "in_candidates": self._rank_of(ranked, key) is not None,
                }
            return result

    def evaluate(self, case_id: str, model_version_id: str = "active") -> dict[str, Any]:
        """在模型保留的測試序列上逐步評估：實際的下一次修改排在候選的第幾名。"""
        with self.core.session_factory() as session:
            record, schema, model, library, predict, method, bigram = self._context(session, case_id, model_version_id)
            names = ("assist", "assist_result", "bigram", "frequency")
            by_previous: dict[str, dict[str, int]] = defaultdict(lambda: {"steps": 0, "assist_hit3": 0,
                                                                          "assist_result_hit3": 0})
            totals = {n: {"hit1": 0, "hit3": 0, "hit5": 0, "rr": 0.0} for n in names}
            steps = covered = candidate_total = invalid = 0
            by_operation: dict[str, dict[str, int]] = defaultdict(lambda: {"steps": 0, "covered": 0, "hit3": 0})
            per_episode = []
            changed_steps = []
            for episode_id in library.test_episode_ids:
                events = self.core._events(session, episode_id)
                snapshots = self._snapshots(session, episode_id)
                tokens = [event_token(e.operation_type, e.features or {}, schema) for e in events]
                tokens = [t if t in model.observations else UNKNOWN for t in tokens]
                episode_hit3 = 0
                previous = previous_results(session.get(EpisodeRecord, episode_id), events)
                for t, event in enumerate(events):
                    step = self._step(snapshots[t].nodes, snapshots[t].edges, tokens[:t], schema, model, library,
                                      predict, bigram, previous[t])
                    key = (event.operation_type, str((event.features or {}).get("activity_type", "none")))
                    steps += 1
                    candidate_total += len(step["assist"])
                    invalid += step["stats"]["invalid"]
                    op_stats = by_operation[event.operation_type]
                    op_stats["steps"] += 1
                    ranks = {n: self._rank_of(step[n], key) for n in names}
                    if ranks["assist"] is not None:
                        covered += 1
                        op_stats["covered"] += 1
                    for n in names:
                        r = ranks[n]
                        if r is not None:
                            totals[n]["hit1"] += r == 1
                            totals[n]["hit3"] += r <= 3
                            totals[n]["hit5"] += r <= 5
                            totals[n]["rr"] += 1 / r
                    if ranks["assist"] is not None and ranks["assist"] <= 3:
                        episode_hit3 += 1
                        op_stats["hit3"] += 1
                    if ranks["assist"] != ranks["assist_result"]:
                        changed_steps.append({"episode_id": episode_id, "prefix_length": t,
                                              "previous_result": previous[t], "assist_rank": ranks["assist"],
                                              "assist_result_rank": ranks["assist_result"]})
                    prev_stats = by_previous[previous[t] or "unknown"]
                    prev_stats["steps"] += 1
                    for name in ("assist", "assist_result"):
                        prev_stats[f"{name}_hit3"] += ranks[name] is not None and ranks[name] <= 3
                per_episode.append({"episode_id": episode_id, "events": len(events),
                                    "assist_hit3": episode_hit3 / len(events) if events else None})
            n = steps or 1
            return {
                "case_id": case_id, "model_version_id": record.model_version_id,
                "method": f"pattern_library x {method}", "test_episodes": len(library.test_episode_ids),
                "train_episodes": len(library.train_episode_ids), "steps": steps,
                "coverage": covered / n, "mean_candidates": candidate_total / n, "invalid_candidates_dropped": invalid,
                "pattern_library": {"patterns": len(library.keys), "templates": len(library.templates),
                                    "unsupported_training_events": library.unsupported_events},
                "methods": {name: {"hit1": v["hit1"] / n, "hit3": v["hit3"] / n, "hit5": v["hit5"] / n,
                                   "mrr": v["rr"] / n} for name, v in totals.items()},
                "by_operation": {op: {**v, "coverage": v["covered"] / v["steps"], "assist_hit3": v["hit3"] / v["steps"]}
                                 for op, v in sorted(by_operation.items())},
                "by_previous_result": {label: {"steps": v["steps"], "assist_hit3": v["assist_hit3"] / v["steps"],
                                               "assist_result_hit3": v["assist_result_hit3"] / v["steps"]}
                                       for label, v in sorted(by_previous.items())},
                "changed_steps": changed_steps,
                "episodes": per_episode,
                "notes": "未命中包含兩種情況：實際修改不在候選中（coverage 的缺口），或排名在門檻之外。",
            }

    @staticmethod
    def validate(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
        problems = validate_workflow(nodes, edges)
        return {"valid": not problems, "problems": problems, "checks": ["node ids", "edge endpoints", "self-loops",
                                                                        "duplicate edges", "cycles (DAG)"]}
