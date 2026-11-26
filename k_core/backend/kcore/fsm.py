"""有限狀態機：把一條 AoV 演變過程呈現成「狀態（流程圖）＋轉移（修改）」。

一條演變過程是一條馬可夫鏈：每個狀態是一張 AoV（CVflow 案例裡就是一張流程圖），每次修改是一次轉移。
完整的流程圖很少重複出現，所以狀態可以切換三種粒度（granularity）：
- full：完整流程圖。節點類型、參數、連線都相同才算同一個狀態（不看節點代號）。
- structure：結構。節點類型與連線相同就算同一個狀態，參數不同也一樣；調參數因此成為停在原地的自迴圈。
- category：節點類別。只看流程圖用到哪幾類節點（依模型特徵規格的 activity_map；沒有模型時用原類型）。

轉移機率＝這條演變過程裡，從同一個狀態出去的修改中，走這條轉移的比例。

兩種範圍：
- 單一演變過程（StateMachineService.episode）：一位解題者解一題的過程。
- 同一題的所有演變過程（StateMachineService.task）：work_order_id 相同的演變過程合併，
  轉移機率改用所有演變過程累計；每個狀態記下最早走到它的演變過程（lane）與當時是第幾版（column），供泳道圖排版。

有可用的模型時，另外標示模型內部的看法（只用於呈現，不影響其他功能）：
- 每次修改由哪個隱藏狀態產生：Viterbi 路徑，以及用整條序列推算的後驗機率。
- 隱藏狀態之間的轉移機率：第一次修改用初始機率，之後用轉移矩陣。
- 模型當時對這次修改的預測：只用前面的修改，預測這次修改類型的機率與排名（模型的預設預測方法）。
一個流程圖狀態被多次經過時，它的隱藏狀態依「產生這一版的修改」統計；第 0 版沒有修改，不標示。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sqlalchemy import select

from .database import EpisodeRecord, WorkflowSnapshotRecord
from .encoding import UNKNOWN, FeatureSchema, event_token
from .hmm.hybrid import DEFAULT_PREDICTOR
from .hmm.model import HiddenMarkovModel
from .service import KCoreError, KCoreService

GRANULARITIES = ("full", "structure", "category")
# 狀態摘要裡省略的通用節點：讀圖、計數，以及只負責輸出顯示的疊圖、畫框（每張流程圖都有，放進標籤只會讓標籤變長）
GENERIC_PREFIXES = ("io.", "cv.draw_")
GENERIC_TYPES = {"flow.count", "flow.overlay_detections", "flow.overlay_keypoints", "cv.overlay_mask", "cv.put_text"}


def _node_signature(node: dict[str, Any], granularity: str) -> str:
    if granularity == "full":
        return json.dumps([node["activity_type"], node.get("parameters") or {}], sort_keys=True, ensure_ascii=False)
    return node["activity_type"]


def state_key(nodes: list[dict], edges: list[dict], granularity: str, schema: FeatureSchema | None = None) -> str:
    """同一個粒度下，key 相同就是同一個狀態。不依賴節點代號，所以改名不影響。"""
    if granularity == "category":
        return json.dumps(sorted({_category(n["activity_type"], schema) for n in nodes}))
    signature = {n["node_id"]: _node_signature(n, granularity) for n in nodes}
    node_part = sorted(signature.values())
    edge_part = sorted([signature.get(e["source"], "?"), signature.get(e["target"], "?")] for e in edges)
    return json.dumps([node_part, edge_part], ensure_ascii=False)


def _category(activity_type: str, schema: FeatureSchema | None) -> str:
    return schema.map_activity(activity_type) if schema else activity_type


def _topological(nodes: list[dict], edges: list[dict]) -> list[dict]:
    """依連線排出前後順序（Kahn）；有環或孤立節點時，剩下的依原順序接在後面。"""
    by_id = {n["node_id"]: n for n in nodes}
    indegree = {n["node_id"]: 0 for n in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["source"] in by_id and e["target"] in by_id:
            indegree[e["target"]] += 1
            children[e["source"]].append(e["target"])
    queue = [n["node_id"] for n in nodes if indegree[n["node_id"]] == 0]
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    order += [n["node_id"] for n in nodes if n["node_id"] not in order]
    return [by_id[i] for i in order]


def state_summary(nodes: list[dict], edges: list[dict], granularity: str,
                  schema: FeatureSchema | None = None) -> list[str]:
    """畫面上的狀態摘要：節點類別，或依流程順序排列、去掉通用節點後的節點類型。"""
    if granularity == "category":
        ordered = [_category(n["activity_type"], schema) for n in _topological(nodes, edges)]
        return list(dict.fromkeys(ordered))
    types = [n["activity_type"] for n in _topological(nodes, edges)]
    key = [t for t in types if not t.startswith(GENERIC_PREFIXES) and t not in GENERIC_TYPES]
    return list(dict.fromkeys(key or types))


def build_state_machine(versions: list[tuple[str, list[dict], list[dict]]], events: list[dict[str, Any]],
                        granularity: str, schema: FeatureSchema | None = None,
                        model_steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """versions：第 0 版到最後一版（版本代號, 節點, 連線）；events[t] 把第 t 版改成第 t+1 版。

    model_steps[t]（可省略）：第 t 次修改的模型資訊（隱藏狀態、後驗、轉移機率、預測機率與排名）。
    """
    if granularity not in GRANULARITIES:
        raise KCoreError("invalid_granularity", f"granularity must be one of {list(GRANULARITIES)}", 422)
    if len(versions) != len(events) + 1:
        raise KCoreError("invalid_episode", "versions must be one more than events", 422)
    keys = [state_key(nodes, edges, granularity, schema) for _, nodes, edges in versions]
    ids: dict[str, str] = {}
    for key in keys:  # 狀態依第一次出現的順序編號
        ids.setdefault(key, f"S{len(ids) + 1}")
    path = [ids[key] for key in keys]

    states: dict[str, dict[str, Any]] = {}
    for index, (version_id, nodes, edges) in enumerate(versions):
        state = states.setdefault(path[index], {
            "id": path[index], "order": int(path[index][1:]), "summary": state_summary(nodes, edges, granularity, schema),
            "n_nodes": len(nodes), "visits": [], "version_ids": [], "hidden_states": Counter(),
        })
        state["visits"].append(index)
        state["version_ids"].append(version_id)
        if index > 0 and model_steps:
            state["hidden_states"][model_steps[index - 1]["hidden_state"]] += 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t, event in enumerate(events):
        item = {"step": t + 1, "from_version": t, "to_version": t + 1, **event}
        if model_steps:
            item.update(model_steps[t])
        grouped[(path[t], path[t + 1])].append(item)
    outgoing = Counter(path[t] for t in range(len(events)))
    transitions = []
    for number, ((source, target), items) in enumerate(sorted(grouped.items(), key=lambda kv: kv[1][0]["step"]), 1):
        hidden = Counter(i["hidden_state"] for i in items if i.get("hidden_state"))
        transitions.append({
            "id": f"T{number}", "source": source, "target": target, "count": len(items),
            "probability": len(items) / outgoing[source], "self_loop": source == target,
            "operations": dict(Counter(i["operation_type"] for i in items)),
            "hidden_states": dict(hidden), "events": items,
        })

    revisits = sum(1 for t in range(1, len(path)) if path[t] != path[t - 1] and path[t] in path[:t])
    for state in states.values():
        state["hidden_states"] = dict(state["hidden_states"])
        state["is_start"] = state["id"] == path[0]
        state["is_final"] = state["id"] == path[-1]
    return {
        "granularity": granularity, "path": path,
        "states": sorted(states.values(), key=lambda s: s["order"]),
        "transitions": transitions,
        "versions": [{"index": i, "version_id": v, "state": path[i], "nodes": nodes, "edges": edges}
                     for i, (v, nodes, edges) in enumerate(versions)],
        "stats": {"versions": len(versions), "states": len(states), "transitions": len(transitions),
                  "self_loop_steps": sum(1 for t in range(len(events)) if path[t] == path[t + 1]),
                  "revisits": revisits},
    }


def model_annotations(model: HiddenMarkovModel, tokens: list[str], predict) -> list[dict[str, Any]]:
    """每次修改的模型資訊：Viterbi 隱藏狀態、後驗、隱藏狀態轉移機率、當時的預測機率與排名。"""
    if not tokens:
        return []
    _, path = model.viterbi(tokens)
    posterior = model.posterior(tokens)
    index = {state: i for i, state in enumerate(model.states)}
    steps = []
    for t, token in enumerate(tokens):
        current = index[path[t]]
        if t == 0:
            hidden_transition = {"from": None, "to": path[t], "probability": float(model.initial[current])}
        else:
            hidden_transition = {"from": path[t - 1], "to": path[t],
                                 "probability": float(model.transition[index[path[t - 1]], current])}
        distribution = predict(tokens[:t])
        actual = model.observation_id(token)
        steps.append({
            "hidden_state": path[t],
            "posterior": dict(zip(model.states, np.round(posterior[t], 6).tolist(), strict=True)),
            "hidden_transition": hidden_transition,
            "model_probability": float(distribution[actual]),
            "model_rank": int((distribution > distribution[actual]).sum()) + 1,
        })
    return steps


def build_task_state_machine(episodes: list[dict[str, Any]], granularity: str,
                             schema: FeatureSchema | None = None) -> dict[str, Any]:
    """同一題的多條演變過程（不同解題者）合併成一個狀態機。

    episodes 依泳道順序排列，每個元素：{"episode_id", "teacher", "scope", "run", "versions", "events", "model_steps"}，
    versions、events、model_steps 的意思與 build_state_machine 相同。
    狀態依「最早在第幾版、被哪一條演變過程走到」編號（同一版時，泳道在前的優先），並記下
    lane（最早走到它的演變過程）與 column（當時是第幾版）。同一條演變過程在同一版只會在一個狀態，
    所以 (lane, column) 不會重複，前端可以直接排成泳道圖：每位解題者一列，別人走到同一個狀態時就是路線交會。
    轉移機率＝所有演變過程中，從同一個狀態出去的修改裡走這條轉移的比例。
    """
    if granularity not in GRANULARITIES:
        raise KCoreError("invalid_granularity", f"granularity must be one of {list(GRANULARITIES)}", 422)
    if not episodes:
        raise KCoreError("task_not_found", "no episodes for this task", 404)
    keyed: list[list[str]] = []
    first: dict[str, tuple[int, int]] = {}
    for lane, episode in enumerate(episodes):
        if len(episode["versions"]) != len(episode["events"]) + 1:
            raise KCoreError("invalid_episode", f"{episode['episode_id']}: versions must be one more than events", 422)
        keys = [state_key(nodes, edges, granularity, schema) for _, nodes, edges in episode["versions"]]
        keyed.append(keys)
        for index, key in enumerate(keys):
            if key not in first or (index, lane) < first[key]:
                first[key] = (index, lane)
    order = sorted(first, key=lambda key: first[key])
    ids = {key: f"S{number}" for number, key in enumerate(order, start=1)}

    states: dict[str, dict[str, Any]] = {}
    for number, key in enumerate(order, start=1):
        column, lane = first[key]
        _, nodes, edges = episodes[lane]["versions"][column]
        states[ids[key]] = {
            "id": ids[key], "order": number, "summary": state_summary(nodes, edges, granularity, schema),
            "n_nodes": len(nodes), "lane": lane, "column": column, "visits": [], "episodes": [],
            "hidden_states": Counter(), "starts": [], "finals": [],
        }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    outgoing: Counter = Counter()
    self_loops = revisits = 0
    lanes = []
    for lane, (episode, keys) in enumerate(zip(episodes, keyed, strict=True)):
        eid = episode["episode_id"]
        steps = episode.get("model_steps")
        path = [ids[key] for key in keys]
        for index, state_id in enumerate(path):
            state = states[state_id]
            state["visits"].append({"episode_id": eid, "version": index})
            if eid not in state["episodes"]:
                state["episodes"].append(eid)
            if index > 0 and steps:
                state["hidden_states"][steps[index - 1]["hidden_state"]] += 1
        states[path[0]]["starts"].append(eid)
        states[path[-1]]["finals"].append(eid)
        for t, event in enumerate(episode["events"]):
            item = {"episode_id": eid, "lane": lane, "step": t + 1, "from_version": t, "to_version": t + 1, **event}
            if steps:
                item.update(steps[t])
            grouped[(path[t], path[t + 1])].append(item)
            outgoing[path[t]] += 1
        self_loops += sum(1 for t in range(len(path) - 1) if path[t] == path[t + 1])
        revisits += sum(1 for t in range(1, len(path)) if path[t] != path[t - 1] and path[t] in path[:t])
        lanes.append({
            "episode_id": eid, "lane": lane, "teacher": episode.get("teacher"), "scope": episode.get("scope"),
            "run": episode.get("run"), "path": path, "events": len(episode["events"]),
            "versions": [{"index": i, "version_id": v, "state": path[i], "nodes": nodes, "edges": edges}
                         for i, (v, nodes, edges) in enumerate(episode["versions"])],
        })

    first_seen = {pair: min((item["step"], item["lane"]) for item in items) for pair, items in grouped.items()}
    transitions = []
    for number, ((source, target), items) in enumerate(sorted(grouped.items(), key=lambda kv: first_seen[kv[0]]), 1):
        hidden = Counter(item["hidden_state"] for item in items if item.get("hidden_state"))
        transitions.append({
            "id": f"T{number}", "source": source, "target": target, "count": len(items),
            "probability": len(items) / outgoing[source], "self_loop": source == target,
            "operations": dict(Counter(item["operation_type"] for item in items)),
            "hidden_states": dict(hidden), "episodes": list(dict.fromkeys(item["episode_id"] for item in items)),
            "events": items,
        })
    for state in states.values():
        state["hidden_states"] = dict(state["hidden_states"])
        state["is_start"] = bool(state["starts"])
        state["is_final"] = bool(state["finals"])
    return {
        "granularity": granularity, "episodes": lanes,
        "states": sorted(states.values(), key=lambda s: s["order"]),
        "transitions": transitions,
        "stats": {"episodes": len(episodes), "versions": sum(len(e["versions"]) for e in episodes),
                  "states": len(states), "transitions": len(transitions),
                  "shared_states": sum(1 for s in states.values() if len(s["episodes"]) > 1),
                  "self_loop_steps": self_loops, "revisits": revisits},
    }


SCOPE_ORDER = {"tasks": 0, "retake": 1}  # CVflow：第一次考試在前，補考在後；其他案例沒有這個欄位


class StateMachineService:
    def __init__(self, core: KCoreService) -> None:
        self.core = core

    def _model_context(self, session, case_id: str, model_version_id: str) -> dict[str, Any] | None:
        """有可用的模型時，回傳模型、特徵規格與預設預測方法；案例沒有使用中的模型時回傳 None。"""
        try:
            record = self.core._model(session, case_id, model_version_id)
        except KCoreError as error:
            if error.code != "no_active_model":
                raise
            return None
        model = HiddenMarkovModel.from_dict(record.parameters)
        predictors = self.core._predictors(record, model)
        method = DEFAULT_PREDICTOR if DEFAULT_PREDICTOR in predictors else "hmm"
        return {"record": record, "model": model, "schema": FeatureSchema.from_dict(record.feature_schema),
                "predict": predictors[method], "method": method}

    @staticmethod
    def _hmm_info(context: dict[str, Any] | None) -> dict[str, Any] | None:
        if context is None:
            return None
        model = context["model"]
        return {
            "model_version_id": context["record"].model_version_id, "states": list(model.states),
            "method": context["method"],
            "top_observations": {
                state: [{"observation": model.observations[i], "probability": float(model.emission[row, i])}
                        for i in np.argsort(-model.emission[row])[:3]]
                for row, state in enumerate(model.states)
            },
        }

    def _load(self, session, episode: EpisodeRecord, context: dict[str, Any] | None) -> dict[str, Any]:
        """一條演變過程的各版流程圖、修改，以及（有模型時）每次修改的模型資訊。"""
        core = self.core
        records = core._events(session, episode.episode_id)
        replay = core._replay(session, episode, records)
        if not replay.ok:
            raise KCoreError("invalid_episode", f"{episode.episode_id}: events cannot be replayed", 422, replay.problems)
        initial = session.scalar(select(WorkflowSnapshotRecord).where(
            WorkflowSnapshotRecord.episode_id == episode.episode_id,
            WorkflowSnapshotRecord.version_id == episode.initial_version_id))
        events = [{"event_id": e.event_id, "operation_type": e.operation_type,
                   "activity_type": replay.features[e.event_id].get("activity_type"),
                   "change": e.change, "reason": e.reason} for e in records]
        model_steps = None
        if context is not None:
            raw = [event_token(e.operation_type, replay.features[e.event_id], context["schema"]) for e in records]
            tokens = [t if t in context["model"].observations else UNKNOWN for t in raw]
            model_steps = model_annotations(context["model"], tokens, context["predict"])
            for step, token in zip(model_steps, raw, strict=True):
                step["token"] = token
        info = episode.initial_context or {}
        return {"episode_id": episode.episode_id, "teacher": info.get("teacher"), "scope": info.get("scope"),
                "run": info.get("run"), "request": info.get("request"),
                "versions": [(initial.version_id, initial.nodes, initial.edges), *replay.snapshots],
                "events": events, "model_steps": model_steps}

    def episode(self, case_id: str, episode_id: str, granularity: str = "structure",
                model_version_id: str = "active") -> dict[str, Any]:
        if granularity not in GRANULARITIES:
            raise KCoreError("invalid_granularity", f"granularity must be one of {list(GRANULARITIES)}", 422)
        with self.core.session_factory() as session:
            episode = self.core._episode(session, case_id, episode_id)
            context = self._model_context(session, case_id, model_version_id)
            data = self._load(session, episode, context)
            result = build_state_machine(data["versions"], data["events"], granularity,
                                         context["schema"] if context else None, data["model_steps"])
            result.update({
                "case_id": case_id, "episode_id": episode_id, "work_order_id": episode.work_order_id,
                "teacher": data["teacher"], "request": data["request"], "hmm": self._hmm_info(context),
            })
            return result

    def task(self, case_id: str, task_id: str, granularity: str = "structure",
             model_version_id: str = "active") -> dict[str, Any]:
        """同一題（work_order_id 相同）所有演變過程合併的狀態機。"""
        if granularity not in GRANULARITIES:
            raise KCoreError("invalid_granularity", f"granularity must be one of {list(GRANULARITIES)}", 422)
        with self.core.session_factory() as session:
            self.core._case(session, case_id)
            episodes = list(session.scalars(select(EpisodeRecord).where(
                EpisodeRecord.case_id == case_id, EpisodeRecord.work_order_id == task_id)))
            if not episodes:
                raise KCoreError("task_not_found", f"no episodes with work_order_id {task_id} in case {case_id}", 404)

            def lane_key(e: EpisodeRecord):
                info = e.initial_context or {}
                return (SCOPE_ORDER.get(info.get("scope"), 9), str(info.get("teacher") or ""), e.started_at, e.episode_id)

            context = self._model_context(session, case_id, model_version_id)
            data = [self._load(session, e, context) for e in sorted(episodes, key=lane_key)]
            data = [d for d in data if d["events"]]
            result = build_task_state_machine(data, granularity, context["schema"] if context else None)
            result.update({"case_id": case_id, "task_id": task_id, "request": data[0]["request"] if data else None,
                           "hmm": self._hmm_info(context)})
            return result
