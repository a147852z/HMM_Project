"""CVflow 轉接器：把「AI 代理人逐輪設計 cvflow 流程圖」的紀錄轉成 K 的演變序列。

資料來源：佳元實驗材料包（teachers/<模型>/{tasks,retake}/<題目>/）。
- 一條演變序列 = 一個模型解一題的過程；每個流程圖版本 g{N}.json、g{N}f{K}.json（同輪修正）依序排列。
- 一次修改事件 = 相鄰兩個版本的差異，由程式計算（不使用 LLM）。
- 修改原因：traj.json 第 N 輪的 hypothesis（修改者為 g{N} 寫的設計理由），標為 reason_source="actor"；
  上一輪的修正計畫 fix 另存為 planned_fix（實測常與實際做法不同）。
- 每個版本的執行結果（能否執行、是否通過）放在事件 attributes；整題成敗放在 provenance["outcome"]。
  這些都是結果資料，特徵編碼不會讀取，避免把答案洩漏進觀測。

轉換時的處理：
- 節點改名：先依代號比對；新出現的代號若與消失的代號類型相同，視為同一節點改名，沿用舊代號（原代號存於 source_id）。
- 多個 port 連接同一對節點時合併為一條連線，port 資訊保存在 ports。
- 前後版本完全相同時略過該版本，不產生空的修改事件。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..aov import diff_workflows, is_empty_change

VERSION_FILE = re.compile(r"^g(\d+)(?:f(\d+))?\.json$")

# 77 種節點類型歸成 8 類，供特徵編碼使用（FeatureSchema.activity_map）
CATEGORY_MEMBERS: dict[str, list[str]] = {
    "io": ["io.load_image", "io.blank_image"],
    "preprocess": [
        "cv.grayscale", "cv.resize", "cv.crop", "cv.pyr_down", "cv.transpose", "cv.add_border", "cv.hconcat",
        "cv.gaussian_blur", "cv.median_blur", "cv.blur", "cv.bilateral_filter", "cv.clahe", "cv.gamma_correct",
        "cv.brightness_contrast", "cv.normalize_image", "cv.equalize_color", "cv.dehaze_dcp", "cv.convert_color",
        "cv.split_channel", "cv.extract_hsv_channel", "cv.adjust_hue", "cv.apply_colormap", "cv.invert",
        "cv.add_weighted", "cv.absdiff", "cv.sobel",
    ],
    "segment": [
        "cv.color_mask_hsv", "cv.in_range", "cv.threshold", "cv.threshold_otsu", "cv.threshold_triangle",
        "cv.threshold_value", "cv.adaptive_threshold", "cv.canny", "cv.mask_to_image", "cv.matrix_to_mask",
        "cv.bitwise_images", "cv.bitwise_apply", "cv.contours_to_mask",
    ],
    "morphology": ["cv.morphology", "cv.fill_holes", "cv.remove_small_objects", "cv.distance_transform",
                   "cv.watershed", "cv.local_maxima"],
    "region": ["cv.find_contours", "cv.largest_contour", "cv.contour_stats", "cv.connected_components",
               "cv.connected_components_stats", "cv.bounding_boxes", "flow.filter_contours"],
    "detect": ["cv.dnn_detect", "cv.hog_svm_detect", "cv.template_match", "cv.hough_circles",
               "cv.find_chessboard_corners"],
    "postprocess": ["flow.filter_detections", "flow.nms", "flow.merge_detections", "flow.transform_detections",
                    "flow.expand_detections", "flow.iou_filter", "flow.largest_detection", "flow.sort_detections",
                    "flow.keypoints_to_detections", "flow.detections_to_mask", "flow.detections_to_points"],
    "output": ["flow.count", "flow.overlay_detections", "flow.overlay_keypoints", "cv.overlay_mask",
               "cv.draw_rectangle", "cv.draw_contours", "cv.put_text"],
}
CVFLOW_ACTIVITY_MAP = {member: category for category, members in CATEGORY_MEMBERS.items() for member in members}
CVFLOW_ACTIVITY_DEFAULT = "other"

# 解題策略：依流程圖中「主要辨識手段」判斷，用來辨識「換策略」這種修改
STRATEGY_MARKERS = [
    ("dnn", ("cv.dnn_detect", "cv.hog_svm_detect")),
    ("template", ("cv.template_match",)),
    ("hough", ("cv.hough_circles", "cv.find_chessboard_corners")),
    ("watershed", ("cv.watershed", "cv.distance_transform")),
    ("color", ("cv.color_mask_hsv", "cv.in_range")),
    ("threshold", ("cv.threshold", "cv.threshold_otsu", "cv.threshold_triangle", "cv.threshold_value",
                   "cv.adaptive_threshold", "cv.canny")),
]


def strategy(node_types: set[str]) -> str:
    for name, markers in STRATEGY_MARKERS:
        if node_types & set(markers):
            return name
    return "other"


@dataclass
class Version:
    round: int
    fix: int
    path: Path
    graph: dict[str, Any]


@dataclass
class AdapterStats:
    episodes: int = 0
    events: int = 0
    skipped_single_version: int = 0
    skipped_identical_versions: int = 0
    renamed_nodes: int = 0
    merged_port_edges: int = 0
    operation_types: Counter = field(default_factory=Counter)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_versions(task_dir: Path) -> list[Version]:
    versions = []
    for path in task_dir.iterdir():
        match = VERSION_FILE.match(path.name)
        if match:
            graph = _load(path)
            if graph.get("nodes"):
                versions.append(Version(int(match.group(1)), int(match.group(2) or 0), path, graph))
    return sorted(versions, key=lambda v: (v.round, v.fix))


def to_aov(graph: dict[str, Any], stats: AdapterStats | None = None) -> tuple[list[dict], list[dict]]:
    nodes = [{"node_id": n["id"], "activity_type": n["type"], "parameters": n.get("params") or {}}
             for n in graph["nodes"]]
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for e in graph.get("edges", []):
        key = (e["source"], e["target"])
        port = f"{e.get('source_port')}->{e.get('target_port')}"
        if key in edges:
            edges[key]["ports"].append(port)
            if stats:
                stats.merged_port_edges += 1
        else:
            edges[key] = {"source": key[0], "target": key[1], "ports": [port]}
    return nodes, list(edges.values())


def canonicalize(prev_nodes: list[dict], nodes: list[dict], edges: list[dict],
                 stats: AdapterStats | None = None) -> tuple[list[dict], list[dict]]:
    """新版本中代號改變、但類型與某個消失節點相同者，視為改名並沿用舊代號。"""
    prev_ids = {n["node_id"] for n in prev_nodes}
    new_ids = {n["node_id"] for n in nodes}
    vanished: dict[str, list[str]] = {}
    for n in prev_nodes:
        if n["node_id"] not in new_ids:
            vanished.setdefault(n["activity_type"], []).append(n["node_id"])
    rename: dict[str, str] = {}
    for n in nodes:
        if n["node_id"] not in prev_ids and vanished.get(n["activity_type"]):
            rename[n["node_id"]] = vanished[n["activity_type"]].pop(0)
    if not rename:
        return nodes, edges
    if stats:
        stats.renamed_nodes += len(rename)
    renamed_nodes = [{**n, "node_id": rename.get(n["node_id"], n["node_id"]),
                      **({"source_id": n["node_id"]} if n["node_id"] in rename else {})} for n in nodes]
    renamed_edges = [{**e, "source": rename.get(e["source"], e["source"]),
                      "target": rename.get(e["target"], e["target"])} for e in edges]
    return renamed_nodes, renamed_edges


def classify(prev_nodes: list[dict], nodes: list[dict], change: dict[str, Any]) -> str:
    """CVflow 修改類型：換策略、調參、加入步驟、刪除步驟、替換步驟、調整連線。"""
    before = {n["activity_type"] for n in prev_nodes}
    after = {n["activity_type"] for n in nodes}
    if strategy(before) != strategy(after):
        return "strategy_switch"
    added, removed = bool(change["added_nodes"]), bool(change["removed_nodes"])
    if added and removed:
        return "replace_step"
    if added:
        return "add_step"
    if removed:
        return "remove_step"
    if change["updated_node_attributes"]:
        return "param_tune"
    return "rewire"


def _try_results(task_dir: Path, score: dict[str, Any]) -> list[dict[str, Any]]:
    """每次執行的結果，附上評分（v1s 視為通過）。用流程圖內容對應到版本。"""
    by_index = {t["try_index"]: t for t in score.get("tries", [])}
    results = []
    for path in task_dir.glob("try*.json"):
        if not re.match(r"^try\d+\.json$", path.name):
            continue
        data = _load(path)
        scored = by_index.get(data.get("try_index"), {})
        results.append({
            "graph": data.get("graph"), "ts": data.get("ts"), "round": data.get("round"),
            "executable": scored.get("executable", not data.get("validator_errors") and not data.get("exec_error")),
            "passed": bool(scored.get("v1s")),
            "validator_errors": data.get("validator_errors") or [],
        })
    return results


def _short_id(teacher: str, scope: str, task: str, prefix: str = "CVF") -> str:
    digest = hashlib.sha1(f"{teacher}/{scope}/{task}".encode()).hexdigest()[:6]
    return f"{prefix}-{teacher}-{scope[0]}-{task[:28]}-{digest}"


def build_episode(task_dir: Path, teacher: str, scope: str, stats: AdapterStats,
                  id_prefix: str = "CVF") -> dict[str, Any] | None:
    versions = load_versions(task_dir)
    if len(versions) < 2:
        stats.skipped_single_version += 1
        return None
    card = _load(task_dir / "card.json") if (task_dir / "card.json").exists() else {}
    score_file = task_dir / ("score_sealed4.json" if scope == "tasks" else "score_retake.json")
    score = _load(score_file) if score_file.exists() else {}
    traj = _load(task_dir / "traj.json") if (task_dir / "traj.json").exists() else {}
    steps = {step["round"]: step for step in traj.get("steps", [])}
    tries = _try_results(task_dir, score)

    def outcome(graph: dict[str, Any]) -> dict[str, Any]:
        for t in tries:
            if t["graph"] == graph:
                return {"executable": t["executable"], "passed": t["passed"], "ts": t["ts"],
                        "validator_errors": t["validator_errors"][:3]}
        return {"executable": None, "passed": None, "ts": None, "validator_errors": []}

    episode_id = _short_id(teacher, scope, task_dir.name, id_prefix)
    first = versions[0]
    nodes, edges = to_aov(first.graph, stats)
    first_outcome = outcome(first.graph)
    started = first_outcome["ts"] or datetime.fromtimestamp(first.path.stat().st_mtime, timezone.utc).isoformat()

    events: list[dict[str, Any]] = []
    for version in versions[1:]:
        raw_nodes, raw_edges = to_aov(version.graph, stats)
        new_nodes, new_edges = canonicalize(nodes, raw_nodes, raw_edges, stats)
        change = diff_workflows(nodes, edges, new_nodes, new_edges)
        if is_empty_change(change):
            stats.skipped_identical_versions += 1
            continue
        index = len(events) + 1
        operation = classify(nodes, new_nodes, change)
        stats.operation_types[operation] += 1
        # 修改原因 = 修改者為這一輪設計寫下的理由（traj 第 N 輪的 hypothesis，描述的正是 g{N}）。
        # 上一輪寫的修正計畫（fix）常與實際做法不同，另存為 planned_fix 供比較。
        # 同輪修正（g{N}f{K}）是驗證失敗後的修補，沒有對應的文字理由。
        reason = None
        if version.fix == 0 and steps.get(version.round, {}).get("hypothesis"):
            reason = steps[version.round]["hypothesis"].strip()
        result = outcome(version.graph)
        events.append({
            "event_id": f"{episode_id}-E{index:02d}",
            "sequence_index": index,
            "occurred_at": result["ts"],
            "actor_type": "ai_agent",
            "operation_type": operation,
            "before_version_id": f"{episode_id}-V{index - 1}",
            "after_version_id": f"{episode_id}-V{index}",
            "change": change,
            "reason": reason,
            "reason_source": "actor" if reason else None,
            "attributes": {
                "source_file": version.path.name, "round": version.round, "fix_attempt": version.fix,
                "strategy_before": strategy({n["activity_type"] for n in nodes}),
                "strategy_after": strategy({n["activity_type"] for n in new_nodes}),
                "result": result,
                "exec_report": steps.get(version.round - 1, {}).get("exec_report") if version.fix == 0 else None,
                "planned_fix": steps.get(version.round - 1, {}).get("fix") if version.fix == 0 else None,
            },
        })
        nodes, edges = new_nodes, new_edges

    if not events:
        stats.skipped_single_version += 1
        return None
    stats.episodes += 1
    stats.events += len(events)
    flags = [p.stem.replace("STATUS_", "") for p in task_dir.glob("STATUS_*.md")]
    return {
        "episode_id": episode_id,
        "work_order_id": task_dir.name,
        "started_at": started,
        "is_synthetic": False,
        "initial_context": {
            "task_id": task_dir.name, "teacher": teacher, "request": card.get("nl_request"), "metric": card.get("metric"),
            "category": card.get("category"), "scope": scope, "status_flags": flags,
        },
        "initial_workflow": {"workflow_id": f"{episode_id}-WF", "version_id": f"{episode_id}-V0",
                             "nodes": to_aov(first.graph)[0], "edges": to_aov(first.graph)[1]},
        "events": events,
        "provenance": {
            "method": "adapter", "adapter": "cvflow-v1", "source": "2026-10-13-jiayuan-package",
            "path": f"teachers/{teacher}/{scope}/{task_dir.name}", "teacher": teacher,
            "first_version_result": first_outcome,
            # 結果資料：只供分析，特徵編碼不讀取
            "outcome": {"final_ok": score.get("final_ok"), "first_ok_round": score.get("first_ok_round"),
                        "stop_reason": traj.get("stop_reason"), "confidence": traj.get("confidence")},
        },
    }


def iter_task_dirs(root: Path, teacher: str):
    for scope in ("tasks", "retake"):
        base = root / "teachers" / teacher / scope
        if base.exists():
            for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                yield scope, task_dir


def ingest_teacher(service: Any, root: Path, teacher: str, case_id: str, id_prefix: str = "CVF") -> AdapterStats:
    """匯入某個模型的全部解題過程到指定案例。已匯入的序列會略過，可以重複執行。"""
    from ..service import KCoreError

    stats = AdapterStats()
    for scope, task_dir in iter_task_dirs(root, teacher):
        payload = build_episode(task_dir, teacher, scope, stats, id_prefix)
        if payload is None:
            continue
        try:
            service.ingest_episode(case_id, payload)
        except KCoreError as error:
            if error.code != "episode_exists":
                raise KCoreError(error.code, f"{task_dir}: {error.message}", error.status, error.details) from error
    return stats
