"""CVflow AoV 演化轉接器：讀取 2026-11-17 更新包（AoV 格式的逐版流程圖）並轉成 K 的演變序列。

資料來源：`<更新包>/d171-negative-20261029/evolution/<來源>/<解題者>/<run>/<題目>/`
- `EVOLUTION.json`：每一版的輪次、判分、修改理由（模型自述）。
- `aov_v<N>/graphs/L0.json`：第 N 版的流程圖（aov-graph/1.0）。轉換失敗的版本沒有這個檔，直接略過；
  下一版的修改會跨過失敗的版本，略過的版數記在事件的 attributes["skipped_versions_before"]。
- `tries.jsonl`、`attempts.jsonl`：每次執行的錯誤類型、整題的結束原因。

與 cvflow.py（原始 g{N}.json 格式）一致的處理：
- 節點改名沿用舊代號、修改類型的判斷（classify）、修改原因 = 該版自己的設計理由（hypothesis），
  上一輪寫的修正計畫（fix）另存為 planned_fix。
- 執行結果只放在 attributes 與 provenance，特徵編碼不讀取。

AoV 特有的處理：
- 連線由節點的輸入資料（I）對應到產生該資料的節點（O）還原；同一對節點的多條資料合併成一條連線。
- AoV 匯出時加上的終點節點（kind = sink）與連到終點的連線不納入。
- 照片題（picgen）沒有標準答案：passed 記為 None，不當成失敗。

評分欄位的定義依 CVflow 的回覆（docs/REPLY-jiayuan.md，2026-11-24）：
- 程式題（d171）的「通過」＝ tries.jsonl 的 error_types 只有 ok（數量完全相等，且有框題 P、R 各 ≥ 0.8）。
  判分器自己的 judge_v1i（數量有容差、舊判分檔不查數量）另外保留，兩者有 35 個 try 不同。
- 整題：final_ok＝最後一個 try 通過、any_ok＝任一 try 通過（都用 error_types）。
  attempts.jsonl 的 ok 是任一 try 的 judge_v1i，另存為 package_ok；照片題的 ok 只代表解題者宣告滿意。
- tries.jsonl 的 P、R 是不套忽略區的嚴格口徑；count_only 題的 P、R 是補的 0，這裡記為 None。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..aov import diff_workflows, is_empty_change
from .cvflow import AdapterStats, _short_id, canonicalize, classify, strategy

SOURCE_NAME = "2026-11-17-jiayuan-package-update/d171-negative-20261029"
EXCLUDED_SOLVERS = {"ollama", "openai"}  # 2026-11-18 決定不納入：ollama 轉換失敗過多，openai 沒有可用的修改


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def aov_to_workflow(graph_file: Path, stats: AdapterStats | None = None) -> tuple[list[dict], list[dict]]:
    graph = _load(graph_file)["graph"]
    kept = [n for n in graph["nodes"] if n.get("kind") != "sink"]
    kept_ids = {n["id"] for n in kept}
    producer = {data: n["id"] for n in kept for data in n.get("O", [])}
    nodes = [{"node_id": n["id"], "activity_type": n["F"], "parameters": n.get("params") or {}} for n in kept]
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for n in kept:
        for data in n.get("I", []):
            source = producer.get(data)
            if source is None or n["id"] not in kept_ids:
                continue
            key = (source, n["id"])
            if key in edges:
                edges[key]["ports"].append(data)
                if stats:
                    stats.merged_port_edges += 1
            else:
                edges[key] = {"source": source, "target": n["id"], "ports": [data]}
    return nodes, list(edges.values())


def _run_time(run: str, fallback: Path) -> str:
    match = re.search(r"(\d{8})(?:T(\d{6}))?", run)
    if match:
        stamp = datetime.strptime(match.group(1) + (match.group(2) or "000000"), "%Y%m%d%H%M%S")
        return stamp.replace(tzinfo=timezone.utc).isoformat()
    return datetime.fromtimestamp(fallback.stat().st_mtime, timezone.utc).isoformat()


def passes(trial: dict[str, Any] | None) -> bool | None:
    """程式題的通過標準：error_types 只有 ok。沒有錯誤類型或沒有正解（unjudged）時為 None。"""
    types = (trial or {}).get("error_types")
    if not types or "unjudged" in types:
        return None
    return types == ["ok"]


def _result(version: dict[str, Any], trial: dict[str, Any] | None, judged: bool) -> dict[str, Any]:
    score = version.get("score") or {}
    raw = version.get("result") or {}
    trial = trial or {}
    executable = score.get("executable") if "executable" in score else (
        not raw.get("exec_error") and not raw.get("validator_errors"))
    has_boxes = trial.get("metric") != "count_only"
    return {"executable": executable, "passed": passes(trial) if judged else None,
            "count": raw.get("count"), "error_types": trial.get("error_types"),
            "judge_v1i": trial.get("judge_v1i"), "gt_count": trial.get("gt_count"),
            "pred_count": trial.get("pred_count"), "count_source": trial.get("count_source"),
            "precision": trial.get("P") if has_boxes else None, "recall": trial.get("R") if has_boxes else None,
            "validator_errors": (raw.get("validator_errors") or [])[:3]}


class TrialIndex:
    """tries.jsonl 與 attempts.jsonl 的查詢表。"""

    def __init__(self, batch_root: Path):
        self.tries: dict[tuple, dict[str, Any]] = {}
        self.by_attempt: dict[tuple, list[dict[str, Any]]] = {}
        self.attempts: dict[str, dict[str, Any]] = {}
        tries_file, attempts_file = batch_root / "tries.jsonl", batch_root / "attempts.jsonl"
        if tries_file.exists():
            for line in tries_file.read_text(encoding="utf-8").splitlines():
                t = json.loads(line)
                self.tries[(t["source"], t["solver"], t["run"], t["task_id"], t["try_index"])] = t
                self.by_attempt.setdefault((t["source"], t["solver"], t["run"], t["task_id"]), []).append(t)
            for rows in self.by_attempt.values():
                rows.sort(key=lambda t: t["try_index"])
        if attempts_file.exists():
            for line in attempts_file.read_text(encoding="utf-8").splitlines():
                a = json.loads(line)
                if a.get("evolution"):
                    self.attempts[a["evolution"].replace("\\", "/")] = a


def build_episode(task_dir: Path, source: str, solver: str, run: str, index: TrialIndex, stats: AdapterStats,
                  id_prefix: str = "CVF2") -> dict[str, Any] | None:
    evolution = _load(task_dir / "EVOLUTION.json")
    task = task_dir.name
    versions = []
    skipped = 0
    for v in evolution["versions"]:
        graph_file = task_dir / f"aov_v{v['version']}" / "graphs" / "L0.json"
        if not graph_file.exists():
            skipped += 1
            continue
        versions.append((v, graph_file, skipped))
        skipped = 0
    if len(versions) < 2:
        stats.skipped_single_version += 1
        return None

    judged = source == "d171"  # 照片題沒有正解

    def result_of(v: dict[str, Any]) -> dict[str, Any]:
        return _result(v, index.tries.get((source, solver, run, task, v.get("try_index"))), judged)

    episode_id = _short_id(solver, f"{source}/{run}", task, id_prefix)
    first, first_file, _ = versions[0]
    nodes, edges = aov_to_workflow(first_file, stats)
    initial_nodes, initial_edges = nodes, edges
    started = _run_time(run, first_file)
    events: list[dict[str, Any]] = []
    for v, graph_file, skipped_before in versions[1:]:
        raw_nodes, raw_edges = aov_to_workflow(graph_file, stats)
        new_nodes, new_edges = canonicalize(nodes, raw_nodes, raw_edges, stats)
        change = diff_workflows(nodes, edges, new_nodes, new_edges)
        if is_empty_change(change):
            stats.skipped_identical_versions += 1
            continue
        position = len(events) + 1
        operation = classify(nodes, new_nodes, change)
        stats.operation_types[operation] += 1
        reason_info = v.get("reason") or {}
        reason = (reason_info.get("hypothesis") or "").strip() or None
        events.append({
            "event_id": f"{episode_id}-E{position:02d}",
            "sequence_index": position,
            "occurred_at": None,
            "actor_type": "ai_agent",
            "operation_type": operation,
            "before_version_id": f"{episode_id}-V{position - 1}",
            "after_version_id": f"{episode_id}-V{position}",
            "change": change,
            "reason": reason,
            "reason_source": "actor" if reason else None,
            "attributes": {
                "source_file": f"aov_v{v['version']}", "round": v.get("round"), "try_index": v.get("try_index"),
                "skipped_versions_before": skipped_before,
                "strategy_before": strategy({n["activity_type"] for n in nodes}),
                "strategy_after": strategy({n["activity_type"] for n in new_nodes}),
                "result": result_of(v),
                "reason_kind": reason_info.get("kind"),
                "planned_fix": reason_info.get("fix"),
                "source_change_kind": v.get("change_kind"),
            },
        })
        nodes, edges = new_nodes, new_edges

    if not events:
        stats.skipped_single_version += 1
        return None
    stats.episodes += 1
    stats.events += len(events)
    relative = f"evolution/{source}/{solver}/{run}/{task}"
    attempt = index.attempts.get(relative, {})
    trials = index.by_attempt.get((source, solver, run, task), [])
    verdicts = [passes(t) for t in trials] if judged else []
    gt_counts = {t.get("gt_count") for t in trials if t.get("gt_count") is not None}
    return {
        "episode_id": episode_id,
        "work_order_id": task,
        "started_at": started,
        "is_synthetic": False,
        "initial_context": {
            "task_id": task, "teacher": solver, "request": None, "metric": attempt.get("metric"),
            "category": attempt.get("category"), "scope": source, "run": run, "status_flags": [],
        },
        "initial_workflow": {"workflow_id": f"{episode_id}-WF", "version_id": f"{episode_id}-V0",
                             "nodes": initial_nodes, "edges": initial_edges},
        "events": events,
        "provenance": {
            "method": "adapter", "adapter": "cvflow-aov-v2", "source": SOURCE_NAME, "path": relative,
            "teacher": solver, "first_version_result": result_of(first),
            # 結果資料：只供分析，特徵編碼不讀取
            "outcome": {"pass_rule": "error_types_ok" if judged else None,
                        "final_ok": verdicts[-1] if verdicts else None,
                        "any_ok": any(verdicts) if verdicts else None,
                        "package_ok": attempt.get("ok"), "stop_reason": attempt.get("stop_reason")},
            "ground_truth": {"gt_count": gt_counts.pop() if len(gt_counts) == 1 else None},
        },
    }


def iter_task_dirs(batch_root: Path, excluded: set[str] = EXCLUDED_SOLVERS):
    """依序產生 (來源, 解題者, run, 題目資料夾)。"""
    for source_dir in sorted(p for p in (batch_root / "evolution").iterdir() if p.is_dir()):
        for solver_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            if solver_dir.name in excluded:
                continue
            for run_dir in sorted(p for p in solver_dir.iterdir() if p.is_dir()):
                for task_dir in sorted(p for p in run_dir.iterdir() if (p / "EVOLUTION.json").exists()):
                    yield source_dir.name, solver_dir.name, run_dir.name, task_dir


def ingest_batch(service: Any, batch_root: Path, case_id: str, id_prefix: str = "CVF2",
                 excluded: set[str] = EXCLUDED_SOLVERS) -> AdapterStats:
    """匯入一批 AoV 演化資料到指定案例。已匯入的序列會略過，可以重複執行。"""
    from ..service import KCoreError

    index = TrialIndex(batch_root)
    stats = AdapterStats()
    for source, solver, run, task_dir in iter_task_dirs(batch_root, excluded):
        payload = build_episode(task_dir, source, solver, run, index, stats, id_prefix)
        if payload is None:
            continue
        try:
            service.ingest_episode(case_id, payload)
        except KCoreError as error:
            if error.code != "episode_exists":
                raise KCoreError(error.code, f"{task_dir}: {error.message}", error.status, error.details) from error
    return stats
