"""依 CVflow 的評分定義回覆（docs/REPLY-jiayuan.md），原地更新 CVFLOW 案例裡的執行結果與標記。

改了什麼（轉接器 cvflow.py v2、cvflow_aov.py v2）：
- 10/13 包：「通過」由 v1s 改為 v1i；保留 v0／v1c／v1s、P、R、K（配對數）、重播框數 n_pred、解題當下框數 count；
  沒有評分紀錄的版本記為 None（原本誤記為未通過）；provenance 加上正解框數（K ÷ R）；
  status_flags 依 10/14 裁定（超出範圍）與正解事後更正（更正 7、8、9）。
- 11/17 包：程式題的「通過」由 v1s 改為 error_types 只有 ok；保留 judge_v1i、gt_count、預測數；
  整題 final_ok／any_ok 改由 error_types 計算，原本 attempts.jsonl 的 ok 另存為 package_ok。

做法：用新版轉接器重新產生每條序列（不寫入），逐條確認修改內容（類型、差異、原因、順序、時間）與資料庫完全相同，
才更新事件的 attributes 與 payload_hash、序列的 initial_context 與 provenance。快照、特徵、已訓練的模型都不動。
內容對不上的序列不更新，列在報告裡。

用法（在 k_core/backend 執行，預設只檢查不寫入）：
    .venv\\Scripts\\python.exe scripts\\cvflow_refresh_results.py            # 檢查
    .venv\\Scripts\\python.exe scripts\\cvflow_refresh_results.py --apply    # 先備份資料庫，再寫入
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kcore.adapters import cvflow, cvflow_aov  # noqa: E402
from kcore.database import ChangeEventRecord, EpisodeRecord, make_session_factory  # noqa: E402
from kcore.service import EVENT_FIELDS, _payload_hash, _parse_time  # noqa: E402

PACKAGE_1013 = Path(r"E:\程式碼\2026-10-13-jiayuan-package")
PACKAGE_1117 = Path(r"E:\程式碼\2026-11-17-jiayuan-package-update\d171-negative-20261029")
TEACHERS = ["codex", "gemini", "opus", "sonnet", "haiku"]
CASE_ID = "CVFLOW"


def payloads():
    stats = cvflow.AdapterStats()
    for teacher in TEACHERS:
        for scope, task_dir in cvflow.iter_task_dirs(PACKAGE_1013, teacher):
            payload = cvflow.build_episode(task_dir, teacher, scope, stats, id_prefix="CVFLOW")
            if payload:
                yield "10/13", payload
    index = cvflow_aov.TrialIndex(PACKAGE_1117)
    stats = cvflow.AdapterStats()
    for source, solver, run, task_dir in cvflow_aov.iter_task_dirs(PACKAGE_1117):
        payload = cvflow_aov.build_episode(task_dir, source, solver, run, index, stats)
        if payload:
            yield "11/17", payload


def utc(value) -> str | None:
    if value is None:
        return None
    moment = _parse_time(value) if isinstance(value, str) else value
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.isoformat()


def differences(record: ChangeEventRecord, event: dict) -> list[str]:
    """修改內容（attributes 以外的欄位）是否完全相同；attributes 只允許 result 不同。"""
    problems = []
    for key in EVENT_FIELDS:
        if key in ("attributes", "occurred_at"):
            continue
        stored, new = getattr(record, key), event.get(key)
        if key in ("context_delta",):
            stored, new = stored or {}, new or {}
        if key in ("related_ui_event_ids",):
            stored, new = stored or [], new or []
        if json.dumps(stored, sort_keys=True, ensure_ascii=False) != json.dumps(new, sort_keys=True, ensure_ascii=False):
            problems.append(key)
    if utc(record.occurred_at) != utc(event.get("occurred_at")):
        problems.append("occurred_at")
    old_attrs = {k: v for k, v in (record.attributes or {}).items() if k != "result"}
    new_attrs = {k: v for k, v in (event.get("attributes") or {}).items() if k != "result"}
    if json.dumps(old_attrs, sort_keys=True, ensure_ascii=False) != json.dumps(new_attrs, sort_keys=True, ensure_ascii=False):
        problems.append("attributes(other than result)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "kcore.db")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.apply:
        backup = args.database.with_name(f"kcore.backup-{datetime.now():%Y%m%d-%H%M}.db")
        shutil.copy2(args.database, backup)
        print(f"backup: {backup.name}")

    session_factory = make_session_factory(f"sqlite:///{args.database}")
    counts: Counter = Counter()
    mismatched: list[tuple[str, str, list[str]]] = []
    changes: Counter = Counter()
    seen = set()
    with session_factory() as session:
        for package, payload in payloads():
            episode = session.get(EpisodeRecord, payload["episode_id"])
            if episode is None or episode.case_id != CASE_ID:
                counts[f"{package}: not in database"] += 1
                continue
            seen.add(episode.episode_id)
            records = sorted(session.query(ChangeEventRecord).filter_by(episode_id=episode.episode_id),
                             key=lambda r: r.sequence_index)
            events = payload["events"]
            problems = [] if len(records) == len(events) else [f"event count {len(records)} vs {len(events)}"]
            for record, event in zip(records, events):
                if record.event_id != event["event_id"]:
                    problems.append(f"event id {record.event_id} vs {event['event_id']}")
                    break
                problems += [f"{record.event_id}: {p}" for p in differences(record, event)]
            if problems:
                mismatched.append((package, episode.episode_id, problems[:3]))
                continue
            counts[f"{package}: episodes ok"] += 1
            # 統計通過標準改變的版本
            pairs = [((episode.provenance or {}).get("first_version_result"), payload["provenance"]["first_version_result"])]
            pairs += [((r.attributes or {}).get("result"), e["attributes"]["result"]) for r, e in zip(records, events)]
            for old, new in pairs:
                changes[f"{package}: passed {(old or {}).get('passed')} -> {(new or {}).get('passed')}"] += 1
            if args.apply:
                for record, event in zip(records, events):
                    record.attributes = event["attributes"]
                    record.payload_hash = _payload_hash(event)
                episode.initial_context = payload["initial_context"]
                episode.provenance = payload["provenance"]
                counts[f"{package}: events updated"] += len(records)
        in_case = {eid for (eid,) in session.query(EpisodeRecord.episode_id).filter_by(case_id=CASE_ID)}
        counts["database episodes not rebuilt"] = len(in_case - seen)
        if args.apply:
            session.commit()

    print("\n== episodes")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    print("\n== per-version 'passed' (old -> new), including first versions")
    for key, value in sorted(changes.items()):
        print(f"  {key}: {value}")
    if mismatched:
        print(f"\n== {len(mismatched)} episodes NOT updated (content differs)")
        for package, eid, problems in mismatched[:20]:
            print(f"  [{package}] {eid}: {problems}")
    print("\n" + ("applied" if args.apply else "dry run only (use --apply to write)"))


if __name__ == "__main__":
    main()
