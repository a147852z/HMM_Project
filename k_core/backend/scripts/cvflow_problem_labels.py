"""用本地 LLM 把「上一版執行後的觀察」歸類成問題類型，結果存成 JSON 供後續實驗重複使用。

觀察文字（exec_report）是 AI 代理人在提出下一次修改之前寫的，所以用它當條件不算偷看答案。
只有 codex 與 Gemini 的非同輪修正步驟有這段文字。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_problem_labels.py --base-url http://192.168.178.212:1234/v1 ^
        --model google/gemma-4-26b-a4b --out ..\\..\\docs\\cvflow_problem_labels.json
已經標過的事件會略過，可以中斷後重跑。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.conversion.llm_client import LLMError, OpenAICompatibleClient, extract_json  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.service import KCoreService  # noqa: E402

PROMPT_VERSION = "problem-v1"
LABELS = {
    "missed": "有目標沒有被框到或被少算（漏偵測）",
    "false_positive": "框到背景、陰影或不是目標的東西，或數量多算（誤偵測）",
    "merged": "多個相鄰目標被合成同一個框或同一個區域",
    "split": "同一個目標被拆成多個框，或被重複框選",
    "box_misfit": "有框到目標，但框的大小或位置不準（太大、太小、只框到一部分）",
    "not_executable": "流程無法執行或沒有產生結果",
    "correct": "結果正確，沒有需要修正的問題",
}
SYSTEM = (
    "你是影像辨識流程的品質分析助理。使用者會給你一段文字，描述某個流程執行後在圖片上看到的結果。\n"
    "請判斷這段文字提到了哪些問題，並選出最主要的一個。只能使用以下問題類型：\n"
    + "\n".join(f"- {key}：{text}" for key, text in LABELS.items())
    + "\n規則：只根據文字內容判斷，不要推測文字沒有提到的問題。"
      "primary 必須是 problems 其中之一；如果文字同時提到多種問題，選文字中描述最多、影響最大的那一種。"
      "只輸出 JSON。"
)
SCHEMA = {
    "type": "object",
    "properties": {
        "problems": {"type": "array", "items": {"type": "string", "enum": list(LABELS)}, "minItems": 1},
        "primary": {"type": "string", "enum": list(LABELS)},
    },
    "required": ["problems", "primary"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--case", default="CVFLOW")
    parser.add_argument("--out", type=Path, default=Path("cvflow_problem_labels.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    service = KCoreService(make_session_factory(args.database_url))
    reports: list[tuple[str, str]] = []
    with service.session_factory() as session:
        for _, events in service._training_sequences(session, args.case):
            for event in events:
                report = (event.attributes or {}).get("exec_report")
                if report:
                    reports.append((event.event_id, report))
    store = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    store.setdefault("labels", {})
    store.update({"model": args.model, "prompt_version": PROMPT_VERSION, "label_definitions": LABELS,
                  "system_prompt": SYSTEM})
    client = OpenAICompatibleClient(args.base_url, args.model, temperature=0.0, max_tokens=200, timeout=180)
    todo = [(eid, text) for eid, text in reports if eid not in store["labels"]]
    if args.limit:
        todo = todo[:args.limit]
    print(f"reports={len(reports)} already={len(reports) - len(todo)} todo={len(todo)}", flush=True)
    started = time.time()
    failures = 0
    for i, (event_id, text) in enumerate(todo, start=1):
        try:
            answer = extract_json(client.complete(SYSTEM, text, SCHEMA))
            problems = [p for p in answer.get("problems", []) if p in LABELS]
            primary = answer.get("primary")
            if primary not in LABELS or not problems:
                raise ValueError(f"invalid answer: {answer}")
            if primary not in problems:
                problems.append(primary)
            store["labels"][event_id] = {"primary": primary, "problems": problems}
        except (LLMError, ValueError, json.JSONDecodeError) as error:
            failures += 1
            print(f"  failed {event_id}: {error}", flush=True)
        if i % 25 == 0 or i == len(todo):
            args.out.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {i}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)
    args.out.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = Counter(v["primary"] for v in store["labels"].values())
    print("labelled:", len(store["labels"]), "failures:", failures)
    print("primary:", dict(counts.most_common()))
    print("any:", dict(Counter(p for v in store["labels"].values() for p in v["problems"]).most_common()))


if __name__ == "__main__":
    main()
