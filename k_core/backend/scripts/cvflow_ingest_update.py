"""把 2026-11-17 佳元更新包的第二批（D171 程式題與照片題，AoV 格式）匯入 CVFLOW 案例。

第一批（aov-evolution-20261013）與 10/13 那包是同樣 41 題、同一批解題紀錄，已經在 CVFLOW 案例裡，不重複匯入。
ollama 與 openai 的解題紀錄不納入（見 kcore.adapters.cvflow_aov.EXCLUDED_SOLVERS）。可以重複執行，已匯入的序列會略過。

用法（在 k_core/backend 執行）：
    .venv\\Scripts\\python.exe scripts\\cvflow_ingest_update.py "E:\\程式碼\\2026-11-17-jiayuan-package-update"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcore.adapters.cvflow_aov import ingest_batch  # noqa: E402
from kcore.database import make_session_factory  # noqa: E402
from kcore.service import KCoreService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--database-url", default="sqlite:///./kcore.db")
    parser.add_argument("--case", default="CVFLOW")
    args = parser.parse_args()
    service = KCoreService(make_session_factory(args.database_url))
    stats = ingest_batch(service, args.package / "d171-negative-20261029", args.case)
    print(f"episodes {stats.episodes}  events {stats.events}  skipped(single version) {stats.skipped_single_version}  "
          f"identical versions {stats.skipped_identical_versions}  renamed nodes {stats.renamed_nodes}")
    print("operation types:", dict(stats.operation_types))


if __name__ == "__main__":
    main()
