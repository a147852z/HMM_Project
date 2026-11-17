"""SQLite 資料層。原始紀錄（事件、快照）與衍生資料（特徵、模型）分開保存。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class CaseRecord(Base):
    """場域案例：一個場域／一套 ERP，或一個非 ERP 案例來源（CVflow、硬體生成）。"""

    __tablename__ = "cases"
    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    domain_kind: Mapped[str] = mapped_column(String(32))
    erp_instance_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recording_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EpisodeRecord(Base):
    """一次演變序列：從初始流程到人員確認定案的一次完整調整過程。"""

    __tablename__ = "episodes"
    episode_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    work_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    workflow_id: Mapped[str] = mapped_column(String(64))
    initial_version_id: Mapped[str] = mapped_column(String(64))
    final_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    initial_context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open / finalized
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    # 資料來源：{"method": "api" | "synthetic" | "llm_conversion", ...轉換紀錄、文件、提示詞版本、審核人}
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowSnapshotRecord(Base):
    """完整 AoV 版本。初始版本於建立序列時保存，其餘版本於定案時依修改重建。"""

    __tablename__ = "workflow_snapshots"
    __table_args__ = (UniqueConstraint("episode_id", "version_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.episode_id"), index=True)
    workflow_id: Mapped[str] = mapped_column(String(64))
    version_id: Mapped[str] = mapped_column(String(64))
    nodes: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    edges: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChangeEventRecord(Base):
    """一次修改事件。payload 保存原始提交內容；features 是定案時衍生的結構特徵。"""

    __tablename__ = "change_events"
    __table_args__ = (UniqueConstraint("episode_id", "sequence_index"),)
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.episode_id"), index=True)
    sequence_index: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(16))  # human / system / synthetic / ai_agent
    operation_type: Mapped[str] = mapped_column(String(64))
    before_version_id: Mapped[str] = mapped_column(String(64))
    after_version_id: Mapped[str] = mapped_column(String(64))
    change: Mapped[dict[str, Any]] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_source: Mapped[str | None] = mapped_column(String(16), nullable=True)  # human / llm / actor / None
    context_delta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    related_ui_event_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    payload_hash: Mapped[str] = mapped_column(String(64))
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingRunRecord(Base):
    __tablename__ = "training_runs"
    training_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    status: Mapped[str] = mapped_column(String(32))  # running / succeeded / failed / insufficient_data
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON)
    dataset: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"
    model_version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    training_run_id: Mapped[str] = mapped_column(String(64))
    feature_schema: Mapped[dict[str, Any]] = mapped_column(JSON)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)  # states, observations, pi, A, B
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="inactive")  # active / inactive
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LLMProfileRecord(Base):
    """某個案例的 LLM 轉換設定。內容不可修改：每次更新都建立新版本，轉換紀錄才能追溯到當時的提示詞。"""

    __tablename__ = "llm_profiles"
    __table_args__ = (UniqueConstraint("profile_id", "version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    base_url: Mapped[str] = mapped_column(String(500))
    model: Mapped[str] = mapped_column(String(200))
    api_key_env: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 只存環境變數名稱，不存金鑰
    system_prompt: Mapped[str] = mapped_column(Text)
    user_prompt_template: Mapped[str] = mapped_column(Text)
    temperature: Mapped[float] = mapped_column(default=0.0)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    structured_output: Mapped[bool] = mapped_column(Boolean, default=True)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=600)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SourceDocumentRecord(Base):
    """尚未整理的原始資料（文字內容）。document_id 由案例與內容雜湊決定，同一份檔案重傳不會重複。"""

    __tablename__ = "source_documents"
    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    media_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    encoding: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConversionRecord(Base):
    """一次 LLM 轉換：原始文件 + 提示詞版本 → JSON 草稿 → 驗證 → 人工確認 → 匯入。"""

    __tablename__ = "conversions"
    conversion_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("source_documents.document_id"))
    profile_id: Mapped[str] = mapped_column(String(64))
    profile_version: Mapped[int] = mapped_column(Integer)
    # queued / running / failed / needs_review / approved / rejected
    status: Mapped[str] = mapped_column(String(16))
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    draft: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    draft_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    validation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_episode_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def make_session_factory(database_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
