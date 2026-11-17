"""LLM 轉換流程：轉換設定（含提示詞版本）→ 上傳原始文件 → 轉換 → 驗證 → 人工確認 → 匯入。"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select

from ..database import ConversionRecord, EpisodeRecord, LLMProfileRecord, SourceDocumentRecord, utcnow
from ..service import KCoreError, KCoreService
from .draft import build_messages, build_preview, draft_json_schema, validate_and_normalize
from .llm_client import LLMClient, LLMError, client_from_profile, extract_json

MAX_DOCUMENT_CHARS = 200_000
TEXT_ENCODINGS = ("utf-8-sig", "cp950", "big5hkscs")
PROFILE_FIELDS = ("name", "base_url", "model", "api_key_env", "system_prompt", "user_prompt_template",
                  "temperature", "max_tokens", "structured_output", "timeout_seconds")

ClientFactory = Callable[[LLMProfileRecord], LLMClient]


def decode_text(data: bytes) -> tuple[str, str]:
    """依序嘗試 UTF-8、Big5（cp950）等常見編碼；UTF-16 只在有 BOM 時使用。都失敗時視為非文字檔。"""
    candidates = (("utf-16",) if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ()) + TEXT_ENCODINGS
    for encoding in candidates:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text, encoding
    raise KCoreError("unsupported_document", "document is not text in a supported encoding "
                     f"({', '.join(TEXT_ENCODINGS)}); convert it to text first", 415)


class ConversionService:
    def __init__(self, core: KCoreService, client_factory: ClientFactory = client_from_profile) -> None:
        self.core = core
        self.session_factory = core.session_factory
        self.client_factory = client_factory

    # ------------------------------------------------------------------ 轉換設定
    def _profile(self, session, case_id: str, profile_id: str, version: int | None = None) -> LLMProfileRecord:
        self.core._case(session, case_id)
        query = select(LLMProfileRecord).where(LLMProfileRecord.profile_id == profile_id)
        query = query.where(LLMProfileRecord.version == version) if version else query.order_by(
            LLMProfileRecord.version.desc())
        record = session.scalars(query).first()
        if record is None or record.case_id != case_id:
            raise KCoreError("profile_not_found", f"LLM profile {profile_id} not found for case {case_id}", 404)
        return record

    @staticmethod
    def _profile_dict(record: LLMProfileRecord) -> dict[str, Any]:
        return {"profile_id": record.profile_id, "version": record.version, "case_id": record.case_id,
                **{name: getattr(record, name) for name in PROFILE_FIELDS},
                "created_at": record.created_at.isoformat()}

    def create_profile(self, case_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self.session_factory() as session:
            self.core._case(session, case_id)
            if session.scalar(select(LLMProfileRecord).where(LLMProfileRecord.profile_id == data["profile_id"])):
                raise KCoreError("profile_exists", f"LLM profile {data['profile_id']} already exists", 409)
            record = LLMProfileRecord(profile_id=data["profile_id"], version=1, case_id=case_id,
                                      **{name: data[name] for name in PROFILE_FIELDS if name in data})
            session.add(record)
            session.commit()
            return self._profile_dict(record)

    def update_profile(self, case_id: str, profile_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """建立新版本；舊版本保留，既有轉換紀錄仍指向當時的提示詞。"""
        with self.session_factory() as session:
            latest = self._profile(session, case_id, profile_id)
            values = {name: getattr(latest, name) for name in PROFILE_FIELDS}
            values.update({name: value for name, value in changes.items() if name in PROFILE_FIELDS})
            record = LLMProfileRecord(profile_id=profile_id, version=latest.version + 1, case_id=case_id, **values)
            session.add(record)
            session.commit()
            return self._profile_dict(record)

    def list_profiles(self, case_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            self.core._case(session, case_id)
            rows = session.scalars(select(LLMProfileRecord).where(LLMProfileRecord.case_id == case_id)
                                   .order_by(LLMProfileRecord.profile_id, LLMProfileRecord.version))
            return [self._profile_dict(row) for row in rows]

    def get_profile(self, case_id: str, profile_id: str, version: int | None = None) -> dict[str, Any]:
        with self.session_factory() as session:
            return self._profile_dict(self._profile(session, case_id, profile_id, version))

    def test_profile(self, case_id: str, profile_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            profile = self._profile(session, case_id, profile_id)
        try:
            models = self.client_factory(profile).list_models()
        except LLMError as error:
            return {"reachable": False, "error": str(error), "model_available": False}
        return {"reachable": True, "models": models, "model_available": profile.model in models}

    # ------------------------------------------------------------------ 原始文件
    def add_document(self, case_id: str, filename: str, data: bytes, media_type: str | None) -> dict[str, Any]:
        text, encoding = decode_text(data)
        if not text.strip():
            raise KCoreError("unsupported_document", "document is empty", 422)
        if len(text) > MAX_DOCUMENT_CHARS:
            raise KCoreError("document_too_large", f"document has {len(text)} characters; the limit is "
                             f"{MAX_DOCUMENT_CHARS}. Split it before conversion.", 413)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = f"{case_id}-DOC-{digest[:12]}"
        with self.session_factory() as session:
            self.core._case(session, case_id)
            record = session.get(SourceDocumentRecord, document_id)
            duplicate = record is not None
            if record is None:
                record = SourceDocumentRecord(document_id=document_id, case_id=case_id, filename=filename,
                                              media_type=media_type, encoding=encoding, sha256=digest, content=text)
                session.add(record)
                session.commit()
            return {**self._document_dict(record, include_content=False), "duplicate": duplicate}

    @staticmethod
    def _document_dict(record: SourceDocumentRecord, include_content: bool) -> dict[str, Any]:
        data = {"document_id": record.document_id, "case_id": record.case_id, "filename": record.filename,
                "media_type": record.media_type, "encoding": record.encoding, "sha256": record.sha256,
                "characters": len(record.content), "uploaded_at": record.uploaded_at.isoformat()}
        if include_content:
            data["content"] = record.content
        return data

    def _document(self, session, case_id: str, document_id: str) -> SourceDocumentRecord:
        self.core._case(session, case_id)
        record = session.get(SourceDocumentRecord, document_id)
        if record is None or record.case_id != case_id:
            raise KCoreError("document_not_found", f"document {document_id} not found for case {case_id}", 404)
        return record

    def list_documents(self, case_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            self.core._case(session, case_id)
            rows = session.scalars(select(SourceDocumentRecord).where(SourceDocumentRecord.case_id == case_id)
                                   .order_by(SourceDocumentRecord.uploaded_at))
            return [self._document_dict(row, include_content=False) for row in rows]

    def get_document(self, case_id: str, document_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            return self._document_dict(self._document(session, case_id, document_id), include_content=True)

    # ------------------------------------------------------------------ 轉換
    def create_conversion(self, case_id: str, document_id: str, profile_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            self._document(session, case_id, document_id)
            profile = self._profile(session, case_id, profile_id)
            count = session.scalar(select(func.count()).select_from(ConversionRecord)
                                   .where(ConversionRecord.case_id == case_id))
            record = ConversionRecord(conversion_id=f"{case_id}-C{count + 1:04d}", case_id=case_id,
                                      document_id=document_id, profile_id=profile_id,
                                      profile_version=profile.version, status="queued")
            session.add(record)
            session.commit()
            return self._conversion_dict(record)

    def run_conversion(self, conversion_id: str) -> dict[str, Any]:
        """呼叫 LLM 並驗證結果。LLM 失敗或回應不是 JSON 時狀態為 failed，並保留原始回應。"""
        with self.session_factory() as session:
            record = session.get(ConversionRecord, conversion_id)
            if record is None:
                raise KCoreError("conversion_not_found", f"conversion {conversion_id} not found", 404)
            record.status = "running"
            session.commit()
            profile = self._profile(session, record.case_id, record.profile_id, record.profile_version)
            document = self._document(session, record.case_id, record.document_id)
            system, user = build_messages(profile.system_prompt, profile.user_prompt_template,
                                          document.content, document.filename)
            try:
                record.raw_response = self.client_factory(profile).complete(
                    system, user, draft_json_schema() if profile.structured_output else None)
                record.draft = extract_json(record.raw_response)
            except (LLMError, ValueError) as error:
                record.status, record.error, record.completed_at = "failed", str(error), utcnow()
                session.commit()
                return self._conversion_dict(record)
            self._revalidate(session, record, document)
            record.status, record.completed_at = "needs_review", utcnow()
            session.commit()
            return self._conversion_dict(record)

    def _revalidate(self, session, record: ConversionRecord, document: SourceDocumentRecord) -> list[dict[str, Any]]:
        payloads, report = validate_and_normalize(record.draft, record.case_id, document.sha256, document.content)
        existing = [p["episode_id"] for p in payloads if session.get(EpisodeRecord, p["episode_id"])]
        if existing:
            report["ok"] = False
            report["already_ingested"] = existing
        record.validation = report
        return payloads

    def _conversion(self, session, case_id: str, conversion_id: str) -> ConversionRecord:
        self.core._case(session, case_id)
        record = session.get(ConversionRecord, conversion_id)
        if record is None or record.case_id != case_id:
            raise KCoreError("conversion_not_found", f"conversion {conversion_id} not found for case {case_id}", 404)
        return record

    @staticmethod
    def _conversion_dict(record: ConversionRecord, include_detail: bool = False) -> dict[str, Any]:
        data = {
            "conversion_id": record.conversion_id, "case_id": record.case_id, "document_id": record.document_id,
            "profile_id": record.profile_id, "profile_version": record.profile_version, "status": record.status,
            "error": record.error, "can_approve": record.status == "needs_review" and bool(record.validation.get("ok")),
            "draft_edited": record.draft_edited, "reviewer": record.reviewer, "review_note": record.review_note,
            "ingested_episode_ids": record.ingested_episode_ids, "validation": record.validation,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "reviewed_at": record.reviewed_at.isoformat() if record.reviewed_at else None,
        }
        if include_detail:
            data["draft"] = record.draft
            data["raw_response"] = record.raw_response
        return data

    def list_conversions(self, case_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            self.core._case(session, case_id)
            rows = session.scalars(select(ConversionRecord).where(ConversionRecord.case_id == case_id)
                                   .order_by(ConversionRecord.created_at))
            return [self._conversion_dict(row) for row in rows]

    def get_conversion(self, case_id: str, conversion_id: str) -> dict[str, Any]:
        """詳細資料含草稿、原始回應，以及正規化後逐步重建的 AoV 預覽（供審核畫面）。"""
        with self.session_factory() as session:
            record = self._conversion(session, case_id, conversion_id)
            data = self._conversion_dict(record, include_detail=True)
            data["preview"] = []
            if record.draft is not None:
                document = self._document(session, case_id, record.document_id)
                payloads, _ = validate_and_normalize(record.draft, case_id, document.sha256, document.content)
                data["preview"] = build_preview(payloads)
            return data

    def edit_draft(self, case_id: str, conversion_id: str, draft: dict[str, Any]) -> dict[str, Any]:
        """審核者直接修正草稿，修正後重新驗證。"""
        with self.session_factory() as session:
            record = self._conversion(session, case_id, conversion_id)
            if record.status not in {"needs_review", "failed"}:
                raise KCoreError("conversion_closed", f"conversion is {record.status} and cannot be edited", 409)
            record.draft, record.draft_edited, record.status, record.error = copy.deepcopy(draft), True, "needs_review", None
            self._revalidate(session, record, self._document(session, case_id, record.document_id))
            session.commit()
            return self._conversion_dict(record, include_detail=True)

    def approve(self, case_id: str, conversion_id: str, reviewer: str, note: str | None) -> dict[str, Any]:
        """人工確認後才匯入；每條序列都帶有轉換來源，可追溯到文件、提示詞版本與審核人。"""
        with self.session_factory() as session:
            record = self._conversion(session, case_id, conversion_id)
            if record.status != "needs_review":
                raise KCoreError("conversion_closed", f"conversion is {record.status} and cannot be approved", 409)
            document = self._document(session, case_id, record.document_id)
            profile = self._profile(session, case_id, record.profile_id, record.profile_version)
            payloads = self._revalidate(session, record, document)
            session.commit()
            if not record.validation.get("ok"):
                raise KCoreError("draft_invalid", "draft has validation errors; fix the draft before approval", 422,
                                 record.validation)
            reviewed_at = utcnow()
            provenance = {
                "method": "llm_conversion", "conversion_id": record.conversion_id,
                "document_id": document.document_id, "document_sha256": document.sha256,
                "filename": document.filename, "profile_id": profile.profile_id,
                "profile_version": profile.version, "model": profile.model, "draft_edited": record.draft_edited,
                "reviewer": reviewer, "reviewed_at": reviewed_at.isoformat(),
            }
        ingested: list[str] = []
        for payload in payloads:
            self.core.ingest_episode(case_id, {**payload, "provenance": provenance})
            ingested.append(payload["episode_id"])
        with self.session_factory() as session:
            record = self._conversion(session, case_id, conversion_id)
            record.status, record.reviewer, record.review_note = "approved", reviewer, note
            record.reviewed_at, record.ingested_episode_ids = reviewed_at, ingested
            session.commit()
            return self._conversion_dict(record)

    def reject(self, case_id: str, conversion_id: str, reviewer: str, note: str | None) -> dict[str, Any]:
        with self.session_factory() as session:
            record = self._conversion(session, case_id, conversion_id)
            if record.status in {"approved", "rejected"}:
                raise KCoreError("conversion_closed", f"conversion is already {record.status}", 409)
            record.status, record.reviewer, record.review_note, record.reviewed_at = "rejected", reviewer, note, utcnow()
            session.commit()
            return self._conversion_dict(record)
