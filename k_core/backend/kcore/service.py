"""K 核心服務：案例登記、演變序列匯入與定案、模型訓練、模型版本管理與推論。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .aov import AoVError, apply_change, derive_change_features, is_empty_change, validate_workflow
from .database import (
    CaseRecord, ChangeEventRecord, EpisodeRecord, ModelVersionRecord, TrainingRunRecord, WorkflowSnapshotRecord,
    utcnow,
)
from .encoding import UNKNOWN, FeatureSchema, Vocabulary, event_token
from .evaluation import evaluate_model, label_recovery
from .hmm.hybrid import DEFAULT_PREDICTOR, PREDICTOR_TYPES, MixturePredictor, RouteBigramPredictor
from .hmm.model import HiddenMarkovModel
from .hmm.training import TrainingConfig, train_with_selection
from .synthetic import LATENT_PHASE_KEY, generate_episodes

# ai_agent：由 AI 代理人進行的修改（例如 CVflow 中由大模型設計流程圖）
ACTOR_TYPES = {"human", "system", "synthetic", "ai_agent"}
# human：工作人員寫的原因；llm：K 的 LLM 推論的原因；actor：修改者（人或 AI 代理人）當時自己記錄的原因
REASON_SOURCES = {"human", "llm", "actor"}
DOMAIN_KINDS = {"erp", "cvflow", "hardware", "other"}
HYBRID_BACKOFF = 2.0  # 轉移表資料不足時退回狀態發射分布的強度；固定值，不以測試資料調整
EVENT_FIELDS = ("sequence_index", "occurred_at", "actor_type", "operation_type", "before_version_id",
                "after_version_id", "change", "reason", "reason_source", "context_delta",
                "related_ui_event_ids", "attributes")


class KCoreError(Exception):
    """可辨識的錯誤狀態；code 供呼叫端判斷，details 列出具體原因。"""

    def __init__(self, code: str, message: str, status: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details


def _parse_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _payload_hash(event: dict[str, Any]) -> str:
    canonical = {key: event.get(key) for key in EVENT_FIELDS}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass
class ReplayResult:
    ok: bool
    problems: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = field(default_factory=list)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)


class KCoreService:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    # ------------------------------------------------------------------ 案例
    def create_case(self, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("domain_kind") not in DOMAIN_KINDS:
            raise KCoreError("invalid_case", f"domain_kind must be one of {sorted(DOMAIN_KINDS)}")
        with self.session_factory() as session:
            if session.get(CaseRecord, data["case_id"]):
                raise KCoreError("case_exists", f"case {data['case_id']} already exists", 409)
            record = CaseRecord(
                case_id=data["case_id"], name=data.get("name") or data["case_id"], domain_kind=data["domain_kind"],
                erp_instance_id=data.get("erp_instance_id"), recording_enabled=data.get("recording_enabled", True),
            )
            session.add(record)
            session.commit()
            return self._case_dict(session, record)

    def list_cases(self) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            return [self._case_dict(session, record) for record in session.scalars(select(CaseRecord))]

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            return self._case_dict(session, self._case(session, case_id))

    def _case(self, session: Session, case_id: str) -> CaseRecord:
        record = session.get(CaseRecord, case_id)
        if record is None:
            raise KCoreError("case_not_found", f"case {case_id} does not exist", 404)
        return record

    def _case_dict(self, session: Session, record: CaseRecord) -> dict[str, Any]:
        counts = dict(session.execute(
            select(EpisodeRecord.status, func.count()).where(EpisodeRecord.case_id == record.case_id)
            .group_by(EpisodeRecord.status)).all())
        active = session.scalar(select(ModelVersionRecord.model_version_id).where(
            ModelVersionRecord.case_id == record.case_id, ModelVersionRecord.status == "active"))
        return {
            "case_id": record.case_id, "name": record.name, "domain_kind": record.domain_kind,
            "erp_instance_id": record.erp_instance_id, "recording_enabled": record.recording_enabled,
            "episodes": {"open": counts.get("open", 0), "finalized": counts.get("finalized", 0)},
            "active_model_version_id": active, "created_at": record.created_at.isoformat(),
        }

    # ------------------------------------------------------------------ 演變序列
    def create_episode(self, case_id: str, data: dict[str, Any]) -> dict[str, Any]:
        workflow = data["initial_workflow"]
        problems = validate_workflow(workflow.get("nodes", []), workflow.get("edges", []))
        if problems:
            raise KCoreError("invalid_workflow", "initial workflow is not a valid AoV (DAG)", 422, problems)
        with self.session_factory() as session:
            self._case(session, case_id)
            if session.get(EpisodeRecord, data["episode_id"]):
                raise KCoreError("episode_exists", f"episode {data['episode_id']} already exists", 409)
            session.add(EpisodeRecord(
                episode_id=data["episode_id"], case_id=case_id, work_order_id=data.get("work_order_id"),
                workflow_id=workflow["workflow_id"], initial_version_id=workflow["version_id"],
                initial_context=data.get("initial_context") or {}, is_synthetic=bool(data.get("is_synthetic")),
                provenance=data.get("provenance") or {"method": "api"},
                started_at=_parse_time(data.get("started_at")) or utcnow(),
            ))
            session.add(WorkflowSnapshotRecord(
                episode_id=data["episode_id"], workflow_id=workflow["workflow_id"], version_id=workflow["version_id"],
                nodes=workflow["nodes"], edges=workflow["edges"],
            ))
            session.commit()
        return self.get_episode(case_id, data["episode_id"])

    def _episode(self, session: Session, case_id: str, episode_id: str) -> EpisodeRecord:
        self._case(session, case_id)
        episode = session.get(EpisodeRecord, episode_id)
        if episode is None:
            raise KCoreError("episode_not_found", f"episode {episode_id} does not exist", 404)
        if episode.case_id != case_id:
            raise KCoreError("case_mismatch", f"episode {episode_id} belongs to case {episode.case_id}", 409)
        return episode

    def list_episodes(self, case_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            self._case(session, case_id)
            rows = session.scalars(select(EpisodeRecord).where(EpisodeRecord.case_id == case_id)
                                   .order_by(EpisodeRecord.started_at, EpisodeRecord.episode_id))
            return [self._episode_summary(session, row) for row in rows]

    def _episode_summary(self, session: Session, episode: EpisodeRecord) -> dict[str, Any]:
        event_count = session.scalar(select(func.count()).select_from(ChangeEventRecord)
                                     .where(ChangeEventRecord.episode_id == episode.episode_id))
        return {
            "episode_id": episode.episode_id, "case_id": episode.case_id, "work_order_id": episode.work_order_id,
            "workflow_id": episode.workflow_id, "status": episode.status, "is_synthetic": episode.is_synthetic,
            "initial_version_id": episode.initial_version_id, "final_version_id": episode.final_version_id,
            "event_count": event_count, "provenance": episode.provenance or {},
            "started_at": episode.started_at.isoformat(),
            "finalized_at": episode.finalized_at.isoformat() if episode.finalized_at else None,
        }

    def get_episode(self, case_id: str, episode_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            episode = self._episode(session, case_id, episode_id)
            events = self._events(session, episode_id)
            snapshots = session.scalars(select(WorkflowSnapshotRecord)
                                        .where(WorkflowSnapshotRecord.episode_id == episode_id)
                                        .order_by(WorkflowSnapshotRecord.id))
            return {
                **self._episode_summary(session, episode),
                "initial_context": episode.initial_context,
                "events": [self._event_dict(event) for event in events],
                "snapshots": [{"version_id": s.version_id, "nodes": s.nodes, "edges": s.edges} for s in snapshots],
            }

    def _events(self, session: Session, episode_id: str) -> list[ChangeEventRecord]:
        return list(session.scalars(select(ChangeEventRecord).where(ChangeEventRecord.episode_id == episode_id)
                                    .order_by(ChangeEventRecord.sequence_index)))

    @staticmethod
    def _event_dict(event: ChangeEventRecord) -> dict[str, Any]:
        return {
            "event_id": event.event_id, "sequence_index": event.sequence_index,
            "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
            "actor_type": event.actor_type, "operation_type": event.operation_type,
            "before_version_id": event.before_version_id, "after_version_id": event.after_version_id,
            "change": event.change, "reason": event.reason, "reason_source": event.reason_source,
            "context_delta": event.context_delta, "related_ui_event_ids": event.related_ui_event_ids,
            "attributes": event.attributes, "features": event.features,
        }

    def _check_event(self, event: dict[str, Any]) -> list[str]:
        problems = [f"missing {name}" for name in ("event_id", "sequence_index", "operation_type",
                                                  "before_version_id", "after_version_id", "change")
                    if event.get(name) in (None, "")]
        if problems:
            return problems
        if not isinstance(event["sequence_index"], int) or event["sequence_index"] < 1:
            problems.append("sequence_index must be an integer >= 1")
        if event.get("actor_type", "human") not in ACTOR_TYPES:
            problems.append(f"actor_type must be one of {sorted(ACTOR_TYPES)}")
        if event.get("reason") and event.get("reason_source") not in REASON_SOURCES:
            problems.append(f"reason_source must be one of {sorted(REASON_SOURCES)} when reason is provided")
        if not event.get("reason") and event.get("reason_source"):
            problems.append("reason_source requires a reason")
        if not isinstance(event["change"], dict) or is_empty_change(event["change"]):
            problems.append("change must contain at least one modification")
        if event["before_version_id"] == event["after_version_id"]:
            problems.append("after_version_id must differ from before_version_id")
        return problems

    def add_events(self, case_id: str, episode_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
        """批次匯入事件。可不依順序到達；完整性與順序在定案時檢查。重送相同事件會被辨識為重複。"""
        accepted: list[str] = []
        duplicates: list[str] = []
        rejected: list[dict[str, Any]] = []
        with self.session_factory() as session:
            episode = self._episode(session, case_id, episode_id)
            if episode.status != "open":
                raise KCoreError("episode_finalized", f"episode {episode_id} is already finalized", 409)
            by_index = {event.sequence_index: event.event_id for event in self._events(session, episode_id)}
            for event in events:
                problems = self._check_event(event)
                if problems:
                    rejected.append({"event_id": event.get("event_id"), "problems": problems})
                    continue
                digest = _payload_hash(event)
                existing = session.get(ChangeEventRecord, event["event_id"])
                if existing is not None:
                    if existing.episode_id == episode_id and existing.payload_hash == digest:
                        duplicates.append(event["event_id"])
                    else:
                        rejected.append({"event_id": event["event_id"],
                                         "problems": ["event_id already exists with different content"]})
                    continue
                if event["sequence_index"] in by_index:
                    rejected.append({"event_id": event["event_id"], "problems": [
                        f"sequence_index {event['sequence_index']} already used by {by_index[event['sequence_index']]}"]})
                    continue
                session.add(ChangeEventRecord(
                    event_id=event["event_id"], episode_id=episode_id, sequence_index=event["sequence_index"],
                    occurred_at=_parse_time(event.get("occurred_at")), actor_type=event.get("actor_type", "human"),
                    operation_type=event["operation_type"], before_version_id=event["before_version_id"],
                    after_version_id=event["after_version_id"], change=event["change"],
                    reason=event.get("reason") or None, reason_source=event.get("reason_source") or None,
                    context_delta=event.get("context_delta") or {},
                    related_ui_event_ids=event.get("related_ui_event_ids") or [],
                    attributes=event.get("attributes") or {}, payload_hash=digest,
                ))
                by_index[event["sequence_index"]] = event["event_id"]
                accepted.append(event["event_id"])
            session.commit()
        return {"accepted": accepted, "duplicates": duplicates, "rejected": rejected}

    def _replay(self, session: Session, episode: EpisodeRecord, events: list[ChangeEventRecord]) -> ReplayResult:
        """從初始快照依序套用修改；檢查順序連續、版本鏈一致、每一步都是合法 DAG。"""
        result = ReplayResult(ok=True)
        initial = session.scalar(select(WorkflowSnapshotRecord).where(
            WorkflowSnapshotRecord.episode_id == episode.episode_id,
            WorkflowSnapshotRecord.version_id == episode.initial_version_id))
        nodes, edges = initial.nodes, initial.edges
        current_version = episode.initial_version_id
        seen_versions = {current_version}
        for expected, event in enumerate(events, start=1):
            problems: list[str] = []
            if event.sequence_index != expected:
                problems.append(f"missing sequence_index {expected} (next received is {event.sequence_index})")
            if event.before_version_id != current_version:
                problems.append(f"before_version_id {event.before_version_id} does not match current {current_version}")
            if event.after_version_id in seen_versions:
                problems.append(f"after_version_id {event.after_version_id} was already used")
            if not problems:
                try:
                    new_nodes, new_edges = apply_change(nodes, edges, event.change)
                except AoVError as error:
                    problems.extend(error.problems)
            if problems:
                result.ok = False
                result.problems.append({"event_id": event.event_id, "sequence_index": event.sequence_index,
                                        "problems": problems})
                break
            result.features[event.event_id] = derive_change_features(nodes, new_nodes, event.change)
            nodes, edges = new_nodes, new_edges
            current_version = event.after_version_id
            seen_versions.add(current_version)
            result.snapshots.append((current_version, nodes, edges))
        return result

    def finalize_episode(self, case_id: str, episode_id: str, finalized_at: Any = None) -> dict[str, Any]:
        with self.session_factory() as session:
            episode = self._episode(session, case_id, episode_id)
            if episode.status == "finalized":
                raise KCoreError("episode_finalized", f"episode {episode_id} is already finalized", 409)
            events = self._events(session, episode_id)
            if not events:
                raise KCoreError("finalize_failed", "episode has no change events", 422)
            replay = self._replay(session, episode, events)
            if not replay.ok:
                raise KCoreError("finalize_failed", "episode events cannot be replayed into a valid AoV", 422,
                                 replay.problems)
            for event in events:
                event.features = replay.features[event.event_id]
            for version_id, nodes, edges in replay.snapshots:
                session.add(WorkflowSnapshotRecord(episode_id=episode_id, workflow_id=episode.workflow_id,
                                                   version_id=version_id, nodes=nodes, edges=edges))
            episode.status = "finalized"
            episode.final_version_id = replay.snapshots[-1][0]
            episode.finalized_at = _parse_time(finalized_at) or utcnow()
            session.commit()
        return self.get_episode(case_id, episode_id)

    def ingest_episode(self, case_id: str, payload: dict[str, Any], finalize: bool = True) -> dict[str, Any]:
        """一次提交完整演變序列（初始流程＋事件），方便批次匯入與示意資料。"""
        self.create_episode(case_id, payload)
        result = self.add_events(case_id, payload["episode_id"], payload.get("events", []))
        if result["rejected"]:
            raise KCoreError("invalid_event", "some events were rejected", 422, result["rejected"])
        return self.finalize_episode(case_id, payload["episode_id"]) if finalize else self.get_episode(
            case_id, payload["episode_id"])

    def generate_synthetic(self, case_id: str, count: int, seed: int) -> dict[str, Any]:
        with self.session_factory() as session:
            case = self._case(session, case_id)
            start = session.scalar(select(func.count()).select_from(EpisodeRecord)
                                   .where(EpisodeRecord.case_id == case_id))
            if case.domain_kind != "erp":
                raise KCoreError("invalid_case", "synthetic AoV episodes are only defined for erp cases")
        created = [self.ingest_episode(case_id, {**payload, "provenance": {"method": "synthetic", "seed": seed}})["episode_id"]
                   for payload in generate_episodes(case_id, count, seed, start_index=start)]
        return {"case_id": case_id, "created": created, "is_synthetic": True, "seed": seed}

    # ------------------------------------------------------------------ 訓練
    def _training_sequences(self, session: Session, case_id: str) -> list[tuple[EpisodeRecord, list[ChangeEventRecord]]]:
        episodes = session.scalars(select(EpisodeRecord).where(
            EpisodeRecord.case_id == case_id, EpisodeRecord.status == "finalized")
            .order_by(EpisodeRecord.started_at, EpisodeRecord.episode_id))
        return [(episode, self._events(session, episode.episode_id)) for episode in episodes]

    def train(self, case_id: str, schema: FeatureSchema, config: TrainingConfig, test_fraction: float = 0.2,
              min_train_episodes: int = 5, activate: bool = True) -> dict[str, Any]:
        if not 0 < test_fraction < 1:
            raise KCoreError("invalid_training", "test_fraction must be between 0 and 1")
        configuration = {"feature_schema": schema.to_dict(), "training": config.to_dict(),
                         "test_fraction": test_fraction, "min_train_episodes": min_train_episodes,
                         "split_method": "chronological by episode started_at"}
        with self.session_factory() as session:
            self._case(session, case_id)
            run_count = session.scalar(select(func.count()).select_from(TrainingRunRecord)
                                       .where(TrainingRunRecord.case_id == case_id))
            run = TrainingRunRecord(training_run_id=f"{case_id}-R{run_count + 1:03d}", case_id=case_id,
                                    status="running", configuration=configuration)
            session.add(run)
            data = self._training_sequences(session, case_id)
            n_test = max(1, round(len(data) * test_fraction))
            train_part, test_part = data[:len(data) - n_test], data[len(data) - n_test:]
            run.dataset = {
                "finalized_episodes": len(data), "train_episodes": [e.episode_id for e, _ in train_part],
                "test_episodes": [e.episode_id for e, _ in test_part],
                "synthetic_episodes": sum(episode.is_synthetic for episode, _ in data),
            }
            if len(train_part) < min_train_episodes:
                run.status = "insufficient_data"
                run.message = (f"need at least {min_train_episodes} training episodes plus a test split; "
                               f"have {len(data)} finalized episodes")
                run.finished_at = utcnow()
                session.commit()
                return self._run_dict(run)

            def tokens(events: list[ChangeEventRecord]) -> list[str]:
                return [event_token(event.operation_type, event.features or {}, schema) for event in events]

            train_tokens = [tokens(events) for _, events in train_part]
            vocabulary = Vocabulary.build(train_tokens, schema.min_count)
            train_seq = [[vocabulary.normalize(token) for token in seq] for seq in train_tokens]
            test_seq = [[vocabulary.normalize(token) for token in tokens(events)] for _, events in test_part]
            try:
                result = train_with_selection([vocabulary.encode(seq) for seq in train_seq], vocabulary.symbols, config)
            except (RuntimeError, ValueError) as error:
                run.status, run.message, run.finished_at = "failed", str(error), utcnow()
                session.commit()
                return self._run_dict(run)

            # HMM＋上一次修改：每個隱藏狀態一張「上一次 → 下一次」轉移表，只用訓練資料估計
            hybrid = RouteBigramPredictor.fit(result.model, train_seq, backoff=HYBRID_BACKOFF)
            mixture = MixturePredictor.fit(result.model, train_seq, config.smoothing)
            metrics: dict[str, Any] = {
                "training": result.summary(),
                "test": evaluate_model(result.model, train_seq, test_seq, config.smoothing, hybrid),
                "vocabulary_size": len(vocabulary.symbols),
                "test_unknown_rate": (sum(token == UNKNOWN for seq in test_seq for token in seq)
                                      / max(1, sum(len(seq) for seq in test_seq))),
            }
            labels = [[event.attributes.get(LATENT_PHASE_KEY) for event in events] for _, events in test_part]
            if all(label is not None for seq in labels for label in seq):
                metrics["synthetic_label_recovery"] = label_recovery(result.model, test_seq, labels)

            model_count = session.scalar(select(func.count()).select_from(ModelVersionRecord)
                                         .where(ModelVersionRecord.case_id == case_id))
            model_id = f"{case_id}-M{model_count + 1:03d}"
            if activate:
                for other in session.scalars(select(ModelVersionRecord).where(ModelVersionRecord.case_id == case_id)):
                    other.status = "inactive"
            session.add(ModelVersionRecord(
                model_version_id=model_id, case_id=case_id, training_run_id=run.training_run_id,
                feature_schema=schema.to_dict(), parameters={**result.model.to_dict(), "predictors": {
                    hybrid.method: hybrid.to_dict(), mixture.method: mixture.to_dict()}},
                metrics=metrics,
                status="active" if activate else "inactive",
            ))
            run.status, run.metrics, run.model_version_id, run.finished_at = "succeeded", metrics, model_id, utcnow()
            session.commit()
            return self._run_dict(run)

    @staticmethod
    def _run_dict(run: TrainingRunRecord) -> dict[str, Any]:
        return {
            "training_run_id": run.training_run_id, "case_id": run.case_id, "status": run.status,
            "message": run.message, "configuration": run.configuration, "dataset": run.dataset,
            "metrics": run.metrics, "model_version_id": run.model_version_id,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }

    def get_training_run(self, case_id: str, run_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            self._case(session, case_id)
            run = session.get(TrainingRunRecord, run_id)
            if run is None or run.case_id != case_id:
                raise KCoreError("training_run_not_found", f"training run {run_id} not found for case {case_id}", 404)
            return self._run_dict(run)

    # ------------------------------------------------------------------ 模型
    def list_models(self, case_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            self._case(session, case_id)
            rows = session.scalars(select(ModelVersionRecord).where(ModelVersionRecord.case_id == case_id)
                                   .order_by(ModelVersionRecord.created_at))
            return [self._model_dict(row, include_parameters=False) for row in rows]

    def get_model(self, case_id: str, model_version_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            return self._model_dict(self._model(session, case_id, model_version_id), include_parameters=True)

    def _model(self, session: Session, case_id: str, model_version_id: str) -> ModelVersionRecord:
        self._case(session, case_id)
        if model_version_id == "active":
            record = session.scalar(select(ModelVersionRecord).where(
                ModelVersionRecord.case_id == case_id, ModelVersionRecord.status == "active"))
            if record is None:
                raise KCoreError("no_active_model", f"case {case_id} has no active model", 404)
            return record
        record = session.get(ModelVersionRecord, model_version_id)
        if record is None or record.case_id != case_id:
            raise KCoreError("model_not_found", f"model {model_version_id} not found for case {case_id}", 404)
        return record

    @staticmethod
    def _model_dict(record: ModelVersionRecord, include_parameters: bool) -> dict[str, Any]:
        data = {
            "model_version_id": record.model_version_id, "case_id": record.case_id,
            "training_run_id": record.training_run_id, "feature_schema": record.feature_schema,
            "status": record.status, "created_at": record.created_at.isoformat(),
            "n_states": len(record.parameters["states"]), "n_observations": len(record.parameters["observations"]),
            "selected_k": record.metrics.get("training", {}).get("selected_k"),
            "test_metrics": record.metrics.get("test"),
        }
        if include_parameters:
            model = HiddenMarkovModel.from_dict(record.parameters)
            data["parameters"] = record.parameters
            data["metrics"] = record.metrics
            data["stationary_distribution"] = model.stationary_distribution().tolist()
            data["top_observations_per_state"] = {
                state: [{"observation": model.observations[i], "probability": float(model.emission[row, i])}
                        for i in np.argsort(-model.emission[row])[:5]]
                for row, state in enumerate(model.states)
            }
        return data

    def activate_model(self, case_id: str, model_version_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            target = self._model(session, case_id, model_version_id)
            for record in session.scalars(select(ModelVersionRecord).where(ModelVersionRecord.case_id == case_id)):
                record.status = "active" if record.model_version_id == target.model_version_id else "inactive"
            session.commit()
            return self._model_dict(target, include_parameters=False)

    def infer(self, case_id: str, model_version_id: str, episode_id: str, prefix_length: int | None = None,
              top_k: int = 5, method: str = "auto") -> dict[str, Any]:
        """只使用該序列前 prefix_length 個事件（當時可取得的前綴）進行推論。"""
        with self.session_factory() as session:
            record = self._model(session, case_id, model_version_id)
            episode = self._episode(session, case_id, episode_id)
            events = self._events(session, episode_id)
            limit = len(events) if prefix_length is None else prefix_length
            if not 0 <= limit <= len(events):
                raise KCoreError("invalid_prefix", f"prefix_length must be between 0 and {len(events)}", 422)
            prefix_events = events[:limit]
            replay = self._replay(session, episode, prefix_events)
            if not replay.ok:
                raise KCoreError("invalid_prefix", "prefix events cannot be replayed", 422, replay.problems)
            schema = FeatureSchema.from_dict(record.feature_schema)
            model = HiddenMarkovModel.from_dict(record.parameters)
            raw_tokens = [event_token(event.operation_type, replay.features[event.event_id], schema)
                          for event in prefix_events]
            tokens = [token if token in model.observations else UNKNOWN for token in raw_tokens]
            beliefs = model.filter(tokens) if tokens else None
            hmm_distribution = model.next_observation_distribution(tokens)
            # 預測方法：auto = 模型有保存結合方法時用預設的結合方法，否則只用 HMM
            saved = dict(record.parameters.get("predictors") or {})
            if "hybrid" in record.parameters:  # 相容較早保存的單一結合方法
                saved.setdefault(record.parameters["hybrid"]["type"], record.parameters["hybrid"])
            if method == "auto":
                method = DEFAULT_PREDICTOR if DEFAULT_PREDICTOR in saved else (next(iter(saved), "hmm"))
            if method == "hmm":
                next_distribution = hmm_distribution
            elif method in saved:
                next_distribution = PREDICTOR_TYPES[method].from_dict(model, saved[method]).predict(tokens)
            else:
                raise KCoreError("invalid_method", f"method must be one of {['auto', 'hmm', *saved]}", 422)
            order = np.argsort(-next_distribution)[:top_k]
            hmm_order = np.argsort(-hmm_distribution)[:top_k]
            return {
                "method": method, "case_id": case_id, "episode_id": episode_id,
                "model_version_id": record.model_version_id, "feature_schema_version": schema.version,
                "prefix_length": limit, "is_synthetic_episode": episode.is_synthetic,
                "steps": [
                    {"event_id": event.event_id, "token": raw, "encoded_as": token,
                     "belief": dict(zip(model.states, beliefs[i].round(6).tolist(), strict=True))}
                    for i, (event, raw, token) in enumerate(zip(prefix_events, raw_tokens, tokens, strict=True))
                ],
                "viterbi_path": model.viterbi(tokens)[1],
                "prefix_log_likelihood": model.log_likelihood(tokens),
                "next_observation": [{"observation": model.observations[i], "probability": float(next_distribution[i])}
                                     for i in order],
                "available_methods": ["hmm", *saved],
                "next_observation_hmm_only": [{"observation": model.observations[i],
                                               "probability": float(hmm_distribution[i])} for i in hmm_order],
            }

    # ------------------------------------------------------------------ 研究頁面
    @staticmethod
    def _predictors(record: ModelVersionRecord, model: HiddenMarkovModel) -> dict[str, Any]:
        """模型版本可用的所有預測方法：method -> predict(prefix) 函式。"""
        saved = dict(record.parameters.get("predictors") or {})
        if "hybrid" in record.parameters:
            saved.setdefault(record.parameters["hybrid"]["type"], record.parameters["hybrid"])
        methods: dict[str, Any] = {"hmm": model.next_observation_distribution}
        for name, data in saved.items():
            methods[name] = PREDICTOR_TYPES[name].from_dict(model, data).predict
        return methods

    def replay(self, case_id: str, model_version_id: str, episode_id: str, top_k: int = 5) -> dict[str, Any]:
        """逐步重播一條序列：第 t 步只用前 t 個修改預測第 t+1 個，並與實際修改比對。

        每一步回傳：當時的路線判斷（濾波信念，只用已知前綴）、各方法的前幾名預測、實際修改在各方法中的排名。
        """
        with self.session_factory() as session:
            record = self._model(session, case_id, model_version_id)
            episode = self._episode(session, case_id, episode_id)
            events = self._events(session, episode_id)
            replay = self._replay(session, episode, events)
            if not replay.ok:
                raise KCoreError("invalid_episode", "episode events cannot be replayed", 422, replay.problems)
            schema = FeatureSchema.from_dict(record.feature_schema)
            model = HiddenMarkovModel.from_dict(record.parameters)
            predictors = self._predictors(record, model)
            raw = [event_token(e.operation_type, replay.features[e.event_id], schema) for e in events]
            tokens = [t if t in model.observations else UNKNOWN for t in raw]
            beliefs = model.filter(tokens) if tokens else np.empty((0, len(model.states)))
            totals = {name: {"hit1": 0, "hit3": 0, "log": 0.0} for name in predictors}
            steps = []
            for t, event in enumerate(events):
                prefix = tokens[:t]
                route = model.initial if t == 0 else beliefs[t - 1] @ model.transition
                actual = model.observation_id(tokens[t])
                predictions = {}
                for name, predict in predictors.items():
                    distribution = predict(prefix)
                    rank = int((distribution > distribution[actual]).sum()) + 1
                    totals[name]["hit1"] += rank == 1
                    totals[name]["hit3"] += rank <= 3
                    totals[name]["log"] += float(np.log(max(distribution[actual], 1e-300)))
                    order = np.argsort(-distribution)[:top_k]
                    predictions[name] = {
                        "top": [{"observation": model.observations[i], "probability": float(distribution[i])}
                                for i in order],
                        "actual_rank": rank, "actual_probability": float(distribution[actual]),
                    }
                steps.append({
                    "sequence_index": event.sequence_index, "event_id": event.event_id,
                    "operation_type": event.operation_type, "token": raw[t], "encoded_as": tokens[t],
                    "route_before": dict(zip(model.states, np.round(route, 6).tolist(), strict=True)),
                    "belief_after": dict(zip(model.states, np.round(beliefs[t], 6).tolist(), strict=True)),
                    "predictions": predictions,
                })
            n = len(events)
            return {
                "case_id": case_id, "episode_id": episode_id, "model_version_id": record.model_version_id,
                "feature_schema_version": schema.version, "states": list(model.states),
                "methods": list(predictors), "default_method": DEFAULT_PREDICTOR if DEFAULT_PREDICTOR in predictors
                else "hmm", "steps": steps,
                "summary": {name: {"top1": v["hit1"] / n if n else None, "top3": v["hit3"] / n if n else None,
                                   "mean_log_probability": v["log"] / n if n else None} for name, v in totals.items()},
            }

    def model_analysis(self, case_id: str, model_version_id: str, group_by: str | None = None) -> dict[str, Any]:
        """以 Viterbi 解碼案例中所有已定案的序列，描述每個隱藏狀態。

        group_by：依序列的某個屬性分組（先找 initial_context，再找 provenance），例如 CVflow 的 teacher。
        整題成敗取自 provenance.outcome.final_ok（若有）。這些屬性只用於分析，不參與推論。
        """
        with self.session_factory() as session:
            record = self._model(session, case_id, model_version_id)
            data = self._training_sequences(session, case_id)
            schema = FeatureSchema.from_dict(record.feature_schema)
            model = HiddenMarkovModel.from_dict(record.parameters)
            state_events = {s: 0 for s in model.states}
            operations = {s: {} for s in model.states}
            groups: dict[str, dict[str, int]] = {}
            outcomes: dict[str, dict[str, int]] = {}
            switched = 0
            for episode, events in data:
                tokens = [event_token(e.operation_type, e.features or {}, schema) for e in events]
                tokens = [t if t in model.observations else UNKNOWN for t in tokens]
                _, path = model.viterbi(tokens)
                switched += any(a != b for a, b in zip(path, path[1:]))
                group = None
                if group_by:
                    group = (episode.initial_context or {}).get(group_by, (episode.provenance or {}).get(group_by))
                outcome = ((episode.provenance or {}).get("outcome") or {}).get("final_ok")
                for state, event in zip(path, events, strict=True):
                    state_events[state] += 1
                    operations[state][event.operation_type] = operations[state].get(event.operation_type, 0) + 1
                    if group is not None:
                        groups.setdefault(str(group), {s: 0 for s in model.states})[state] += 1
                    if outcome is not None:
                        key = "final_ok" if outcome else "final_fail"
                        outcomes.setdefault(key, {s: 0 for s in model.states})[state] += 1
            share = lambda counts: {k: (v / sum(counts.values()) if sum(counts.values()) else 0.0)  # noqa: E731
                                    for k, v in counts.items()}
            states = []
            for row, state in enumerate(model.states):
                top = np.argsort(-model.emission[row])[:6]
                states.append({
                    "state": state, "events": state_events[state],
                    "self_transition": float(model.transition[row, row]), "initial": float(model.initial[row]),
                    "top_observations": [{"observation": model.observations[i],
                                          "probability": float(model.emission[row, i])} for i in top],
                    "operation_share": share(operations[state]),
                })
            return {
                "case_id": case_id, "model_version_id": record.model_version_id, "episodes": len(data),
                "events": sum(state_events.values()), "states": states,
                "transition": np.round(model.transition, 6).tolist(),
                "episodes_with_state_switch": switched / len(data) if data else 0.0,
                "group_by": group_by, "state_share_by_group": {g: share(c) for g, c in sorted(groups.items())},
                "state_share_by_outcome": {k: share(c) for k, c in outcomes.items()},
                "test_metrics": record.metrics.get("test"),
                "training": {"selected_k": record.metrics.get("training", {}).get("selected_k"),
                             "candidates": [{k: c.get(k) for k in ("k", "log_likelihood", "bic", "aic", "converged")}
                                            for c in record.metrics.get("training", {}).get("candidates", [])]},
                "feature_schema": record.feature_schema,
            }
