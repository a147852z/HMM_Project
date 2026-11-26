"""K 核心 HTTP 介面（/api/v1）。路徑依規格書第 11 節草案，並檢查路徑與資料內的案例一致。"""
from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .assist import AssistService
from .conversion import ConversionService
from .conversion.service import ClientFactory
from .database import make_session_factory
from .encoding import ALLOWED_FIELDS, FeatureSchema
from .fsm import StateMachineService
from .hmm.training import TrainingConfig
from .service import KCoreError, KCoreService


class CaseIn(BaseModel):
    case_id: str = Field(min_length=1, max_length=64)
    name: str | None = None
    domain_kind: Literal["erp", "cvflow", "hardware", "other"]
    erp_instance_id: str | None = None
    recording_enabled: bool = True


class WorkflowIn(BaseModel):
    workflow_id: str
    version_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class EpisodeIn(BaseModel):
    episode_id: str = Field(min_length=1, max_length=64)
    case_id: str | None = None
    work_order_id: str | None = None
    is_synthetic: bool = False
    started_at: str | None = None
    initial_context: dict[str, Any] = Field(default_factory=dict)
    initial_workflow: WorkflowIn


class EventsIn(BaseModel):
    events: list[dict[str, Any]]


class FinalizeIn(BaseModel):
    finalized_at: str | None = None


class SyntheticIn(BaseModel):
    count: int = Field(default=60, ge=1, le=2000)
    seed: int = 0


class TrainingIn(BaseModel):
    feature_fields: list[Literal[ALLOWED_FIELDS]] = ["operation_type", "activity_type"]  # type: ignore[valid-type]
    feature_schema_version: str = "fs-v1"
    min_count: int = Field(default=1, ge=1)
    activity_map: dict[str, str] | None = None
    activity_default: str | None = None
    k_min: int = Field(default=1, ge=1, le=20)
    k_max: int = Field(default=5, ge=1, le=20)
    restarts: int = Field(default=3, ge=1, le=50)
    n_iter: int = Field(default=200, ge=1, le=2000)
    tol: float = Field(default=1e-4, gt=0)
    smoothing: float = Field(default=0.1, ge=0)
    seed: int = 0
    criterion: Literal["bic", "aic"] = "bic"
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    min_train_episodes: int = Field(default=5, ge=1)
    activate: bool = True


class InferIn(BaseModel):
    episode_id: str
    prefix_length: int | None = Field(default=None, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    method: str = "auto"  # auto / hmm / hmm_bigram_mix / hmm_route_bigram


class ProfileIn(BaseModel):
    profile_id: str = Field(min_length=1, max_length=64)
    name: str
    base_url: str = "http://localhost:1234/v1"
    model: str
    api_key_env: str | None = None
    system_prompt: str
    user_prompt_template: str = "以下是檔案 {filename} 的內容，請依系統指示整理：\n\n{document}"
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    structured_output: bool = True
    timeout_seconds: int = Field(default=600, ge=10, le=7200)


class ProfileUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    system_prompt: str | None = None
    user_prompt_template: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    structured_output: bool | None = None
    timeout_seconds: int | None = Field(default=None, ge=10, le=7200)


class ConversionIn(BaseModel):
    document_id: str
    profile_id: str


class DraftIn(BaseModel):
    draft: dict[str, Any]


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    note: str | None = None


class ReplayIn(BaseModel):
    episode_id: str
    top_k: int = Field(default=5, ge=1, le=20)


class AssistIn(BaseModel):
    episode_id: str
    prefix_length: int = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=50)
    model_version_id: str = "active"
    use_previous_result: bool = False  # 實驗選項：納入上一版的執行結果


class WorkflowCheckIn(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def build_assist_router(assist: AssistService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/cases/{case_id}")

    @router.post("/aov/assist")
    def assist_step(case_id: str, body: AssistIn) -> dict[str, Any]:
        return assist.assist(case_id, body.episode_id, body.prefix_length, body.top_k, body.model_version_id,
                             body.use_previous_result)

    @router.post("/aov/validate")
    def validate(case_id: str, body: WorkflowCheckIn) -> dict[str, Any]:
        assist.core.get_case(case_id)
        return assist.validate(body.nodes, body.edges)

    @router.get("/aov/assist-evaluation")
    def evaluation(case_id: str, model_version_id: str = "active") -> dict[str, Any]:
        return assist.evaluate(case_id, model_version_id)

    return router


def build_state_machine_router(machines: StateMachineService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/cases/{case_id}")

    @router.get("/episodes/{episode_id}/state-machine")
    def state_machine(case_id: str, episode_id: str,
                      granularity: Literal["full", "structure", "category"] = "structure",
                      model_version_id: str = "active") -> dict[str, Any]:
        """一條演變過程的有限狀態機：狀態＝流程圖（依粒度合併），轉移＝修改；有模型時附上隱藏狀態。"""
        return machines.episode(case_id, episode_id, granularity, model_version_id)

    @router.get("/tasks/{task_id}/state-machine")
    def task_state_machine(case_id: str, task_id: str,
                           granularity: Literal["full", "structure", "category"] = "structure",
                           model_version_id: str = "active") -> dict[str, Any]:
        """同一題（work_order_id 相同）所有演變過程合併的有限狀態機：每條演變過程一個泳道，共用的狀態就是路線交會。"""
        return machines.task(case_id, task_id, granularity, model_version_id)

    return router


def build_conversion_router(conversions: ConversionService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/cases/{case_id}")

    @router.post("/llm-profiles", status_code=201)
    def create_profile(case_id: str, body: ProfileIn) -> dict[str, Any]:
        return conversions.create_profile(case_id, body.model_dump())

    @router.get("/llm-profiles")
    def list_profiles(case_id: str) -> list[dict[str, Any]]:
        return conversions.list_profiles(case_id)

    @router.get("/llm-profiles/{profile_id}")
    def get_profile(case_id: str, profile_id: str, version: int | None = None) -> dict[str, Any]:
        return conversions.get_profile(case_id, profile_id, version)

    @router.put("/llm-profiles/{profile_id}")
    def update_profile(case_id: str, profile_id: str, body: ProfileUpdate) -> dict[str, Any]:
        return conversions.update_profile(case_id, profile_id, body.model_dump(exclude_unset=True))

    @router.post("/llm-profiles/{profile_id}/test")
    def test_profile(case_id: str, profile_id: str) -> dict[str, Any]:
        return conversions.test_profile(case_id, profile_id)

    @router.post("/documents", status_code=201)
    async def upload_document(case_id: str, file: UploadFile) -> dict[str, Any]:
        return conversions.add_document(case_id, file.filename or "upload.txt", await file.read(), file.content_type)

    @router.get("/documents")
    def list_documents(case_id: str) -> list[dict[str, Any]]:
        return conversions.list_documents(case_id)

    @router.get("/documents/{document_id}")
    def get_document(case_id: str, document_id: str) -> dict[str, Any]:
        return conversions.get_document(case_id, document_id)

    @router.post("/conversions", status_code=202)
    def create_conversion(case_id: str, body: ConversionIn, background: BackgroundTasks) -> dict[str, Any]:
        record = conversions.create_conversion(case_id, body.document_id, body.profile_id)
        background.add_task(conversions.run_conversion, record["conversion_id"])
        return record

    @router.get("/conversions")
    def list_conversions(case_id: str) -> list[dict[str, Any]]:
        return conversions.list_conversions(case_id)

    @router.get("/conversions/{conversion_id}")
    def get_conversion(case_id: str, conversion_id: str) -> dict[str, Any]:
        return conversions.get_conversion(case_id, conversion_id)

    @router.put("/conversions/{conversion_id}/draft")
    def edit_draft(case_id: str, conversion_id: str, body: DraftIn) -> dict[str, Any]:
        return conversions.edit_draft(case_id, conversion_id, body.draft)

    @router.post("/conversions/{conversion_id}/approve")
    def approve(case_id: str, conversion_id: str, body: ReviewIn) -> dict[str, Any]:
        return conversions.approve(case_id, conversion_id, body.reviewer, body.note)

    @router.post("/conversions/{conversion_id}/reject")
    def reject(case_id: str, conversion_id: str, body: ReviewIn) -> dict[str, Any]:
        return conversions.reject(case_id, conversion_id, body.reviewer, body.note)

    return router


def build_router(service: KCoreService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.post("/cases", status_code=201)
    def create_case(body: CaseIn) -> dict[str, Any]:
        return service.create_case(body.model_dump())

    @router.get("/cases")
    def list_cases() -> list[dict[str, Any]]:
        return service.list_cases()

    @router.get("/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        return service.get_case(case_id)

    @router.post("/cases/{case_id}/episodes", status_code=201)
    def create_episode(case_id: str, body: EpisodeIn) -> dict[str, Any]:
        if body.case_id is not None and body.case_id != case_id:
            raise KCoreError("case_mismatch", f"body case_id {body.case_id} does not match path {case_id}", 409)
        return service.create_episode(case_id, body.model_dump())

    @router.get("/cases/{case_id}/episodes")
    def list_episodes(case_id: str) -> list[dict[str, Any]]:
        return service.list_episodes(case_id)

    @router.get("/cases/{case_id}/episodes/{episode_id}")
    def get_episode(case_id: str, episode_id: str) -> dict[str, Any]:
        return service.get_episode(case_id, episode_id)

    @router.post("/cases/{case_id}/episodes/{episode_id}/events")
    def add_events(case_id: str, episode_id: str, body: EventsIn) -> dict[str, Any]:
        return service.add_events(case_id, episode_id, body.events)

    @router.post("/cases/{case_id}/episodes/{episode_id}/finalize")
    def finalize(case_id: str, episode_id: str, body: FinalizeIn | None = None) -> dict[str, Any]:
        return service.finalize_episode(case_id, episode_id, body.finalized_at if body else None)

    @router.post("/cases/{case_id}/synthetic-episodes", status_code=201)
    def synthetic(case_id: str, body: SyntheticIn) -> dict[str, Any]:
        return service.generate_synthetic(case_id, body.count, body.seed)

    @router.post("/cases/{case_id}/training-runs", status_code=201)
    def train(case_id: str, body: TrainingIn) -> dict[str, Any]:
        try:
            schema = FeatureSchema(version=body.feature_schema_version, fields=tuple(body.feature_fields),
                                   min_count=body.min_count, activity_map=body.activity_map,
                                   activity_default=body.activity_default)
            config = TrainingConfig(k_min=body.k_min, k_max=body.k_max, restarts=body.restarts, n_iter=body.n_iter,
                                    tol=body.tol, smoothing=body.smoothing, seed=body.seed, criterion=body.criterion)
        except ValueError as error:
            raise KCoreError("invalid_training", str(error), 422) from error
        return service.train(case_id, schema, config, body.test_fraction, body.min_train_episodes, body.activate)

    @router.get("/cases/{case_id}/training-runs/{run_id}")
    def get_run(case_id: str, run_id: str) -> dict[str, Any]:
        return service.get_training_run(case_id, run_id)

    @router.get("/cases/{case_id}/models")
    def list_models(case_id: str) -> list[dict[str, Any]]:
        return service.list_models(case_id)

    @router.get("/cases/{case_id}/models/{model_version_id}")
    def get_model(case_id: str, model_version_id: str) -> dict[str, Any]:
        return service.get_model(case_id, model_version_id)

    @router.post("/cases/{case_id}/models/{model_version_id}/activate")
    def activate(case_id: str, model_version_id: str) -> dict[str, Any]:
        return service.activate_model(case_id, model_version_id)

    @router.post("/cases/{case_id}/models/{model_version_id}/replay")
    def replay(case_id: str, model_version_id: str, body: ReplayIn) -> dict[str, Any]:
        return service.replay(case_id, model_version_id, body.episode_id, body.top_k)

    @router.get("/cases/{case_id}/models/{model_version_id}/analysis")
    def analysis(case_id: str, model_version_id: str, group_by: str | None = None) -> dict[str, Any]:
        return service.model_analysis(case_id, model_version_id, group_by)

    @router.post("/cases/{case_id}/models/{model_version_id}/infer")
    def infer(case_id: str, model_version_id: str, body: InferIn) -> dict[str, Any]:
        return service.infer(case_id, model_version_id, body.episode_id, body.prefix_length, body.top_k, body.method)

    return router


def create_app(database_url: str | None = None, llm_client_factory: ClientFactory | None = None) -> FastAPI:
    url = database_url or os.environ.get("KCORE_DATABASE_URL", "sqlite:///./kcore.db")
    service = KCoreService(make_session_factory(url))
    conversions = ConversionService(service, llm_client_factory) if llm_client_factory else ConversionService(service)
    app = FastAPI(title="K: HMM Knowledge Core", version="0.1.0")
    app.state.service = service

    @app.exception_handler(KCoreError)
    async def handle_error(_: Request, error: KCoreError) -> JSONResponse:
        return JSONResponse(status_code=error.status,
                            content={"error": {"code": error.code, "message": error.message, "details": error.details}})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(build_router(service))
    app.include_router(build_conversion_router(conversions))
    app.include_router(build_assist_router(AssistService(service)))
    app.include_router(build_state_machine_router(StateMachineService(service)))
    return app
