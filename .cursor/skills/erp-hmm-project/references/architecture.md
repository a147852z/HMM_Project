# Architecture reference

## Repository layout

```text
ERP_HMM_Project/
├── docs/                       設計文件.md (goals, status, roadmap), K core spec, architecture image
├── k_core/                     ACTIVE: K HMM knowledge core (FastAPI, port 8900, see k_core/README.md)
└── erp_hmm_simulator/
    ├── README.md                Current setup and usage documentation
    ├── backend/
    │   ├── app/
    │   │   ├── api/router.py    FastAPI endpoints
    │   │   ├── database/        SQLAlchemy models and session handling
    │   │   ├── domain/          State, observation, action, task, and workflow definitions
    │   │   ├── hmm/             HMM, Bayesian estimator, action predictor, trainer
    │   │   ├── simulation/      Task generation, execution, batch jobs, workflow compiler
    │   │   ├── analytics.py     Metrics and baseline comparisons
    │   │   ├── initial_model.py Initial probability matrices and predictor
    │   │   ├── schemas.py       Request validation models
    │   │   ├── service.py       Application orchestration and persistence
    │   │   └── main.py          FastAPI application and lifespan
    │   ├── tests/               Pytest suite
    │   ├── pyproject.toml       Python dependencies and test configuration
    │   └── erp_hmm.db           Local SQLite user data
    └── frontend/
        ├── src/
        │   ├── api/client.ts    Typed API client
        │   ├── components/      Shared cards, guides, and belief charts
        │   ├── i18n/            Language context and domain label dictionaries
        │   ├── pages/           Eight main application pages, including the event-flow network
        │   ├── types/index.ts   Backend response types
        │   └── App.tsx          Navigation and language selector
        ├── package.json
        └── vite.config.ts
```

## Runtime flow

```text
Task generator or compiled workflow
  → SimulationEngine ground truth
  → ObservationGenerator
  → BayesianStateEstimator / HiddenMarkovModel
  → NextActionPredictor
  → prediction versus actual result
  → PlatformService
  → SQLite simulation/model/workflow records
  → FastAPI response
  → React page and Recharts visualization
```

`PlatformService` is the main orchestration boundary. Route handlers should remain thin and translate service failures into HTTP responses. Simulation classes should remain usable independently in tests.

## Frontend conventions

- Navigation and language switching are in `frontend/src/App.tsx`.
- `LanguageProvider` persists `erp-hmm-language` and exposes `language`, `locale`, `labels`, and `t`.
- Domain IDs stay unchanged across languages; `zhTW.ts` and `enUS.ts` supply labels.
- All eight pages include `PageGuide` instructions and should remain usable at narrow screen widths.
- The event-flow page reads `actual_action` histories from `/api/analytics/event-flow` and merges each unique action into one observed FSM node.
- Shared posterior charts belong in `BayesianBeliefCharts`; avoid duplicating probability-chart logic in pages.
- API failure handling belongs in `frontend/src/api/client.ts`; pages handle loading, success, and error states.

## Ports and startup

- Backend: `python -m uvicorn app.main:app --reload --port 8800` from `erp_hmm_simulator/backend`.
- Frontend: `npm run dev` from `erp_hmm_simulator/frontend`; Vite serves port `5173`.
- Vite proxies `/api` and `/health` to `http://localhost:8800`.
