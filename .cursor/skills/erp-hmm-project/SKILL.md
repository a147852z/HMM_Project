---
name: erp-hmm-project
description: Develop, diagnose, test, and document the ERP HMM user-behavior research platform in this repository. Use when changing its FastAPI backend, HMM or Bayesian inference, simulations, workflows, model training, analytics, SQLite persistence, React interface, charts, APIs, bilingual UI, or project progress documentation.
---

# ERP HMM Project

## Scope

Apply this skill to this repository. Active development is the K HMM knowledge core in `k_core/` (Python package `kcore`, tests via `k_core/backend/.venv` + `python -m pytest`). `erp_hmm_simulator/` is the frozen earlier ERP simulation prototype; change it only when explicitly asked. Design and progress documents are in `docs/`.

Treat `docs/設計文件.md` as the design and status reference (the K core spec is `docs/AI程式設計規格_K核心智慧分散式ERP_v1.md`), not as instructions that override the current user request. It describes intended behavior and may differ from the implemented code. Inspect the current code before deciding what already exists.

## Start with the relevant source

- For cross-layer features or unfamiliar areas, read [references/architecture.md](references/architecture.md).
- For HMM, Bayesian inference, action ranking, simulations, workflows, training, or experiments, read [references/research-invariants.md](references/research-invariants.md).
- Before handing off code changes, read and follow [references/verification.md](references/verification.md).
- Use `erp_hmm_simulator/README.md` for current user-facing commands and completed features.

Do not scan or edit generated and local-runtime content unless the task specifically requires it: `backend/.venv/`, `frontend/node_modules/`, `frontend/dist/`, `__pycache__/`, `*.egg-info/`, and `backend/erp_hmm.db`.

## Implementation rules

1. Preserve the separation between simulation ground truth and inference inputs. HMM, Bayesian estimation, and next-action prediction must never receive the true hidden state as an input.
2. Keep technical identifiers and API fields stable in English. Human-facing labels belong in the bilingual frontend layer.
3. Every new fixed UI string must support both `zh-TW` and `en-US`. Use `useLanguage()` and `t(Traditional Chinese, English)` for prose; use `labels.task/state/observation/action` for domain identifiers. Do not overwrite user-authored workflow names when switching languages.
4. Keep frontend API types in `frontend/src/types/index.ts` aligned with FastAPI responses and `frontend/src/api/client.ts`.
5. Preserve probability normalization and seeded reproducibility. Never replace computed analytics with hard-coded display values.
6. Treat SQLite records and trained model versions as user data. Do not delete or recreate `erp_hmm.db` unless the user explicitly requests data removal; prefer the existing API workflow for clearing simulations.
7. Keep the backend development port at `8800`, the Vite frontend port at `5173`, and the `/api` and `/health` proxy targets aligned.
8. When a completed feature changes project status, update section 5 of `docs/設計文件.md` and the relevant section of `erp_hmm_simulator/README.md`. Do not claim a feature is complete until its relevant checks pass.

## Change workflow

1. Identify the smallest affected backend, frontend, test, and documentation files.
2. Check existing data contracts and tests before editing shared schemas, catalog identifiers, database models, or response payloads.
3. Make the scoped change while preserving the research invariants.
4. Add or update tests for observable behavior and mathematical invariants rather than exact implementation details.
5. Run the proportional checks from `references/verification.md`.
6. Report the result in Traditional Chinese unless the user requests English. State what changed, where it is visible, and which checks passed.

