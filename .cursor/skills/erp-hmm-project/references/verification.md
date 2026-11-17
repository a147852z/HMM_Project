# Verification reference

Run checks proportional to the change. Do not start long-running development servers only to prove compilation.

## Backend

From `erp_hmm_simulator/backend`:

```powershell
python -m pytest
```

For a focused iteration, run the closest file first, then the full suite before handoff when shared model, service, schema, persistence, or API behavior changed:

```powershell
python -m pytest tests/test_hmm.py
python -m pytest tests/test_simulation.py
python -m pytest tests/test_trainer.py
python -m pytest tests/test_workflow_compiler.py
python -m pytest tests/test_api.py
```

Model or simulation changes must also verify:

- matrix and belief normalization within floating-point tolerance
- seeded reproducibility
- exactly the requested number of Top-K predictions
- no true-state input reaches inference
- non-persistent experiments leave simulation-record counts unchanged

## Frontend

From `erp_hmm_simulator/frontend`:

```powershell
npm.cmd run build
```

The build runs TypeScript checking and Vite production bundling. For bilingual changes, inspect both `zh-TW` and `en-US` modes and check:

- navigation, page titles, controls, help text, charts, tables, tooltips, notices, and empty/error states
- task, state, observation, and action labels
- locale-sensitive dates and counts
- language persistence after reload
- user-authored workflow text remains unchanged

## API and data contracts

When a backend response changes:

1. Update its Pydantic schema or service serializer.
2. Update `frontend/src/types/index.ts`.
3. Update `frontend/src/api/client.ts` and all consumers.
4. Add an assertion in `tests/test_api.py` for the observable response.

Use `/health` for a running-backend smoke check and `/docs` for manual endpoint inspection. The backend development URL is `http://localhost:8800`.

## Documentation and progress

- Update `erp_hmm_simulator/README.md` when commands, ports, user workflows, metrics, or completed features change.
- Update section 5 (feature status) of `docs/設計文件.md` only when implementation status changes.
- Describe verified behavior, not planned behavior, as completed.

