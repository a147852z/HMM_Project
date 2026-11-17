# Research and domain invariants

## Current catalog

The authoritative catalog is `backend/app/domain/catalog.py`:

- 32 hidden states
- 20 discrete observations
- 33 user actions
- 6 task types: `CreateQuotation`, `CreateProduct`, `CreateCustomer`, `CreateCustomerAndQuotation`, `CreateProductAndQuotation`, and `UpdateInventory`

When adding or renaming an identifier, update all dependent matrices, workflows, serializers, frontend types, and both language dictionaries. Prefer adding identifiers over renaming persisted ones.

## Ground truth isolation

`SimulationEngine` may know the task, real action, and true hidden state. Inference may use only:

- current and previous observations
- previous actions
- current state belief
- task context
- workflow information
- saved model parameters

The true state is allowed only when scoring predictions, saving supervised training labels, or constructing evaluation output after inference. A test that achieves accuracy by passing the true state into inference is invalid.

## Probability requirements

- Initial probabilities sum to 1.
- Every transition-matrix row sums to 1.
- Every emission-matrix row sums to 1.
- Bayesian prior and posterior each sum to 1 at every step.
- Likelihood is `P(observation | state)` across states and therefore does not need to sum to 1.
- Laplace smoothing must remain positive for normal training requests and prevents unseen events from receiving an exact zero probability.
- Initial HMM matrices encode ERP behavior; do not replace them with arbitrary random matrices.

The first Bayesian update uses the model initial distribution as its prior. Later updates propagate the previous posterior through the transition matrix before multiplying by emission likelihood.

## Next-action ranking

`backend/app/hmm/predictor.py` is authoritative. The current ranking combines:

- 60% state-belief × state-action scores
- 25% previous-action transition scores when a previous action exists
- 15% workflow successor scores when workflow candidates exist

The combined score is then adjusted by task context and explicit allowed-action constraints before normalization and Top-K selection. Actions outside the workflow are down-weighted rather than always forced to zero, preserving mistakes and alternate paths.

If changing these weights or multipliers, update tests, UI explanations, and README wording together.

## Workflow rules

The workflow compiler requires:

- exactly one start node
- at least one end node and one action node
- end nodes with no outgoing edge
- ordinary nodes with exactly one outgoing edge
- condition nodes with exactly one `true` and one `false` edge
- valid actions on action nodes
- no cycles, duplicate node IDs, invalid edge targets, or unreachable nodes

Persisted workflow labels are user data. Translate editor controls and defaults, not previously saved labels.

## Training and experiments

- Supervised HMM training uses stored true-state sequences and Laplace smoothing.
- Seeded simulations with identical parameters must be reproducible.
- Baseline evaluation uses the earlier 70% of chronological data to build rules and the later 30% to test.
- Learning curves use the first 80% as the training pool and the final 20% as a fixed test set.
- Noise, learning-curve, and ambiguous-state experiments that promise not to affect history must not insert simulation records.
- Model comparisons should identify which model version produced each stored simulation; historical predictions are not silently recomputed by the active model.

## Observed event-flow graph

The event-flow network is descriptive analytics, not HMM training. It merges each unique recorded `actual_action` into one node, aggregates directed consecutive-action counts, and reports each edge probability as its count divided by all outgoing counts from the same source. Task filters may select sessions before aggregation; display thresholds must not silently renormalize the original probabilities.

## Persistence safety

Simulation records, workflow definitions, model matrices, metrics, and active-version state are stored in SQLite. Clearing simulations resets the active model according to the service behavior but must not silently delete saved workflows or model-version records. Use the established confirmation flow for destructive clearing.
