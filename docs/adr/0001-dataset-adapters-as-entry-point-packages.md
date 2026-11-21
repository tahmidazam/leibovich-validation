# Datasets are adapter packages found by entry point

Status: accepted

Each dataset is an adapter package in the uv workspace. The analysis app, the API package `lv-datasets` and the adapters depend in one direction only.

- **Entry-point discovery.** An adapter registers a `Dataset` in the `leibovich_validation.datasets` entry-point group. The app lists and loads datasets through `lv_datasets.names` and `lv_datasets.get`. Adding a dataset edits no app code. Tests register a dataset with `lv_datasets.override`.
- **One-way dependencies.** Adapters depend on `lv-datasets`. The app depends on `lv-datasets`. Neither imports the other: ruff bans the app from importing adapter packages and adapters from importing `leibovich_validation`. The only exceptions are the test files that write an adapter's raw files.
- **Return every raw row.** `harmonise` returns one row per raw patient in the harmonised columns, with an `excluded` column that is null for retained rows and otherwise names the first exclusion step. The app, not the adapter, drops excluded rows and derives the flow counts, the computed Leibovich score and the score agreement in `lv_datasets.finalise`. Exclusions stay auditable and no adapter can disagree with another about how a flow is counted.

## Consequences

An adapter declares every exclusion step and note in `flow_steps`; `finalise` raises on an undeclared one. Adapters report data problems as `SchemaError`.

Ruff's `banned-api` rule matches module names exactly and has no wildcard, so the app's ban lists each adapter package by name. A new adapter is not covered until its package is added to the root `pyproject.toml`.
