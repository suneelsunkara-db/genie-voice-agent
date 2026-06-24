# Lakebase (Autoscaling) — serving layer

Lakebase Autoscaling is a **serverless Postgres** project (scale-to-zero, instant
restore, Unity Catalog governed). We use it for the low-latency reads the Agent
Assist UI needs:

- **`call_state`** — live enrichment per call, upserted by the serving layer
  (`genie_voice.serve.LakebaseServing`).
- **operational call tables** — `call_facts` and `live_call_utterances`, loaded
  directly into Lakebase under the configured schema.
- **Lakebase CDF** — started in the Lakebase UI to publish `lb_<table>_history`
  into Unity Catalog for task-based analytics refresh.

## Provisioning

```bash
python infra/lakebase/setup_lakebase.py
```

This resolves the Autoscaling project and ensures the configured Postgres schema
exists. If the project does not exist, create it in the UI, then re-run setup.

## Connecting (U2M — default)

No password needed. With OAuth U2M the serving layer **mints a short-lived
Postgres token at runtime** via `/api/2.0/postgres/credentials` and connects as
`databricks.run_as`. Just set `lakebase.enabled: true` in
`config/config.yaml`.

To pin a static connection instead, set these in `.env` (overrides the minted
token):

```
LAKEBASE_HOST=...
LAKEBASE_PORT=5432
LAKEBASE_DATABASE=databricks_postgres
LAKEBASE_USER=...
LAKEBASE_PASSWORD=...
```

With `lakebase.enabled: false` the serving layer falls back to an in-process
store so the local end-to-end flow still works offline.

## Notes

- New Lakebase instances are **Autoscaling** projects by default (2026+).
- UC analytics reads Lakebase CDF history; this repo does not create duplicate
  UC-to-Lakebase managed synced tables.
