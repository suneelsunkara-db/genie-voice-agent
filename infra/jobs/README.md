# Lakebase-first orchestration job (Databricks)

`infra/jobs/deploy_pipeline.py` deploys one serverless orchestration job running
as your U2M identity:

- **Reference UC ingest task** — reads `raw_batch_data` files into UC reference
  Delta tables.
- **Lakebase ingest task** — reads `raw_streaming_data` files and upserts primary
  Lakebase call tables. Runs in parallel with reference UC ingest.
- **Lakebase CDF sync check task** — resolves the project/branch, verifies
  `REPLICA IDENTITY FULL`, requires `wal2delta.tables` status to be `STREAMING`
  or `SNAPSHOTTING`, then waits for `lb_<table>_history` tables in UC.
- **Gold insights refresh task** — creates regular UC Delta `gold_call_insights`
  directly from Lakebase call and utterance history.
- **UC constraints task** — adds informational PK/FK metadata after table refresh
  so Genie sees relationships.
- **Data quality task** — validates PK/FK metadata, data integrity, vocabularies,
  and call consistency.
- **Genie reconcile task** — recreates the Genie space only after DQ passes.

Lakebase is the low-latency serving path. UC is the asynchronous analytics path.

## Automated deploy

`infra/jobs/deploy_pipeline.py`:

1. builds the `genie_voice` wheel from `backend/`
2. uploads the wheel to a stable UC Volume `libs/` path and copies config into a
   workspace folder.
3. creates/updates `pipeline.orchestration_job_name`.
5. optionally runs the job in this order: reference UC ingest + Lakebase ingest
   in parallel → Lakebase CDF sync check → gold refresh → UC constraints →
   data quality → Genie reconcile.

```bash
python infra/jobs/deploy_pipeline.py                 # deploy + run orchestration
python infra/jobs/deploy_pipeline.py --full-refresh  # accepted for compatibility
python infra/jobs/deploy_pipeline.py --no-run        # deploy only
python infra/jobs/deploy_pipeline.py --paused        # create paused
```

## Serverless compute + source in the workspace

The wheel tasks run on **serverless** compute: a job environment
(`environments`) whose only dependency is the wheel installed from the stable UC
Volume path. `pip` pulls the wheel's declared deps from PyPI; `pyspark`/`pandas`
come preinstalled in the serverless base image. There is no cluster to size.

The task reads config from the `config.yaml` copied into the same workspace folder
via a `--config /Workspace/.../config.yaml` CLI argument (workspace files are
FUSE-mounted on serverless, so `open()` just works). `GENIE_<SECTION>__<KEY>`
overrides still apply if set.

## Prerequisites

- The UC schema + Volume + typed tables exist (run `genie_voice.databricks.bootstrap`,
  which `local-deploy.sh` does) so the jobs only move data.
- `pip` can build the wheel (built with `--no-deps`).
- Your identity can create serverless jobs and write to its workspace home.
