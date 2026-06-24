"""Console entry point for Databricks python_wheel_task helpers.

Orchestration helper stages:
    genie-pipeline --stage batch-reference-ingest
    genie-pipeline --stage call-lakebase-ingest
    genie-pipeline --stage lakebase-cdf-sync-check
    genie-pipeline --stage gold-insights-refresh
    genie-pipeline --stage uc-constraints
    genie-pipeline --stage data-quality-check
    genie-pipeline --stage genie-space

Config is read from `--config` (a config.yaml the deployer copied into the job's
workspace folder; exposed to settings via GENIE_CONFIG) plus any
`GENIE_<SECTION>__<KEY>` env overrides set on the job.
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="genie-pipeline")
    ap.add_argument(
        "--stage",
        choices=[
            "batch-reference-ingest",
            "call-lakebase-ingest",
            "lakebase-cdf-sync-check",
            "gold-insights-refresh",
            "uc-constraints",
            "data-quality-check",
            "genie-space",
        ],
        default="call-lakebase-ingest",
        help="helper stage to run in a Databricks wheel task",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="path to config.yaml (sets GENIE_CONFIG before settings load)",
    )
    args = ap.parse_args()

    # Point settings at the uploaded config BEFORE anything reads get_settings().
    if args.config:
        os.environ["GENIE_CONFIG"] = args.config

    if args.stage == "batch-reference-ingest":
        from genie_voice.databricks.reference_ingest import ingest_reference_tables

        ingest_reference_tables()
        return
    if args.stage == "call-lakebase-ingest":
        from genie_voice.lakebase.call_ingest import ingest_call_stream

        ingest_call_stream()
        return
    if args.stage == "lakebase-cdf-sync-check":
        from genie_voice.lakebase.cdf import wait_for_lakebase_cdf

        wait_for_lakebase_cdf()
        return
    if args.stage == "gold-insights-refresh":
        from genie_voice.databricks.gold_insights import refresh_gold_insights

        refresh_gold_insights()
        return
    if args.stage == "uc-constraints":
        from genie_voice.databricks.constraints import apply_constraints

        apply_constraints()
        return
    if args.stage == "data-quality-check":
        from genie_voice.databricks.data_quality import run_data_quality

        run_data_quality()
        return
    if args.stage == "genie-space":
        from genie_voice.genie.space import ensure_space

        ensure_space(require_quality=False)
        return


if __name__ == "__main__":
    main()
