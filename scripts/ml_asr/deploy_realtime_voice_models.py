"""Deploy an isolated realtime voice agent endpoint.

Promotes the ``candidate`` alias of a registered ResponsesAgent to its Model
Serving endpoint. Two modes:

* Local: ``deploy_realtime_voice_models.py <candidate_id>`` — resolves everything
  from the merged ``realtime_voice:`` config block.
* Serverless job: pass ``--registered-model`` (fully-qualified) and ``--endpoint``
  (plus optional workload flags); no config file is required on the job.
"""
from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_id", nargs="?")
    parser.add_argument("--registered-model", help="Fully-qualified UC model (catalog.schema.name)")
    parser.add_argument("--endpoint")
    parser.add_argument("--workload-type", default="GPU_MEDIUM")
    parser.add_argument("--workload-size", default="Small")
    parser.add_argument("--scale-to-zero", action="store_true")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    if args.registered_model and args.endpoint:
        registered_model = args.registered_model
        endpoint = args.endpoint
        workload_type, workload_size, scale_to_zero = args.workload_type, args.workload_size, args.scale_to_zero
        profile = args.profile
    else:
        if not args.candidate_id:
            parser.error("Provide a candidate_id, or --registered-model and --endpoint")
        from _realtime_config import databricks, find_candidate, realtime_voice, registered_model_name

        candidate = find_candidate(args.candidate_id)
        serving = realtime_voice().get("serving") or {}
        registered_model = registered_model_name(candidate)
        endpoint = candidate["endpoint"]
        workload_type = candidate.get("workload_type", serving.get("workload_type", "GPU_MEDIUM"))
        workload_size = candidate.get("workload_size", serving.get("workload_size", "Small"))
        scale_to_zero = bool(candidate.get("scale_to_zero", serving.get("scale_to_zero", False)))
        profile = args.profile or databricks().get("profile")

    client = WorkspaceClient(profile=profile or None)
    alias = client.api_client.do("GET", f"/api/2.1/unity-catalog/models/{registered_model}/aliases/candidate")
    version = str(alias["version"])
    served_model_name = f"{endpoint}-candidate"[:63]
    endpoint_config = {
        "served_entities": [
            {
                "name": served_model_name,
                "entity_name": registered_model,
                "entity_version": version,
                "workload_type": workload_type,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
            }
        ],
        "traffic_config": {"routes": [{"served_model_name": served_model_name, "traffic_percentage": 100}]},
    }
    try:
        client.serving_endpoints.get(name=endpoint)
    except Exception:  # noqa: BLE001
        client.api_client.do(
            "POST",
            "/api/2.0/serving-endpoints",
            body={"name": endpoint, "config": endpoint_config, "route_optimized": False},
        )
        print(f"created {endpoint} from {registered_model}@{version}")
    else:
        client.api_client.do("PUT", f"/api/2.0/serving-endpoints/{endpoint}/config", body=endpoint_config)
        print(f"updated {endpoint} from {registered_model}@{version}")


if __name__ == "__main__":
    main()
