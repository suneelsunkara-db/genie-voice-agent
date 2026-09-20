# Fresh customer installation

`deploy_app.sh` is the supported end-to-end installer. It uses
`config/config.yaml` exclusively for the app, deploy-time helpers, serverless
jobs, and model jobs. `config.local.yaml` is for local development and is never
used or synced by this installer.

## Before running

Edit `config/config.yaml` for the customer workspace. At minimum, set:

- `databricks.host`, `catalog`, `schema`, and `sql_warehouse_id`
- both `ai_gateway.model_services.*.service` FQNs to that catalog and schema
- `lakebase.instance` and the desired Lakebase schema
- customer-specific Genie space names

The deploying user must be able to create schemas and volumes in the existing
catalog, use the SQL warehouse, create Apps and serving endpoints, create or
administer the Lakebase project, create Genie spaces, and grant permissions to
the App service principal. The workspace must have GPU Model Serving capacity,
outbound access to download the configured Hugging Face models, and access to
the configured `system.ai.*` foundation models.

Authenticate the CLI, then run:

```bash
DATABRICKS_PROFILE=<customer-profile> \
APP_EXTERNAL_USERS=user1@example.com,user2@example.com \
./deploy_app.sh
```

`APP_EXTERNAL_USERS`, `APP_EXTERNAL_GROUPS`, and `APP_EXTERNAL_SPS` are optional
comma-separated principals. The installer grants all three `CAN_USE` on the App;
users and groups also receive `CAN_RUN` on every configured Genie space. Grant
failures stop deployment. No demo principal is granted by default.

## What the installer owns

The installer is idempotent and performs these phases:

1. validates the committed deployment config and runs frontend/Python gates;
2. creates the UC schemas and Volumes, lands telco/card data, provisions
   Lakebase, deploys and runs the orchestration job, snapshots serving data, and
   reconciles both Genie spaces;
3. registers, deploys, waits for, and smoke-tests the Qwen3-ASR and VoxCPM2
   ResponsesAgent endpoints;
4. reconciles the app-owned Unity AI Gateway model services and probes their
   policy behavior;
5. creates/updates the Databricks App, grants its service principal UC,
   Lakebase, Genie, model-service, warehouse, and serving-endpoint access,
   deploys the source, and smoke-tests the running App.

All runtime grants are fail-closed. Registered voice models use Unity Catalog's
SQL `FUNCTION` securable for `EXECUTE`; serving endpoints and model services
retain their own required runtime grants.

For a normal code-only redeploy, `DEPLOY_DATA=0` skips data/Lakebase/jobs.
`DEPLOY_REALTIME_MODELS=auto` (the default) reuses model endpoints only when
they are READY **and** each endpoint routes the configured Unity Catalog model's
`candidate` alias. Otherwise it downloads the configured Hugging Face snapshot,
registers a new ResponsesAgent version in UC, updates the alias, deploys the GPU
endpoint, waits for READY, and runs a multilingual TTS→STT contract smoke test.
Use `FORCE_REALTIME_MODELS=1` only when a new registered model version must be
promoted.

## Manual workspace checkpoints

Databricks does not currently expose every required operation through a stable
public API. The installer continues through these UI-only checkpoints so that it
can deploy the App's **Setup & Readiness** page (`#/setup`). Complete the actions
shown there, rerun the installer when instructed, and use **Re-check**. The page
does not report **Ready to use** until infrastructure, data, jobs, model
registration/serving/invocation, Gateway behavior, OBO viewer Genie, and Agent
Mode checks pass.

### 1. Lakebase Change Data Feed

Start CDF from the Lakebase UI for the Gold inputs in the configured serving
schema:

- `call_facts`
- `live_call_utterances`

Also start the optional Genie billing-audit feed:

- `billing_adjustments`

Publish each table to the configured UC catalog/schema using the configured
`lb_` prefix and `_history` suffix. The orchestration job blocks only until the
two Gold inputs are streaming and populated sources have history. The Setup
page reports `billing_adjustments` as a warning if its source has rows that have
not reached UC; that warning affects Genie billing-adjustment analytics, not the
live voice path or Gold. `call_state` and `resolution_events` remain
Lakebase-only operational tables and do not need CDF for this application.

### 2. Unity AI Gateway policy attachments

The installer creates/reconciles the two app-owned model services, but policy
attachment writes remain UI-only during the current Beta. In the AI Gateway UI,
attach the policy bundle printed by `infra/apps/provision_ai_gateway.py` to each
service. The required source of truth is `config/guardrails.yaml`.

The installer reports this as a warning rather than preventing the Setup page
from being deployed. The Setup page runs the behavioral allow, deny, and
redaction probes and remains blocked until they conform.

### 3. Databricks Apps User Authorization

Enable User Authorization for the App with the `genie` and `sql` scopes. Users
must consent when first opening the App (and again if the requested scopes
change). The installer warns if effective scopes are absent but continues so the
Setup page can explain the repair. Full readiness verifies the forwarded token
and performs real viewer-scoped queries against both configured Genie spaces.

This OBO token is required for Genie and SQL calls made as the signed-in viewer;
the App service principal is not used as a fallback.

### 4. Genie Agent Mode preview

Enable **Agent Mode APIs for Genie Agents** in workspace Previews if the
long-running “why”/deep-dive experience is required. Standard Genie Space
Conversation and Genie One do not depend on this preview.

### 5. Workspace identity and App access

Customer users and groups must already exist in the same Databricks account
through the customer’s normal SCIM/JIT process. Grant App `CAN_USE` by setting
`APP_EXTERNAL_USERS`, `APP_EXTERNAL_GROUPS`, or `APP_EXTERNAL_SPS`, or use the
App permissions UI. Users and groups receive Genie `CAN_RUN` as well.

If users target an additional Genie space with `space_name`, those users also
need `CAN_RUN` on that space. Full readiness proves permission by asking a real
question as the signed-in viewer; it does not infer access from space existence.

## Permission-related stops

- If this machine's public IP is not on the workspace IP access list, Databricks
  rejects every workspace API — including the IP Access Lists API itself. The
  installer adds the current `/32` to an installer-owned ALLOW list named
  `genie-voice-deploy` when the workspace is still reachable. If you are already
  blocked, connect through an allowed network (VPN/office) or have an admin add
  the IP, then rerun. Set `UPDATE_IP_ACCESS_LIST=0` to skip the mutation.
- If SCIM cannot add `workspace-access` to the App service principal, an account
  admin must enable that entitlement and rerun. Lakebase OAuth token minting
  cannot work without it.
- If any UC, Lakebase, Genie, Model Serving, model-service, warehouse, or App ACL
  grant fails, correct the deploying user’s ownership/admin privileges and
  rerun. These failures are not optional warnings.
- A missing GPU entitlement/quota, unavailable `system.ai.*` model, blocked
  Hugging Face egress, or inaccessible existing catalog/warehouse is a customer
  workspace prerequisite, not an App runtime permission.
