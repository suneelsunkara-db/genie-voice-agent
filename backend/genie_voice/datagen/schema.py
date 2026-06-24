"""Declarative enterprise data model.

Ownership (single producer per table):
  - REFERENCE tables are batch-ingested from raw_batch_data into UC Delta:
        customers, agents, invoices, payments
  - CALL tables are operational Lakebase tables first:
        call_facts, live_call_utterances
    Lakebase CDF publishes them to UC history tables, and a gold refresh task
    derives gold_call_insights from the history tables.
  - DERIVED analytics tables are produced in UC:
        gold_call_insights

This single spec drives BOTH the Databricks DDL (column COMMENTs + informational
PRIMARY/FOREIGN KEY constraints) and the data-model docs. Genie relies on this
metadata: comments give semantics and PK/FK tell it how to join.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Logical table names.
T_CUSTOMERS = "customers"
T_AGENTS = "agents"
T_INVOICES = "invoices"
T_PAYMENTS = "payments"
T_BILLING_ADJUSTMENTS = "billing_adjustments"
T_CALL_FACTS = "call_facts"
T_GOLD = "gold_call_insights"


@dataclass
class Column:
    name: str
    type: str
    comment: str
    nullable: bool = True


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class TableSpec:
    name: str
    comment: str
    columns: list[Column]
    primary_key: list[str]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)

    def render_schema_ddl(self, table_name, *, include_constraints: bool = True) -> str:
        """Render only the SQL schema body used by UC table creators.

        `table_name` maps a logical table name to the identifier used inside the
        pipeline (usually unqualified, because the pipeline supplies catalog +
        schema defaults).
        """
        col_lines = []
        for c in self.columns:
            null = "" if c.nullable else " NOT NULL"
            safe = c.comment.replace("'", "''")
            col_lines.append(f"  {c.name} {c.type}{null} COMMENT '{safe}'")

        constraints = []
        if include_constraints and self.primary_key:
            pk = ", ".join(self.primary_key)
            constraints.append(f"  CONSTRAINT pk_{self.name} PRIMARY KEY ({pk})")
        if include_constraints:
            for fk in self.foreign_keys:
                constraints.append(
                    f"  CONSTRAINT fk_{self.name}_{fk.column} "
                    f"FOREIGN KEY ({fk.column}) "
                    f"REFERENCES {table_name(fk.ref_table)}({fk.ref_column})"
                )
        return ",\n".join(col_lines + constraints)

    def render_ddl(self, fqtn) -> str:
        """Render Databricks CREATE TABLE with comments + PK/FK constraints.

        `fqtn` is a callable: table_name -> fully-qualified name.
        """
        body = self.render_schema_ddl(fqtn)
        tbl_comment = self.comment.replace("'", "''")
        props = ""
        if self.properties:
            body_props = ", ".join(
                f"'{k}' = '{v}'" for k, v in sorted(self.properties.items())
            )
            props = f" TBLPROPERTIES ({body_props})"

        return (
            f"CREATE TABLE IF NOT EXISTS {fqtn(self.name)} (\n{body}\n) "
            f"COMMENT '{tbl_comment}'{props}"
        )


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
MODEL: dict[str, TableSpec] = {
    T_AGENTS: TableSpec(
        name=T_AGENTS,
        comment="Contact-center agents who handle customer calls.",
        primary_key=["agent_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        columns=[
            Column("agent_id", "STRING", "Unique agent identifier.", nullable=False),
            Column("full_name", "STRING", "Agent full name."),
            Column("team", "STRING", "Team: billing | retention | technical."),
            Column("hire_date", "DATE", "Date the agent was hired."),
        ],
    ),
    T_CUSTOMERS: TableSpec(
        name=T_CUSTOMERS,
        comment="Enterprise customer master. One row per customer account.",
        primary_key=["customer_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        columns=[
            Column("customer_id", "STRING", "Unique customer/account identifier.", nullable=False),
            Column("full_name", "STRING", "Customer full name."),
            Column("segment", "STRING", "Customer segment: consumer | smb | enterprise."),
            Column("region", "STRING", "Sales region: NA | EMEA | APAC."),
            Column("plan", "STRING", "Subscription plan: basic | pro | premium."),
            Column("monthly_charge", "DECIMAL(10,2)", "Recurring monthly charge in USD."),
            Column("tenure_months", "INT", "Months since signup."),
            Column("status", "STRING", "Account status: active | at_risk | churned."),
            Column("autopay_enabled", "BOOLEAN", "Whether automatic payment is enabled."),
            Column("email", "STRING", "Customer email address."),
            Column("signup_date", "DATE", "Date the customer signed up."),
        ],
    ),
    T_INVOICES: TableSpec(
        name=T_INVOICES,
        comment="Monthly invoices issued to customers.",
        primary_key=["invoice_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("customer_id", T_CUSTOMERS, "customer_id")],
        columns=[
            Column("invoice_id", "STRING", "Unique invoice identifier (e.g. INV-90231).", nullable=False),
            Column("customer_id", "STRING", "Customer this invoice belongs to.", nullable=False),
            Column("period", "STRING", "Billing period (YYYY-MM)."),
            Column("issue_date", "DATE", "Date the invoice was issued."),
            Column("due_date", "DATE", "Date payment is due."),
            Column("amount", "DECIMAL(10,2)", "Total invoice amount in USD (incl. late_fee)."),
            Column("late_fee", "DECIMAL(10,2)", "Late fee applied to this invoice, USD (0 if none)."),
            Column("status", "STRING", "Invoice status: paid | open | overdue | disputed | refunded."),
            Column("paid_date", "DATE", "Date the invoice was paid (null if unpaid)."),
        ],
    ),
    T_PAYMENTS: TableSpec(
        name=T_PAYMENTS,
        comment="Payment attempts against invoices (including declined attempts).",
        primary_key=["payment_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("invoice_id", T_INVOICES, "invoice_id"),
            ForeignKey("customer_id", T_CUSTOMERS, "customer_id"),
        ],
        columns=[
            Column("payment_id", "STRING", "Unique payment identifier.", nullable=False),
            Column("invoice_id", "STRING", "Invoice this payment is for.", nullable=False),
            Column("customer_id", "STRING", "Customer who made the payment.", nullable=False),
            Column("amount", "DECIMAL(10,2)", "Payment amount in USD."),
            Column("payment_date", "DATE", "Date of the payment attempt."),
            Column("method", "STRING", "Payment method: card | bank_transfer | autopay."),
            Column("status", "STRING", "Payment status: succeeded | declined | refunded."),
        ],
    ),
    T_BILLING_ADJUSTMENTS: TableSpec(
        name=T_BILLING_ADJUSTMENTS,
        comment=(
            "Live agent-assist billing adjustments (waiver / payment plan). "
            "Written by the voice API via SQL warehouse; mirrors Lakebase billing_adjustments."
        ),
        primary_key=["adjustment_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("customer_id", T_CUSTOMERS, "customer_id"),
            ForeignKey("invoice_id", T_INVOICES, "invoice_id"),
        ],
        columns=[
            Column("adjustment_id", "STRING", "Unique adjustment id (call_id-invoice_id).", nullable=False),
            Column("call_id", "STRING", "Call that triggered the adjustment.", nullable=False),
            Column("customer_id", "STRING", "Customer receiving the adjustment.", nullable=False),
            Column("invoice_id", "STRING", "Invoice adjusted.", nullable=False),
            Column("waiver_applied", "BOOLEAN", "Whether a late-fee waiver was applied."),
            Column("payment_plan_applied", "BOOLEAN", "Whether a payment arrangement was set."),
            Column("amount_before", "DECIMAL(10,2)", "Invoice amount before adjustment."),
            Column("late_fee_before", "DECIMAL(10,2)", "Late fee before adjustment."),
            Column("status_before", "STRING", "Invoice status before adjustment."),
            Column("amount_after", "DECIMAL(10,2)", "Invoice amount after adjustment."),
            Column("late_fee_after", "DECIMAL(10,2)", "Late fee after adjustment."),
            Column("status_after", "STRING", "Invoice status after adjustment."),
            Column("applied_at", "TIMESTAMP", "When the adjustment was applied."),
            Column("reverted_at", "TIMESTAMP", "When the demo/session revert restored the invoice."),
        ],
    ),
    T_CALL_FACTS: TableSpec(
        name=T_CALL_FACTS,
        comment=(
            "Semantic current-state telephony/CTI metadata per call, exposed as "
            "a UC view over Lakebase CDF history. One row per call."
        ),
        primary_key=["call_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("customer_id", T_CUSTOMERS, "customer_id"),
            ForeignKey("agent_id", T_AGENTS, "agent_id"),
        ],
        columns=[
            Column("call_id", "STRING", "Unique call identifier.", nullable=False),
            Column("customer_id", "STRING", "Customer on the call.", nullable=False),
            Column("agent_id", "STRING", "Agent who handled the call."),
            Column("call_ts", "TIMESTAMP", "Call start timestamp."),
            Column("duration_sec", "INT", "Call duration (handle time) in seconds."),
            Column("csat", "INT", "Post-call CSAT score 1-5 (null if not collected)."),
            Column("audio_path", "STRING", "UC Volume path to the call audio file."),
            Column("transcript_path", "STRING", "UC Volume path to the call transcript file."),
        ],
    ),
    # ---- DERIVED (produced from Lakebase utterance history) ---- #
    T_GOLD: TableSpec(
        name=T_GOLD,
        comment=(
            "Conversation-derived insights, one row per call. Produced by the "
            "gold refresh task (NLP over the transcript). Joins to Lakebase "
            "call_facts history on call_id, customers on customer_id, invoices "
            "on mentioned_invoice_id."
        ),
        primary_key=["call_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("customer_id", T_CUSTOMERS, "customer_id"),
            ForeignKey("mentioned_invoice_id", T_INVOICES, "invoice_id"),
        ],
        columns=[
            Column("call_id", "STRING", "Call these insights describe.", nullable=False),
            Column("customer_id", "STRING", "Customer on the call (denormalized for convenience)."),
            Column("primary_intent", "STRING", "Primary detected intent (billing_dispute, late_fee, autopay_issue, refund, payment_arrangement, billing_inquiry, plan_inquiry, cancellation_risk)."),
            Column("all_intents", "ARRAY<STRING>", "All detected intents."),
            Column("sentiment_score", "DOUBLE", "Customer sentiment in [-1, 1]."),
            Column("sentiment_label", "STRING", "negative | neutral | positive."),
            Column("disposition", "STRING", "Call disposition: resolved | follow_up | escalated."),
            Column("resolution_status", "STRING", "resolved | open."),
            Column("next_best_action", "STRING", "Recommended next action."),
            Column("mentioned_invoice_id", "STRING", "Invoice referenced during the call (FK)."),
            Column("mentioned_amount", "DECIMAL(10,2)", "Dollar amount discussed on the call."),
            Column("summary", "STRING", "Short natural-language call summary."),
        ],
    ),
}

# Batch reference tables ingested from raw_batch_data into UC Delta.
REFERENCE_TABLES = [T_CUSTOMERS, T_AGENTS, T_INVOICES, T_PAYMENTS]
# Derived business insight tables.
DERIVED_TABLES = [T_GOLD]
# Live-assist operational UC tables (SQL warehouse writes from the API).
OPERATIONAL_UC_TABLES = [T_BILLING_ADJUSTMENTS]
# All modeled tables (used for DDL creation with constraints + comments).
ALL_TABLES = REFERENCE_TABLES + OPERATIONAL_UC_TABLES + DERIVED_TABLES

# Curated questions the dataset can answer - seeded into the Genie space.
SAMPLE_QUESTIONS = [
    "How many calls did we get last month broken down by primary_intent?",
    "What is the average handle time (duration_sec) by agent team?",
    "Which customers called about a billing_dispute and also have an overdue invoice?",
    "What share of late_fee calls ended with resolution_status = resolved?",
    "List customers with status = at_risk who had a cancellation_risk call.",
    "Total disputed invoice amount by region this quarter.",
    "What is the average customer sentiment by plan?",
    "For autopay_issue calls, how many customers had a declined payment?",
    "Which agents on the retention team have the highest CSAT?",
    "Sum of overdue invoice amounts for customers in the enterprise segment.",
]
