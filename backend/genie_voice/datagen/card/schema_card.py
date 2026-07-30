"""Declarative credit-card issuer data model.

One spec drives BOTH the Databricks DDL (column COMMENTs + informational PK/FK
constraints) and the Genie Agent curation. Genie relies on this metadata:
comments give semantics and PK/FK tell it how to join for multi-step reasoning.

Ownership: every table here is datagen-produced reference data, batch-ingested
from the card-issuer raw landing volume into UC Delta. The hot "fast facts"
subset (cardholders + latest statement + rewards balance) is additionally
snapshotted into the Lakebase serving cache for the live voice lookup path.

Reuses the generic ``Column`` / ``ForeignKey`` / ``TableSpec`` primitives from
the telco schema module so there is a single DDL renderer.
"""
from __future__ import annotations

from genie_voice.datagen.schema import Column, ForeignKey, TableSpec

# Logical table names.
T_CARD_PRODUCTS = "card_products"
T_REWARD_CATEGORY_RATES = "reward_category_rates"
T_CARDHOLDERS = "cardholders"
T_CARDHOLDER_CARDS = "cardholder_cards"
T_TRANSACTIONS = "transactions"
T_STATEMENTS = "statements"
T_REWARDS_LEDGER = "rewards_ledger"
T_REWARD_ACTIVATIONS = "reward_activations"
T_SUBSCRIPTIONS = "subscriptions"
T_SPENDING_BY_CATEGORY = "spending_by_category"

# Controlled vocabularies (kept small + stable so Genie entity matching is sharp).
CATEGORIES = ["dining", "groceries", "gas", "travel", "streaming", "electronics", "fees", "other"]
TXN_TYPES = ["purchase", "fee", "interest", "payment", "refund", "return", "cash_advance"]
FEE_TYPES = ["none", "annual_fee", "foreign_tx", "late_fee", "interest", "cash_advance_fee"]


CARD_MODEL: dict[str, TableSpec] = {
    T_CARD_PRODUCTS: TableSpec(
        name=T_CARD_PRODUCTS,
        comment="Credit-card products the issuer offers. One row per product/tier.",
        primary_key=["product_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        columns=[
            Column("product_id", "STRING", "Unique card product identifier (e.g. PROD-CORE).", nullable=False),
            Column("product_name", "STRING", "Marketing name of the card product."),
            Column("tier", "STRING", "Product tier: core | preferred | reserve."),
            Column("annual_fee", "DECIMAL(10,2)", "Annual fee in USD (0 if none)."),
            Column("base_earn_rate", "DECIMAL(4,2)", "Base reward points earned per $1 spent (all categories)."),
            Column("foreign_tx_fee_pct", "DECIMAL(4,2)", "Foreign-transaction fee as a percent of the foreign purchase amount."),
            Column("signup_bonus_points", "BIGINT", "Sign-up bonus points, when the spend threshold is met."),
            Column("apr_pct", "DECIMAL(5,2)", "Purchase APR percent applied to revolving balances."),
        ],
    ),
    T_REWARD_CATEGORY_RATES: TableSpec(
        name=T_REWARD_CATEGORY_RATES,
        comment=(
            "Per-category reward earn rates for a card product. Lets Genie compute "
            "the OPTIMAL points a purchase could have earned versus what it did earn."
        ),
        primary_key=["rate_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("product_id", T_CARD_PRODUCTS, "product_id")],
        columns=[
            Column("rate_id", "STRING", "Unique rate identifier.", nullable=False),
            Column("product_id", "STRING", "Card product this earn rate belongs to.", nullable=False),
            Column("category", "STRING", "Spend category: dining|groceries|gas|travel|streaming|electronics|other."),
            Column("earn_multiplier", "DECIMAL(4,2)", "Points earned per $1 spent in this category on this product."),
            Column("quarterly_cap", "DECIMAL(10,2)", "Spend cap the multiplier applies to per quarter (null = uncapped)."),
            Column("requires_activation", "BOOLEAN", "Whether the cardholder must ACTIVATE this bonus to earn it."),
        ],
    ),
    T_CARDHOLDERS: TableSpec(
        name=T_CARDHOLDERS,
        comment="Cardholder master. One row per customer/account holder.",
        primary_key=["customer_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("primary_product_id", T_CARD_PRODUCTS, "product_id")],
        columns=[
            Column("customer_id", "STRING", "Unique cardholder/account identifier.", nullable=False),
            Column("full_name", "STRING", "Cardholder full name."),
            Column("segment", "STRING", "Cardholder segment: consumer | affluent | private."),
            Column("region", "STRING", "Region: NA | EMEA | APAC."),
            Column("primary_product_id", "STRING", "The cardholder's primary card product (FK card_products)."),
            Column("credit_limit", "DECIMAL(10,2)", "Total credit limit in USD across the account."),
            Column("apr_pct", "DECIMAL(5,2)", "Purchase APR percent on the account."),
            Column("autopay_type", "STRING", "Autopay setting: none | minimum | fixed | full_balance."),
            Column("points_balance", "BIGINT", "Current lifetime redeemable rewards points balance."),
            Column("status", "STRING", "Account status: active | at_risk | delinquent."),
            Column("tenure_months", "INT", "Months since the account was opened."),
            Column("email", "STRING", "Cardholder email address."),
            Column("signup_date", "DATE", "Date the account was opened."),
        ],
    ),
    T_CARDHOLDER_CARDS: TableSpec(
        name=T_CARDHOLDER_CARDS,
        comment=(
            "Cards a cardholder holds. A cardholder may hold MORE THAN ONE product, "
            "which is why some spend earns fewer points than it could on their other card."
        ),
        primary_key=["holding_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("customer_id", T_CARDHOLDERS, "customer_id"),
            ForeignKey("product_id", T_CARD_PRODUCTS, "product_id"),
        ],
        columns=[
            Column("holding_id", "STRING", "Unique holding identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder who holds this card.", nullable=False),
            Column("product_id", "STRING", "Card product held.", nullable=False),
            Column("card_id", "STRING", "Account card identifier used on transactions/statements.", nullable=False),
            Column("opened_date", "DATE", "Date this card was opened."),
            Column("is_primary", "BOOLEAN", "Whether this is the cardholder's primary card."),
        ],
    ),
    T_TRANSACTIONS: TableSpec(
        name=T_TRANSACTIONS,
        comment=(
            "Transaction-level ledger: purchases, fees, interest, payments, refunds "
            "and returns. THIS is the substrate Genie Agent mode decomposes to explain "
            "why a statement balance changed and where rewards were earned or lost."
        ),
        primary_key=["txn_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("customer_id", T_CARDHOLDERS, "customer_id"),
            ForeignKey("product_id", T_CARD_PRODUCTS, "product_id"),
        ],
        columns=[
            Column("txn_id", "STRING", "Unique transaction identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder the transaction belongs to.", nullable=False),
            Column("card_id", "STRING", "Card the transaction was made on.", nullable=False),
            Column("product_id", "STRING", "Card product of the card used (denormalized for reward-rate joins)."),
            Column("cycle", "STRING", "Statement cycle this transaction falls in (YYYY-MM)."),
            Column("posted_date", "DATE", "Date the transaction posted."),
            Column("merchant", "STRING", "Merchant / descriptor."),
            Column("category", "STRING", "Spend category: dining|groceries|gas|travel|streaming|electronics|fees|other."),
            Column("mcc", "STRING", "Merchant category code."),
            Column("amount", "DECIMAL(10,2)", "Amount in USD. Positive = charge/fee/interest; negative = payment/refund/return."),
            Column("currency", "STRING", "Original transaction currency (ISO-4217)."),
            Column("is_foreign", "BOOLEAN", "Whether this was a foreign (non-domestic-currency) purchase."),
            Column("txn_type", "STRING", "purchase | fee | interest | payment | refund | return | cash_advance."),
            Column("fee_type", "STRING", "When txn_type=fee/interest: none | annual_fee | foreign_tx | late_fee | interest | cash_advance_fee."),
            Column("is_reversed", "BOOLEAN", "Whether this purchase was later reversed (e.g. a return), clawing back its points."),
            Column("reversal_of_txn_id", "STRING", "For a return/refund, the txn_id of the original purchase it reverses (else null)."),
        ],
    ),
    T_STATEMENTS: TableSpec(
        name=T_STATEMENTS,
        comment=(
            "Monthly statement per card. Carries the balance roll-forward "
            "(prev_balance - payments + purchases + fees + interest = new_balance) "
            "so Genie can compare this cycle to prior cycles and explain the change."
        ),
        primary_key=["statement_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("customer_id", T_CARDHOLDERS, "customer_id")],
        columns=[
            Column("statement_id", "STRING", "Unique statement identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder the statement belongs to.", nullable=False),
            Column("card_id", "STRING", "Card the statement is for."),
            Column("cycle", "STRING", "Statement cycle (YYYY-MM)."),
            Column("statement_date", "DATE", "Statement closing date."),
            Column("due_date", "DATE", "Payment due date."),
            Column("prev_balance", "DECIMAL(10,2)", "Balance carried in from the prior statement, USD."),
            Column("purchases", "DECIMAL(10,2)", "Total purchases posted this cycle, USD."),
            Column("fees", "DECIMAL(10,2)", "Total fees posted this cycle (annual, foreign-tx, late), USD."),
            Column("interest", "DECIMAL(10,2)", "Interest charged this cycle on any revolving balance, USD."),
            Column("payments", "DECIMAL(10,2)", "Total payments/credits applied this cycle, USD."),
            Column("new_balance", "DECIMAL(10,2)", "Closing statement balance, USD."),
            Column("min_payment", "DECIMAL(10,2)", "Minimum payment due, USD."),
            Column("paid_amount", "DECIMAL(10,2)", "Amount the cardholder paid toward this statement, USD."),
            Column("paid_in_full", "BOOLEAN", "Whether the statement was paid in full (no interest next cycle)."),
        ],
    ),
    T_REWARDS_LEDGER: TableSpec(
        name=T_REWARDS_LEDGER,
        comment=(
            "Per-cycle, per-category rewards accounting: points earned versus the "
            "points that were POSSIBLE (optimal card/activation), plus reversed and "
            "expired points. Explains exactly where and why rewards leaked."
        ),
        primary_key=["ledger_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("customer_id", T_CARDHOLDERS, "customer_id")],
        columns=[
            Column("ledger_id", "STRING", "Unique ledger row identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder these rewards belong to.", nullable=False),
            Column("card_id", "STRING", "Card the eligible spend was made on."),
            Column("cycle", "STRING", "Statement cycle (YYYY-MM)."),
            Column("category", "STRING", "Spend category for this rewards line."),
            Column("eligible_spend", "DECIMAL(10,2)", "Spend that was eligible to earn rewards in this category/cycle, USD."),
            Column("points_earned", "BIGINT", "Points actually earned on this eligible spend."),
            Column("points_possible", "BIGINT", "Points that COULD have been earned with the optimal card/activation."),
            Column("reversed_points", "BIGINT", "Points clawed back this cycle due to a return/refund."),
            Column("expired_points", "BIGINT", "Points that expired this cycle."),
            Column("missed_reason", "STRING", "Why points were left on the table: none | wrong_card | inactive_bonus | excluded_category | returned_purchase | expired."),
        ],
    ),
    T_REWARD_ACTIVATIONS: TableSpec(
        name=T_REWARD_ACTIVATIONS,
        comment=(
            "Rotating quarterly bonus categories that require the cardholder to "
            "ACTIVATE them. A non-activated bonus is a common reason rewards are missed."
        ),
        primary_key=["activation_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[
            ForeignKey("customer_id", T_CARDHOLDERS, "customer_id"),
            ForeignKey("product_id", T_CARD_PRODUCTS, "product_id"),
        ],
        columns=[
            Column("activation_id", "STRING", "Unique activation identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder the bonus is offered to.", nullable=False),
            Column("product_id", "STRING", "Card product the bonus applies to.", nullable=False),
            Column("promo_id", "STRING", "Promotion identifier."),
            Column("category", "STRING", "Bonus spend category (e.g. groceries)."),
            Column("quarter", "STRING", "Bonus quarter (YYYY-Qn)."),
            Column("bonus_multiplier", "DECIMAL(4,2)", "EXTRA points per $1 in the category when activated (added to the base rate)."),
            Column("window_start", "DATE", "First date the bonus is valid."),
            Column("window_end", "DATE", "Last date the bonus is valid."),
            Column("activated", "BOOLEAN", "Whether the cardholder activated the bonus."),
            Column("activated_date", "DATE", "Date the cardholder activated it (null if never)."),
        ],
    ),
    T_SUBSCRIPTIONS: TableSpec(
        name=T_SUBSCRIPTIONS,
        comment=(
            "Detected recurring-merchant subscriptions on the account. A NEW "
            "subscription that started this cycle is a common statement-increase driver."
        ),
        primary_key=["subscription_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("customer_id", T_CARDHOLDERS, "customer_id")],
        columns=[
            Column("subscription_id", "STRING", "Unique subscription identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder the subscription belongs to.", nullable=False),
            Column("card_id", "STRING", "Card the subscription bills to."),
            Column("merchant", "STRING", "Recurring merchant / service name."),
            Column("category", "STRING", "Spend category of the subscription."),
            Column("monthly_amount", "DECIMAL(10,2)", "Recurring monthly charge, USD."),
            Column("started_cycle", "STRING", "Cycle (YYYY-MM) the subscription first appeared."),
            Column("is_active", "BOOLEAN", "Whether the subscription is currently active."),
            Column("first_txn_id", "STRING", "The first transaction for this subscription (FK transactions)."),
        ],
    ),
    T_SPENDING_BY_CATEGORY: TableSpec(
        name=T_SPENDING_BY_CATEGORY,
        comment=(
            "Pre-aggregated spending by category per customer per cycle. Lets the "
            "UI render expense-trend charts instantly and gives Genie a direct path "
            "to identify WHICH categories spiked — the first question customers ask."
        ),
        primary_key=["spend_cat_id"],
        properties={"delta.enableChangeDataFeed": "true"},
        foreign_keys=[ForeignKey("customer_id", T_CARDHOLDERS, "customer_id")],
        columns=[
            Column("spend_cat_id", "STRING", "Unique row identifier.", nullable=False),
            Column("customer_id", "STRING", "Cardholder.", nullable=False),
            Column("cycle", "STRING", "Statement cycle (YYYY-MM)."),
            Column("category", "STRING", "Spend category."),
            Column("total_amount", "DECIMAL(10,2)", "Total spend in this category/cycle, USD."),
            Column("txn_count", "INT", "Number of transactions in this category/cycle."),
            Column("largest_merchant", "STRING", "Merchant with the highest single charge in this category/cycle."),
            Column("largest_amount", "DECIMAL(10,2)", "Amount of the largest single charge."),
            Column("is_new_category", "BOOLEAN", "True if this category had zero spend in the prior cycle (new expense)."),
            Column("pct_change_vs_prior", "DECIMAL(6,2)", "Percent change vs same category in prior cycle (null if no prior)."),
        ],
    ),
}

# Batch reference tables ingested from the card-issuer raw landing volume into UC Delta.
CARD_REFERENCE_TABLES = [
    T_CARD_PRODUCTS,
    T_REWARD_CATEGORY_RATES,
    T_CARDHOLDERS,
    T_CARDHOLDER_CARDS,
    T_TRANSACTIONS,
    T_STATEMENTS,
    T_REWARDS_LEDGER,
    T_REWARD_ACTIVATIONS,
    T_SUBSCRIPTIONS,
    T_SPENDING_BY_CATEGORY,
]
# Parents first so FOREIGN KEY ... REFERENCES resolves at CREATE TABLE time.
CARD_ALL_TABLES = CARD_REFERENCE_TABLES

# Hot subset served from the Lakebase fast cache on the live voice lookup path.
CARD_FAST_FACTS_TABLES = [T_CARDHOLDERS, T_STATEMENTS, T_SPENDING_BY_CATEGORY]

# Curated questions seeded into the Genie Agent. The first two are the diagnostic
# ANCHORS the voice use cases depend on (validated against the ground-truth ledger).
CARD_SAMPLE_QUESTIONS = [
    # Statement Insights anchor
    "Why did this customer's expenses spike this cycle compared with their prior three cycles? Itemize the drivers by category.",
    # Rewards Optimizer anchor
    "Where is this customer losing rewards they qualify for this cycle, and why? Quantify the points gap by reason.",
    "What were the largest expenses this cycle and which are one-off versus recurring?",
    "Which spending categories increased the most compared to prior months?",
    "How much did this customer pay in fees this cycle, broken down by fee type?",
    "Which recurring subscriptions started this cycle?",
    "For each spend category this cycle, how many points were earned versus possible?",
    "Which rotating bonus categories did this customer fail to activate, and how many points did that cost?",
    "How much interest was charged this cycle and why?",
    "What share of foreign purchases incurred a foreign-transaction fee, and how much in total?",
    "Which cardholders have the largest gap between points earned and points possible?",
]
