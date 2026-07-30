"""Credit-card issuer domain dataset (voice-first "why" assistant).

A self-contained, parallel data domain that lives ALONGSIDE the telco/contact-
center dataset (``datagen/schema.py`` + ``datagen/generators.py``) without
touching it. It reuses the generic primitives — ``Column`` / ``ForeignKey`` /
``TableSpec`` and ``TableSpec.render_ddl`` — and is selected by the ``card_issuer:``
config block (own catalog schema + own Genie Agent), so the two demos never
collide.

The data is engineered so Databricks Genie Agent mode can dig into *why*:
  - Statement Insights   — "why did my statement balance go up this cycle?"
  - Rewards Optimizer    — "why am I not earning all my rewards?"

Two archetype cardholders carry hand-seeded, tie-out ground truth for those two
questions (see ``generators_card.GROUND_TRUTH``); a deterministic random
population provides realistic volume for aggregate reasoning.
"""
from __future__ import annotations
