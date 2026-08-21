"""Databricks Knowledge Agent tools for the realtime voice LLM (knowledge profile).

TIER 1: a real, navigable knowledge voice experience backed by a MOCK curated
corpus, so the end-to-end flow (greeting + voice + cited answers) works today.
The corpus is the single source of truth for both this tool and the
``/knowledge/corpus`` UI endpoint, so spoken answers and on-screen cards agree.

Cite-or-silence: ``knowledge_search`` only ever returns entries from the corpus,
each carrying its own ``citation``. The agent is instructed to answer from those
entries and to say it does not know when the search comes back empty — it must
not improvise Databricks behaviour.

SEAM for Tier 2: replace ``KNOWLEDGE_CORPUS`` / ``_run_knowledge_search`` with
Databricks Vector Search (or a Genie space) over real docs; the profile
registration, greeting, and frontend contract stay the same.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .tool_registry import ToolContext, attach_session_identity, register, run_tool, tools_spec

_PROFILE = "knowledge"

KNOWLEDGE_BRAND = "Databricks Knowledge Agent"
KNOWLEDGE_AGENT_NAME = "Genie"

# --- Two lanes, one page ---------------------------------------------------- #
# Every question the PAGE publishes is a Genie One question: asking it runs a live
# MCP round-trip against the caller's own governed workspace (OBO), so the answer
# describes their workspace rather than a brochure.
#
#   workspace lane — ``WORKSPACE_PROMPTS``: everything on screen. Genie One answers.
#   pack lane      — ``KNOWLEDGE_CORPUS``: curated platform documentation, kept as
#                    the cited fallback for spoken concept questions ("what is
#                    Unity Catalog"). Not shown as cards.
CAT_PLATFORM = "Platform & Concepts"
CAT_COVERAGE = "Coverage & Scope"
CAT_TRUST = "Data & Trust"
CAT_ANSWERS = "Answers & Explainability"
CAT_DOCUMENTS = "Documents & Connectors"
CAT_ACTIONS = "Actions & Automation"
CAT_LIMITS = "Limits & Boundaries"

# Stable display order for the on-screen groups.
KNOWLEDGE_CATEGORIES: tuple[str, ...] = (
    CAT_COVERAGE,
    CAT_TRUST,
    CAT_ANSWERS,
    CAT_DOCUMENTS,
    CAT_ACTIONS,
    CAT_LIMITS,
    CAT_PLATFORM,
)

# Curated platform documentation (Tier 1). Every entry carries its own citation so
# the agent can attribute each spoken answer.
KNOWLEDGE_CORPUS: list[dict[str, Any]] = [
    {
        "id": "unity-catalog",
        "category": CAT_PLATFORM,
        "topic": "Unity Catalog",
        "question": "What is Unity Catalog?",
        "answer": (
            "Unity Catalog is the unified governance layer for data and AI on Databricks. "
            "It centralises access control, auditing, lineage, and discovery across "
            "catalogs, schemas, tables, volumes, and models, so one permission model "
            "covers every workspace in an account."
        ),
        "citation": "Databricks Docs · Data Governance · Unity Catalog",
        "keywords": ["unity", "catalog", "governance", "permissions", "lineage", "access"],
    },
    {
        "id": "delta-lake",
        "category": CAT_PLATFORM,
        "topic": "Delta Lake",
        "question": "Why Delta Lake instead of plain Parquet?",
        "answer": (
            "Delta Lake adds a transaction log on top of Parquet, which brings ACID "
            "transactions, schema enforcement, and time travel. That is what makes "
            "concurrent reads and writes safe, and lets you query or restore an earlier "
            "version of a table."
        ),
        "citation": "Databricks Docs · Delta Lake · Table Format",
        "keywords": ["delta", "lake", "parquet", "acid", "transaction", "time travel", "schema"],
    },
    {
        "id": "lakebase",
        "category": CAT_PLATFORM,
        "topic": "Lakebase",
        "question": "What is Lakebase used for?",
        "answer": (
            "Lakebase is managed Postgres on Databricks for low-latency serving. You "
            "sync a governed Delta table into it so applications can read single rows in "
            "milliseconds, while Unity Catalog stays the source of truth for analytics."
        ),
        "citation": "Databricks Docs · Lakebase · Synced Tables",
        "keywords": ["lakebase", "postgres", "serving", "latency", "synced", "oltp"],
    },
    {
        "id": "genie",
        "category": CAT_PLATFORM,
        "topic": "Genie",
        "question": "How does Genie answer questions about my data?",
        "answer": (
            "Genie is a conversational interface over a curated set of tables called a "
            "Genie space. It translates a natural-language question into governed SQL, "
            "runs it, and returns the result with the query it used, so every answer can "
            "be checked."
        ),
        "citation": "Databricks Docs · AI/BI · Genie Spaces",
        # No bare "sql": it is too generic to identify THIS entry, and it pulled
        # workspace questions ("the slowest SQL queries today") into a docs answer.
        "keywords": ["genie", "space", "natural language", "text to sql", "aibi"],
    },
    {
        "id": "vector-search",
        "category": CAT_PLATFORM,
        "topic": "Vector Search",
        "question": "What does Vector Search do?",
        "answer": (
            "Vector Search is a managed vector database on Databricks. It builds and "
            "keeps an index in sync with a Delta table, then serves nearest-neighbour "
            "lookups so retrieval-augmented applications can ground answers in your own "
            "governed documents."
        ),
        "citation": "Databricks Docs · Generative AI · Vector Search",
        "keywords": ["vector", "search", "embedding", "index", "rag", "retrieval", "similarity"],
    },
    {
        "id": "model-serving",
        "category": CAT_PLATFORM,
        "topic": "Model Serving",
        "question": "How do I put a model behind an API?",
        "answer": (
            "Mosaic AI Model Serving hosts a registered Unity Catalog model as an "
            "autoscaling REST endpoint. It handles scaling, versioning, and traffic "
            "splitting, and it can scale to zero when idle so you only pay while it is "
            "serving requests."
        ),
        "citation": "Databricks Docs · Mosaic AI · Model Serving",
        "keywords": ["model", "serving", "endpoint", "deploy", "mlflow", "rest", "autoscale"],
    },
    {
        "id": "dlt",
        "category": CAT_PLATFORM,
        "topic": "Declarative Pipelines",
        "question": "What are declarative pipelines?",
        "answer": (
            "Declarative pipelines let you define the tables you want and the quality "
            "expectations they must meet, and the platform works out the execution order "
            "and incremental processing. Failed expectations are recorded as data-quality "
            "metrics rather than silently passing through."
        ),
        "citation": "Databricks Docs · Data Engineering · Declarative Pipelines",
        "keywords": ["pipeline", "dlt", "declarative", "expectations", "quality", "etl", "ingest"],
    },
    {
        "id": "photon",
        "category": CAT_PLATFORM,
        "topic": "Photon",
        "question": "What is Photon?",
        "answer": (
            "Photon is a vectorised query engine written in C++ that runs underneath the "
            "same SQL and DataFrame code you already have. It speeds up scans, joins, and "
            "aggregations without any change to your queries."
        ),
        "citation": "Databricks Docs · Compute · Photon Engine",
        "keywords": ["photon", "engine", "vectorised", "vectorized", "query engine"],
    },
]

# --- Published questions (all Genie One MCP) -------------------------------- #
# Every question on the Knowledge page lives here, and every one of them is a LIVE
# Genie One round-trip against the caller's own governed workspace (OBO). They are
# deliberately capability questions — what Genie One can reach, trust, explain, and
# act on — because the useful first answer is "here is what I can do with YOUR data",
# not a canned platform fact.
#
#   source   — the Genie One surface the answer comes from.
#   preview  — what asking will actually do (there is no canned answer, by design).
#   keywords — route a SPOKEN version of the question into this same lane.
WORKSPACE_PROMPTS: list[dict[str, Any]] = [
    # ---- Coverage & Scope --------------------------------------------------- #
    {
        "id": "answerable-questions",
        "category": CAT_COVERAGE,
        "topic": "What you can answer",
        "question": "What kinds of business questions can you answer using the data I can access?",
        "source": "Genie One · accessible domains",
        "preview": (
            "Genie One reports the question types it can serve from the data your own "
            "permissions reach — scoped to you, not to the catalog at large."
        ),
        "keywords": ["business questions", "kinds of questions", "can you answer", "data access"],
    },
    {
        "id": "domain-inventory",
        "category": CAT_COVERAGE,
        "topic": "What is available to me",
        "question": (
            "Which business domains, Genie Agents, dashboards, metric views, and "
            "queries are available to me?"
        ),
        "source": "Genie One · domain & asset inventory",
        "preview": (
            "Lists the governed assets you can actually use: business domains, Genie "
            "Agents, dashboards, metric views, and saved queries."
        ),
        "keywords": [
            "business domains", "genie agents", "dashboards", "metric views", "available"
        ],
    },
    {
        "id": "analysis-types",
        "category": CAT_COVERAGE,
        "topic": "Supported analysis",
        "question": "What types of analysis are supported across these domains?",
        "source": "Genie One · analysis capabilities",
        "preview": (
            "Describes the analysis it supports per domain — aggregation, trend, "
            "comparison, ranking, cohort — rather than promising anything generic."
        ),
        "keywords": ["types analysis", "analysis supported", "across domains"],
    },
    {
        "id": "combine-domains",
        "category": CAT_COVERAGE,
        "topic": "Combining domains",
        "question": "Which domains can you combine when answering a question?",
        "source": "Genie One · cross-domain joins",
        "preview": (
            "Names the domains it can join in one answer, and where a governed "
            "relationship exists to join them on."
        ),
        "keywords": ["combine", "domains combine", "combine domains", "cross domain"],
    },
    # ---- Data & Trust ------------------------------------------------------- #
    {
        "id": "sources-and-trust",
        "category": CAT_TRUST,
        "topic": "Sources it trusts",
        "question": (
            "What data sources do you use to answer questions, and how do you decide "
            "which source to trust?"
        ),
        "source": "Genie One · source selection",
        "preview": (
            "Explains which sources back its answers and the precedence it applies when "
            "two sources disagree — the difference between an answer and a guess."
        ),
        "keywords": ["data sources", "trust", "which source", "decide source"],
    },
    {
        "id": "business-terms",
        "category": CAT_TRUST,
        "topic": "Terms it understands",
        "question": "What business terms, metrics, and relationships do you understand?",
        "source": "Genie One · semantic model",
        "preview": (
            "Reports the vocabulary it has been taught: business terms, metric "
            "definitions, and the relationships between entities."
        ),
        "keywords": ["business terms", "relationships", "terms metrics", "vocabulary"],
    },
    {
        "id": "certified-metrics",
        "category": CAT_TRUST,
        "topic": "Certified metrics",
        "question": "Which metrics have governed or certified definitions?",
        "source": "Genie One · certified metric definitions",
        "preview": (
            "Separates metrics with a governed, certified definition from ones computed "
            "ad hoc, so you know which numbers are safe to quote."
        ),
        "keywords": ["governed", "certified", "certified definitions", "metric definitions"],
    },
    # ---- Answers & Explainability ------------------------------------------ #
    {
        "id": "visualizations",
        "category": CAT_ANSWERS,
        "topic": "Visualizations",
        "question": "What visualizations can you generate from analytical questions?",
        "source": "Genie One · visualization support",
        "preview": (
            "Describes the chart types it can produce from an analytical question, and "
            "when a result is better read as a table."
        ),
        "keywords": ["visualizations", "visualisations", "charts", "generate"],
    },
    {
        "id": "explainability",
        "category": CAT_ANSWERS,
        "topic": "Show your work",
        "question": "Can you explain the reasoning, SQL, assumptions, and sources behind an answer?",
        "source": "Genie One · answer provenance",
        "preview": (
            "Asks for the audit trail: the reasoning it followed, the SQL it ran, the "
            "assumptions it made, and the sources it read."
        ),
        "keywords": ["reasoning", "assumptions", "explain reasoning", "behind answer"],
    },
    {
        "id": "follow-ups",
        "category": CAT_ANSWERS,
        "topic": "Follow-up questions",
        "question": "What kinds of follow-up questions can I ask after an initial answer?",
        "source": "Genie One · conversational context",
        "preview": (
            "Explains what it keeps in context between turns, so you know what you can "
            "refine without restating the whole question."
        ),
        "keywords": ["follow up", "followup", "initial answer", "after answer"],
    },
    {
        "id": "ambiguity",
        "category": CAT_ANSWERS,
        "topic": "When it must ask",
        "question": "Which questions would require clarification because the data or terminology is ambiguous?",
        "source": "Genie One · ambiguity handling",
        "preview": (
            "Names the cases where it will ask a clarifying question instead of "
            "guessing — ambiguous terms, overlapping metrics, undated ranges."
        ),
        "keywords": ["clarification", "ambiguous", "terminology", "require clarification"],
    },
    # ---- Documents & Connectors -------------------------------------------- #
    {
        "id": "structured-vs-documents",
        "category": CAT_DOCUMENTS,
        "topic": "Data vs documents",
        "question": "What questions can you answer from structured Databricks data versus connected documents?",
        "source": "Genie One · structured & unstructured coverage",
        "preview": (
            "Draws the line between what it answers from governed tables and what it "
            "answers from connected documents."
        ),
        "keywords": ["structured", "connected documents", "versus", "documents"],
    },
    {
        "id": "connectors",
        "category": CAT_DOCUMENTS,
        "topic": "Connected sources",
        "question": (
            "Can you use connected sources such as Google Drive, SharePoint, Slack, "
            "Glean, Jira, or Confluence?"
        ),
        "source": "Genie One · connected sources",
        "preview": (
            "Reports which external sources are actually connected for you, rather than "
            "which ones are theoretically supported."
        ),
        "keywords": [
            "google drive", "sharepoint", "slack", "glean", "jira", "confluence",
            "connected sources",
        ],
    },
    {
        "id": "file-with-data",
        "category": CAT_DOCUMENTS,
        "topic": "File + your data",
        "question": "Can you analyze a file together with my Databricks data?",
        "source": "Genie One · file analysis",
        "preview": (
            "Covers whether an uploaded file can be analysed alongside governed tables, "
            "and what governance applies when it is."
        ),
        "keywords": ["analyze file", "analyse file", "file together", "upload file"],
    },
    # ---- Actions & Automation ---------------------------------------------- #
    {
        "id": "create-agent",
        "category": CAT_ACTIONS,
        "topic": "Reusable agent",
        "question": "Can you create a reusable Genie Agent from this conversation?",
        "source": "Genie One · agent authoring",
        "preview": (
            "Asks whether this conversation can be promoted into a reusable Genie Agent "
            "others can run."
        ),
        "keywords": ["reusable genie agent", "create agent", "this conversation"],
    },
    {
        "id": "scheduled-briefing",
        "category": CAT_ACTIONS,
        "topic": "Scheduled briefing",
        "question": "Can you create a scheduled briefing or recurring report?",
        "source": "Genie One · scheduling",
        "preview": (
            "Covers turning an answer into a recurring briefing or report on a schedule, "
            "and who receives it."
        ),
        "keywords": ["scheduled briefing", "recurring report", "schedule", "briefing"],
    },
    {
        "id": "draft-document",
        "category": CAT_ACTIONS,
        "topic": "Draft a document",
        "question": "Can you draft a document based on an analysis?",
        "source": "Genie One · document generation",
        "preview": (
            "Covers writing an analysis up as a document, with the figures and sources "
            "carried through rather than retyped."
        ),
        "keywords": ["draft", "draft document", "document based", "write up"],
    },
    {
        "id": "external-actions",
        "category": CAT_ACTIONS,
        "topic": "Acting in other systems",
        "question": (
            "Can you take actions in external systems, and what permissions or "
            "approvals would be required?"
        ),
        "source": "Genie One · action permissions",
        "preview": (
            "The effect question: which actions it can take outside Databricks, and the "
            "permissions or approvals each one demands first."
        ),
        "keywords": ["take actions", "external systems", "approvals", "permissions required"],
    },
    # ---- Limits & Boundaries ----------------------------------------------- #
    {
        "id": "limitations",
        "category": CAT_LIMITS,
        "topic": "Known limits",
        "question": "What are the known limitations, data freshness constraints, and permission boundaries?",
        "source": "Genie One · capability boundaries",
        "preview": (
            "Asks for the honest edges: what it cannot do, how stale the data may be, "
            "and where your permissions stop the answer."
        ),
        "keywords": ["known limitations", "data freshness", "permission boundaries", "constraints"],
    },
    {
        "id": "avoid-questions",
        "category": CAT_LIMITS,
        "topic": "What not to ask",
        "question": "Which questions should I avoid asking because the available data cannot answer them reliably?",
        "source": "Genie One · reliability boundaries",
        "preview": (
            "The inverse of the first question: which questions the available data "
            "cannot support, so you do not act on a shaky number."
        ),
        "keywords": ["avoid", "cannot answer", "reliably", "should avoid"],
    },
]

_KNOWLEDGE_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": (
            "Search the governed Databricks knowledge base and return matching entries, "
            "each with its own citation. CALL THIS IMMEDIATELY whenever the caller asks "
            "anything about Databricks — a product, a concept, or how something works. "
            "Answer ONLY from the entries returned; if it returns no matches, say you do "
            "not have that in the knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The caller's question, in their own words.",
                }
            },
            "required": ["query"],
        },
    },
}

# Words that carry no topical signal; scoring them would match every entry.
_STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from how i in is it me my of on or "
    "please tell that the to use used using was what when where which who why with you "
    "your about into".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOPWORDS]


def _score(entry: dict[str, Any], query: str) -> int:
    """Keyword overlap between the query and one entry.

    Deliberately simple and explainable: keyword hits are weighted above topic and
    question-text hits so "what is unity catalog" ranks the Unity Catalog entry
    first. Tier 2 replaces the whole function with a vector index.
    """
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0
    score = 0
    for keyword in entry["keywords"]:
        keyword_tokens = set(_tokens(keyword))
        if keyword_tokens and keyword_tokens <= query_tokens:
            score += 3
    score += 2 * len(query_tokens & set(_tokens(entry["topic"])))
    score += len(query_tokens & set(_tokens(entry["question"])))
    return score


# Minimum score that counts as a match: a keyword hit (3) or a topic-name hit (2).
# A lone generic word shared with an entry's question text (1) is NOT evidence that
# the entry answers the question — "which tables hold sensitive data" overlapping
# the Genie entry's question on "data" must not be served a Genie docs answer.
_MIN_RELEVANCE = 2


def search_knowledge(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Top corpus entries for ``query``, best first; empty when nothing matches."""
    ranked = sorted(
        ((_score(e, query), e) for e in KNOWLEDGE_CORPUS),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [
        {
            "topic": entry["topic"],
            "answer": entry["answer"],
            "citation": entry["citation"],
            "category": entry["category"],
        }
        for score, entry in ranked[:limit]
        if score >= _MIN_RELEVANCE
    ]


_TOPIC_LOCALES_PATH = Path(__file__).resolve().parent / "phrases" / "knowledge_topics.json"


@lru_cache(maxsize=1)
def _topic_locales() -> dict[str, Any]:
    """Offline-reviewed display translations keyed by BCP-47 language tag."""
    try:
        loaded = json.loads(_TOPIC_LOCALES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _topic_locale(language: str | None) -> dict[str, Any]:
    """Resolve one display catalog without ever making localization a runtime call."""
    from genie_voice.i18n import content_language

    try:
        tag = content_language(language)
    except ValueError:
        tag = "en-US"
    catalog = _topic_locales()
    localized = catalog.get(tag)
    if isinstance(localized, dict):
        return localized
    base = tag.split("-", 1)[0].lower()
    return next(
        (
            value
            for key, value in catalog.items()
            if key.split("-", 1)[0].lower() == base and isinstance(value, dict)
        ),
        {},
    )


def knowledge_categories(language: str = "en-US") -> list[str]:
    """Category headings in stable order, localized for display."""
    locale = _topic_locale(language)
    labels = locale.get("categories")
    if not isinstance(labels, dict):
        labels = {}
    return [str(labels.get(category) or category) for category in KNOWLEDGE_CATEGORIES]


def knowledge_topics(language: str = "en-US") -> list[dict[str, Any]]:
    """The localized questions the page publishes, in display order — all Genie One.

    Only the workspace lane is published: every card is a live round-trip against
    the caller's own governed workspace. ``preview`` says what asking will DO, and
    there is deliberately no canned answer — the answer only exists once Genie One
    runs, so the page can never show a number the workspace did not produce. The
    docs corpus stays behind the spoken concept lane, not on screen.

    Display text is selected from a committed offline catalog. The canonical English
    question travels separately as ``canonical_question`` for stable content identity
    and auditing; ``question`` is the caller-language version and is the one a future
    click-to-ask interaction should send.
    """
    locale = _topic_locale(language)
    category_labels = locale.get("categories")
    topic_labels = locale.get("topics")
    if not isinstance(category_labels, dict):
        category_labels = {}
    if not isinstance(topic_labels, dict):
        topic_labels = {}

    return [
        {
            "id": prompt["id"],
            "category": str(category_labels.get(prompt["category"]) or prompt["category"]),
            "topic": str(
                (topic_labels.get(prompt["id"]) or {}).get("topic") or prompt["topic"]
            ),
            "question": str(
                (topic_labels.get(prompt["id"]) or {}).get("question")
                or prompt["question"]
            ),
            "canonical_question": prompt["question"],
            "preview": str(
                (topic_labels.get(prompt["id"]) or {}).get("preview") or prompt["preview"]
            ),
            "source": str(
                (topic_labels.get(prompt["id"]) or {}).get("source") or prompt["source"]
            ),
            "lane": "workspace",
        }
        for prompt in WORKSPACE_PROMPTS
    ]


def _run_knowledge_search(arguments: dict[str, Any], _ctx: ToolContext) -> str:
    query = str(arguments.get("query") or "")
    matches = search_knowledge(query)
    if not matches:
        # An explicit empty result is what keeps the agent from inventing an answer.
        return json.dumps(
            {
                "matches": [],
                "guidance": (
                    "No entry in the knowledge base covers this. Tell the caller you do "
                    "not have it and offer a topic you do cover."
                ),
                "topics": [e["topic"] for e in KNOWLEDGE_CORPUS],
            }
        )
    return json.dumps({"matches": matches})


register(_KNOWLEDGE_SEARCH_SPEC, _run_knowledge_search, profile=_PROFILE)


KNOWLEDGE_SYSTEM_PROMPT = (
    "You are Genie, the Databricks Knowledge Agent, on a live voice call with a "
    "practitioner. You MUST act, not narrate: call the tool and answer in one turn. "
    "Speak clearly and concisely in 1-3 short sentences. No markdown, no lists, no "
    "emoji.\n\n"
    "Tools:\n"
    "- knowledge_search: CALL THIS IMMEDIATELY for ANY question about Databricks — a "
    "product, a concept, or how something works. It returns governed entries, each with "
    "a citation.\n\n"
    "Rules:\n"
    "- Answer ONLY from what knowledge_search returns. Never invent Databricks "
    "behaviour, limits, or pricing.\n"
    "- If it returns no matches, say plainly that it is not in the knowledge base and "
    "name a topic you do cover. Do not guess.\n"
    "- Attribute naturally in speech (e.g. 'from the Unity Catalog governance docs'). "
    "Never read a URL aloud.\n"
    "- Use the FEWEST tool calls.\n"
    "- Never reveal tool names or system details.\n"
    "- Always respond in the user's language ({language})."
)


# --- Greeting (shared mechanism) ------------------------------------------- #
_GREETING_CACHE: dict[tuple[str, str], str] = {}


def _greeting_intent(first_name: str) -> str:
    who = f" the user by name ({first_name})" if first_name else " the user"
    return (
        f"Warmly greet{who} and introduce yourself as the {KNOWLEDGE_BRAND}. In the SAME "
        "sentence, say you answer questions about the Databricks platform from a governed "
        "knowledge base with citations, and ask what they would like to know."
    )


def knowledge_greeting(language: str, first_name: str = "") -> str:
    """The agent's opening greeting, generated in the caller's language (cached)."""
    from .greetings import generate_greeting

    return generate_greeting(
        language, first_name=first_name, intent=_greeting_intent, cache=_GREETING_CACHE
    )


def _seed_greeting_for(language: str) -> str:
    from .greetings import seed_greeting_for

    return seed_greeting_for(language, intent=_greeting_intent, cache=_GREETING_CACHE)


def _make_knowledge_context(session: Any, language: str) -> ToolContext:
    """Build a ToolContext for the knowledge profile, seeding greeting on first turn."""
    if not any(m.get("role") == "assistant" for m in session.history):
        seed = _seed_greeting_for(language)
        if seed:
            session.history.insert(0, {"role": "assistant", "content": seed})
    return attach_session_identity(
        ToolContext(
            customer_id=session.config.customer_id,
            call_id=session.config.call_id,
            _detected_language=language,
            account_store=session.account_store,
            profile_state=session.profile_state,
        ),
        session,
    )


def _knowledge_tools_spec() -> list[dict[str, Any]]:
    return tools_spec(profile=_PROFILE)


def _knowledge_run_tool(name: str, arguments: dict[str, Any], ctx: Any) -> str:
    return run_tool(name, arguments, ctx, profile=_PROFILE)


def register_profile() -> None:
    from .profiles import VoiceProfile, register_profile as _register_profile

    _register_profile(
        VoiceProfile(
            name=_PROFILE,
            system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
            tools_spec=_knowledge_tools_spec,
            tool_runner=_knowledge_run_tool,
            make_context=_make_knowledge_context,
            after_turn=None,
        )
    )


register_profile()
