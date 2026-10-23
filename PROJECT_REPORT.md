# PRECEDENT V1 — Project Report (as inspected)

**Report scope:** implementation visible in this workspace during this review, including the follow-up analysis of confidence scoring.  
**Status:** descriptive report, not a requirements-compliance certification.  
**Evidence convention:** file links below point to implementation or configuration inspected in the repository. Statements about behavior are limited to that code. Where this session did not inspect the complete implementation, that is called out explicitly.

## 1. Executive summary

PRECEDENT is a Python/FastAPI service for ingesting internal policy text and retrieving policy passages in response to user queries. Its codebase includes SQLModel persistence, document chunking, query decomposition and retrieval/filtering logic, confidence/refusal data structures, execution-budget helpers, error and validation helpers, and API trace retrieval. The package description calls it an “Internal Policy Assurance Engine” and identifies the goal as policy QA with auditability ([`pyproject.toml`](pyproject.toml), [`app/main.py`](app/main.py)).

The implementation is a V1 application, not evidence of a production-ready system. The inspected configuration defaults to mock LLM and embedding providers, retrieval is described by the endpoint as simple text matching, and traces are held in process memory ([`app/core/config.py`](app/core/config.py), [`app/api/v1/endpoints/retrieval_phase2.py`](app/api/v1/endpoints/retrieval_phase2.py), [`app/api/v1/endpoints/trace.py`](app/api/v1/endpoints/trace.py)). The README and `SRS_v1.txt` were empty when inspected; the SRS Word document’s content was not assessed in this report.

## 2. Technology and entry points

- **Runtime and framework:** Python 3.11+, FastAPI, Pydantic v2, SQLModel, and Uvicorn are declared in [`pyproject.toml`](pyproject.toml).
- **Application startup:** [`app/main.py`](app/main.py) creates the FastAPI application, initializes the database during lifespan startup, installs CORS middleware and a generic exception handler, registers v1 routers, and exposes `/` and `/health`.
- **Configuration:** [`app/core/config.py`](app/core/config.py) reads settings from environment variables and `.env`; it defines database, provider, retrieval, confidence, budget, chunking, and logging settings.
- **Database:** [`app/core/database.py`](app/core/database.py) creates a SQLModel engine, creates tables during initialization, and provides session dependencies/context management. SQLite gets `check_same_thread=False`.

## 3. Implemented components

### 3.1 Ingestion and chunking

[`app/api/v1/endpoints/ingest.py`](app/api/v1/endpoints/ingest.py) defines the ingestion router and helpers. The inspected code computes a SHA-256 content hash, looks up documents by hash, creates policy chunks carrying request metadata, and can mark active chunks as superseded. Chunk IDs are formed from the document ID and chunk index. The ingestion request schema in [`app/schemas.py`](app/schemas.py) includes policy ID/version, owner, authority level, jurisdiction, department, audience, status, effective dates, and optional `supersedes`.

[`app/core/chunking.py`](app/core/chunking.py) includes character-based chunking with overlap and a sentence-oriented helper. The character chunker rejects blank text and invalid size/overlap parameters; a short trailing fragment is merged into the previous chunk when possible. Sentence splitting uses a simple punctuation-based regular expression, with long sentences passed back through character chunking. This is a heuristic splitter, not a linguistic sentence parser.

### 3.2 Stored policy data

[`app/models.py`](app/models.py) defines two SQLModel tables:

- **Document:** document ID, name, unique content hash, ingestion timestamp.
- **PolicyChunk:** text and document linkage; policy ID/version/status; authority level, jurisdiction, department, audience; effective dates; supersession references; and owner.

Indexes are declared for commonly filtered metadata including policy/version, status/effective dates, and authority/jurisdiction/department. These model fields establish what metadata can be stored; they do not by themselves prove every field is used correctly throughout retrieval.

### 3.3 Retrieval and query processing

The router exported as `retrieval_router` comes from `retrieval_phase2.py` ([`app/api/v1/endpoints/__init__.py`](app/api/v1/endpoints/__init__.py)). Its `/retrieve` endpoint requires a bearer token, accepts query and `k`, plus optional jurisdiction/department/authority context, constructs an `ExecutionBudget`, calls `run_retrieval_pipeline`, and stores the returned trace in the trace module’s in-memory store ([`app/api/v1/endpoints/retrieval_phase2.py`](app/api/v1/endpoints/retrieval_phase2.py)). The endpoint docstring describes the intended sequence as decomposition, retrieval, authority/applicability filtering, conflict detection, evidence assessment, confidence calculation, and response with trace.

[`app/core/retrieval.py`](app/core/retrieval.py) contains the pipeline module. The inspected portion defines regex-based question decomposition patterns for question intents such as definition, process, timing, location, actor, reason, eligibility, maximum/minimum, approval, and reimbursement. Decomposition also splits on conjunctions and punctuation in some cases, with a fallback to the original question. The endpoint describes V1 retrieval as simple text matching. The previous review observed text-term/bigram matching and a relevance score based on overlap; this report does not claim semantic/vector retrieval.

### 3.4 Confidence, applicability, and refusals

The schemas define refusal categories and outcomes in [`app/schemas.py`](app/schemas.py), including no evidence, low confidence, conflict, inapplicable/stale evidence, insufficient authority, budget/provider/validation errors, and a generation low-confidence outcome.

[`app/core/confidence_scorer.py`](app/core/confidence_scorer.py) documents an intentional phase distinction: missing claim-verification data is neutral during retrieval because claim extraction occurs later. The inspected function `compute_authority_score` starts with a neutral score when flags are absent, then applies multiplicative penalties for conflict, lower/unknown authority level, jurisdiction or department mismatch, being outside the effective window, and supersession. It clamps the resulting score to the interval [0, 1]. The module also computes a claim-verification score from valid-claim ratio and confidence values; with no verification results it returns 1.0 and no penalty.

The earlier confidence walkthrough identified the following component weighting as the intended/currently reported calculation:

| Phase | Components | Weights reported in the earlier code review |
|---|---|---|
| Retrieval | Retrieval relevance, evidence sufficiency, authority score | 0.35, 0.35, 0.30 |
| Generation | Retrieval relevance, evidence sufficiency, claim-verification score, authority score | 0.25 each |

These weights and the overall combination function were not re-opened in full during preparation of this file because the required `rtk` executable was unavailable. Treat the table as a record of the previous review, not an independently re-verified code citation. Parameters declared in [`app/core/config.py`](app/core/config.py) include `confidence_threshold=0.5`, `min_evidence_sufficiency=0.3`, and `min_retrieval_relevance=0.3`; actual use should be confirmed against the full decision code before treating these as exact live decision thresholds.

### 3.5 Execution limits, timeout, validation, and errors

- [`app/core/budget.py`](app/core/budget.py) defines per-run accounting for LLM calls, input/output tokens, retries, elapsed wall time, and estimated cost. The settings defaults are 10 LLM calls, 50,000 tokens, 3 retries, 30,000 ms, and estimated cost 10.0.
- [`app/core/timeouter.py`](app/core/timeouter.py) supplies async timeout wrapping and sync/async budget checks, raising categorized infrastructure errors on timeout/budget failure.
- [`app/core/validation.py`](app/core/validation.py) validates JSON/dictionary output into Pydantic models and includes helpers for required fields, confidence ranges, enums, and non-empty strings.
- [`app/core/errors.py`](app/core/errors.py) defines a categorized error hierarchy with retryability, safe user-facing messages, diagnostic codes, and optional details. The inspected examples distinguish non-retryable authentication/model failures from retryable provider rate-limit/timeout failures.

These helpers exist in the codebase; their presence alone does not establish that every pipeline operation consistently invokes them.

### 3.6 Trace/audit access

[`app/api/v1/endpoints/trace.py`](app/api/v1/endpoints/trace.py) stores `TraceRecord` objects in a module-level dictionary and exposes authenticated `GET /{trace_id}`. The endpoint returns 404 when the ID is absent. Since storage is in memory, traces are process-local and are lost on restart; multi-worker consistency is not provided by this implementation.

## 4. End-to-end workflow represented by the code

### Ingestion path

1. Client submits text and policy metadata to the ingestion route.
2. Request validation checks required/nonblank text and applies schema defaults/enums.
3. The ingestion helper hashes document text and can identify a duplicate by hash.
4. Text is divided into chunks using the configured chunking utility.
5. Document and chunk records are persisted; chunk records carry authority and validity metadata.
6. Supersession helpers can mark existing active chunks as superseded when a new version is ingested.

The precise transaction/rollback behavior and every branch of the endpoint were not fully re-audited for this report.

### Retrieval path

1. Client submits a query plus optional applicability filters and bearer token.
2. The endpoint validates the token and assembles query context.
3. It creates an execution budget from application settings.
4. `run_retrieval_pipeline` receives the database session, query, `k`, context, and budget.
5. The pipeline module provides decomposition, text retrieval, metadata applicability/filtering, conflict/evidence/confidence/refusal-related logic.
6. The endpoint stores the returned trace in memory and returns retrieval information.

This sequence follows endpoint documentation and code structure. A full line-by-line validation of each pipeline branch and the separate `/query` response path was not completed in this report.

## 5. API surface established by inspected sources

| Route | Purpose/evidence |
|---|---|
| `GET /` | Service/version and endpoint overview, defined in [`app/main.py`](app/main.py). |
| `GET /health` | Static healthy status response, defined in [`app/main.py`](app/main.py). |
| `POST /api/v1/ingest` | Policy document ingestion, router defined in [`app/api/v1/endpoints/ingest.py`](app/api/v1/endpoints/ingest.py). |
| `POST /api/v1/retrieval/retrieve` | Retrieval endpoint, defined in [`app/api/v1/endpoints/retrieval_phase2.py`](app/api/v1/endpoints/retrieval_phase2.py). |
| `GET /api/v1/trace/{trace_id}` | Authenticated trace lookup, defined in [`app/api/v1/endpoints/trace.py`](app/api/v1/endpoints/trace.py). |

The earlier source inspection reported an additional `/api/v1/retrieval/query` endpoint. It is not documented as an independently verified route here because the full endpoint source was not available for re-check during this report pass.

## 6. Configuration and security observations

The checked-in `.env` contents observed earlier in the session included a development API token and SQLite database URL. [`app/core/config.py`](app/core/config.py) also supplies a development-token default. The trace and retrieval endpoints compare a bearer token directly to this configured value. This documents the current authentication mechanism; no role-based authorization or user identity model was observed in the inspected files.

[`app/main.py`](app/main.py) configures CORS with all origins, methods, and headers, while also allowing credentials. The code comment says to configure this appropriately for production. Do not treat the development token or wildcard CORS configuration as production hardening.

Provider settings are labeled as swappable and include mock, Groq, OpenAI, and Anthropic values, but declared dependencies and inspected source do not establish that each provider adapter is implemented or operational. Configuration choices must not be confused with completed integrations.

## 7. Tests and verification status

The repository contains `tests/test_suite.py`, whose header describes a synthetic controlled corpus, mock/fake-based tests, and coverage of ten named failure modes. This report did not run tests. Older compiled test cache files were present, but cache artifacts are not evidence that tests pass against the current source. The README and `SRS_v1.txt` were empty in the earlier inspection, so they do not provide usable project documentation in their current inspected state.

## 8. Known boundaries and items requiring a fresh source verification

The following are limitations of this report’s evidence, rather than claims that the functionality is absent:

- The terminal helper required by the supplied RTK instruction was unavailable, preventing a fresh command-line scan of all current files.
- The complete confidence combination/refusal code was not reopened line by line; the weighting table is explicitly attributed to the prior review.
- The complete ingestion endpoint, retrieval pipeline, and `/query` implementation were not exhaustively checked in this pass.
- No test suite, static analysis, or live API request was run for this report.
- Requirements compliance cannot be certified from the empty README/SRS text and partial code review.

## 9. Source index

- Application wiring: [`app/main.py`](app/main.py)
- API schemas: [`app/schemas.py`](app/schemas.py)
- Database models: [`app/models.py`](app/models.py)
- Settings: [`app/core/config.py`](app/core/config.py)
- Database lifecycle: [`app/core/database.py`](app/core/database.py)
- Chunking: [`app/core/chunking.py`](app/core/chunking.py)
- Ingestion endpoint: [`app/api/v1/endpoints/ingest.py`](app/api/v1/endpoints/ingest.py)
- Retrieval endpoint: [`app/api/v1/endpoints/retrieval_phase2.py`](app/api/v1/endpoints/retrieval_phase2.py)
- Retrieval logic: [`app/core/retrieval.py`](app/core/retrieval.py)
- Confidence logic: [`app/core/confidence_scorer.py`](app/core/confidence_scorer.py)
- Budget tracking: [`app/core/budget.py`](app/core/budget.py)
- Timeout checks: [`app/core/timeouter.py`](app/core/timeouter.py)
- Structured validation: [`app/core/validation.py`](app/core/validation.py)
- Error classes: [`app/core/errors.py`](app/core/errors.py)
- Trace endpoint/store: [`app/api/v1/endpoints/trace.py`](app/api/v1/endpoints/trace.py)
- Test-suite description and corpus: [`tests/test_suite.py`](tests/test_suite.py)
- Dependency and tool configuration: [`pyproject.toml`](pyproject.toml)
