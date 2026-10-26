# Parallax Labs Internship — Codex Project Instructions

## Active Week 4 scope - 2026-10-24

User supplied Week 4 (Oct 26-Nov 1) screenshot and explicitly requested local work
only: DO NOT PUSH to GitHub until the user asks. Required: topic modeling, sentiment
or NER with a small manually labelled evaluation, manual topic validation, edge
cases, vector metadata filters and documented effectiveness. Implement LDA plus
spaCy NER; sentiment is not required when choosing the screenshot's NER alternative.
Keep the Week 2/3 collection intact and enrich a separate collection by copying
vectors. Draft annotation provenance must stay explicit: user agreed to review
labels but has not yet confirmed corrections/completion. Never call Codex-created
labels human-reviewed or claim the human review requirement passed prematurely.
Frontend, FastAPI and deployment remain out of scope; CUDA remains unresolved.

## Purpose and current status

This is Scheeza Ahmad's Parallax Labs internship project. Scheeza and Codex will build it together, step by step. Use this file as project context and working instructions when it is placed in the repository root.

Source of requirements: three internship portal screenshots supplied by Scheeza: Week 1 tasks, overall project brief, and tech-stack/submission rules. This file transcribes and organizes those requirements. Later official roadmap details may refine them; do not invent missing weekly requirements or deadlines.

Current status (2026-10-15): existing Git repository and Python 3.11.9 venv inspected. Week 1 cleaned JSONL contains 6,954 unique, nonempty records. Required imports and pip check pass; CUDA remains unavailable. Preserve Week 1 code and data. See README and reports for executed retrieval results.

Immediate objective: implement the Week 3 screenshot supplied on 2026-10-18: LLM API integration, system prompts and context injection, robust API errors, hallucination checks, out-of-domain handling, and end-to-end latency logs. The user selected Groq. Preserve Week 1/2 code and the full vector database. Frontend and deployment remain out of scope. CUDA remains unresolved.

## Overall project: source-grounded RAG AI assistant

Build an end-to-end Retrieval-Augmented Generation system with the following capabilities from the project brief:

1. Ingest and clean a real-world text dataset of 5,000+ documents from sources such as ArXiv, Wikipedia, or Reddit dumps.
2. Chunk documents and embed them into a vector database; the roadmap names ChromaDB.
3. Retrieve semantically relevant context for complex user queries.
4. Generate answers through an LLM API, grounded strictly in retrieved sources. OpenRouter and Groq are examples in the brief, not confirmed provider selections.
5. Provide corpus-level NLP analysis, including topic modeling and sentiment analysis.
6. Serve the system through a production-ready API; the roadmap names FastAPI, with comprehensive documentation.
7. Address the learning outcomes: data cleaning and exploratory data analysis (EDA), chunking and embedding strategies, vector database management, prompt engineering, LLM integration, RAG evaluation (precision, recall, hallucination detection), NLP analysis, and API performance benchmarking.

These are overall goals, not all Week 1 deliverables. Retrieval-week requirements were supplied by the user on 2026-10-15; later generation/API requirements and weekly dates remain unspecified. The retrieval baseline now uses pinned all-MiniLM-L6-v2. No frontend design, LLM provider, evaluation threshold, deployment platform or final project name has been confirmed.

## Mandatory internship submission rules

- Maintain ONE GitHub repository for the entire internship. Do not create a new repository for each week.
- Push each week's new commits to that same repository when submitting.
- Update README.md for EVERY weekly submission with that week's progress, dependencies, and instructions to run the work.
- Submit the same repository link each week.
- The portal explicitly says submissions without an updated README will be rejected.
- AGENTS.md is Codex context; it does not replace README.md.
- Verified origin: https://github.com/Scheeza-Ahmad/Parallax_Labs.git. Reuse this repository. Never claim a push happened without verifying it.

## Tech-stack policy

The portal permits alternative stacks: evaluation focuses on the final product. However:

1. All functional and visual requirements in weekly deliverables must still be met.
2. Interns must independently troubleshoot custom-stack issues; mentors may not provide framework-specific debugging assistance.
3. The chosen stack must support proper cloud deployment.

Working default for this project: follow the roadmap's Python-oriented stack, using pandas, sentence-transformers, ChromaDB, spaCy, and later FastAPI. This is a practical proposed default, not a claim that alternatives are prohibited. Verify compatible versions from official documentation when implementing; do not invent or blindly pin versions in advance.

## Week 1: Environment, Data Ingestion & Preprocessing

The screenshot displays Oct 5–Oct 11. Confirm any changed deadline through Scheeza or the portal; do not infer an extension or submission status.

### Official tasks

1. Initialize a clean Git repository with a virtual environment (venv or conda), a .gitignore that ignores data files and .env, and a locked requirements.txt.
2. Write AND run scripts/verify_env.py to assert CUDA/GPU availability and test basic imports of pandas, sentence-transformers, ChromaDB, and spaCy.
3. Acquire a real-world text dataset of 5,000+ documents from sources such as ArXiv, Wikipedia, or Reddit dumps. Save raw data in data/raw/ and keep it Git-ignored.
4. Implement modular cleaning functions in src/preprocessing.py covering HTML stripping, Unicode normalization, whitespace cleanup, and language filtering.
5. Write unit tests in tests/test_preprocessing.py. Output a validated clean corpus to data/processed/clean_corpus.parquet OR data/processed/clean_corpus.jsonl.

The weekly deliverable is a verified environment with test scripts, a modular text-cleaning pipeline with unit tests, and a validated clean dataset.

All paths above are relative to the project root, not operating-system root directories.

### Implementation guidance — proposed approach

These details make the work reproducible; they are implementation choices rather than extra portal requirements:

- Inspect the operating system, Python version, Git state, existing files, available disk space, and GPU access before setup. Use commands appropriate to the user's actual shell.
- Create an isolated environment. Record the Python version and pin the exact installed dependency versions after establishing a working installation.
- Use a small real sample for development, then run the pipeline on the full dataset. Sample-only success does not satisfy the 5,000+ document requirement.
- A document means a source record, such as an article or paper abstract. Do not count words or artificially split records into chunks to reach 5,000 documents.
- Choose a source with accessible, permitted downloads and useful text. Record source URL, dataset revision/snapshot if available, acquisition date, license/attribution, and selection method. Preserve provenance for later RAG citations.
- Selected source: Simple English Wikipedia, immutable 20231101.simple snapshot; target language: English. Preserve article IDs/URLs and document the limits of sentiment analysis on neutral encyclopedic text.
- Preserve raw data. Keep cleaned output separate and avoid overwriting raw files.
- Suggested record fields: doc_id, title when available, text, source, source_url, and language. Preserve original source IDs where available; do not fabricate missing metadata.
- Use separate cleaning functions plus a composable clean_text/pipeline function. Preserve useful punctuation, case, and paragraph boundaries unless there is a documented reason to change them. Do not add stemming, stop-word removal, or aggressive lowercasing by default.
- Explicitly choose and document the Unicode normalization form and language-detection method. Handle empty, null, malformed, and too-short inputs predictably; report uncertain language cases.
- Ensure deterministic selection and processing where practical. Record any random seed and transformation configuration.
- Log input count, accepted count, rejected count and reasons, empty-text count, duplicate count, and a basic text-length summary. Explain any reduction in document count.
- The screenshot states 5,000+ documents at acquisition. Aim to retain at least 5,000 usable unique documents after filtering too; acquire extra records if needed. Do not silently present fewer clean records as 5,000.
- Select either JSONL or Parquet and document the choice. Validate that the final file can be read back and has the expected schema and nonempty text.
- Add a reproducible ingestion script and a pipeline entry point if useful. Keep paths configurable and relative to the repository.

### GPU requirement: report reality

- Verify actual CUDA availability using an appropriate runtime and, when available, a small GPU operation; do not equate importing a package with a working GPU.
- Report detected device/runtime information and import failures clearly. Return a failing exit status when a required check fails.
- If CUDA is unavailable, mark that official requirement as blocked and explain the finding to Scheeza. Other CPU-capable Week 1 tasks can continue.
- A CPU fallback may help development, but it must not silently bypass or falsely satisfy the screenshot's GPU assertion. Obtain mentor clarification or use an approved GPU environment before claiming full compliance.

### Unit tests and validation

Use meaningful tests for HTML removal while retaining readable content, Unicode normalization, whitespace normalization, language filtering, and empty/missing input behavior. Cover malformed HTML and short/uncertain language input where relevant. Tests should be repeatable and should not require a live dataset download or paid API calls.

Run both unit tests and an actual pipeline execution. Distinguish passing tests, environment checks, data validation, and any blocked steps in the README. Never invent outputs, dataset counts, GPU availability, or passing test results.

## Suggested repository layout

This is a proposed structure; adapt to existing code without discarding user work.

- AGENTS.md — these project instructions.
- README.md — setup, usage, results, limitations, and weekly progress.
- .gitignore — at least virtual environment, .env/secrets, datasets, caches, and generated local artifacts.
- .env.example — placeholder configuration only, if environment variables are needed.
- requirements.txt — pinned, verified dependencies.
- scripts/verify_env.py — environment and CUDA checks.
- scripts/ingest_data.py — reproducible data acquisition (proposed).
- scripts/run_preprocessing.py — cleaning entry point (proposed).
- src/__init__.py — package initialization if needed.
- src/preprocessing.py — reusable cleaning functions.
- tests/test_preprocessing.py — meaningful unit tests.
- data/raw/ — original downloaded records; ignored by Git.
- data/processed/clean_corpus.jsonl OR clean_corpus.parquet — validated output; ignored by Git.
- reports/ — small, non-sensitive validation summaries if useful.

Keep large datasets, future vector stores, model caches, credentials, and API keys out of Git. Provide reproducible acquisition instructions and clarify any separate dataset-submission route instead of committing ignored data.

## README content for Week 1

Include project overview and scope; Week 1 status; Python version; environment setup; pinned dependency installation; GPU prerequisites; dataset source and license; acquisition steps; cleaning choices and language filter; exact commands for environment checks, ingestion, preprocessing, and tests; actual validation results; output paths; known limitations or blockers; and next steps.

Maintain a weekly progress section as later work is added. Describe only implemented and verified features as complete.

## Collaboration with Scheeza

- Explain actions and concepts in simple Roman Urdu. Keep code, identifiers, technical documentation, and the submission README in clear English.
- Work in small, understandable steps. Before a substantial step, briefly state what it will accomplish; afterward explain changed files, how to run them, and what the results mean.
- Help Scheeza understand the implementation well enough to explain it in an internship review.
- Inspect existing work before editing and preserve unrelated changes. Do not overwrite datasets, delete files, reset Git history, change an established stack, or incur paid-service charges without appropriate user authorization.
- Make routine reversible implementation decisions autonomously, state assumptions, and keep progress moving. Ask only for information that genuinely affects the work, such as unavailable account access or a required dataset/domain decision.
- Use tools that are available and suitable. Do not assume a particular editor, GPU provider, API subscription, account connection, or secret exists.
- Never put real secrets in code, prompts, README examples, logs, or Git. Request configuration through environment variables when needed.
- Update README progress as work is verified. End each work session with completed steps, verification evidence, remaining blockers, and the next concrete step.

## Definition of Week 1 completion

- [x] One existing or initialized Git repository identified for the whole internship.
- [x] Isolated Python environment and reproducible pinned requirements prepared.
- [x] .gitignore excludes datasets, environment files, credentials, and caches.
- [ ] CUDA/GPU requirement: verifier runs and imports pass, but CUDA is unavailable; this is still a failing mandatory requirement.
- [x] 5,000+ real source documents acquired with provenance; raw files retained outside Git.
- [x] All four required cleaning operations implemented in modular functions.
- [x] Meaningful unit tests run and pass.
- [x] Full preprocessing run completed and clean output validated with actual counts.
- [x] README updated with runnable instructions, source details, results, and limitations.
- [x] Same-repository Week 2 implementation commit pushed; public repository and updated README verified on 2026-10-18.

An unresolved mandatory check means the task remains partially complete; a checklist entry documenting a blocker is not a passing requirement.

## First action when implementation starts

Read this file and README, inspect existing code, environment and actual cleaned data, then explain the current requested scope in Roman Urdu. Reuse the established environment and corpus. Verify sample ingestion before full indexing, rerun tests, check persistence in a new process, benchmark real queries, and update README with evidence. Keep the mandatory CUDA blocker explicit.


## Retrieval-week verified progress — 2026-10-16

- Implemented `src/chunking.py`, `src/embeddings.py`, `src/vector_store.py`, and
  `scripts/retrieval.py`; commands are ingest, search and benchmark.
- Chunks use 220 content tokens with 32-token overlap; source text/metadata,
  character offsets and stable SHA256 IDs are retained. No silent truncation.
- Model: sentence-transformers/all-MiniLM-L6-v2, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions, normalized vectors,
  CPU, batch 32, four threads. Chroma uses persistent cosine-distance storage.
- All 51 tests passed on 2026-10-15; all 124 installed dependency pins matched.
- Real 25-document sample: 191 chunks; external-process search and resume passed.
- Full corpus now complete: 6,954 documents, 31,984 chunks, all expected chunk
  IDs/text/metadata read back successfully (`reports/index_full_resume.json`).
- An interrupted initial full run retained 28,672 chunks. Resume safely skipped
  them and encoded the remaining 3,312 in 201.26142 seconds (16.456 chunks/s).
  Initial interrupted-run total encoding time was not saved; do not infer it.
- CUDA remains blocked. LLM generation is now authorized for Week 3; frontend and deployment remain out of scope.
- Do not claim a new GitHub commit/push or public visibility verification without
  actually checking it. See README for final benchmark evidence and submission steps.

- Full-database benchmark completed in a new process: 8 queries, 3 repeats,
  24 measurements; mean total 37.45395 ms, median 30.99065 ms, p95 79.2688 ms.
  Evidence: `reports/retrieval_benchmark.json`. No precision/recall claimed.
- Full identical rerun passed: 31,984 skipped, 0 encoded, count unchanged, all
  chunks read back (`reports/index_rerun.json`). Standalone search after restart
  returned five results with Alan Turing ranked first (`reports/search_restart.json`).
- Publication completed on 2026-10-18. Remaining: user portal submission and
  resolution or mentor clarification of the earlier CUDA blocker.


## Submission verification - 2026-10-18

- Reran 51 tests: all passed in 42.88 seconds; pip check and required imports pass.
- Full ingestion/read-back rerun: 31,984 chunks, all skipped, no new vectors.
- Real query "What is an atom made of?" returned Atom/Atom/Proton source chunks.
- Fresh benchmark: 24 measured searches; mean 32.853 ms, median 26.692 ms,
  p95 66.413 ms. Evidence files use the `reports/submission_` prefix.
- GitHub API verified that the existing repository is public; remote main matched
  the Week 1 base before publication. CUDA remains unavailable (verifier exit 1).

- Published implementation commit `0e385766b4b95e19fd4e4bb07a68fd52bc47fdf1`;
  GitHub main and the updated README were independently verified through the API.
  Submit https://github.com/Scheeza-Ahmad/Parallax_Labs and disclose the CUDA blocker.


## Week 3 requirements and implementation - 2026-10-18

The supplied portal screenshot is titled "LLM Integration & Prompt Engineering"
and displays Oct 19-Oct 25; do not infer submission status or changed deadlines.
Required: LLM API over retrieved chunks, system prompt/context injection, handling
rate limits/timeouts/token limits, hallucination checks, off-topic abstention,
and measured retrieval-plus-generation latency. User selected Groq.

- Added `src/llm.py`, `src/rag.py`, `scripts/rag.py`, `.env.example`,
  `rag_queries.json`, and HTTP/grounding tests. Existing dependency lock unchanged;
  httpx and python-dotenv were already installed and pinned.
- Current tests: 100 passed; HTTP/LLM fixtures are offline tests, not live evidence.
- Groq documentation checked for active model discovery, JSON mode, context limits
  and rate limits. Default candidate is `openai/gpt-oss-20b`; each live run verifies
  the ID and context window through authenticated /models before sending prompts.
- Generated answers require exact quote/source-ID validation plus a separate LLM
  relevance/entailment check. These are fallible safeguards, never a guarantee of
  truth or zero hallucinations. No quality percentages without labelled evaluation.
- No secrets in logs/Git/chat. Load local `.env` without printing it. Do not make
  paid calls without authorization. User has a Groq account/key, but local key and
  Free-plan confirmation were still pending when this implementation was prepared.
- Do not claim Week 3 is fully verified until real authenticated generation,
  citations, abstention and end-to-end latency have been executed and inspected.

## Week 3 live verification - 2026-10-24

- User confirmed configured key and Free plan. Moved credential from example to
  ignored `.env`; restored `.env.example` to a blank key placeholder.
- 100 tests passed in 9.03 seconds using workspace-local test temporary files;
  pip check passed. Evidence: `reports/week3_tests_20261024.xml`.
- Real five-query benchmark passed: two cited answers with exact quotations and
  LLM verification, two off-topic abstentions and one live-price abstention.
- Active model `openai/gpt-oss-20b` verified through authenticated discovery,
  context window 131072. Actual answer end-to-end times: 2452.854 and 1891.465 ms;
  successful-answer mean 2172.160 ms. No quality percentage claimed.
- Evidence: `reports/week3_live_20261024.json`; README updated. Week 3 implementation
  commit `a6903d9` was pushed to origin/main on 2026-10-24.
- CUDA remains unresolved; frontend and deployment remain out of scope.
