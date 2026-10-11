# Parallax Labs Internship — Codex Project Instructions

## Purpose and current status

This is Scheeza Ahmad's Parallax Labs internship project. Scheeza and Codex will build it together, step by step. Use this file as project context and working instructions when it is placed in the repository root.

Source of requirements: three internship portal screenshots supplied by Scheeza: Week 1 tasks, overall project brief, and tech-stack/submission rules. This file transcribes and organizes those requirements. Later official roadmap details may refine them; do not invent missing weekly requirements or deadlines.

Current handoff status: planning/context file prepared. No project repository, dependency installation, GPU check, dataset download, implementation, tests, GitHub push, or deployment has been verified in this handoff. Inspect the actual workspace before assuming anything exists.

Immediate objective: implement and verify WEEK 1. Keep the architecture extensible for the complete project, but do not implement all future phases before Week 1 is complete.

## Overall project: source-grounded RAG AI assistant

Build an end-to-end Retrieval-Augmented Generation system with the following capabilities from the project brief:

1. Ingest and clean a real-world text dataset of 5,000+ documents from sources such as ArXiv, Wikipedia, or Reddit dumps.
2. Chunk documents and embed them into a vector database; the roadmap names ChromaDB.
3. Retrieve semantically relevant context for complex user queries.
4. Generate answers through an LLM API, grounded strictly in retrieved sources. OpenRouter and Groq are examples in the brief, not confirmed provider selections.
5. Provide corpus-level NLP analysis, including topic modeling and sentiment analysis.
6. Serve the system through a production-ready API; the roadmap names FastAPI, with comprehensive documentation.
7. Address the learning outcomes: data cleaning and exploratory data analysis (EDA), chunking and embedding strategies, vector database management, prompt engineering, LLM integration, RAG evaluation (precision, recall, hallucination detection), NLP analysis, and API performance benchmarking.

These are overall goals, not all Week 1 deliverables. Detailed requirements and dates for later weeks have not been supplied. No specific frontend design, model, provider, evaluation threshold, deployment platform, or final project name has been confirmed.

## Mandatory internship submission rules

- Maintain ONE GitHub repository for the entire internship. Do not create a new repository for each week.
- Push each week's new commits to that same repository when submitting.
- Update README.md for EVERY weekly submission with that week's progress, dependencies, and instructions to run the work.
- Submit the same repository link each week.
- The portal explicitly says submissions without an updated README will be rejected.
- AGENTS.md is Codex context; it does not replace README.md.
- Repository URL and remote configuration are currently unknown. Inspect existing configuration before creating or changing anything. Never invent a GitHub URL or claim a push happened when it did not.

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
- Dataset source and target language are not yet selected. English is a possible default, to be explicitly documented. Prefer a corpus whose content supports meaningful retrieval and explain limitations of sentiment analysis on neutral academic text.
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

- [ ] One existing or initialized Git repository identified for the whole internship.
- [ ] Isolated Python environment and reproducible pinned requirements prepared.
- [ ] .gitignore excludes datasets, environment files, credentials, and caches.
- [ ] scripts/verify_env.py runs; required imports pass and CUDA/GPU requirement is satisfied, or its unresolved status is explicitly recorded.
- [ ] 5,000+ real source documents acquired with provenance; raw files retained outside Git.
- [ ] All four required cleaning operations implemented in modular functions.
- [ ] Meaningful unit tests run and pass.
- [ ] Full preprocessing run completed and clean output validated with actual counts.
- [ ] README updated with runnable instructions, source details, results, and limitations.
- [ ] Same-repository commits pushed and repository link ready for weekly submission when authorized/access is available.

An unresolved mandatory check means the task remains partially complete; a checklist entry documenting a blocker is not a passing requirement.

## First action when implementation starts

Read this file, inspect the workspace and environment, then summarize the exact Week 1 plan in Roman Urdu. Start with repository/environment setup and GPU/import verification. Continue through dataset acquisition, modular preprocessing, tests, validation, and README. Do not invent later-week milestones or build unrequested future features before finishing this foundation.
