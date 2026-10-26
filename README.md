# Parallax Labs — Source-grounded RAG Assistant

Scheeza Ahmad's internship project. Week 1 implements environment verification,
real document acquisition and conservative preprocessing. The retrieval week adds
token-window chunking, sentence-transformer embeddings, persistent ChromaDB and
latency benchmarking. Week 3 adds live-verified Groq generation with evidence
checks. Week 4 adds LDA topics, named entities and filtered retrieval locally;
human annotation review is pending. FastAPI remains future scope.

**Retrieval baseline (2026-10-16):** the entire 6,954-document clean corpus is
indexed as 31,984 chunks in local ChromaDB. Retrieval benchmarking on the reopened
full database completed: 24 searches, mean 37.45 ms, median 30.99 ms, p95 79.27 ms.
51 tests pass. The earlier mandatory CUDA requirement remains unresolved.
See the retrieval-week sections below for commands, evidence and timing limitations.

Latest Week 3/4 results and limitations are documented in their dated sections
below. Week 4 changes must not be pushed until the user requests publication.

Windows persistence note: the rebuilt `wikipedia_nlp_v2` collection reopened in a
fresh process with 31,984 chunks. Chroma's current Rust binding then produced a
native access violation during a large filtered semantic query, while the source
collection and all small fixture tests remained usable. The failure is in the
local Chroma/native index path, not caught as a Python exception. The collection
is retained for diagnosis; Week 4 must not claim a successful full-size filtered
semantic query until this environment issue is resolved. No collection was
deleted and no GitHub push was made.

## Week 1 progress — 2026-10-11

Initial inspection: only AGENTS.md existed. Initialized one local Git repository
and a Python 3.11.9 virtual environment on Windows. The local branch is `main`;
`origin` is `https://github.com/Scheeza-Ahmad/Parallax_Labs.git`.
The remote was reachable and advertised no existing refs during inspection.
Dependency installation completed; `pip check` passed with no broken requirements.
All 124 installed packages (including bootstrap tools) are pinned in `requirements.txt`.
This lock was resolved on Windows/Python 3.11.9; other platforms, especially GPU
environments, need their own compatibility verification before replacing it.
Acquisition succeeded: 7,000 whole source articles selected from a retained
241,787-article snapshot (156,885,218 bytes). The first acquisition used the
existing system Python/pyarrow while venv dependencies were downloading.
The snapshot SHA256 is recorded in `reports/acquisition.json`.
The existing system PyTorch reports CUDA unavailable (CPU runtime, zero devices).
The mandatory GPU requirement is unresolved; a CPU run is not GPU compliance.

| Actual check | Result |
| --- | --- |
| Unit and pipeline tests | 16 passed in 3.71 seconds |
| Development sample | 100 input, 100 accepted and read back |
| Full pipeline | 7,000 input, 6,954 accepted and read back |
| Rejections | 43 uncertain/short language, 3 detected non-English |
| Empty / duplicate / malformed / invalid-schema records | 0 / 0 / 0 / 0 |
| Clean text lengths (characters) | min 68, median 1,552.5, mean 3,582.97, max 51,461 |
| Dependency consistency | `pip check`: no broken requirements |
| Required imports | pandas 3.0.6, sentence-transformers 6.1.0, ChromaDB 1.5.9, spaCy 3.8.16: all passed |
| CUDA | **BLOCKED**: PyTorch 2.14.1+cpu, runtime null, availability false, zero CUDA devices; verifier exit 1 |
| GitHub | Submission repository: `Scheeza-Ahmad/Parallax_Labs`, branch `main` |

Evidence: `reports/acquisition.json`, `reports/tests.json`, `reports/sample.json`,
`reports/preprocessing.json` and `reports/environment.json`. Mandatory CUDA remains
unresolved, so Week 1 is **partially complete**, not fully compliant.
`nvidia-smi` was not found. Windows hardware inventory access was denied in the
sandbox; an additional elevated inventory request was not executed because the
automatic approval review service hit its usage limit. This was a review failure,
not a finding that the action was unsafe. Physical GPU hardware remains unverified;
the installed runtime's inability to run CUDA is verified.

### Submission scope

All CPU-capable Week 1 deliverables are implemented and verified. At the project
owner's request, CUDA remediation is deferred for this submission. This is not a
mentor waiver: the official GPU requirement remains unmet, and the verifier
continues to fail honestly when CUDA is unavailable.

The repository contains source code, tests, pinned dependencies, documentation and
small validation reports. Raw and processed datasets remain local and Git-ignored;
reviewers can reproduce them with the commands below. If the portal separately
requires the dataset file, provide `data/processed/clean_corpus.jsonl` through its
approved upload route; no separate dataset-submission route has been supplied.

### Demonstration rerun — 2026-10-11

Reran all 16 tests (passed in 2.75 seconds), `pip check` (passed), and the environment
verifier (all imports passed; CUDA still blocked, exit 1). Reran the complete
7,000-record pipeline into `data/processed/demo-20261011-211759.jsonl` while
preserving the original corpus. Again 6,954 records passed validation, with the
same 46 language rejections. SHA256 comparison confirmed the demo output is
byte-for-byte identical to the original clean corpus.
Report: `reports/demo-20261011-211759.json`.

To demonstrate again without overwriting an existing output:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe scripts/verify_env.py
$demoTag = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv\Scripts\python.exe scripts/run_preprocessing.py --output "data/processed/demo-$demoTag.jsonl" --report "reports/demo-$demoTag.json"
```

## Setup and commands (PowerShell, project root)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_env.py
.\.venv\Scripts\python.exe scripts/ingest_data.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_preprocessing.py
```

Use the venv executable directly; activation is optional. `requirements.in` lists
direct dependencies; `requirements.txt` locks the installed versions.
The environment verifier writes `reports/environment.json` and returns exit code 1
when any import or CUDA check fails. It tests a tensor operation on CUDA when
available. CUDA needs compatible NVIDIA hardware, driver and a CUDA-enabled PyTorch
installation. Obtain mentor clarification or use an approved GPU environment before
claiming the GPU requirement is satisfied. Do not use CPU results as a replacement.

## Dataset and provenance

Source: [Wikimedia Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia),
`20231101.simple` (Simple English Wikipedia, snapshot November 1, 2023).
Pinned dataset revision: `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
The dataset card lists CC-BY-SA-3.0 and GFDL. Attribution belongs to Wikipedia
contributors; original article IDs, titles and URLs are retained. Consult the
dataset card and article histories for attribution and redistribution obligations.
The clean corpus is a transformed derivative; no ownership of source text is claimed.

The ingestion script retains the source Parquet snapshot in `data/raw/` and selects
the first 7,000 whole articles in stable source row order into
`data/raw/wikipedia-7000.jsonl`. No articles are split to inflate counts. The source
already contains extracted article text, not the original Wikimedia XML dump.
Acquisition time, SHA256, revision, source URL and counts go into a raw manifest
and `reports/acquisition.json`. Selection is deterministic but is not a random or
representative sample of Wikipedia. The complete Parquet download contains more
articles than the selected pipeline input.

## Cleaning and output

`src/preprocessing.py` provides separate HTML stripping, NFC Unicode normalization,
whitespace cleanup and language filtering functions. Script/style contents are
removed. Case, punctuation and paragraph boundaries are preserved; stemming,
stop-word removal and lowercasing are intentionally absent.

Language detection uses `langdetect`, seed 0, the first 10,000 characters, at least
50 alphabetic characters and confidence >= 0.90. Only English is accepted. Short
or uncertain cases are reported separately; detector confidence is heuristic and
does not guarantee correctness. Null and non-string text becomes empty text.
Malformed JSON and missing source metadata receive distinct rejection counts.
Duplicates share a source ID or identical cleaned-text SHA256.

Output uses UTF-8 JSONL for easy inspection and streaming:
`data/processed/clean_corpus.jsonl`. Each record has `doc_id`, `title`, `text`,
`source`, `source_url`, `language`, and `language_confidence`.
The pipeline reads back every record and checks required fields, nonempty text,
unique IDs/text and a minimum of 5,000 accepted records before publishing output.
`reports/preprocessing.json` records counts, rejection reasons, configuration and
character-length statistics. A below-minimum run exits nonzero and retains a
`.partial` file for diagnosis, without publishing a validated corpus.

Scripts refuse to overwrite existing datasets. To experiment, use new paths:

```powershell
.\.venv\Scripts\python.exe scripts/ingest_data.py --limit 100 --raw-dir data/sample
.\.venv\Scripts\python.exe scripts/run_preprocessing.py --input data/sample/wikipedia-100.jsonl --output data/sample/clean.jsonl --report reports/sample.json --minimum 1
```

A sample run does not satisfy the 5,000-document requirement. If a download fails,
its `.download` file is retained; inspect it and choose a fresh `--raw-dir` to retry
without overwriting anything. Data, virtual environments, caches and secrets are
Git-ignored. Commit code and small reports; reproduce data using the ingestion
script rather than committing it.

## Limitations and next steps

Wikipedia is mostly neutral encyclopedic writing, so later sentiment analysis may
have limited usefulness and must be interpreted cautiously. This historical,
ordered English sample cannot support claims about all Wikipedia or current facts.
Week 1 tested base imports without downloading language models. The retrieval
week now uses the pinned MiniLM model described below. No spaCy language model
is needed for the implemented cleaning or retrieval pipeline.

Deferred next step: verify an approved CUDA-capable environment (or obtain mentor
clarification), install a compatible CUDA-enabled PyTorch build there, and rerun
`scripts/verify_env.py` until the real tensor operation passes. CPU preprocessing and
its tests are already verified. Retrieval-week work is described below; the user
authorized proceeding while the earlier CUDA requirement remains unresolved.

One repository is used for the entire internship:
[Scheeza-Ahmad/Parallax_Labs](https://github.com/Scheeza-Ahmad/Parallax_Labs).
Submit this same repository link every week with that week's committed work and
an updated README. For this submission, disclose the outstanding CUDA check.

Installation references: [Sentence Transformers](https://www.sbert.net/docs/installation.html),
[spaCy](https://spacy.io/usage.html), [Chroma](https://docs.trychroma.com/docs/overview/getting-started),
[pandas](https://pandas.pydata.org/docs/getting_started/install.html).


## Retrieval week — 2026-10-15

Scope: chunk the existing cleaned corpus, embed it, populate persistent ChromaDB,
search it and measure retrieval latency. No answer-generation API, frontend or
cloud deployment is included. The same repository and existing venv are reused.

### Architecture and chunking

`data/processed/clean_corpus.jsonl` → `src/chunking.py` →
`src/embeddings.py` → `src/vector_store.py` → ranked source chunks.
`scripts/retrieval.py` exposes ingestion, search and benchmarking commands.

The strategy is overlapping model-token windows, not recursive splitting.
Defaults: **220 content tokens, 32-token overlap**, with the pinned model's fast
WordPiece tokenizer. Special tokens are excluded from those settings; the model
limit is 256 including special tokens. Every source slice is retokenized and
checked before encoding; oversized queries fail instead of silently truncating.
The window shrinks if retokenization would exceed its budget. Pathological inputs
that cannot advance within the overlap budget fail explicitly.

Original Unicode, case, punctuation and whitespace remain in stored text: chunks
are slices of the source string, never decoded token IDs. Tokenizer normalization
still applies internally when computing embeddings. Boundaries may cut sentences
or words; overlap supplies nearby context but does not guarantee whole sentences.
Python character offsets (`char_start`, `char_end`) are half-open, and token
positions refer to the original full-document tokenization. Chunk index is zero
based. IDs are SHA256 of source metadata, exact slice, settings and strategy
version. All source fields are retained alongside positions and token counts.

### Model and environment

[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
is a compact English sentence/paragraph encoder suitable for this CPU baseline.
It produces 384-dimensional vectors. Revision is pinned to
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; model license is Apache-2.0.
Documents and queries use the same encoder, normalized float32 embeddings and
cosine distance. Lower distance means closer vectors; it is **not a confidence
probability**. This is a baseline choice, not a measured model-quality comparison.

CPU is explicit (`--device cpu`, default), with batch size 32 and four PyTorch
threads. Use `--batch-size`, `--threads` and `--device` to configure execution.
CUDA selection fails if unavailable. The Week 1 CUDA requirement remains blocked.
Model loading/download time is separate from encoding time; loading excludes
Python imports. Encoding timing includes token-limit validation and conversion
of vectors to NumPy. GPU-to-CPU conversion, if used, completes before timing ends.
Ingestion timing fields cover only the current invocation and its newly encoded
chunks. Skipped vectors contribute no encoding time. A process interrupted before
its final report may leave valid database batches without a saved timing report;
resume timings must not be presented as total initial indexing time.

No package upgrades were needed. The existing pinned `requirements.txt` remains
valid: all 124 pins match installed distribution versions (see
`reports/dependencies_retrieval.json`). NumPy (already pinned at 2.4.6) is now
an explicit direct dependency in
`requirements.in`. Verified stack: Python 3.11.9, sentence-transformers 6.1.0,
transformers 5.19.0, ChromaDB 1.5.9, PyTorch 2.14.1+cpu. `pip check` passes.

API references checked: [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
and [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure).
The installed Chroma version emits a legacy embedding-function configuration
DeprecationWarning during tests. Explicit vectors are used throughout; no Chroma
embedding model is downloaded or used.

### Exact commands (PowerShell, repository root)

First installation uses the existing setup commands above. First model use needs
internet access; the public model is cached in ignored `.cache/huggingface/hub`.
No API key or paid service is required. Omit `--offline` for the first model download,
then use it to avoid network requests once the pinned snapshot is cached.

```powershell
# Unit tests (including real local Chroma storage tests; no downloads)
.\.venv\Scripts\python.exe -m pytest -q

# Small real sample; use a separate collection from the full corpus
.\.venv\Scripts\python.exe scripts/retrieval.py ingest --limit 25 --collection wikipedia_sample_v1 --report reports/index_sample.json

# Full corpus, explicit CPU development mode
.\.venv\Scripts\python.exe scripts/retrieval.py ingest --device cpu --offline --report reports/index_full.json

# A separate process reopens persistent storage and searches it
.\.venv\Scripts\python.exe scripts/retrieval.py search --query "How did Alan Turing contribute to computing?" --top-k 5 --offline --report reports/search_restart.json

# Eight corpus-related queries, one warm-up per query, then three repeats
.\.venv\Scripts\python.exe scripts/retrieval.py benchmark --offline --queries benchmark_queries.json --repeats 3 --top-k 5 --report reports/retrieval_benchmark.json

# Verify identical ingestion skips records without increasing the stored count
.\.venv\Scripts\python.exe scripts/retrieval.py ingest --offline --report reports/index_rerun.json
```

Use `--input`, `--db`, `--collection`, `--chunk-size`, `--overlap`, and
`--write-batch` to change ingestion settings. Relative paths resolve from the
current directory. Default storage is `vector_store/`, collection
`wikipedia_minilm_v1`. Cleaned data, database and model cache are Git-ignored.
Reports are small JSON files suitable for Git. Report paths may be overwritten on
rerun; choose a new `--report` path to retain a previous measurement.

### Storage safety, recovery and edge cases

Collections record model/revision/dimension/normalization/input limit, complete
selected-corpus fingerprint, chunk configuration and expected counts. Different
settings, changed source records or another model are rejected before ingestion
writes. To rebuild, choose a **new collection name**, for example:

```powershell
.\.venv\Scripts\python.exe scripts/retrieval.py ingest --collection wikipedia_minilm_v2 --chunk-size 200 --overlap 24 --offline --report reports/index_v2.json
.\.venv\Scripts\python.exe scripts/retrieval.py search --collection wikipedia_minilm_v2 --query "What is an atom?" --offline
```

Old collections are preserved; there is no automatic deletion, replacement or
migration. Changing only batch size/device does not require a new collection.
Use matching `--model` and `--revision` on all commands if selecting another model;
verify its language, tokenizer and input constraints before adopting it.

Ingestion validates the selected corpus and all chunks before creating a
collection. Empty documents/corpora, duplicate source IDs, missing source fields,
nested/null/nonfinite metadata, invalid token settings and bad vector shapes fail
clearly. Existing chunk IDs are checked against text/metadata and skipped, not
upserted. Float metadata allows a relative tolerance of 1e-14 solely for Chroma
serialization round-off; source text and string IDs require exact equality.
All expected IDs/text/metadata are read back and the count is checked before the
collection is marked ready. Interrupted batches retain partial data: rerun the
same command to resume. Search refuses a collection still marked building.
Use one ingestion writer per collection; concurrent mutations are not supported.

Blank queries, nonpositive top-k, missing/empty/incomplete collections and
incompatible query encoders fail explicitly. Top-k larger than the collection is
clamped to available records. Results contain rank, chunk ID, source metadata,
text and labelled cosine distance. Multiple chunks may belong to the same article.

### Benchmark interpretation

`benchmark_queries.json` contains eight exploratory English queries based on
actual corpus subjects. They are not relevance labels. No precision, recall or
hallucination score is claimed. Reports retain per-query/repeat timings, top-five
350-character snippets, source URLs/IDs and distances for manual review. Wikipedia
attribution and the Week 1 dataset license also apply to these snippets.

Model/client startup is excluded from repeated retrieval timing. Each query has
one unmeasured warm-up search before three measured repeats in fixed query order.
`perf_counter` measures query encoding, Chroma query/result deserialization and
their sum; validation/count checks, result formatting, printing and report writes
are outside these boundaries. Mean, median and nearest-rank p95 summarize the 24
measured searches. These are warm local single-process timings, not cold-start,
concurrent API performance or a production SLA. OS load/cache effects can change
results; use the JSON evidence rather than treating one measurement as a guarantee.


### Verified retrieval results — 2026-10-16

| Check | Actual result |
| --- | --- |
| Existing clean corpus | 6,954 unique, nonempty English documents |
| Real development sample | 25 documents, 191 chunks; encoding 21.152 s, model loading 1.837 s |
| Full populated database | 31,984 chunks; every expected ID/text/metadata read back successfully |
| Recovery | 28,672 persisted chunks skipped; remaining 3,312 encoded without duplicates |
| Resume encoding | 201.261 s; 16.456 newly encoded chunks/s; CPU, batch 32, four threads |
| Resume model loading / chunking / DB writes | 1.427 s / 46.642 s / 16.694 s |
| Tests | 51 passed, 0 failed; pytest 17.11 s (2026-10-15), including Week 1 regression tests |
| Dependencies | All 124 pinned versions match installed packages; pip check passes |
| Full-database benchmark | New process reopened 31,984 chunks; 8 queries × 3 repeats = 24 measurements |
| Full identical rerun | 31,984 existing chunks skipped, 0 encoded, count unchanged; read-back passed |
| Separate search process | Returned five ranked results; first source Alan Turing; stored count 31,984 |
| Benchmark startup exclusions | Model loading 0.939 s; 8 warm-up searches took 1.006 s |

| Warm retrieval phase | Mean (ms) | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| Query embedding | 31.187 | 24.844 | 73.371 |
| Database search | 6.267 | 5.942 | 7.668 |
| Total retrieval | 37.454 | 30.991 | 79.269 |

The initial full indexing process was interrupted after 28,672 chunks were
persisted, before its final report was written. The successful resume is recorded
in `reports/index_full_resume.json`. Its 201.261 seconds measure only the last
3,312 chunks: **total original full-corpus encoding time is unavailable**. No
estimate is substituted. All 31,984 stored chunks are nevertheless validated.

Evidence files:
- `reports/index_sample.json`: fresh real sample encoding and read-back.
- `reports/index_sample_resume.json`: recovery/idempotence check of the initial sample.
- `reports/search_sample_restart.json`: separate-process sample search.
- `reports/benchmark_sample.json`: benchmark command smoke test on 191 chunks,
  performed while full indexing was running; do not use it as the full benchmark.
- `reports/index_full_resume.json`: completed full corpus and measured resume work.
- `reports/index_rerun.json`: full identical ingestion, no new vectors or duplicates.
- `reports/search_restart.json`: standalone full-database search after process restart.
- `reports/retrieval_benchmark.json`: full-database per-query/repeat timings,
  retrieved snippets, source IDs/URLs, distances and summary statistics.
- `reports/retrieval_tests.json` and `reports/retrieval_tests.xml`: test evidence.
- `reports/dependencies_retrieval.json`: installed-version/lock comparison.

Manual inspection of the first repeat returned Alan Turing, Atom, August,
Aquaculture, Astronomy, Afghanistan, Addition and Archaeology as the first source
articles for their respective queries. This is an illustrative relevance check,
not labelled precision/recall. No ground-truth evaluation dataset exists yet.

### Files and submission checklist

Created: `src/chunking.py`, `src/embeddings.py`, `src/vector_store.py`,
`scripts/retrieval.py`, `tests/test_chunking.py`, `tests/test_retrieval.py`,
`benchmark_queries.json`, and the retrieval reports listed above.
Updated: `README.md`, `AGENTS.md`, `requirements.in`. The existing pinned
`requirements.txt` required no version changes. Week 1 source/data are preserved.

- [x] Implement and test configurable chunking and metadata preservation.
- [x] Encode and validate the complete available clean corpus in persistent Chroma.
- [x] Reopen the database in another process and benchmark actual retrieval.
- [x] Update README with runnable commands, measured results and limitations.
- [x] Confirm data, vector storage and model caches are Git-ignored.
- [x] Reviewed and committed source, tests, README and small reports to the same repository.
- [x] Verified public visibility, pushed the Week 2 commit and verified the updated README on GitHub.
- [ ] Submit the same repository URL and disclose the unresolved Week 1 CUDA requirement.

The repository was verified public through the GitHub API on 2026-10-18.
Week 2 implementation commit `0e385766b4b95e19fd4e4bb07a68fd52bc47fdf1` was
pushed and its updated README verified through the GitHub API on 2026-10-18.
The submission repository remains:
https://github.com/Scheeza-Ahmad/Parallax_Labs.git. Do not add ignored data or caches
to Git to make the submission look complete; reviewers should reproduce them.


### Submission demonstration - 2026-10-18

Reran all 51 tests: **51 passed in 42.88 seconds**, with 14 non-failing Chroma
configuration deprecation warnings. `pip check` passed and all required imports
passed. The environment verifier still exits 1 because CUDA is unavailable; this
is explicitly an unresolved Week 1 requirement, not a passing GPU check.

The full ingestion command read back all **31,984 chunks from 6,954 documents**,
skipped all existing vectors and generated **zero duplicates/new vectors**.
A new search process queried "What is an atom made of?" and returned three chunks:
Atom, Atom, Proton (cosine distances 0.3218, 0.3280, 0.3968, rounded).
These are retrieved sources, not an LLM-generated answer or confidence probabilities.

A separate benchmark after ingestion completed measured 8 queries x 3 repeats:
mean total retrieval **32.853 ms**, median **26.692 ms**, p95 **66.413 ms**.
Model loading (0.893 s) and eight warm-up searches (0.823 s) were excluded.
Previous dated reports are preserved; latency naturally varies between runs.

Fresh evidence: `reports/submission_tests.xml`, `reports/submission_verification.json`,
`reports/submission_ingestion.json`, `reports/submission_search.json`, and
`reports/submission_benchmark.json`. The public submission repository is
[Scheeza-Ahmad/Parallax_Labs](https://github.com/Scheeza-Ahmad/Parallax_Labs).


## Week 3 - LLM integration and prompt engineering

Screenshot scope: Oct 19-Oct 25, "LLM Integration & Prompt Engineering". The user
requested implementation on 2026-10-18 and selected Groq. Frontend, API serving
and deployment are not part of this screenshot. Week 1 CUDA remains unresolved.

**Current status (2026-10-24): implemented, offline-tested and live-tested on the
user-confirmed Groq Free plan.** See the live results below. The initial
offline run had 100 tests pass in 6.53 seconds, including Week 1/2 regressions.
`reports/week3_tests.json` and `.xml` record this run. Real retrieval from the
31,984-chunk persistent database also passed the Week 3 dry-run. No API key was
configured at inspection time, and no Groq generation latency or answer-quality
result is claimed. This weekly deliverable remains partially complete until the
live checks below pass.

The initial key-related limitations in the preceding historical inspection were
resolved on 2026-10-24; the live evidence below supersedes that pending status.

### Flow and safeguards

`scripts/rag.py` -> existing Week 2 retrieval -> relevance gate -> budgeted source
context -> Groq draft -> exact evidence validation -> Groq evidence/relevance
review -> cited answer or explicit abstention.

- `src/llm.py`: authenticated Groq HTTP client. Credentials come from local `.env`
  or environment; environment values take precedence. API keys and raw provider
  error bodies are never printed or put in reports. Only the fixed HTTPS Groq
  endpoint is used, redirects are disabled, and no provider/model fallback occurs.
- `src/rag.py`: factual questions against the historical 2023-11-01 Wikipedia
  corpus. The system prompt identifies sources and question as untrusted data,
  requires source-only claims and exact supporting quotes, and instructs the
  model to abstain on unsupported or off-topic requests. Source text is encoded
  inside JSON in a separate user message; it is not inserted into the system role.
- Each released claim must have a known S1/S2 source ID and a contiguous exact
  quote (12-500 characters). The application attaches citations and original
  document/chunk IDs and article URLs. Unknown citations, fabricated quotes,
  invalid JSON/schema and incomplete/truncated responses withhold the answer.
- A second LLM call checks every claim's support by its quote and whether the
  claims answer the question. Every boolean must be true; uncertain, malformed or
  failed verification withholds the draft. Raw failed drafts are not returned.
- A quote match alone does not prove the claim is true. The same-model verifier
  can also make mistakes; these are hallucination-resistance safeguards, not a
  proof, zero-hallucination guarantee or independent factual audit. Wikipedia may
  itself be wrong or outdated. Human inspection is still required.
- Narrow English rules reject clear personal-password requests, creative writing
  and live stock/weather/news requests. Other questions use retrieval relevance
  and the LLM checks. The rules are intentionally not a universal intent classifier.
- Retrieval keeps chunks with cosine distance <= 0.55. This is an explicit,
  provisional heuristic, not a calibrated confidence score. Change with
  `--max-distance` only after evaluation; false rejection/acceptance is possible.
  No eligible chunks returns `insufficient_context` without generation.

### Model discovery, token budgets and API failures

Default candidate: `openai/gpt-oss-20b`. Groq's official model/JSON-mode docs were
checked on 2026-10-18; do not assume an old model ID remains active. Before each
live answer/benchmark process, authenticated `/models` must confirm the exact ID
and its context window. Missing/deprecated/inaccessible models fail with an
instruction to rerun `models`. No automatic paid upgrade is attempted. Groq's
Free-plan availability and account limits must be checked in the user's console;
model existence alone does not establish billing status.

Existing pins already include httpx 0.28.1 and python-dotenv 1.2.4; these are now
explicit in `requirements.in`. No package upgrades were needed, and `pip check`
passes. Week 2 embedding model/configuration are unchanged.

Defaults: 8,192 context-budget tokens, 1,024 completion tokens, 256-token reserve.
Prompts use a conservative UTF-8-byte count plus 128 per message and 128 framing
allowance, NOT the provider's exact tokenization. Whole source chunks are packed
until the budget is full; no source slice is silently truncated. The configured
budget is capped by the live model context window. Verification prompts undergo
the same check. A local budget failure or provider context rejection returns an
explicit error rather than an unsupported answer. The embedding query limit still
applies; oversized questions must be shortened. GPT-OSS uses low reasoning effort;
completion limits include reasoning, so an exhausted budget causes withholding.

API behavior: maximum three attempts (configurable 1-5), 20-second HTTP operation
timeout (configurable up to 60), exponential waits of 1/2 seconds by default.
429 and 500/502/503/504 plus network timeouts/transport failures are retried.
Numeric or HTTP-date Retry-After is honored when <=10 seconds; longer waits exit
with a retry-later error instead of retrying too early. Authentication, permissions,
404, malformed requests and oversized requests are not retried. No raw API error
body is logged. Timeout retries can duplicate provider work if a response was lost;
use a Free-plan account or explicitly approve any paid usage before live testing.

### Setup and exact commands

1. In Groq Console, confirm the account plan/limits and create a local API key.
2. Copy `.env.example` to `.env` only if `.env` does not already exist, then edit
   `.env` locally. Set `GROQ_API_KEY` to the real key. Never paste it into chat,
   commit it or put it directly into a terminal command/history.
3. Run these commands from the repository root. `--offline` applies only to the
   cached embedding model; a live Groq request still needs internet access.

```powershell
# Offline tests: no API key, network or paid calls required
.\.venv\Scripts\python.exe -m pytest -q

# Actual local retrieval and prompt packing; explicitly NOT a generated answer
.\.venv\Scripts\python.exe scripts/rag.py ask --query "What is an atom made of?" --offline --dry-run
.\.venv\Scripts\python.exe scripts/rag.py benchmark --offline --dry-run --report data/rag_runs/preview.json

# Requires configured key: check live IDs first (does not generate text)
.\.venv\Scripts\python.exe scripts/rag.py models

# Real source-grounded answer plus evidence verification
.\.venv\Scripts\python.exe scripts/rag.py ask --query "What is an atom made of?" --offline

# Two factual and three out-of-scope/unsupported queries, one pass by default
.\.venv\Scripts\python.exe scripts/rag.py benchmark --offline --repeats 1 --report data/rag_runs/live-benchmark.json
```

Override model with `--model` or `GROQ_MODEL`. Other controls: `--top-k`,
`--max-distance`, `--context-tokens`, `--output-tokens`, `--db`, `--collection`,
`--queries`, `--repeats` and `--report`. `.env.example` documents environment knobs.
Keep reports containing private questions in ignored `data/rag_runs/` (the default).
Only reviewed, non-sensitive demonstration reports belong in Git. User questions
and the packed Wikipedia excerpts are transmitted to Groq for generation; the
verification request contains the question, proposed claims and evidence quotes.

### Latency, tests and current evidence

Each query records retrieval, generation, verification and end-to-end milliseconds.
End-to-end includes prompt packing, validation and API retries/waits. Model discovery,
embedding/database startup and report/console I/O are excluded and reported separately.
No LLM warm-up requests are hidden. All statuses are retained; successful answers
have a separate mean/median/nearest-rank p95 summary so fast refusals do not inflate
apparent answer performance. Authentication/model discovery failures exit before
query timing begins. First-query cache effects can affect the timings.

Offline tests cover bounded retries, rate limits, timeouts, credential redaction,
model mismatch, malformed responses, incomplete output, budgets, source metadata,
fabricated citations/quotes, false claims despite matching quotes, relevance
rejection, off-topic rules, dry-run labeling and failure timings. Stub LLM responses
and MockTransport fixtures are test inputs, not evidence of actual API quality.

Actual dry-run (`reports/week3_dry_run.json`):
- Atom and Alan Turing queries retrieved real sources and packed prompts; status
  `preview_only`, no answer and zero API attempts.
- Personal password and poem requests: `off_topic` without retrieval/generation.
- Current Apple stock price: `insufficient_context` because the corpus is historical.
- Database reopened with 31,984 chunks. Live answer latency remains unmeasured.

Official references: [models](https://console.groq.com/docs/models),
[JSON output](https://console.groq.com/docs/structured-outputs),
[API parameters](https://console.groq.com/docs/api-reference),
[rate limits](https://console.groq.com/docs/rate-limits),
[error codes](https://console.groq.com/docs/errors).

Week 3 completion checklist:
- [x] Implement generation, prompts/context, error handling and grounding checks.
- [x] Run offline unit tests and actual local retrieval/prompt integration.
- [x] Preserve Week 1/2 and document setup, caveats and latency boundaries.
- [x] Configure local Groq key and confirm Free-plan use or authorize paid usage.
- [x] Verify authenticated active model and run real generation/verification.
- [x] Inspect citations/abstentions and record actual end-to-end live latency.
- [x] Update verified results and publish this week's changes to the same repository.

### Reverification - 2026-10-24

The same Week 3 deliverables were reconfirmed from the supplied screenshot.
All **100 tests passed in 9.03 seconds**, with 14 Chroma deprecation warnings;
`pip check` also passed. The first run hit a Windows sandbox temporary-directory
rename permission error (99 passed, one failed). Rerunning with a workspace-local,
Git-ignored temporary directory passed without changing preprocessing code:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=data/test_tmp_20261024 --junitxml=reports/week3_tests_20261024.xml
.\.venv\Scripts\python.exe scripts/rag.py benchmark --offline --dry-run --report reports/week3_dry_run_20261024.json
```

The real database reopened with 31,984 chunks. Atom and Alan Turing queries
returned `preview_only`; password and poem queries returned `off_topic`; the
live Apple stock-price query returned `insufficient_context`. All five recorded
zero API attempts. This validates retrieval/prompt integration and local scope
rules, not live generation or LLM latency. No local Groq key was configured.
Authenticated generation, citation inspection and live end-to-end measurements
remain pending; do not submit this as fully verified Week 3 work yet.

If Groq key creation reports "We couldn't verify this request", complete its
browser verification in an updated browser, temporarily disable interfering
ad/privacy extensions or VPN, and retry with another browser or network.
If it persists, use the console's Contact support option. See
[Cloudflare verification troubleshooting](https://developers.cloudflare.com/turnstile/troubleshooting/client-side-errors/).
Keep the resulting key only in local `.env`; confirm the account plan before
live tests. No paid API usage has been authorized.

### Live verification - 2026-10-24

After the preceding dry-run, the user configured a key and confirmed Free-plan
use. Credentials were moved from `.env.example` into Git-ignored `.env`, and the
example was restored to an empty placeholder. Authenticated Groq model discovery
verified `openai/gpt-oss-20b`, context window 131,072. An initial connection failure
was followed by successful authenticated discovery and this benchmark:

```powershell
.\.venv\Scripts\python.exe scripts/rag.py benchmark --offline --repeats 1 --report reports/week3_live_20261024.json
```

| Query | Actual outcome | End-to-end |
| --- | --- | ---: |
| What is an atom made of? | Answered, exact quote and verifier approved, citation S2 | 2,452.854 ms |
| How did Alan Turing contribute to computing? | Answered, three supported claims, citations S1/S2 | 1,891.465 ms |
| Personal email password | off_topic; no API attempts | 0.029 ms |
| Romantic poem | off_topic; no API attempts | 0.009 ms |
| Apple stock price right now | insufficient_context; no API attempts | 0.299 ms |

The two successful answers averaged **2,172.160 ms** including retrieval,
generation and separate evidence verification; each used two completion requests.
The atom answer was: "An atom is made of protons, neutrons, and electrons. [S2]".
Manual inspection matched its quotation to the returned Atom source. The Turing
answer quoted the 1936 theoretical machine, Turing test and wartime Enigma work;
the claims' quotations and stored source IDs were inspected. All four claims
across the two answers passed exact-quote validation and LLM verification.

Evidence: `reports/week3_live_20261024.json`, containing full source text,
source URLs, supporting quotes and phase timings. This small demonstration is
not a labelled quality evaluation or a zero-hallucination guarantee. Error paths
are verified with offline HTTP fixtures, not deliberately induced live failures.
Week 3 implementation and live checks are complete. Implementation commit
`a6903d9` was pushed to the existing repository on 2026-10-24. Submit
[Scheeza-Ahmad/Parallax_Labs](https://github.com/Scheeza-Ahmad/Parallax_Labs).
The earlier CUDA blocker remains unresolved.


## Week 4 - NLP analysis and filtered retrieval (local only)

The supplied screenshot specifies Oct 26-Nov 1: topic discovery with BERTopic or
LDA, sentiment analysis **or NER**, small manually labelled evaluation, manual
topic review, edge-case handling and vector metadata filtering. This implementation
chooses **LDA and NER**. Named entities are useful for encyclopedic text; sentiment
is not implemented or implied. User explicitly requested no GitHub push.

### Approach and reproducibility

- `src/nlp.py`: CountVectorizer + scikit-learn LDA, 10 topics, seed 42, batch
  learning, 10 iterations, one worker, English stop words, minimum document
  frequency 2, maximum document frequency .95, vocabulary capped at 15,000.
  Topic IDs are run-specific, not permanent semantic categories. Top terms aid
  interpretation; topic weight is mixture mass, not classification confidence.
- Documents with fewer than 20 whitespace/word tokens or no retained vocabulary
  receive topic -1 and an explicit short/out-of-vocabulary status. Technical jargon
  is retained when it occurs in at least two documents; rare technical words may
  be discarded. Mixed-topic documents retain only their dominant topic for filtering.
- spaCy `en_core_web_sm` 3.8.0 extracts PERSON, ORG, GPE, LOC and DATE entities.
  Other entity types are intentionally excluded. Every document is processed;
  long text uses consecutive 50,000-character windows, retaining global offsets.
  An entity crossing a window boundary may be missed/split. Names and dates may
  be mistaken or omitted; metadata must not be treated as factual ground truth.
- Models and full metadata are saved to ignored `data/nlp/`; only load joblib
  artifacts produced by trusted local code. Dependency pins include the official
  spaCy model wheel, existing scikit-learn and joblib versions. No LLM calls or
  additional paid services are used for this analysis.
- `scripts/nlp.py enrich` copies the existing embeddings, IDs, source metadata and
  text into a separate `wikipedia_nlp_v2` collection. It adds document-level
  `topic_id`, `topic_weight`, `topic_status`, `entity_count` and boolean
  `has_person/org/gpe/loc/date` fields. No re-embedding is performed. A chunk can
  inherit a document entity/topic even when that entity is not in the chunk.
- Source corpus digest and NLP-artifact signature prevent incompatible reuse.
  Each copied batch is read back, checking metadata, text and vectors. Interrupted
  runs can resume with the same artifact; changed artifacts require a new collection.
  The source collection is never updated. Use one writer at a time.
- Filters are applied by Chroma before similarity ranking, using topic equality,
  entity presence, minimum topic weight and AND combinations. Zero matches return
  an empty result; there is no silent fallback to unfiltered retrieval.

### Commands

Run from the repository root after installing `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe scripts/nlp.py analyze --report reports/week4_topics.json
.\.venv\Scripts\python.exe scripts/nlp.py enrich --report reports/week4_enrichment.json
.\.venv\Scripts\python.exe scripts/nlp.py evaluate --report reports/week4_ner_evaluation.json
.\.venv\Scripts\python.exe scripts/nlp.py search --query "How did Alan Turing contribute to computing?" --entity PERSON --report reports/week4_filtered_search.json
# Choose a topic ID after inspecting reports/week4_topics.json
.\.venv\Scripts\python.exe scripts/nlp.py search --query "astronomy" --topic 0 --min-weight 0.4 --report data/nlp/topic_search.json
# Week 3 integration: dry-run, no API call
.\.venv\Scripts\python.exe scripts/rag.py ask --collection wikipedia_nlp_v2 --entity PERSON --query "Who was Alan Turing?" --offline --dry-run
# Remove --dry-run to generate using the configured Groq account
.\.venv\Scripts\python.exe -m pytest -q --basetemp=data/week4_tests --junitxml=reports/week4_tests.xml
```

Both search and RAG support topic/entity filtering (RAG uses
`--min-topic-weight`, NLP search uses `--min-weight`). Filters belong to the enriched
collection. Original Week 2/3 commands continue using `wikipedia_minilm_v1`.

### Evaluation and human review

`evaluation/ner_labels.json` contains 12 real corpus excerpts and independently
assigned **Codex draft annotations made before NER predictions**.
`evaluation/NER_REVIEW.md` is the readable review sheet. The user has agreed to
review; no completed human review is claimed. Corrections must update text spans
and offsets consistently before rerunning evaluate. The report preserves annotation
provenance; human-reviewed accuracy remains pending until that review is recorded.

The evaluation uses exact span + entity type precision/recall/F1, and exact
whole-example match accuracy. It includes no-entity scientific passages and
ambiguous dates/geographical names. It is a purposive convenience sample, not an
independent random holdout; it contains no gold ORG examples, so organizational
entity accuracy is not established. Do not generalize its metrics to the corpus.

Initial draft-label result: 24 true positives, 12 false positives, 12 false negatives;
precision/recall/F1 each 66.7%, exact-example accuracy 5/12 (41.7%). Boundary and
label disagreements matter: model predictions must not be copied into gold labels
to improve the score. Keep annotation decisions independent of model outputs.

Official references: [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html),
[spaCy English model](https://spacy.io/models/en),
[Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).

Observed draft-set errors include `OBE FRS`, `BC` and `the American Civil War`
predicted as ORG; missing Wilmslow; Oceania predicted as GPE instead of LOC;
and date/leading-article boundary mismatches. These examples explain why metadata
filters can create false positives and exclude otherwise relevant results.


### Full-corpus evidence - 2026-10-25

Analyzed all **6,954 documents**, retaining 6,877 topic assignments and 77 short
unassigned documents (topic -1). Extracted **543,794 mentions** across the five
NER categories. Every stored entity span was checked against its exact source
substring; this is an offset/integrity check, not a semantic accuracy measurement.
The reported elapsed wall time spans a long desktop session and is not a controlled
uninterrupted runtime benchmark; no throughput claim is made.

| Topic ID | Proposed interpretation | Documents | Qualitative assessment |
| --- | --- | ---: | --- |
| 0 | Biology / natural materials | 807 | broad but interpretable |
| 1 | Religion plus physical concepts | 618 | mixed |
| 2 | People / health / colors | 614 | weak |
| 3 | Calendar events and biographical lists | 940 | structural artifact |
| 4 | Numbers / computing / games | 509 | broad but interpretable |
| 5 | Entertainment and competitive teams | 588 | mixed |
| 6 | Geography | 1270 | interpretable |
| 7 | American people / culture | 478 | broad |
| 8 | Politics / government / war | 545 | interpretable |
| 9 | Music and language | 508 | mixed |

Codex inspected 30 representative source excerpts (the top three mixture weights
per topic), not a human-labelled or random topic-quality evaluation. In particular,
topic 1 mixes religion and relativity, topic 2 is weak, topic 3 reflects calendar
list structure, and topic 5 mixes entertainment and sports. Very high mixture
weights do not imply coherent or correct topics. Treat these as exploratory
filtering facets; restrictive filters can reduce recall. No topic accuracy or
retrieval precision percentage is asserted. Human review remains pending.

Evidence: `reports/week4_topics.json`, `reports/week4_topic_review.json`,
`reports/week4_ner_evaluation.json`, `evaluation/NER_REVIEW.md`, and
`evaluation/TOPIC_REVIEW.md`. Full document metadata and trained LDA artifacts
remain in ignored `data/nlp/`.
