# Parallax Labs — Source-grounded RAG Assistant

Scheeza Ahmad's internship project. Week 1 implements environment verification,
real document acquisition and conservative preprocessing. Retrieval, generation,
NLP analysis and FastAPI are future scope, not implemented features.

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
No embedding model or spaCy language model is downloaded in Week 1; required base
imports are tested. Later work must follow the official roadmap when supplied.

Deferred next step: verify an approved CUDA-capable environment (or obtain mentor
clarification), install a compatible CUDA-enabled PyTorch build there, and rerun
`scripts/verify_env.py` until the real tensor operation passes. CPU preprocessing and
its tests are already verified; no later-week features have been implemented.

One repository is used for the entire internship:
[Scheeza-Ahmad/Parallax_Labs](https://github.com/Scheeza-Ahmad/Parallax_Labs).
Submit this same repository link every week with that week's committed work and
an updated README. For this submission, disclose the outstanding CUDA check.

Installation references: [Sentence Transformers](https://www.sbert.net/docs/installation.html),
[spaCy](https://spacy.io/usage.html), [Chroma](https://docs.trychroma.com/docs/overview/getting-started),
[pandas](https://pandas.pydata.org/docs/getting_started/install.html).
