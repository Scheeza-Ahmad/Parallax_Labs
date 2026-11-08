# Week 5 final local result — 2026-11-08

The official Week 5 (Nov 2–8) deliverable is a tested FastAPI service with
automated retrieval and generation evaluation reports. The local implementation
and verification are complete. This result is included in the same internship
repository for weekly submission.

| Requirement | Verified result | Evidence |
| --- | --- | --- |
| FastAPI endpoints | `/health`, `/retrieve` and `/ask`; live `/ask` returned HTTP 200 with four sources and a request ID | `reports/week5_api_live_20261108.json` |
| Structured logging and errors | JSON request ID, method, path, status and duration; pytest covers 400, 422, 503 and sanitized 500 | `tests/test_api.py` |
| Retrieval Precision@K / Recall@K | 10 Codex-labelled known-article questions: Precision@5 0.18; Recall@5 0.90 | `reports/week5_retrieval.json` |
| Live end-to-end evaluation | 5/5 expected statuses; two answered cases passed exact-quote and LLM verifier checks; two off-topic and one live-price query abstained | `reports/week5_final_generation_20261108.json` |
| End-to-end latency | Mean 519.022 ms across all five outcomes; answered cases 1728.68 and 865.80 ms | `reports/week5_final_generation_20261108.json` |
| Local HTTP retrieval benchmark | 10 calls: mean 71.904 ms, p95 91.453 ms | `reports/api_benchmark.json` |
| Tests and dependencies | 116 pytest tests passed; `pip check` reported no broken requirements | Executed 2026-11-08; see README |

The retrieval labels are Codex-authored and purposively selected, with one named
article marked relevant per query. Precision and recall describe only this small
reference set. Expected generation statuses are also Codex-authored; exact quote
checks and LLM verification reduce unsupported output but do not prove factual
accuracy. Off-topic and live-price abstentions take no Groq generation calls, so
the all-case latency mean includes three near-zero outcomes. The local HTTP
benchmark excludes model/database startup and Groq generation.

The currently configured Groq model worked on the final live run. An earlier
HTTP 401 was caused by the credential being put in `.env.example` instead of the
ignored `.env`; the example file is now blank. No key is in this report.
Week 4's independent human annotation review and the earlier CUDA requirement
remain separate unresolved items. Public deployment is outside the Week 5
screenshot. The user authorized publication for Week 5 submission on 2026-11-08.
