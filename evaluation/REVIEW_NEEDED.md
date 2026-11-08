# Human review needed for Week 4

The draft NLP outputs are ready, but they have **not** been approved by Scheeza.
This file is a handoff, not a record of completed human review.
Codex performed a second pass on 2026-11-07: all 36 draft entity spans match
their source excerpts exactly; no label changes were made. See
`reports/week4_codex_second_review.json`. This remains a Codex review.

1. Read `evaluation/NER_REVIEW.md` (12 short real article excerpts). For each
   highlighted span, check both the exact words and PERSON/ORG/GPE/LOC/DATE type.
   List corrections by article title, or confirm after reading all 12 that no
   corrections are needed. Empty examples should also be checked.
2. Read `reports/week4_topic_review.json` alongside `reports/week4_topics.json`.
   Check whether each of the 10 proposed topic names and examples makes sense.
   Note mixed or misleading topics; there is no target accuracy threshold.
3. After corrections are supplied, update the draft labels without copying model
   predictions into gold labels, record who reviewed them and rerun the evaluation.
   Until then, the reported NER score is against **Codex draft labels**, and no
   human-reviewed result should be claimed.

For additional RAG precision/recall, a separate human-reviewed set of questions
and relevant Wikipedia document IDs is needed. `scripts/evaluate_rag.py` rejects
unreviewed labels. This is distinct from the Week 4 NER evaluation requirement.
