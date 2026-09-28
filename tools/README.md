# Tools

Diagnostics, dataset checks and model smoke tests. None of these produce reported results; they exist to validate
inputs and catch failures before spending GPU time. Run from the repo root.

## Dataset and label checks

| Script | Purpose |
|---|---|
| `check_dataset.py` | Gate report for a new dataset: schema, gold coverage, sample fields |
| `check_gold_coverage.py` | Quick gold-coverage check on labelled artifacts |
| `check_parser_diff.py` | Confirm Docling and PyMuPDF produced different text for the same PDFs |
| `compute_doc_intersection.py` | Documents successfully labelled by every model (writes `intersection_docs.txt`) |
| `inspect_labels.py` | Print label distributions |
| `inspect_tokens.py` | Show which token the `last_token` activation position lands on |
| `review_doc.py` | Show value-mismatch and hallucination errors for one document |
| `qwen_checks.py` | Read-only consistency checks on the Qwen runs |
| `patch_matcher_depth_guard.py` | One-off patch adding a recursion-depth guard to the matcher |
| `mini_deck_examples.py` | Pull concrete examples from artifacts for presentations |

## Model smoke tests

Run one document through a new model to check that it emits schema-conformant JSON before a full run.

| Script | Model |
|---|---|
| `smoke_test_gemma.py` | Gemma 3 |
| `gemma4_smoke.py` | Gemma 4 E4B |
| `smoke_test_r1.py` | DeepSeek-R1-Distill-Qwen-7B |
| `smoke_test_insurance.py` | Insurance-claims schema |
