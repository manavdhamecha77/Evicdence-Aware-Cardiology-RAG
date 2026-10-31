# Remaining Research Work

Completed evaluation and current measured results are in [REPORT.md](REPORT.md). The list below distinguishes follow-up research from completed implementation.

## Highest priority

- Tune the evidence gate on D2 validation using answer correctness and a risk–coverage analysis, not score percentiles alone. The current threshold is frozen and reported; do not retune against D2 test and present the same test as untouched.
- Manually annotate a relevance subset for the hierarchical D4 corpus. D5's candidate IDs target the fixed corpus and are not human-validated gold labels.
- Obtain blinded human review of semantic support and citation entailment; current citation checks are structural only.
- Build and validate a D1 diagnosis matcher and review title/abstract leakage before reporting D1 diagnostic accuracy. The current D1 result is retrieval coverage only.
- Add a broader perturbation evaluation. The 21-item P5 set produced no retries or abstentions under the frozen thresholds.

## Retrieval and reproducibility

- If source terms permit, build a dense-index comparison with a pinned embedding model revision; current headline results use lexical BM25-style retrieval only.
- Record immutable model-serving revision, token/cost data, and decoding/provider details for future runs.
- Review the 4 flagged OCR-damage markers in each of D2 validation and test (MedMCQA records).
- Confirm external API processing and redistribution terms for D4-derived guideline chunks before repeating or publishing generated content.

## Dataset scope

- D6 is linked to D1 (100/100 DOI links) and should not be counted as an independent cohort.
- Keep any MedQA-USMLE or textbook-derived assets within their applicable license terms. Keyword candidate lists require human review before use as benchmarks.
- Add sources only with verified provenance and terms permitting the planned use; do not redistribute source text or derived passages without permission.
