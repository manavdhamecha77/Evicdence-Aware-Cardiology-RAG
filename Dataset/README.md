# Benchmark Dataset

A multi-tiered, contamination-controlled evaluation benchmark designed to assess clinical diagnostic reasoning, perturbation sensitivity, retrieval-augmented generation (RAG), and progressive multi-stage inquiry in cardiovascular medicine.

---

## 1. Benchmark Suite Overview

The dataset consists of six distinct, cryptographically verified assets (**D1–D6**). All splits have undergone deduplication, near-duplicate scrubbing ($\text{word-Jaccard} \ge 0.80$), OCR damage repair, and audit validation against external cross-split leakage.

| Asset | Sub-Split / Arm | Items ($N$) | Format | Key Evaluation Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **D1** | Staged Real Cases | **118** | `.jsonl` | Longitudinal, real-world clinical reasoning (2025–2026 cases; CC-BY). |
| **D2** | Holdout Test Split | **609** | `.jsonl` | Primary evaluation MCQ benchmark (273 MedQA vignettes + 336 MedMCQA items). |
| | Validation Split | **613** | `.jsonl` | Hyperparameter tuning, prompt engineering, and few-shot calibration. |
| | Training Split | **4,910** | `.jsonl` | Clean, deduplicated supervised fine-tuning / exemplar pool. |
| **D3** | P1 (Option Permutation) | **609** | `.jsonl` | Position bias and answer-order sensitivity across all test MCQs. |
| | P3 (Noise Injection) | **273** | `.jsonl` | Resilience against non-contributory clinical history distractors. |
| | P4 (Safe Numeric Jitter) | **200** | `.jsonl` | Robustness to non-critical vital sign variations without crossing clinical cutoffs. |
| | P5 (Finding Stripping) | **21** | `.jsonl` | Diagnostic robustness when pathognomonic physical exam and biopsy anchors are stripped. |
| **D4** | Arm 1 (Naive Fixed Window) | **959** | `.jsonl` | Sliding-window retrieval corpus (380 words/chunk, 50-word overlap). |
| | Arm 2 (Hierarchical) | **569** | `.jsonl` | Heading-aware structured guideline index from ESC and ACC/AHA guidelines. |
| **D5** | Retrieval Pairs | **46** | `.jsonl` | Golden query-to-chunk pairs for evaluation of Recall@$K$, MRR, and Hit Rate. |
| **D6** | Progressive Reveal | **118** | `.jsonl` | Multi-stage diagnostic trajectories measuring dynamic belief updates (Steps 1 $\to$ 2 $\to$ 3). |

---

## 2. Directory Layout

```text
Dataset/
├── README.md                              # Dataset specification and audit documentation
├── audit_data.py                          # Automated sanity check and verification script
├── audit_report.txt                       # Output log of the data audit
│
├── artifacts/                             # Core frozen data splits
│   ├── D1_cases_staged.jsonl              # 118 real-world cases structured into 3 clinical stages
│   ├── D2_test_clean.jsonl                # 609 clean holdout test MCQs
│   ├── D2_val_clean.jsonl                 # 613 clean validation MCQs
│   ├── D2_train_clean.jsonl               # 4,910 clean training MCQs
│   ├── D3_P1_permutations.jsonl           # 609 option-shuffled test items
│   ├── D3_P3_noise.jsonl                  # 273 vignettes with distractor noise
│   ├── D3_P4_jitter.jsonl                 # 200 vignettes with band-safe numerical vitals jitter
│   ├── D3_P5_stripped.jsonl               # 21 vignettes with pathognomonic anchors removed
│   ├── D5_retrieval_benchmark.jsonl       # 46 query-to-chunk ground-truth pairs
│   └── D6_progressive_reveal.jsonl        # 118 multi-stage sequential trajectories
│
├── Knowledge_Base/                        # D4 clinical guidelines corpus
│   ├── raw_guidelines/                    # Source Open Access guideline PDFs
│   │   ├── ehad191.pdf                    # 2023 ESC Acute Coronary Syndromes Guideline (107 pages)
│   │   ├── ehag100.pdf                    # 2026 ESC Heart Failure Guideline (112 pages)
│   │   └── otto-et-al-2020...pdf          # 2020 ACC/AHA Valvular Heart Disease Guideline (156 pages)
│   └── indexed_chunks/
│       ├── D4_cardiology_kb.jsonl         # 959 naive 380-word chunks
│       └── D4_cardiology_kb_hierarchical.jsonl # 569 heading-aware section chunks
│
├── manifests/                             # Cryptographic SHA-256 manifests
│   ├── D1_manifest.txt
│   ├── D2_manifest.txt
│   ├── D3_manifest.txt
│   ├── D4_manifest.txt
│   ├── D5_manifest.txt
│   └── D6_manifest.txt
│
└── scripts/                               # Reproducible generation & cleaning pipelines
    ├── build_d2.py                        # MedQA/MedMCQA deduplication & leak-free split
    ├── fetch_d1_raw.py                    # Europe PMC API query script
    ├── fix_d1_cases.py                    # XML parser, license enricher, and stage structurer
    ├── fix_p4_jitter.py                   # Physiological band-safe numerical jitter pipeline
    ├── fix_p5_stripped.py                 # Clinical anchor finding removal engine
    ├── build_d4_chunks.py                 # Naive sliding-window PDF chunker
    ├── build_d4_heading_aware.py          # Heading-aware hierarchical PDF chunker
    ├── build_d5_retrieval.py              # Query-chunk golden pair builder
    └── purge_near_dups.py                 # Cross-split near-duplicate eliminator
```

---

## 3. Schema Specifications

### D1: Human Clinical Cases (`D1_cases_staged.jsonl`)
Cases sourced from Europe PMC (published 2025–2026 under CC-BY licenses; non-human/veterinary cases filtered out):
```json
{
  "id": "D1_PMC13715631",
  "pmcid": "PMC13715631",
  "pmid": "42953764",
  "doi": "10.1093/ehjcr/ytag741",
  "year": "2026",
  "license": "cc-by",
  "title": "Decompensated heart failure after one year of successful therapy with mavacamten: a case report.",
  "gold_primary": "Decompensated heart failure after one year of successful therapy with mavacamten",
  "stage_1_presentation": "Chief complaint and presenting clinical history...",
  "stage_2_diagnostics": "Physical exam findings, ECG rhythms, and diagnostic labs...",
  "stage_3_intervention": "Therapeutic management and follow-up course...",
  "ground_truth_abstract": "Full peer-reviewed publication abstract..."
}
```

### D2: Benchmark Multiple-Choice Questions (`D2_test_clean.jsonl`)
Standardized 4-option clinical MCQs. Train, validation, and test splits have 0 exact-match and 0 near-duplicate overlaps ($\text{Jaccard} < 0.80$):
```json
{
  "id": "medqa_1042",
  "source": "medqa",
  "question": "A 62-year-old man presents with progressive dyspnea...",
  "options": {
    "A": "Aortic stenosis",
    "B": "Hypertrophic obstructive cardiomyopathy",
    "C": "Dilated cardiomyopathy",
    "D": "Constrictive pericarditis"
  },
  "answer": "A",
  "explanation": "Rationale text (excluded from Knowledge Base to prevent RAG leakage)",
  "category": "Structural Heart Disease"
}
```

### D3: Perturbation Benchmarks (`D3_P1` through `D3_P5`)
* **P1 (Option Permutation, $N=609$):** Options shuffled randomly; ground-truth key remapped.
* **P3 (Noise Injection, $N=273$):** Long MedQA vignettes injected with non-contributory medical history (e.g., mild managed seasonal allergies).
* **P4 (Numerical Jitter, $N=200$):** Physiological vitals jittered within strict bands without crossing diagnostic cutoffs:
  * Heart Rate: Bradycardic stays $<60$ bpm, normal stays $60\text{--}100$ bpm, tachycardic stays $>100$ bpm.
  * Blood Pressure: Normal stays $90\text{--}139 / 60\text{--}89$ mmHg; hypertensive crisis remains $\ge 180 / 120$ mmHg.
* **P5 (Finding Stripped, $N=21$):** High-yield clinical anchors (e.g., *holosystolic murmur*, *thrill*, *Aschoff bodies*, *apical ballooning*) replaced with neutral clinical text.

### D4: Guidelines Knowledge Base
Indexed across 375 pages of ESC (ACS 2023, Heart Failure 2026) and ACC/AHA (Valvular Heart Disease 2020) guidelines:
* **Arm 1 (`D4_cardiology_kb.jsonl`, $N=959$):** Fixed 380-word windows with 50-word overlap.
* **Arm 2 (`D4_cardiology_kb_hierarchical.jsonl`, $N=569$):** Parsed by primary and secondary section headings.

### D5: Retrieval Evaluation Benchmark (`D5_retrieval_benchmark.jsonl`)
Contains 46 golden pairs linking test questions to positive guideline chunk IDs in D4:
```json
{
  "query_id": "medqa_412",
  "domain": "Aortic Stenosis",
  "query": "A 74-year-old male with symptomatic severe aortic stenosis...",
  "positive_chunk_ids": [
    "KB_otto-et-al-2020-2_0114",
    "KB_otto-et-al-2020-2_0115"
  ]
}
```

### D6: Progressive Diagnostic Reveal Trajectories (`D6_progressive_reveal.jsonl`)
Standardized 3-step prompt trajectory to test dynamic multi-step updating:
* **Step 1:** Initial Presentation & HPI $\to$ Model predicts top-3 differential diagnoses.
* **Step 2:** Presentation + Physical Exam & Diagnostic Labs $\to$ Model updates and re-ranks differential.
* **Step 3:** Full Trajectory + Outcome $\to$ Model confirms definitive diagnosis and management.

---

## 4. Cryptographic Manifest & Integrity Hashes

All file contents are frozen. Checksums can be validated using SHA-256:

| Artifact Path | Record Count ($N$) | SHA-256 Checksum |
| :--- | :---: | :--- |
| `artifacts/D1_cases_staged.jsonl` | 118 | `b62b4b4b2dcf6a3ea5b8fb838b396e0b10eec21bce90e93c2c952f6aa5c15003` |
| `artifacts/D2_test_clean.jsonl` | 609 | `e2fa56d15a5bf85e9e22e3de96ce554586a86165e2b8ae48823a67603e6437b1` |
| `artifacts/D2_val_clean.jsonl` | 613 | `c306b38e2f25ba7144d18fa1c7a055ab4a75f0ad96b611acab7a1f8ff06be4d1` |
| `artifacts/D2_train_clean.jsonl` | 4,910 | `e0f3ffe98b28a36d4c76c1fc581360da689d19fe645d4537ebbf89dd59abc154` |
| `artifacts/D3_P1_permutations.jsonl`| 609 | `947c8cc38c4329c0c084e099f359359c6b4ff90ad04b6c0872376519e08ebcbc` |
| `artifacts/D3_P3_noise.jsonl` | 273 | `5418e169a15a068dda44def8a6540ea13c81248a6e7a8be3c8d376be56d3c648` |
| `artifacts/D3_P4_jitter.jsonl` | 200 | `aca27db002f8a3460986655e53953a776e318bbdecc6e0b5166657239af6dff7` |
| `artifacts/D3_P5_stripped.jsonl` | 21 | `ef4cba0fe48d0cd6ce420d57f3849a63bab6be8d19eeb45ef7c2ebc126daa20d` |
| `artifacts/D5_retrieval_benchmark.jsonl`| 46 | `f86dc24eb0b0e1911dba4b5d49b36900a4d5ea832a27a0e81f0f73fbe6cdb5b5` |
| `artifacts/D6_progressive_reveal.jsonl`| 118 | `60ddb5c67a671126e6b11e7b21c67734555ba60fbc0e472059dc2805c3554b92` |
| `Knowledge_Base/indexed_chunks/D4_cardiology_kb.jsonl` | 959 | `8977352040efbee3926d6c4e0641f833c0903ba5abaaf280de080e228e1901b0` |
| `Knowledge_Base/indexed_chunks/D4_cardiology_kb_hierarchical.jsonl` | 569 | `aca591bdb8bb131a4b18a7c06a31162ced5399dcdefd1b226ceab5ff9618c45c` |

---

## 5. Verification & Audit Execution

The pipeline includes a standalone audit script checking for format compliance, hash fidelity, answer-key balance, cross-split text overlap, and knowledge-base contamination:

```bash
python audit_data.py --root "."
```

### Audit Status
* **Integrity Status:** `0 FAILs`, `0 WARNs`
* **Cross-Split Leakage:** 0 exact matches, 0 near-duplicates between test and train/val.
* **Contamination Control:** 0 shared 13-grams between test sets and the D4 knowledge base.
* **Confidence Interval:** $95\%\text{ CI half-width} = \pm 3.9\text{ pp}$ at $n = 609$ (for baseline accuracy $\approx 0.60$).