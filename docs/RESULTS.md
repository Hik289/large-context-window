# Results and Evaluation

## Protocol Separation

The artifact preserves separate protocols rather than pooling incompatible runs:

| Protocol | Purpose |
| --- | --- |
| Full-system 10M evaluation | Headline comparison with the public EnterpriseRAG-Bench reference |
| Pure-mini six-tier evaluation | Controlled cost-oriented scaling from 10M to 250M tokens |
| Leave-one-out component study | Identifies which mechanisms carry answer quality |
| Reader-by-evidence control | Separates retrieval failure from answer-model capacity |
| External probes | Tests transfer under a separately reported embedding protocol |

## Headline Comparison

| Metric | Full system at 10M | Public reference | Difference |
| --- | ---: | ---: | ---: |
| Overall | 78.29 | 68.22 | +10.07 |
| Correctness | 88.00 | 81.60 | +6.40 |
| Completeness | 80.88 | 72.86 | +8.02 |
| Document recall | 87.29 | 79.02 | +8.27 |
| Invalid documents, lower is better | 3.80 | 0.47 | worse |

The method improves four of the five reported comparisons. The invalid-document gap is
not absorbed into a universal superiority claim.

## Scaling Diagnosis

Correctness from 20M to 250M tokens changes as follows:

| Reader | Evidence source | 20M | 250M | Change |
| --- | --- | ---: | ---: | ---: |
| Smaller reader | Gold documents | 82.50 | 82.25 | −0.25 |
| Smaller reader | Retrieved documents | 84.00 | 73.50 | −10.50 |
| Larger reader | Gold documents | 85.75 | 83.75 | −2.00 |
| Larger reader | Retrieved documents | 70.25 | 56.25 | −14.00 |

Gold evidence keeps both readers nearly flat while retrieved evidence degrades sharply.
The control therefore localizes the primary scale bottleneck to retrieval recall.

## Context Efficiency

Under an aligned candidate depth:

| Corpus | Mixed context | Full-detail context | Token reduction |
| --- | ---: | ---: | ---: |
| 10M | 2,049 tokens/query | 6,532 tokens/query | 3.19× |
| 20M | 2,025 tokens/query | 6,497 tokens/query | 3.21× |

The stable ratio supports a representation-efficiency claim, not a claim that the mixed
context always maximizes correctness.

## External Probes

| Benchmark | Combined quality |
| --- | ---: |
| HotpotQA | 86.37 |
| FinanceBench | 81.05 |
| LoCoMo | 42.34 |
| UltraDomain | 17.73 |

The external probes use a local BGE-large embedding model, whereas the main pipeline uses
the recorded hosted index. They are reported as transfer evidence under a distinct
protocol.

## Disclosure Notes

- The official leaderboard claim uses the gold-answer-only judge protocol.
- The system is higher on four of five comparisons, not five of five.
- Citation extraction affects provenance rather than answer-quality metrics.
- The external embedder differs from the main experimental embedder.
- Raw licensed corpora, generated indices, and model outputs are not distributed here.
