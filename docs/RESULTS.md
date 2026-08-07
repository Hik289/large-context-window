# Evaluation

This repository publishes the UltraMem implementation and a compact summary of the
reported evaluation. Restricted corpora, generated indexes, raw predictions, and model
credentials are not redistributed. Evaluation code and plotting utilities remain
available for authorized validation of dual-view retrieval, bounded evidence packing,
and provenance-preserving answer construction.

## Reported Results

### End-to-End Quality

The primary 10M-token evaluation uses 400 scored questions.

| System | Combined | Correctness | Completeness | Document recall | InvDocs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published reference | 68.22 | 81.60 | 72.86 | 79.02 | 0.470 |
| UltraMem | **82.26** | **86.50** | **86.98** | **81.90** | 0.760 |

UltraMem improves the combined score by 14.04 percentage points. InvDocs is weaker than
the reference, so the gain should not be read as dominance on every metric.

### Evidence Efficiency

The selective-loading study uses 100 questions and compares the evidence presented to
the answer model.

| Tier | Context policy | Tokens/query | Correctness |
| --- | --- | ---: | ---: |
| 10M | Detailed-only | 6,532 | 88.0% |
| 10M | UltraMem selective | **2,049** | 86.5% |
| 20M | Detailed-only | 6,497 | 85.5% |
| 20M | UltraMem selective | **2,025** | 84.0% |

Selective loading uses 68.6% fewer evidence tokens at 10M and 68.8% fewer at 20M,
with a 1.5-point correctness difference in both settings.

### Fixed-Budget Scaling

| Addressable tier | Combined | Correctness | Document recall |
| --- | ---: | ---: | ---: |
| 20M | 71.67 | 84.00 | 84.00 |
| 60M | 64.19 | 78.25 | 78.36 |
| 100M | 63.17 | 77.00 | 75.23 |
| 150M | 60.49 | 75.75 | 72.16 |
| 250M | 58.02 | 73.50 | 66.79 |

The active context remains bounded across tiers. The decline in document recall indicates
that candidate coverage, not an enlarged answer prompt, is the main scaling limitation.

## What Evaluation Should Establish

- **Representation integrity.** Every memory node should preserve both a distilled
  navigation view and a source-linked detailed view.
- **Retrieval discipline.** Candidate discovery should remain separate from loading
  detailed evidence into the answer context.
- **Promotion behavior.** Detailed payloads should enter the context only when the
  query requires them.
- **Grounding quality.** Final answers should be traceable to explicit source evidence
  identifiers.
- **Budget accounting.** Prompt construction and model calls should be recorded through
  the token ledger.
- **Configuration isolation.** Model aliases, provider endpoints, and corpus paths
  should be explicit and reproducible without exposing secrets.

## Protocol Hygiene

Private runs should keep method comparisons, corpus variants, model aliases, and
evaluation suites separated. Do not pool incompatible protocols into a single claim.
When using an external evaluation package, keep its data, code, and license boundary
outside this repository unless redistribution is explicitly permitted.

## Reporting Guidance

When reporting results outside this public artifact, include the repository commit,
corpus manifest, model alias file, embedding configuration, retrieval settings, token
ledger, answer outputs, and citation records. Keep raw credentials and restricted
documents out of the repository.

## Public Claim

The public claim is both methodological and bounded by the reported protocols:
large-context agents can use auditable dual-view memory instead of placing the full
collection in every prompt. The code exposes the representation contract, retrieval
interfaces, evidence-selection path, packing logic, and provenance checks needed to
instantiate and audit that design.
