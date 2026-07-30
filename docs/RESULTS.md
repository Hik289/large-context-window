# Evaluation Boundary

This repository is a solution release, not a public score table. Evaluation code and
plotting utilities are kept to support private, authorized validation, but the public
artifact centers on the method: dual-view memory, on-demand promotion, bounded packing,
and provenance-preserving answer construction.

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

The public claim of this artifact is methodological: large-context agents can be built
around auditable dual-view memory instead of relying on ever-larger prompts. The code
exposes the representation contract, retrieval interfaces, promotion path, packing
logic, and provenance checks needed to instantiate that solution.
