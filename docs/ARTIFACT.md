# Artifact Contract

## Included

- Installable `agent_memory` Python package
- Canonical dual-view node, index, construction, and token-ledger components
- Public model and environment templates
- Credential-free tests and CI
- External benchmark adapters
- Deterministic aggregate plotting scripts
- Result and protocol summaries
- MIT license and citation metadata

## Not Included

- Private or licensed corpora
- Real API credentials
- Generated vector indexes
- Raw predictions or model logs
- Provider-specific account configuration
- The official EnterpriseRAG evaluation package

## Review Checklist

- Run `pytest`.
- Run the three method self-tests in `docs/REPRODUCIBILITY.md`.
- Build a wheel with `python -m build`.
- Confirm `.env`, data, outputs, and generated figures remain untracked.
- Record the commit and configuration manifest for every reported run.
