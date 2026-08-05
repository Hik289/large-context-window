# UltraMem Artifact Contract

This release is scoped as a method artifact for large-context agent memory. It is
intended to make the solution inspectable, installable, and reusable without bundling
private corpora or provider credentials.

## Included

- Installable `ultramem` Python package.
- Canonical dual-view node, dual index, hierarchy builder, and token-ledger components.
- Public model and environment templates.
- Credential-free tests for method invariants and package importability.
- Optional integration adapters for authorized external evaluation suites.
- Method schematic and manuscript plotting utilities.
- MIT license.

## Not Included

- Private or licensed corpora.
- Real API credentials.
- Generated vector indexes.
- Raw predictions or model logs.
- Provider-specific account configuration.
- Upstream evaluation packages that carry separate terms.

## Review Checklist

- Run `pytest`.
- Run the method self-tests listed in `docs/REPRODUCIBILITY.md`.
- Build a wheel with `python -m build`.
- Confirm `.env`, data, outputs, and generated figures remain untracked.
- Record the commit and configuration manifest for any private run.
