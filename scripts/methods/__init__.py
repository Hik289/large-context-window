# V4 methods package.
#
# Submodules:
#   dual_node      — DualNode schema + validation
#   dual_index     — Two ChromaDB collections (distilled + detailed) with strict
#                    "detailed-by-id-only" default for retrieval
#   hierarchy_builder — Build DualNodes from L0 records using gpt_5_4_mini for
#                    distilled_text generation
#   token_ledger   — Per-phase, per-node token accounting (build vs retrieval vs answer)
