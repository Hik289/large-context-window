#!/bin/bash

# Run segment baseline experiment
# This baseline uses conversation segments as episodic memories (no LLM summarization)
# and formats output with both episodic segments and factual memories

python run_ultramem.py \
    openai.model="gpt-4.1-mini" \
    general.debug=True \
    memory.memory_store="segment-baseline-debug" \
    eval.subset_idx=1 \
    memory.enable_cue_index=False \
    memory.enable_episodic_memory=True \
    memory.segment_baseline=True \
    memory.force_rebuild=True \
    retrieval.strategy="semantic"
