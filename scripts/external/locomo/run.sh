#!/bin/bash

# external_baseline
# python run_experiments.py memory.type=external_baseline

# mem0
# python run_experiments.py memory.type=mem0 openai.model="gpt-4.1-mini" memory.memory_store="mem0-4.1-debug" general.debug=True
# python run_experiments.py memory.type=mem0 openai.model="gpt-4.1-mini" memory.memory_store="mem0-4.1-all" eval.subset_idx=-1 memory.force_rebuild=True

# external_baseline debug
# python run_agent_memory.py llm.model="gpt-4.1-mini" memory.memory_store="external_baseline-debug" general.debug=True memory.force_rebuild=True

# external_baseline subset 1
# python run_agent_memory.py llm.model="gpt-4.1-mini" memory.memory_store="external_baseline-subset1" eval.subset_idx=1

# external_baseline full dataset
# python run_agent_memory.py llm.model="gpt-4.1-mini" memory.memory_store="external_baseline-efficient" eval.subset_idx=-1 memory.force_rebuild=True

# external_baseline gpt-4.1-mini with cue index + prompt retrieval
python run_agent_memory.py llm.model="gpt-4.1-mini" memory.memory_store="external_baseline-cue" eval.subset_idx=-1 memory.enable_cue_index=True retrieval.strategy="prompt"
