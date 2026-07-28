# Disable PostHog telemetry to prevent hanging background threads
export MEM0_TELEMETRY=False

for subset in 3 4 5 6 7 8 9 10; do
    echo "Running experiments with subset index: ${subset}"
    python run_experiments.py memory.type=mem0 openai.model="gpt-4.1-mini" memory.memory_store="mem0-4.1-${subset}" eval.subset_idx=${subset}
done