## 🚀 Running Experiments for Locomo Dataset

You can run experiments using the provided commands:

```bash
# Run AgentMemory experiments
python run_ultramem.py openai.model="gpt-4.1-mini" memory.memory_store="external_baseline-4.1-all"
```

This script:
1. Build the memory based on the conversations if the memory store doesn't exist.
2. Generate the answers for each question in "${result_folder}/ultramem_output.json" file.
3. Evaluate the results using BLEU, F1 and LLM-AS-JUDGE in "${result_folder}/ultramem_eval.json" file.
4. Generate the final result scores in "${result_folder}/ultramem_scores.json" file.

Example output:
```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score  count
category                                       
1           0.xxxx    0.xxxx     0.xxxx     xx
2           0.xxxx    0.xxxx     0.xxxx     xx
3           0.xxxx    0.xxxx     0.xxxx     xx

Overall Mean Scores:
bleu_score    0.xxxx
f1_score      0.xxxx
llm_score     0.xxxx
```

## Baselines
Now we support mem0 baseline.
[NOTE] We will support baselines RAG, LangMem, Zep and OpenAI soon.

You can run experiments using the provided commands:

```bash
# Run mem0 experiments
python run_experiments.py memory.type=mem0 openai.model="gpt-4.1-mini" memory.memory_store="mem0-4.1-all"
```

## 📏 Evaluation Metrics

We use several metrics to evaluate the performance of different memory techniques:

1. **BLEU Score**: Measures the similarity between the model's response and the ground truth
2. **F1 Score**: Measures the harmonic mean of precision and recall
3. **LLM Score**: A binary score (0 or 1) determined by an LLM judge evaluating the correctness of responses