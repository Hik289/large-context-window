# EnterpriseRAG Evaluation Adapter

`metrics_based_eval.py` is an adapter for the official EnterpriseRAG evaluation package.
It imports the official package's `src.llm`, `src.prompts`, and `src.utils` modules and is
therefore not a standalone evaluator in this repository.

Run it only from an authorized checkout of the official benchmark with that checkout on
`PYTHONPATH`. This separation prevents the repository from redistributing benchmark code
or data under incompatible terms.
