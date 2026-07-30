# External Evaluation Adapter

`metrics_based_eval.py` is an adapter for an authorized upstream evaluation package. It
imports upstream `src.llm`, `src.prompts`, and `src.utils` modules and is therefore not
a standalone evaluator in this repository.

Run it only from an authorized checkout of the upstream suite with that checkout on
`PYTHONPATH`. This separation prevents the repository from redistributing external code
or data under incompatible terms.
