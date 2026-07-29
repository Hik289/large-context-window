# Figure Reproduction

Run the self-contained publication plot generator from the repository root:

```bash
pip install -e ".[figures]"
python figures/make_all_figs.py
```

The command writes 11 PDF/PNG pairs under `figures/generated/`. Generated outputs are
ignored by Git.

`publication_figures.py` contains the final aggregate values reported by the manuscript.
The older scripts under `figures/`, `figures_v22/`, and `figures_supp/` are preserved as
design history. Some of those scripts expect the private raw experiment tree and are not
part of the self-contained reproduction path.
