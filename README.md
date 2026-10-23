# Surface Eikonal Taylor--Hood comparison

- `main.tex`: report.
- `code/`: original TH, explicit-curvature TH, and reproduction scripts.
- `data/explicit_curvature/quadrature16/`: the twelve reported runs.
- `images/explicit_curvature_comparison.pdf`: the figure used in the report.

From this directory, in a DOLFINx environment with Matplotlib:

```bash
EIKONAL_PYTHON=python bash code/reproduce.sh
python make_explicit_curvature_figures.py --data data/reproduced/summary.json
```

Build the report by running `pdflatex main.tex` twice.
