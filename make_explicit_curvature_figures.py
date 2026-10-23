"""Regenerate the manuscript comparison from recorded FEM data."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path,
                    default=ROOT / 'data/explicit_curvature/quadrature16/summary.json')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'images')
args = parser.parse_args()
rows = json.loads(args.data.read_text())
assert len(rows) == 12
assert all(not r['failed'] and r['S'] == 10000 for r in rows)
data = {(r['geometry_degree'], r['mode'], r['n']): r for r in rows}
assert set(data) == {(g, m, n) for g in (2, 3) for m in ('TH', 'TH-curvature') for n in (2, 4, 8)}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
series = [(2, 'TH', '#ad3946', 's', '--', 'Original TH, g=2'),
          (3, 'TH', '#cd7831', 'o', '-', 'Original TH, g=3'),
          (2, 'TH-curvature', '#236fa1', 's', '--', 'Corrected TH, g=2'),
          (3, 'TH-curvature', '#138477', 'o', '-', 'Corrected TH, g=3')]
for ax, field, title, ylabel in zip(
    axes, ('e_u_L2', 'Z_normal_L2'),
    ('Scalar distance error', 'Normal component'),
    (r'$E_u$', r'$E_n$')):
    for g, mode, color, marker, style, label in series:
        ax.plot([2, 4, 8], [data[g, mode, n][field] for n in (2, 4, 8)],
                color=color, marker=marker, linestyle=style, markersize=5,
                markerfacecolor='white' if g == 2 else color,
                linewidth=1.5, label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator([2, 4, 8]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Refinement index n')
    ax.set_ylabel(ylabel, rotation=0, labelpad=12)
    ax.set_title(title, fontsize=10)
    ax.grid(which='major', alpha=.2, linewidth=.6)
    ax.set_xlim(1.75, 9.15)
handles, labels = axes[0].get_legend_handles_labels()
# Put the two original formulations in the left column and corrected in right.
order = [0, 1, 2, 3]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc='lower center', ncol=2, frameon=False, fontsize=8.5,
           bbox_to_anchor=(.53, -.005), columnspacing=2.5, handlelength=2.5)
fig.subplots_adjust(left=.09, right=.985, top=.9, bottom=.27, wspace=.35)
target = args.output_dir
target.mkdir(parents=True, exist_ok=True)
fig.savefig(target / 'explicit_curvature_comparison.pdf', bbox_inches='tight', pad_inches=.08)
plt.close(fig)
