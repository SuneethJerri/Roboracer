#!/usr/bin/env python3
"""The README's two figures, each written with the CSV it was drawn from.

    python3 plot_results.py            # writes results/fig{1,2}_*.{svg,csv}
    python3 plot_results.py --selftest

Fig 1: SPSA testbed NMSE against noise, per perturbation family (`spsa_testbed.py --sweep`).
Fig 2: circuits finished out of 17 per seed, constrained SPSA against black-box search, read
from the evaluation run logs, whose bytes `runlog.py verify` checks.
"""
import csv
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results')
SWEEP = os.path.join(ROOT, 'scratch/outreach/sweep_quadratic.csv')
TABLE_I = 1.601e-5                    # Chandramouli et al., hadamard-paper at sigma = 0
SIGMA0_AT = 10 ** -4.6                # where sigma = 0 is drawn on the log axis
ARMS = [                              # label, evaluation run
    ('Constrained SPSA', '20261019T114027Z_o19-eval-c3_p7'),
    ('CMA-ES', '20261020T072116Z_o27-eval-cma4_p7'),
    ('CMA-ES, circuits redrawn\nevery generation', '20261020T092435Z_o28-eval-cma4r_p7'),
    ('Random search', '20261019T114155Z_o19-eval-rand4_p7'),
]
SEED_ROW = re.compile(r'^(s\d+)\s+\S+\s+\S+\s+(\d+)/17\s')


def sweep_medians(rows):
    """{family: (sigmas, q25, median, q75)} over the replications at each sigma."""
    by = {}
    for r in rows:
        by.setdefault(r['family'], {}).setdefault(float(r['sigma']), []).append(float(r['nmse']))
    out = {}
    for fam, d in by.items():
        s = sorted(d)
        q = np.array([np.percentile(d[k], [25, 50, 75]) for k in s])
        out[fam] = (np.array(s), q[:, 0], q[:, 1], q[:, 2])
    return out


def seed_counts(log_text):
    """[(seed, finished)] from an eval_runs.py log, in the order printed."""
    return [(m[1], int(m[2])) for m in map(SEED_ROW.match, log_text.splitlines()) if m]


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), metadata={'Date': None}, bbox_inches='tight')


def fig1(plt):
    from spsa_testbed import FAMILIES
    rows = list(csv.DictReader(open(SWEEP)))
    med = sweep_medians(rows)
    with open(os.path.join(OUT, 'fig1_sweep.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'sigma', 'nmse_q25', 'nmse_median', 'nmse_q75'])
        for fam in FAMILIES:
            for row in zip(*med[fam]):
                w.writerow([fam] + [f'{x:.6g}' for x in row])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fam in FAMILIES:
        s, lo, m, hi = med[fam]
        x = np.where(s == 0, SIGMA0_AT, s)
        ax.plot(x, m, marker='o', ms=3.5, label=fam)
        ax.fill_between(x, lo, hi, alpha=0.15)
    ax.plot([SIGMA0_AT], [TABLE_I], 'k*', ms=11, label='Table I (Chandramouli et al.)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = [SIGMA0_AT, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    ax.set_xticks(ticks, ['0', '1e-4', '1e-3', '1e-2', '0.1', '1'])
    ax.minorticks_off()
    ax.set_xlabel('observation noise σ')
    ax.set_ylabel(f'NMSE after {rows[0]["budget"]} evaluations (median, IQR)')
    ax.set_title(f'SPSA on the p = {rows[0]["p"]} quadratic, {len(rows) // len(med) // len(s)} '
                 'reps per point', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, 'fig1_sweep.svg')


def fig2(plt):
    counts = [(label, seed_counts(open(os.path.join(ROOT, 'runs', run, 'log.txt')).read()))
              for label, run in ARMS]
    with open(os.path.join(OUT, 'fig2_reliability.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['arm', 'eval_run', 'seed', 'finished_of_17'])
        for (label, run), (_, cs) in zip(ARMS, counts):
            for seed, n in cs:
                w.writerow([label.replace('\n', ' '), run, seed, n])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (label, cs) in enumerate(counts):
        n = np.array([c for _, c in cs])
        ax.scatter(n, i + np.linspace(-0.12, 0.12, len(n)), s=36, zorder=3)
        ax.plot([n.mean()] * 2, [i - 0.25, i + 0.25], 'k-', lw=1.5)
    ax.set_yticks(range(len(counts)), [lab for lab, _ in counts])
    ax.invert_yaxis()
    ax.set_xlim(0, 17.5)
    ax.set_xticks([0, 5, 10, 15, 17])
    ax.set_xlabel('training circuits finished, of 17 (dots: seeds; bar: mean)')
    ax.set_title('Same circuits, same rollout budget, p = 7', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    _save(fig, 'fig2_reliability.svg')


def main():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = 'plot_results'     # stable ids, so stable bytes
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig1(plt)
    fig2(plt)
    print(f'wrote {sorted(os.listdir(OUT))}')


def selftest():
    rows = [{'family': 'f', 'sigma': '0.0', 'nmse': v} for v in ('1', '2', '3')]
    s, lo, m, hi = sweep_medians(rows)['f']
    assert list(s) == [0.0] and m[0] == 2 and lo[0] == 1.5 and hi[0] == 2.5
    log = 'run  iters lambda feasible mean|feas\ns0   1   nan   13/17   0.5476\ns1  200 1.58  17/17  0.68\n'
    assert seed_counts(log) == [('s0', 13), ('s1', 17)]
    assert seed_counts('  s0           feasible   nan/17 []') == [], 'the grouped block is not a row'
    print('selftest OK')


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    selftest() if '--selftest' in sys.argv else main()
