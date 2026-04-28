#!/usr/bin/env python3
"""
Process batched TTV benchmark results and emit TikZ/pgfplots figures.
Usage: python3 process.py <results.csv> [fig_dir]

Writes to fig_dir (default: fig/):
  fig1.dat, fig1.tex   — speedup over serial at n=1000, all six (m,n) cases
  fig2.dat, fig2.tex   — OMP thread scaling at n=1000 (log-log)
  fig3.dat, fig3.tex   — size scaling for case (m=2,n=1) with VRAM annotation
"""
import os
import sys
from collections import defaultdict


def load(path):
    d = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            method, case, I, J, K, t = parts
            d[(method, case, int(I))].append(float(t))
    return {k: sum(v) / len(v) for k, v in d.items()}


def get(d, method, case, n):
    return d.get((method, case, n))


def fmt(v):
    if v is None:
        return 'nan'
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def write_table(path, header, rows):
    with open(path, 'w') as f:
        f.write(' '.join(header) + '\n')
        for row in rows:
            f.write(' '.join(fmt(v) for v in row) + '\n')


def main():
    path    = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    fig_dir = sys.argv[2] if len(sys.argv) > 2 else 'fig'
    os.makedirs(fig_dir, exist_ok=True)
    d = load(path)

    cases   = ['m1n3', 'm1n2', 'm2n3', 'm3n2', 'm2n1', 'm3n1']
    threads = [1, 2, 4, 8, 16, 32]
    sizes   = [500, 750, 1000, 1250, 1500, 1750, 2000]
    n0 = 1000
    c3 = 'm3n1'

    # ── Figure 1: speedup over serial, all cases at n=1000 ───────────
    rows1 = []
    for c in cases:
        s = get(d, 'Serial',    c, n0)
        o = get(d, 'OpenMP-32', c, n0)
        g = get(d, 'GPU',       c, n0)
        rows1.append((c,
                      s / o if s and o else None,
                      s / g if s and g else None))
    write_table(os.path.join(fig_dir, 'fig1.dat'),
                ['case', 'omp', 'gpu'], rows1)

    # ── Figure 2: OMP thread scaling at n=1000, m1n3 vs m3n1 ────────
    rows2 = []
    s_a = get(d, 'Serial', 'm1n3', n0)
    s_b = get(d, 'Serial', 'm3n1', n0)
    for t in threads:
        o_a = get(d, f'OpenMP-{t}', 'm1n3', n0)
        o_b = get(d, f'OpenMP-{t}', 'm3n1', n0)
        rows2.append((t,
                      s_a / o_a if s_a and o_a else None,
                      s_b / o_b if s_b and o_b else None))
    write_table(os.path.join(fig_dir, 'fig2.dat'),
                ['threads', 'm1n3', 'm3n1'], rows2)

    # ── Figure 3: size scaling for case (m=3, n=1) ───────────────────
    rows3 = []
    for n in sizes:
        rows3.append((n,
                      get(d, 'GPU',       c3, n),
                      get(d, 'OpenMP-32', c3, n),
                      get(d, 'Serial',    c3, n)))
    write_table(os.path.join(fig_dir, 'fig3.dat'),
                ['n', 'gpu', 'omp', 'ser'], rows3)

    # ── Emit ─────────────────────────────────────────────────────────
    # Figure 1 --------------------------------------------------------
    with open(os.path.join(fig_dir, 'fig1.tex'), 'w') as f:
        f.write(r"""% ── Figure 1: Speedup over serial at n=1000 ─────────────────
% Requires: \usepackage{pgfplots}  \pgfplotsset{compat=1.18}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=8pt,
    symbolic x coords={m1n3,m1n2,m2n3,m3n2,m2n1,m3n1},
    xtick=data,
    xticklabels={{$(1,3)$},{$(1,2)$},{$(2,3)$},{$(3,2)$},{$(2,1)$},{$(3,1)$}},
    xlabel={Contraction case},
    ylabel={Speedup over serial},
    legend pos=north east,
    width=0.9\linewidth,
    height=6.5cm,
    ymajorgrids=true,
    ymin=0,
    enlarge x limits=0.15,
]
\addplot[fill=orange!70,draw=orange!90!black] table[x=case,y=omp] {fig/fig1.dat};
\addlegendentry{OMP-32}
\addplot[fill=teal!60,draw=teal!80!black] table[x=case,y=gpu] {fig/fig1.dat};
\addlegendentry{GPU}
\end{axis}
\end{tikzpicture}
""")

    # Figure 2 --------------------------------------------------------
    with open(os.path.join(fig_dir, 'fig2.tex'), 'w') as f:
        f.write(r"""% ── Figure 2: OMP thread scaling at n=1000 (log-log) ────────
% Requires: \usepackage{pgfplots}  \pgfplotsset{compat=1.18}
\begin{tikzpicture}
\begin{axis}[
    xmode=log, ymode=log,
    log basis x=2, log basis y=2,
    xtick={1,2,4,8,16,32},
    xticklabels={1,2,4,8,16,32},
    xlabel={OMP threads},
    ylabel={Speedup over serial},
    legend pos=north west,
    width=0.9\linewidth,
    height=6.5cm,
    xmin=0.8, xmax=48,
    ymin=0.5, ymax=48,
    ymajorgrids=true, xmajorgrids=true,
]
\addplot[dashed,gray,thick,domain=1:32,samples=2] {x};
\addlegendentry{Ideal}
\addplot[mark=o,blue,thick] table[x=threads,y=m1n3] {fig/fig2.dat};
\addlegendentry{$m{=}1,\,n{=}3$}
\addplot[mark=square*,red,thick] table[x=threads,y=m3n1] {fig/fig2.dat};
\addlegendentry{$m{=}3,\,n{=}1$}
\end{axis}
\end{tikzpicture}
""")

    # Figure 3 --------------------------------------------------------
    with open(os.path.join(fig_dir, 'fig3.tex'), 'w') as f:
        f.write(r"""% ── Figure 3: Size scaling for case (m=3,n=1), log-log ──────
% Requires: \usepackage{pgfplots}  \pgfplotsset{compat=1.18}
% Vertical line marks the 8 GB VRAM limit: n^3*4 bytes = 8 GB -> n~1260
\begin{tikzpicture}
\begin{axis}[
    xmode=log, ymode=log,
    xlabel={Tensor mode size $n$ \, (cubic $n\times n\times n$)},
    ylabel={'Kernel' Time (s)},
    legend pos=south east,
    width=0.9\linewidth,
    height=6.5cm,
    xmin=400, xmax=2500,
    xtick={500,1000,1500,2000},
    xticklabels={500,1000,1500,2000},
    ymajorgrids=true, xmajorgrids=true,
    unbounded coords=discard,
]
\addplot[mark=triangle*,green!60!black,thick] table[x=n,y=gpu] {fig/fig3.dat};
\addlegendentry{GPU}
\addplot[mark=square*,orange!80!black,thick] table[x=n,y=omp] {fig/fig3.dat};
\addlegendentry{OMP-32}
\addplot[mark=o,blue,thick] table[x=n,y=ser] {fig/fig3.dat};
\addlegendentry{Serial}
\end{axis}
\end{tikzpicture}
""")


if __name__ == '__main__':
    main()
