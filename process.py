#!/usr/bin/env python3
"""
Process batched TTV benchmark results and emit TikZ/pgfplots figures.
Usage: python3 process.py <results.csv> > figures.tex

Figures produced:
  1. Speedup over serial at n=1000, all six (m,n) cases — OMP-32 vs GPU
  2. OMP thread scaling at n=1000 (log-log), best vs worst case
  3. Size scaling log-log for case (m=2,n=1) — Serial, OMP-32, GPU
     with a vertical annotation at the 8 GB VRAM boundary (~n=1260)
"""
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


def coords(pts):
    return ' '.join(f'({x},{y:.6g})' for x, y in pts if y is not None)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    d = load(path)

    cases   = ['m1n3', 'm1n2', 'm2n3', 'm3n2', 'm2n1', 'm3n1']
    threads = [1, 2, 4, 8, 16, 32]
    sizes   = [500, 750, 1000, 1250, 1500, 1750, 2000]
    n0 = 1000

    # ── Figure 1: speedup over serial, all cases at n=1000 ───────────
    f1_omp, f1_gpu = [], []
    for c in cases:
        s = get(d, 'Serial',    c, n0)
        o = get(d, 'OpenMP-32', c, n0)
        g = get(d, 'GPU',       c, n0)
        if s and o and g:
            f1_omp.append((c, s / o))
            f1_gpu.append((c, s / g))

    # ── Figure 2: OMP thread scaling at n=1000, m1n3 vs m3n1 ────────
    f2 = {}
    for c in ['m1n3', 'm3n1']:
        s = get(d, 'Serial', c, n0)
        f2[c] = [(t, s / get(d, f'OpenMP-{t}', c, n0))
                 for t in threads
                 if s and get(d, f'OpenMP-{t}', c, n0)]

    # ── Figure 3: size scaling for case (m=2, n=1) ───────────────────
    c3 = 'm2n1'
    f3_gpu = [(n, get(d, 'GPU',       c3, n)) for n in sizes]
    f3_omp = [(n, get(d, 'OpenMP-32', c3, n)) for n in sizes]
    f3_ser = [(n, get(d, 'Serial',    c3, n)) for n in [500, 750, 1000]]

    # ── Emit ─────────────────────────────────────────────────────────
    print(r"""% Requires: \usepackage{pgfplots}  \pgfplotsset{compat=1.18}
""")

    # Figure 1 --------------------------------------------------------
    print(r"""% ── Figure 1: Speedup over serial at n=1000 ─────────────────
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=8pt,
    symbolic x coords={m1n3,m1n2,m2n3,m3n2,m2n1,m3n1},
    xtick=data,
    xticklabels={$(1,3)$,$(1,2)$,$(2,3)$,$(3,2)$,$(2,1)$,$(3,1)$},
    xlabel={Contraction case $(m,n)$},
    ylabel={Speedup over serial},
    legend pos=north east,
    width=0.9\linewidth,
    height=6.5cm,
    ymajorgrids=true,
    ymin=0,
    enlarge x limits=0.15,
]""")
    print(f'\\addplot[fill=orange!70,draw=orange!90!black] coordinates {{ {coords(f1_omp)} }};')
    print(r'\addlegendentry{OMP-32}')
    print(f'\\addplot[fill=teal!60,draw=teal!80!black] coordinates {{ {coords(f1_gpu)} }};')
    print(r'\addlegendentry{GPU}')
    print(r"""\end{axis}
\end{tikzpicture}

""")

    # Figure 2 --------------------------------------------------------
    print(r"""% ── Figure 2: OMP thread scaling at n=1000 (log-log) ────────
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
\addlegendentry{Ideal}""")
    print(f'\\addplot[mark=o,blue,thick] coordinates {{ {coords(f2["m1n3"])} }};')
    print(r'\addlegendentry{$(m,n){=}(1,3)$}')
    print(f'\\addplot[mark=square*,red,thick] coordinates {{ {coords(f2["m3n1"])} }};')
    print(r'\addlegendentry{$(m,n){=}(3,1)$}')
    print(r"""\end{axis}
\end{tikzpicture}

""")

    # Figure 3 --------------------------------------------------------
    print(r"""% ── Figure 3: Size scaling for case (m=2,n=1), log-log ──────
% Vertical line marks the 8 GB VRAM limit: n^3*4 bytes = 8 GB -> n~1260
\begin{tikzpicture}
\begin{axis}[
    xmode=log, ymode=log,
    xlabel={Tensor mode size $n$ \, (cubic $n\times n\times n$)},
    ylabel={Time (s)},
    legend pos=north west,
    width=0.9\linewidth,
    height=6.5cm,
    xmin=400, xmax=2500,
    ymajorgrids=true, xmajorgrids=true,
]""")
    print(f'\\addplot[mark=triangle*,green!60!black,thick] coordinates {{ {coords(f3_gpu)} }};')
    print(r'\addlegendentry{GPU}')
    print(f'\\addplot[mark=square*,orange!80!black,thick] coordinates {{ {coords(f3_omp)} }};')
    print(r'\addlegendentry{OMP-32}')
    print(f'\\addplot[mark=o,blue,thick] coordinates {{ {coords(f3_ser)} }};')
    print(r'\addlegendentry{Serial}')
    print(r"""\addplot[densely dashed,gray,no marks,thick] coordinates {(1260,0.01) (1260,200)};
\node[gray,rotate=90,font=\small] at (axis cs:1120,0.07) {8\,GB VRAM limit};
\end{axis}
\end{tikzpicture}""")


if __name__ == '__main__':
    main()
