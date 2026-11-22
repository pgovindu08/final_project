"""
Author: Tejas Srinivasan

Wires SN's Dijkstra implementations and PG's k-d tree into the benchmark
harness (benchmark.py at the repo root).  Run this module directly to
reproduce all timing experiments.

Tasks 3-4 from TS milestone rubric:
  Task 3: repeatable harness -- warm-up, median/min/max -> CSV
  Task 4: timing for all four Dijkstra variants + k-d tree across growing input sizes
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import run_suite           # repo-root harness
from graph     import Graph
from kdtree    import KDTree
from dijkstra  import repeated_dijkstra, multi_source_dijkstra


def _make_random_graph(n_nodes, n_edges, seed=0):
    rng   = random.Random(seed)
    g     = Graph()
    for i in range(n_nodes):
        g.add_node(i)
    nodes = list(range(n_nodes))
    for _ in range(n_edges):
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        w = rng.uniform(0.01, 5.0)
        g.add_edge(u, v, w)
    return g


def _make_random_kdtree(n_points, seed=0):
    rng    = random.Random(seed)
    points = [(i, rng.uniform(0, 1_000_000), rng.uniform(0, 1_000_000))
              for i in range(n_points)]
    return KDTree(points), points


# ---- benchmark wrappers (each takes exactly the args make_args returns) ----

def bench_repeated_lazy(graph, store_nodes):
    repeated_dijkstra(graph, store_nodes, heap_type="lazy")


def bench_repeated_indexed(graph, store_nodes):
    repeated_dijkstra(graph, store_nodes, heap_type="indexed")


def bench_multi_lazy(graph, store_nodes):
    multi_source_dijkstra(graph, store_nodes, heap_type="lazy")


def bench_multi_indexed(graph, store_nodes):
    multi_source_dijkstra(graph, store_nodes, heap_type="indexed")


def bench_kdtree_nearest(kdtree, queries):
    for qx, qy in queries:
        kdtree.nearest(qx, qy)


# ---- public entry points ----

def run_dijkstra_benchmarks(
    out_csv="benchmark_results_dijkstra.csv",
    node_counts=None,
    n_stores=10,
    edge_factor=3,
):
    """Time all four Dijkstra variants across graphs of increasing node count."""
    if node_counts is None:
        node_counts = [100, 250, 500, 1000, 2000]

    methods = {
        "repeated_lazy":    bench_repeated_lazy,
        "repeated_indexed": bench_repeated_indexed,
        "multi_lazy":       bench_multi_lazy,
        "multi_indexed":    bench_multi_indexed,
    }

    def make_args(n):
        g      = _make_random_graph(n, n * edge_factor)
        stores = list(g.nodes())[:n_stores]
        return (g, stores)

    run_suite(methods, node_counts, make_args, out_csv)


def run_kdtree_benchmarks(
    out_csv="benchmark_results_kdtree.csv",
    point_counts=None,
    n_queries=100,
):
    """Time k-d tree nearest-neighbor queries across increasing point-set sizes."""
    if point_counts is None:
        point_counts = [100, 500, 1000, 5000, 10000]

    rng = random.Random(42)

    def make_args(n):
        kdtree, _ = _make_random_kdtree(n)
        queries   = [(rng.uniform(0, 1_000_000), rng.uniform(0, 1_000_000))
                     for _ in range(n_queries)]
        return (kdtree, queries)

    run_suite({"kdtree_nearest": bench_kdtree_nearest}, point_counts, make_args, out_csv)


def plot_dijkstra_results(
    csv_path="benchmark_results_dijkstra.csv",
    out_png=None,
):
    """Read benchmark CSV and produce a linear + log-log complexity plot."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as _pd
    from pathlib import Path as _Path

    df = _pd.read_csv(csv_path)

    colors = {
        "repeated_lazy":    "#d73027",
        "repeated_indexed": "#fc8d59",
        "multi_lazy":       "#1a9850",
        "multi_indexed":    "#91cf60",
    }
    labels = {
        "repeated_lazy":    "Repeated (lazy heap)    O(N·(V+E)logV)",
        "repeated_indexed": "Repeated (indexed heap) O(N·(V+E)logV)",
        "multi_lazy":       "Multi-source (lazy)     O((V+E)logV)",
        "multi_indexed":    "Multi-source (indexed)  O((V+E)logV)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for method, grp in df.groupby("method"):
        grp = grp.sort_values("size")
        xs  = grp["size"].values
        ys  = grp["median_s"].values
        ax1.plot(xs, ys, "o-", color=colors.get(method, "gray"),
                 label=labels.get(method, method))
        ax2.loglog(xs, ys, "o-", color=colors.get(method, "gray"),
                   label=labels.get(method, method))

    # Theoretical curves (normalised to first data point of each family)
    sizes = np.array(sorted(df["size"].unique()), dtype=float)

    def theory_curve(ax, method_name, n_stores=1):
        grp = df[df["method"] == method_name].sort_values("size")
        if grp.empty:
            return
        v0, t0 = grp["size"].values[0], grp["median_s"].values[0]
        c = t0 / (n_stores * 4 * v0 * np.log(v0))
        t_theory = c * n_stores * 4 * sizes * np.log(sizes)
        ax.loglog(sizes, t_theory, "--",
                  color=colors.get(method_name, "gray"), alpha=0.35, linewidth=1)

    theory_curve(ax2, "repeated_lazy",  n_stores=10)
    theory_curve(ax2, "multi_lazy",     n_stores=1)

    for ax in (ax1, ax2):
        ax.set_xlabel("Graph nodes (V)")
        ax.set_ylabel("Median time (s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    ax1.set_title("Dijkstra timing — linear scale")
    ax2.set_title("Dijkstra timing — log-log  (dashed = theory)")
    fig.tight_layout()

    if out_png is None:
        out_png = _Path(csv_path).parent / "benchmark_dijkstra_plot.png"
    _Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_png}")
    plt.show()


def plot_kdtree_results(
    csv_path="benchmark_results_kdtree.csv",
    out_png=None,
):
    """Plot k-d tree nearest-neighbor timing with O(log n) theory overlay."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as _pd
    from pathlib import Path as _Path

    df   = _pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    grp  = df[df["method"] == "kdtree_nearest"].sort_values("size")
    xs   = grp["size"].values.astype(float)
    ys   = grp["median_s"].values

    ax.loglog(xs, ys, "o-", color="#2166ac", label="k-d tree nearest (measured)")

    # O(n_queries * log n) theory
    n_queries = 100
    c = ys[0] / (n_queries * np.log(xs[0]))
    ax.loglog(xs, c * n_queries * np.log(xs), "--", color="#2166ac",
              alpha=0.4, label=f"O({n_queries}·log n) theory")

    ax.set_xlabel("Number of points (n)")
    ax.set_ylabel("Median time (s)  for 100 queries")
    ax.set_title("k-d tree nearest-neighbor timing (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_png is None:
        out_png = _Path(csv_path).parent / "benchmark_kdtree_plot.png"
    _Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_png}")
    plt.show()


if __name__ == "__main__":
    print("=== Dijkstra benchmarks ===")
    run_dijkstra_benchmarks()
    print("\n=== k-d tree benchmarks ===")
    run_kdtree_benchmarks()
    print("\n=== Plotting results ===")
    plot_dijkstra_results()
    plot_kdtree_results()
