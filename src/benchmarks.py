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


if __name__ == "__main__":
    print("=== Dijkstra benchmarks ===")
    run_dijkstra_benchmarks()
    print("\n=== k-d tree benchmarks ===")
    run_kdtree_benchmarks()
