"""
Unit tests for heaps and Dijkstra implementations (SN).
Reference implementations: NetworkX (multi_source_dijkstra_path_length) and
SciPy -- used here only, never inside our algorithms.
"""
import math
import random
import pytest

import networkx as nx

from graph    import Graph
from heaps    import BinaryHeap, IndexedHeap
from dijkstra import dijkstra_lazy, dijkstra_indexed, repeated_dijkstra, multi_source_dijkstra

INF       = math.inf
TOLERANCE = 1e-9


# ---- helpers ----------------------------------------------------------------

def make_toy_graph():
    """
    5-node graph:
      0 --1.0--> 1 --2.0--> 2
      0 --4.0--> 2
      1 --1.5--> 3
      3 --0.5--> 2
    Shortest 0->2: 0->1->3->2 = 3.0  (not the direct 4.0 edge).
    Node 4 is isolated (unreachable from 0).
    """
    g = Graph()
    for i in range(5):
        g.add_node(i)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 2.0)
    g.add_edge(0, 2, 4.0)
    g.add_edge(1, 3, 1.5)
    g.add_edge(3, 2, 0.5)
    return g


def make_random_graph(n, m, seed=0):
    rng   = random.Random(seed)
    g     = Graph()
    for i in range(n):
        g.add_node(i)
    nodes = list(range(n))
    for _ in range(m):
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        w = rng.uniform(0.01, 5.0)
        g.add_edge(u, v, w)
    return g


def to_networkx(g):
    """Convert to NetworkX DiGraph, keeping the minimum weight for parallel edges."""
    G = nx.DiGraph()
    G.add_nodes_from(g.nodes())
    best = {}
    for u in g.nodes():
        for v, w in g.neighbors(u):
            key = (u, v)
            if key not in best or w < best[key]:
                best[key] = w
    for (u, v), w in best.items():
        G.add_edge(u, v, weight=w)
    return G


def nx_single_source(g, source):
    G       = to_networkx(g)
    lengths = nx.single_source_dijkstra_path_length(G, source, weight="weight")
    return {node: lengths.get(node, INF) for node in g.nodes()}


def nx_multi_source(g, sources):
    G       = to_networkx(g)
    lengths = dict(nx.multi_source_dijkstra_path_length(G, sources, weight="weight"))
    return {node: lengths.get(node, INF) for node in g.nodes()}


# ---- BinaryHeap -------------------------------------------------------------

class TestBinaryHeap:
    def test_push_pop_order(self):
        h = BinaryHeap()
        h.push(3.0, "c"); h.push(1.0, "a"); h.push(2.0, "b")
        assert h.pop()[0] == 1.0
        assert h.pop()[0] == 2.0
        assert h.pop()[0] == 3.0

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            BinaryHeap().pop()

    def test_len(self):
        h = BinaryHeap()
        assert len(h) == 0
        h.push(1.0, "x")
        assert len(h) == 1
        h.pop()
        assert len(h) == 0

    def test_duplicates(self):
        h = BinaryHeap()
        h.push(2.0, "a"); h.push(2.0, "b"); h.push(1.0, "c")
        assert h.pop()[0] == 1.0
        assert h.pop()[0] == 2.0
        assert h.pop()[0] == 2.0

    def test_sorted_order_many(self):
        rng = random.Random(42)
        priorities = [rng.uniform(0, 100) for _ in range(200)]
        h = BinaryHeap()
        for i, p in enumerate(priorities):
            h.push(p, i)
        popped = [h.pop()[0] for _ in range(200)]
        assert popped == sorted(priorities)

    def test_is_empty(self):
        h = BinaryHeap()
        assert h.is_empty()
        h.push(1.0, "x")
        assert not h.is_empty()


# ---- IndexedHeap ------------------------------------------------------------

class TestIndexedHeap:
    def test_push_pop_order(self):
        h = IndexedHeap()
        h.push(3.0, "c"); h.push(1.0, "a"); h.push(2.0, "b")
        assert h.pop()[0] == 1.0
        assert h.pop()[0] == 2.0
        assert h.pop()[0] == 3.0

    def test_decrease_key(self):
        h = IndexedHeap()
        h.push(5.0, "a"); h.push(3.0, "b"); h.push(4.0, "c")
        h.decrease_key("a", 1.0)
        assert h.pop() == (1.0, "a")
        assert h.pop() == (3.0, "b")
        assert h.pop() == (4.0, "c")

    def test_contains(self):
        h = IndexedHeap()
        h.push(1.0, "x")
        assert h.contains("x")
        assert not h.contains("y")
        h.pop()
        assert not h.contains("x")

    def test_empty_raises(self):
        with pytest.raises(IndexError):
            IndexedHeap().pop()

    def test_matches_binary_heap_pop_order(self):
        """IndexedHeap pop order must match BinaryHeap on the same pushes."""
        rng   = random.Random(99)
        items = [(rng.uniform(0, 50), i) for i in range(100)]
        bh, ih = BinaryHeap(), IndexedHeap()
        for p, item in items:
            bh.push(p, item)
            ih.push(p, item)
        bh_order = [bh.pop()[0] for _ in range(100)]
        ih_order = [ih.pop()[0] for _ in range(100)]
        assert bh_order == ih_order

    def test_decrease_key_many_randomized(self):
        """After random decrease-keys the heap still pops in sorted order."""
        rng  = random.Random(7)
        vals = {i: rng.uniform(10, 100) for i in range(50)}
        h    = IndexedHeap()
        for item, p in vals.items():
            h.push(p, item)
        for item in rng.sample(range(50), 20):
            new_p      = rng.uniform(0, 10)
            vals[item] = new_p
            h.decrease_key(item, new_p)
        popped = []
        while not h.is_empty():
            popped.append(h.pop()[0])
        assert popped == sorted(popped)


# ---- Single-source Dijkstra -------------------------------------------------

class TestDijkstraLazy:
    def test_toy_graph(self):
        g    = make_toy_graph()
        dist = dijkstra_lazy(g, 0)
        assert abs(dist[0] - 0.0) < TOLERANCE
        assert abs(dist[1] - 1.0) < TOLERANCE
        assert abs(dist[2] - 3.0) < TOLERANCE   # via 0->1->3->2, not direct 4.0
        assert abs(dist[3] - 2.5) < TOLERANCE
        assert dist[4] == INF                    # isolated node

    def test_matches_networkx_random(self):
        g = make_random_graph(50, 150, seed=1)
        for source in [0, 5, 10]:
            our = dijkstra_lazy(g, source)
            ref = nx_single_source(g, source)
            for node in g.nodes():
                assert (our[node] == INF) == (ref[node] == INF), f"reachability mismatch node {node}"
                if our[node] < INF:
                    assert abs(our[node] - ref[node]) < TOLERANCE

    def test_single_node(self):
        g = Graph(); g.add_node(0)
        assert dijkstra_lazy(g, 0)[0] == 0.0


class TestDijkstraIndexed:
    def test_toy_graph(self):
        g    = make_toy_graph()
        dist = dijkstra_indexed(g, 0)
        assert abs(dist[0] - 0.0) < TOLERANCE
        assert abs(dist[1] - 1.0) < TOLERANCE
        assert abs(dist[2] - 3.0) < TOLERANCE
        assert abs(dist[3] - 2.5) < TOLERANCE
        assert dist[4] == INF

    def test_indexed_matches_lazy(self):
        g = make_random_graph(80, 200, seed=2)
        for source in [0, 3, 7]:
            lazy    = dijkstra_lazy(g, source)
            indexed = dijkstra_indexed(g, source)
            for node in g.nodes():
                assert (lazy[node] == INF) == (indexed[node] == INF)
                if lazy[node] < INF:
                    assert abs(lazy[node] - indexed[node]) < TOLERANCE


# ---- Multi-source Dijkstra --------------------------------------------------

class TestMultiSourceDijkstra:
    def test_toy_graph_lazy(self):
        g = make_toy_graph()
        _, dist = multi_source_dijkstra(g, [0, 4], heap_type="lazy")
        assert dist[0] == 0.0
        assert abs(dist[1] - 1.0) < TOLERANCE
        assert abs(dist[2] - 3.0) < TOLERANCE

    def test_multi_matches_repeated_lazy(self):
        g           = make_random_graph(60, 180, seed=3)
        store_nodes = list(range(5))
        _, d_multi  = multi_source_dijkstra(g, store_nodes, heap_type="lazy")
        _, d_rep    = repeated_dijkstra(g,    store_nodes, heap_type="lazy")
        for node in g.nodes():
            assert (d_multi[node] == INF) == (d_rep[node] == INF)
            if d_multi[node] < INF:
                assert abs(d_multi[node] - d_rep[node]) < TOLERANCE

    def test_multi_matches_repeated_indexed(self):
        g           = make_random_graph(60, 180, seed=4)
        store_nodes = list(range(5))
        _, d_multi  = multi_source_dijkstra(g, store_nodes, heap_type="indexed")
        _, d_rep    = repeated_dijkstra(g,    store_nodes, heap_type="indexed")
        for node in g.nodes():
            assert (d_multi[node] == INF) == (d_rep[node] == INF)
            if d_multi[node] < INF:
                assert abs(d_multi[node] - d_rep[node]) < TOLERANCE

    def test_multi_lazy_matches_networkx(self):
        g           = make_random_graph(60, 180, seed=5)
        store_nodes = list(range(5))
        _, d_multi  = multi_source_dijkstra(g, store_nodes, heap_type="lazy")
        ref         = nx_multi_source(g, store_nodes)
        for node in g.nodes():
            assert (d_multi[node] == INF) == (ref[node] == INF), f"reachability mismatch node {node}"
            if d_multi[node] < INF:
                assert abs(d_multi[node] - ref[node]) < TOLERANCE

    def test_multi_indexed_matches_networkx(self):
        g           = make_random_graph(60, 180, seed=6)
        store_nodes = list(range(5))
        _, d_multi  = multi_source_dijkstra(g, store_nodes, heap_type="indexed")
        ref         = nx_multi_source(g, store_nodes)
        for node in g.nodes():
            assert (d_multi[node] == INF) == (ref[node] == INF), f"reachability mismatch node {node}"
            if d_multi[node] < INF:
                assert abs(d_multi[node] - ref[node]) < TOLERANCE

    def test_tie_determinism(self):
        """Two stores equidistant to a node -- result must be stable across calls."""
        g = Graph()
        for i in range(4):
            g.add_node(i)
        g.add_edge(0, 2, 1.0)
        g.add_edge(1, 2, 1.0)
        _, dist1 = multi_source_dijkstra(g, [0, 1], heap_type="lazy")
        _, dist2 = multi_source_dijkstra(g, [0, 1], heap_type="lazy")
        assert dist1[2] == dist2[2]


# ---- Edge cases -------------------------------------------------------------

class TestEdgeCases:
    def test_disconnected_graph(self):
        g = Graph()
        for i in range(6):
            g.add_node(i)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        _, dist = multi_source_dijkstra(g, [0], heap_type="lazy")
        assert dist[0] == 0.0
        assert dist[3] == INF
        assert dist[4] == INF

    def test_duplicate_store_nodes(self):
        """Passing the same store node twice must not crash or double-count."""
        g = Graph()
        for i in range(3):
            g.add_node(i)
        g.add_edge(0, 1, 1.0)
        g.add_edge(0, 2, 2.0)
        _, dist = multi_source_dijkstra(g, [0, 0], heap_type="lazy")
        assert dist[0] == 0.0

    def test_zero_length_edges(self):
        g = Graph()
        for i in range(3):
            g.add_node(i)
        g.add_edge(0, 1, 0.0)
        g.add_edge(1, 2, 1.0)
        dist = dijkstra_lazy(g, 0)
        assert dist[1] == 0.0
        assert dist[2] == 1.0

    def test_empty_store_list(self):
        g = make_toy_graph()
        _, dist = multi_source_dijkstra(g, [], heap_type="lazy")
        assert all(d == INF for d in dist.values())

    def test_single_node_graph(self):
        g = Graph()
        g.add_node(42)
        _, dist = multi_source_dijkstra(g, [42], heap_type="lazy")
        assert dist[42] == 0.0
