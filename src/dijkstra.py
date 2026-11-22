"""
Author: Srinath Narayanan

Three Dijkstra variants over our Graph class:

  dijkstra_lazy         -- single-source, binary heap with lazy deletion
  dijkstra_indexed      -- single-source, indexed heap with true decrease-key
  repeated_dijkstra     -- run single-source once per store (O(N(V+E) log V) baseline)
  multi_source_dijkstra -- single-pass multi-source (O((V+E) log V))

All accept heap_type="lazy"|"indexed" so Tejas's benchmark harness can call
the same function signature for both heaps.

Note on direction: OSMnx walking networks include edges in both directions, so
running multi-source FROM stores (finding distances from stores to intersections)
gives the same distances as running from intersections to stores.
"""

import math
from heaps import BinaryHeap, IndexedHeap

INF = math.inf


def dijkstra_lazy(graph, source):
    """Single-source Dijkstra with lazy-deletion binary heap."""
    dist = {node: INF for node in graph.nodes()}
    dist[source] = 0.0
    heap = BinaryHeap()
    heap.push(0.0, source)

    while not heap.is_empty():
        d, u = heap.pop()
        if d > dist[u]:       # stale entry -- a shorter path was found later
            continue
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heap.push(nd, v)

    return dist


def dijkstra_indexed(graph, source):
    """Single-source Dijkstra with indexed heap and true decrease-key."""
    dist = {node: INF for node in graph.nodes()}
    dist[source] = 0.0
    heap = IndexedHeap()
    heap.push(0.0, source)

    while not heap.is_empty():
        d, u = heap.pop()
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                if heap.contains(v):
                    heap.decrease_key(v, nd)
                else:
                    heap.push(nd, v)

    return dist


def repeated_dijkstra(graph, store_nodes, heap_type="lazy"):
    """
    Baseline: run single-source Dijkstra once per store, then take per-node minimums.
    Complexity: O(N * (V+E) log V)  where N = number of stores.

    Returns
    -------
    nearest_store : dict {node_id -> store_id}
    nearest_dist  : dict {node_id -> distance_miles}  (INF if unreachable)
    """
    dijkstra_fn = dijkstra_indexed if heap_type == "indexed" else dijkstra_lazy

    nearest_store = {}
    nearest_dist  = {node: INF for node in graph.nodes()}

    for store_id in store_nodes:
        if store_id not in graph.nodes():
            continue
        dists = dijkstra_fn(graph, store_id)
        for node, d in dists.items():
            if d < nearest_dist[node]:
                nearest_dist[node]  = d
                nearest_store[node] = store_id

    return nearest_store, nearest_dist


def multi_source_dijkstra(graph, store_nodes, heap_type="lazy"):
    """
    Single-pass multi-source Dijkstra.  All stores are inserted at distance 0 and
    the wavefront propagates outward, labeling every reachable intersection with
    (nearest_store_id, walking_distance_miles).
    Complexity: O((V+E) log V) -- one pass regardless of the number of stores.

    Ties (two stores equidistant to a node) are broken by heap pop order,
    giving a deterministic but arbitrary nearest-store assignment.

    Returns
    -------
    nearest_store : dict {node_id -> store_id}
    nearest_dist  : dict {node_id -> distance_miles}  (INF if unreachable)
    """
    dist          = {node: INF for node in graph.nodes()}
    nearest_store = {}

    heap = IndexedHeap() if heap_type == "indexed" else BinaryHeap()

    # Push each unique store at distance 0
    seen = set()
    for store_id in store_nodes:
        if store_id in graph.nodes() and store_id not in seen:
            seen.add(store_id)
            dist[store_id]          = 0.0
            nearest_store[store_id] = store_id
            heap.push(0.0, store_id)

    while not heap.is_empty():
        d, u = heap.pop()
        if heap_type == "lazy" and d > dist[u]:    # skip stale lazy entries
            continue
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist[v]:
                dist[v]          = nd
                nearest_store[v] = nearest_store[u]
                if heap_type == "indexed":
                    if heap.contains(v):
                        heap.decrease_key(v, nd)
                    else:
                        heap.push(nd, v)
                else:
                    heap.push(nd, v)

    return nearest_store, dist
