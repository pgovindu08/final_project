"""
FoodRoute end-to-end pipeline.

Usage:
    python run_pipeline.py           # use cached data
    python run_pipeline.py --refresh # re-download everything
"""
import sys
sys.path.insert(0, "src")

from data_loading import load_walk_network, load_snap_stores
from graph        import Graph
from kdtree       import KDTree
from snapping     import filter_knox_stores, project_to_graph_crs, snap_points_to_graph
from dijkstra     import multi_source_dijkstra

REFRESH = "--refresh" in sys.argv


def main():
    # 1 -- Load walking network and SNAP retailers
    print("=== Loading data ===")
    G_ox    = load_walk_network(refresh=REFRESH)
    snap_df = load_snap_stores(refresh=REFRESH)

    # 2 -- Build our adjacency-list graph (no NetworkX downstream)
    print("=== Building graph ===")
    graph = Graph.from_osmnx(G_ox)
    print(f"Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

    # 3 -- Build k-d tree from projected node coordinates
    print("=== Building k-d tree ===")
    kdtree = KDTree.from_osmnx(G_ox)

    # 4 -- Snap SNAP stores to nearest graph node
    print("=== Snapping stores ===")
    knox_stores = filter_knox_stores(snap_df)
    print(f"Knox County SNAP supermarkets: {len(knox_stores):,}")
    graph_crs   = G_ox.graph.get("crs", "EPSG:32617")
    knox_stores = project_to_graph_crs(knox_stores, graph_crs)
    snapped     = snap_points_to_graph(knox_stores, kdtree)
    store_nodes = list(set(snapped["snapped_node_id"].tolist()))
    print(f"Unique snapped graph nodes: {len(store_nodes)}")

    # 5 -- Multi-source Dijkstra: label every intersection with nearest store
    print("=== Running multi-source Dijkstra ===")
    nearest_store, nearest_dist = multi_source_dijkstra(
        graph, store_nodes, heap_type="lazy"
    )
    reachable = [d for d in nearest_dist.values() if d < float("inf")]
    print(f"Reachable nodes : {len(reachable):,} / {graph.num_nodes:,}")
    if reachable:
        print(f"Max walking dist: {max(reachable):.2f} mi")

    print("\n=== Pipeline complete ===")
    return graph, nearest_store, nearest_dist


if __name__ == "__main__":
    main()
