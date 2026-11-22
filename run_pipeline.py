"""
FoodRoute end-to-end pipeline.

Usage:
    python run_pipeline.py           # use cached data
    python run_pipeline.py --refresh # re-download everything
    python run_pipeline.py --no-map  # skip map (useful for quick data checks)
"""
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd
import geopandas as gpd

from data_loading   import load_walk_network, load_snap_stores, load_tract_demographics
from graph          import Graph
from kdtree         import KDTree
from snapping       import filter_knox_stores, project_to_graph_crs, snap_points_to_graph
from dijkstra       import multi_source_dijkstra
from classification import classify_all_tracts
from map            import build_map

ROOT             = Path(__file__).parent
PROCESSED        = ROOT / "data" / "processed"
KNOX_STATE_FIPS  = "47"
KNOX_COUNTY_FIPS = "093"
METERS_PER_MILE  = 1609.344

REFRESH = "--refresh" in sys.argv
NO_MAP  = "--no-map"  in sys.argv


# ---- census tract loader ----------------------------------------------------

def load_knox_tracts(refresh=False):
    """Download Knox County census tract geometries from US Census TIGER/Line."""
    path = PROCESSED / "knox_tracts.gpkg"
    if path.exists() and not refresh:
        return gpd.read_file(path)
    print("Downloading Knox County census tract geometries...")
    url = (f"https://www2.census.gov/geo/tiger/TIGER2023/TRACT/"
           f"tl_2023_{KNOX_STATE_FIPS}_tract.zip")
    gdf = gpd.read_file(url)
    gdf = gdf[gdf["COUNTYFP"] == KNOX_COUNTY_FIPS].copy().reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    print(f"Knox County tracts: {len(gdf)}")
    return gdf


# ---- main -------------------------------------------------------------------

def main():

    # 1 -- Load walking network and SNAP retailers
    print("=== Loading data ===")
    G_ox    = load_walk_network(refresh=REFRESH)
    snap_df = load_snap_stores(refresh=REFRESH)
    print(f"SNAP columns: {list(snap_df.columns[:8])} ...")   # debug column names

    # 2 -- Build adjacency-list graph (no NetworkX downstream)
    print("\n=== Building graph ===")
    graph_crs = G_ox.graph.get("crs", "EPSG:32617")
    graph     = Graph.from_osmnx(G_ox)
    print(f"Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

    # 3 -- Build k-d tree from projected graph node coordinates
    print("\n=== Building k-d tree ===")
    node_kdtree = KDTree.from_osmnx(G_ox)

    # 4 -- Snap SNAP stores to nearest graph node
    print("\n=== Snapping stores ===")
    knox_stores = filter_knox_stores(snap_df)

    # Fallback: if strict filter finds nothing, relax to state + county only
    if len(knox_stores) == 0:
        print("WARNING: strict store-type filter returned 0 -- relaxing to county only")
        df = snap_df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        state_col  = next((c for c in df.columns if c in {"state", "store_state"}), None)
        county_col = next((c for c in df.columns if "county" in c), None)
        lat_col    = next((c for c in df.columns if c in {"latitude", "lat"}), None)
        lon_col    = next((c for c in df.columns if c in {"longitude", "lon", "long"}), None)
        if state_col:  df = df[df[state_col].str.upper().str.strip() == "TN"]
        if county_col: df = df[df[county_col].str.upper().str.strip() == "KNOX"]
        if lat_col and lon_col:
            df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
            df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"])
        knox_stores = df.reset_index(drop=True)

    print(f"Knox County SNAP stores: {len(knox_stores)}")
    if len(knox_stores) == 0:
        print("ERROR: no stores found -- cannot run Dijkstra. Check SNAP data columns.")
        return

    knox_stores = project_to_graph_crs(knox_stores, graph_crs)
    snapped     = snap_points_to_graph(knox_stores, node_kdtree)
    store_nodes = list(set(snapped["snapped_node_id"].tolist()))
    print(f"Unique snapped graph nodes: {len(store_nodes)}")

    # 5 -- Multi-source Dijkstra
    print("\n=== Running multi-source Dijkstra ===")
    nearest_store, nearest_dist = multi_source_dijkstra(
        graph, store_nodes, heap_type="lazy"
    )
    reachable = [d for d in nearest_dist.values() if d < float("inf")]
    print(f"Reachable nodes : {len(reachable):,} / {graph.num_nodes:,}")
    if reachable:
        print(f"Max walking dist: {max(reachable):.2f} mi")
        print(f"Avg walking dist: {sum(reachable)/len(reachable):.2f} mi")

    if NO_MAP:
        print("\n=== Pipeline complete (map skipped) ===")
        return

    # 6 -- Load census tract geometries
    print("\n=== Loading census tracts ===")
    tracts_gdf = load_knox_tracts(refresh=REFRESH)

    # Project tract centroids to graph CRS for k-d tree queries
    centroids_proj = tracts_gdf.copy().to_crs(graph_crs)
    centroid_xy    = {
        row["GEOID"]: (centroids_proj.geometry.centroid.x.iloc[i],
                       centroids_proj.geometry.centroid.y.iloc[i])
        for i, row in tracts_gdf.iterrows()
    }

    # Snap tract centroids to graph nodes -> walking distance via nearest_dist
    centroid_df  = pd.DataFrame([
        {"GEOID": gid, "x": x, "y": y} for gid, (x, y) in centroid_xy.items()
    ])
    snapped_tracts = snap_points_to_graph(centroid_df, node_kdtree)

    # Build a k-d tree from store coordinates for straight-line distance
    store_points = [(i, row["x"], row["y"]) for i, (_, row) in enumerate(snapped.iterrows())]
    store_kdtree = KDTree(store_points)

    walking_dists       = {}
    straight_line_dists = {}
    for _, row in snapped_tracts.iterrows():
        gid     = row["GEOID"]
        node_id = row["snapped_node_id"]
        walking_dists[gid]       = nearest_dist.get(node_id, float("inf"))
        cx, cy                   = centroid_xy[gid]
        _, dist_m                = store_kdtree.nearest(cx, cy)
        straight_line_dists[gid] = dist_m / METERS_PER_MILE

    # 7 -- Load real demographic flags (poverty rate, population density)
    print("\n=== Loading tract demographics ===")
    demographics = load_tract_demographics(refresh=REFRESH)

    # Compute tract area in sq miles from projected geometry
    tracts_proj = tracts_gdf.to_crs(graph_crs)
    area_sqmi   = tracts_proj.geometry.area / (METERS_PER_MILE ** 2)

    tracts_meta = tracts_gdf[["GEOID"]].copy().rename(columns={"GEOID": "tract_id"})
    tracts_meta["area_sqmi"] = area_sqmi.values

    if demographics is not None:
        tracts_meta = tracts_meta.merge(demographics, on="tract_id", how="left")
        tracts_meta["pop_density"]   = tracts_meta["population"] / tracts_meta["area_sqmi"]
        # USDA proxies: low-income = poverty rate >= 20%;
        # urban = pop density >= 500 people/sq mile
        tracts_meta["is_low_income"] = tracts_meta["poverty_rate"].fillna(0) >= 0.20
        tracts_meta["is_urban"]      = tracts_meta["pop_density"].fillna(0)  >= 500
        li_count  = tracts_meta["is_low_income"].sum()
        urb_count = tracts_meta["is_urban"].sum()
        print(f"Low-income tracts: {li_count} / {len(tracts_meta)}")
        print(f"Urban tracts     : {urb_count} / {len(tracts_meta)}")
    else:
        # Fallback: Knox County is predominantly urban; no income data available
        tracts_meta["is_low_income"] = False
        tracts_meta["is_urban"]      = True
        print("Using defaults: all urban, no low-income flag (add CENSUS_API_KEY for real data)")

    # 8 -- Classify tracts
    print("\n=== Classifying tracts ===")

    access_df = classify_all_tracts(tracts_meta, walking_dists, straight_line_dists)
    n_low      = (access_df["walking_classification"] == "low_access").sum()
    n_disagree = access_df["disagrees"].sum()
    print(f"Low-access tracts  (walking)       : {n_low} / {len(access_df)}")
    print(f"Straight-line low-access tracts    : {(access_df['straight_line_classification'] == 'low_access').sum()} / {len(access_df)}")
    print(f"Tracts where methods disagree      : {n_disagree}")

    # 9 -- Load CDC PLACES health data (optional -- skip gracefully if unavailable)
    health_df = None
    try:
        from health import load_cdc_places, analyze_health_vs_access
        print("\n=== Loading CDC PLACES health data ===")
        health_df = load_cdc_places(refresh=REFRESH)
        analyze_health_vs_access(health_df, access_df)
    except Exception as e:
        print(f"Health data skipped: {e}")

    # 10 -- Build Folium map and open in browser
    print("\n=== Building map ===")
    out_html = ROOT / "results" / "map.html"
    build_map(tracts_gdf.to_crs("EPSG:4326"), access_df, health_df, out_html)
    print(f"Opening map in browser...")
    webbrowser.open(f"file://{out_html.resolve()}")

    print("\n=== Pipeline complete ===")
    return graph, nearest_store, nearest_dist, access_df


if __name__ == "__main__":
    main()
