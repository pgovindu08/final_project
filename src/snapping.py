"""
Author: Pranav Govindu

Snap SNAP store addresses and census tract centroids to the nearest road-network
node using the k-d tree.  Returns straight-line distances (meters and miles)
for later comparison with Dijkstra walking distances.
"""

import pandas as pd
import geopandas as gpd

METERS_PER_MILE = 1609.344
KNOX_STATE      = "TN"
KNOX_COUNTY     = "KNOX"


def filter_knox_stores(snap_df):
    """
    Filter SNAP retailers DataFrame to Knox County, TN supermarkets.
    Column names are normalised because the USDA CSV format varies by release.
    Returns a DataFrame with at minimum 'latitude' and 'longitude' columns.
    """
    df = snap_df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    state_col  = next((c for c in df.columns if c in {"state", "store_state"}), None)
    county_col = next((c for c in df.columns if "county" in c), None)
    type_col   = next((c for c in df.columns if "store_type" in c or c == "type"), None)
    lat_col    = next((c for c in df.columns if c in {"latitude", "lat"}), None)
    lon_col    = next((c for c in df.columns if c in {"longitude", "lon", "long"}), None)

    if state_col:
        df = df[df[state_col].str.upper().str.strip() == KNOX_STATE]
    if county_col:
        df = df[df[county_col].str.upper().str.strip() == KNOX_COUNTY]
    if type_col:
        supermarket_types = {"SUPERMARKET", "SUPER STORE", "LARGE GROCERY STORE"}
        df = df[df[type_col].str.upper().str.strip().isin(supermarket_types)]

    if lat_col and lon_col:
        df = df.dropna(subset=[lat_col, lon_col])
        df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])

    return df.reset_index(drop=True)


def project_to_graph_crs(df, graph_crs="EPSG:32617"):
    """
    Project a DataFrame with 'latitude'/'longitude' (WGS-84) to the graph CRS.
    Adds 'x' and 'y' columns in meters.  graph_crs may be a string or pyproj CRS.
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(graph_crs)
    out = df.copy()
    out["x"] = gdf.geometry.x.values
    out["y"] = gdf.geometry.y.values
    return out


def snap_points_to_graph(points_df, kdtree):
    """
    Snap each point (with 'x', 'y' in graph CRS) to its nearest graph node.

    Returns the input DataFrame with three extra columns:
        snapped_node_id   -- OSMnx node ID of the nearest graph node
        snap_dist_meters  -- straight-line distance to that node (meters)
        snap_dist_miles   -- same distance in miles
    """
    node_ids, dists_m = [], []
    for _, row in points_df.iterrows():
        nid, dm = kdtree.nearest(row["x"], row["y"])
        node_ids.append(nid)
        dists_m.append(dm)

    out = points_df.copy()
    out["snapped_node_id"]  = node_ids
    out["snap_dist_meters"] = dists_m
    out["snap_dist_miles"]  = [d / METERS_PER_MILE for d in dists_m]
    return out
