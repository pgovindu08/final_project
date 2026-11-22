"""
Author: Pranav Govindu

Grabs the Knox County walking network from OSMnx and projects it to meters
so the k-d tree and Dijkstra code downstream don't have to deal with lat/lon.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import osmnx as ox
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PLACE = "Knox County, Tennessee, USA"
CRS_METERS = "EPSG:32617"  # UTM zone 17N -- meters, not degrees

SNAP_URL = ("https://www.fna.usda.gov/sites/default/files/resource-files/"
            "snap-retailer-locator-data2005-2025.zip")

# Loads the Knox County walking network, projected to meters.
def load_walk_network(refresh=False):
    # First run downloads it from OSM (slow); after that it just reads the cached file.
    path = PROCESSED / "knox_walk.graphml"
    if path.exists() and not refresh:
        G = ox.load_graphml(path)
    else:
        print(f"Downloading walking network for {PLACE} (this can take a few minutes)")
        G = ox.graph_from_place(PLACE, network_type="walk")
        G = ox.project_graph(G, to_crs=CRS_METERS)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, path)
    print(f"Walking network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G



# Downloads the SNAP retailer list and reads the CSV out of the zip. No filtering yet --
# that happens later once we actually need just the Knox County supermarkets.
def load_snap_stores(refresh=False):
    path = RAW / "snap_retailers_2005_2025.zip"
    if not path.exists() or refresh:
        print("Downloading SNAP retailer data (big file, give it a minute)")
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SNAP_URL, headers={"User-Agent": "FoodRoute student project"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())

    with zipfile.ZipFile(path) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(csv_name)
    df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
    print(f"SNAP retailers: {len(df):,} rows (not filtered to Knox County yet)")
    return df


def load_tract_demographics(api_key=None, refresh=False):
    """
    Pull poverty rate and population for Knox County census tracts from ACS 5-year.

    Requires a free Census API key from https://api.census.gov/data/key_signup.html
    Pass it as api_key=, or set the CENSUS_API_KEY environment variable.

    Returns DataFrame with columns:
        tract_id      -- 11-digit FIPS string
        poverty_rate  -- fraction of population below poverty level (0-1)
        population    -- total population estimate
    Returns None if no API key is available.
    """
    import json
    import os

    key = api_key or os.environ.get("CENSUS_API_KEY", "")
    if not key:
        print("NOTE: No CENSUS_API_KEY found -- skipping real demographics.")
        print("      Get a free key at https://api.census.gov/data/key_signup.html")
        print("      Then: export CENSUS_API_KEY=your_key  (or pass api_key=)")
        return None

    path = PROCESSED / "knox_tract_demographics.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"tract_id": str})

    print("Downloading Knox County tract demographics from Census ACS 5-year...")
    # B17001_001E = poverty universe total, B17001_002E = below poverty level
    # B01003_001E = total population
    url = (
        "https://api.census.gov/data/2022/acs/acs5"
        "?get=B17001_001E,B17001_002E,B01003_001E"
        f"&for=tract:*&in=state:47+county:093&key={key}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "FoodRoute student project"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)

    cols = data[0]
    df   = pd.DataFrame(data[1:], columns=cols)

    df["B17001_001E"] = pd.to_numeric(df["B17001_001E"], errors="coerce")
    df["B17001_002E"] = pd.to_numeric(df["B17001_002E"], errors="coerce")
    df["B01003_001E"] = pd.to_numeric(df["B01003_001E"], errors="coerce")

    df["tract_id"]     = (df["state"] + df["county"] + df["tract"]).str.zfill(11)
    df["poverty_rate"] = df["B17001_002E"] / df["B17001_001E"]
    df["population"]   = df["B01003_001E"]

    result = df[["tract_id", "poverty_rate", "population"]].dropna()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"Tract demographics: {len(result)} tracts")
    return result


if __name__ == "__main__":
    load_walk_network(refresh="--refresh" in sys.argv)
    load_snap_stores()
