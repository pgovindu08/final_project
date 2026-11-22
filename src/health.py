"""
Author: Tejas Srinivasan

CDC PLACES 2025 health data: download, tract-ID validation, and Spearman
correlation of diabetes prevalence against food-access walking distances.

Task 1 (TS rubric): load CDC PLACES Tennessee tract estimates; validate 11-digit FIPS.
Task 6 (TS rubric): Spearman rho + p-value for
    (diabetes vs walking dist)  and  (diabetes vs straight-line dist).

NOTE: verify CDC_PLACES_URL before first run -- the view ID may change between
annual releases.  Find the current URL at:
https://chronicdata.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-Census-Tract-D/cwsq-ngmh
"""

import urllib.request
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"

# CDC PLACES Local Data for Better Health -- Census Tract Data 2024 release
CDC_PLACES_URL      = "https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD"
DIABETES_MEASURE_ID = "DIABETES"


def load_cdc_places(refresh=False):
    """
    Download CDC PLACES tract-level data for Tennessee.
    Returns a DataFrame with columns [tract_id (11-digit str), diabetes_pct (float)].
    """
    path = RAW / "cdc_places_tract.csv"
    if not path.exists() or refresh:
        print("Downloading CDC PLACES data (may be large)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            CDC_PLACES_URL, headers={"User-Agent": "FoodRoute student project"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            path.write_bytes(resp.read())

    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Filter to Tennessee and the diabetes measure
    df = df[df["StateAbbr"] == "TN"].copy()
    df = df[df["MeasureId"] == DIABETES_MEASURE_ID].copy()

    # Normalise tract ID to 11-digit zero-padded string (guards against leading-zero loss)
    df["tract_id"]     = df["LocationName"].str.strip().str.zfill(11)
    df["diabetes_pct"] = pd.to_numeric(df["Data_Value"], errors="coerce")

    return df[["tract_id", "diabetes_pct"]].dropna().reset_index(drop=True)


def check_tract_ids(ids):
    """
    Return a list of IDs that are NOT valid 11-digit numeric strings.
    Call this after loading both the health data and PG's tract data to catch
    leading-zero loss early (TS task 1).
    """
    return [i for i in ids if not (str(i).isdigit() and len(str(i)) == 11)]


def spearman_correlation(x, y):
    """Return (rho, p_value) Spearman rank correlation between sequences x and y."""
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def analyze_health_vs_access(health_df, access_df):
    """
    Join health and access data on tract_id and compute Spearman correlations.

    Parameters
    ----------
    health_df  : DataFrame with [tract_id, diabetes_pct]
    access_df  : DataFrame with [tract_id, walking_dist_miles, straight_line_dist_miles]

    Returns
    -------
    dict: n_tracts, walking_rho, walking_p, straight_line_rho, straight_line_p
    """
    merged = (
        health_df
        .merge(access_df, on="tract_id", how="inner")
        .dropna(subset=["diabetes_pct", "walking_dist_miles", "straight_line_dist_miles"])
    )

    rho_walk, p_walk = spearman_correlation(
        merged["diabetes_pct"], merged["walking_dist_miles"]
    )
    rho_line, p_line = spearman_correlation(
        merged["diabetes_pct"], merged["straight_line_dist_miles"]
    )

    print(f"Tracts in analysis : {len(merged)}")
    print(f"Walking dist  -> diabetes : rho={rho_walk:.3f},  p={p_walk:.4f}")
    print(f"Straight-line -> diabetes : rho={rho_line:.3f},  p={p_line:.4f}")

    return {
        "n_tracts":          len(merged),
        "walking_rho":       rho_walk,
        "walking_p":         p_walk,
        "straight_line_rho": rho_line,
        "straight_line_p":   p_line,
    }
