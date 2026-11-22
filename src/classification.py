"""
Author: Pranav Govindu

USDA food-access classification for census tracts.

Low-access thresholds (walking distance to nearest SNAP supermarket):
  Urban tracts : > 0.5 mi for low-income,  > 1 mi otherwise
  Rural tracts : > 10 mi for low-income, > 20 mi otherwise

Source: USDA Economic Research Service, Food Access Research Atlas.
"""

import pandas as pd

URBAN_THRESHOLD_LOW_INCOME = 0.5    # miles
URBAN_THRESHOLD            = 1.0    # miles
RURAL_THRESHOLD_LOW_INCOME = 10.0   # miles
RURAL_THRESHOLD            = 20.0   # miles


def classify_tract(walking_dist_miles, is_urban, is_low_income=False):
    """
    Return 'low_access' or 'adequate_access' for a single tract.

    walking_dist_miles : walking distance to the nearest SNAP-authorized supermarket
    is_urban           : True if the tract is classified as urban by USDA
    is_low_income      : True if the tract meets USDA low-income criteria
    """
    if is_urban:
        threshold = URBAN_THRESHOLD_LOW_INCOME if is_low_income else URBAN_THRESHOLD
    else:
        threshold = RURAL_THRESHOLD_LOW_INCOME if is_low_income else RURAL_THRESHOLD
    return "low_access" if walking_dist_miles > threshold else "adequate_access"


def classify_all_tracts(tracts_df, walking_dists, straight_line_dists):
    """
    Classify every tract and flag disagreements between walking and straight-line methods.

    Parameters
    ----------
    tracts_df           : DataFrame with columns [tract_id, is_urban, is_low_income]
    walking_dists       : dict {tract_id -> walking_dist_miles}  from multi-source Dijkstra
    straight_line_dists : dict {tract_id -> straight_line_dist_miles}  from k-d tree

    Returns
    -------
    DataFrame with columns:
        tract_id, walking_dist_miles, straight_line_dist_miles,
        walking_classification, straight_line_classification, disagrees
    """
    rows = []
    for _, row in tracts_df.iterrows():
        tid         = row["tract_id"]
        walk        = walking_dists.get(tid, float("inf"))
        straight    = straight_line_dists.get(tid, float("inf"))
        is_urban    = bool(row.get("is_urban", True))
        is_low_inc  = bool(row.get("is_low_income", False))

        walk_cls     = classify_tract(walk,     is_urban, is_low_inc)
        straight_cls = classify_tract(straight, is_urban, is_low_inc)

        rows.append({
            "tract_id":                     tid,
            "walking_dist_miles":           walk,
            "straight_line_dist_miles":     straight,
            "walking_classification":       walk_cls,
            "straight_line_classification": straight_cls,
            "disagrees":                    walk_cls != straight_cls,
        })
    return pd.DataFrame(rows)
