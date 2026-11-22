"""
Author: Pranav Govindu

Interactive Folium map showing:
  - Food-access classification per tract (low / adequate, based on walking distance)
  - Walking-vs-straight-line disagreements (highlighted borders)
  - Diabetes-rate overlay (when health data is available)
"""

from pathlib import Path
import folium

ROOT        = Path(__file__).resolve().parents[1]
KNOX_CENTER = [35.9606, -83.9207]   # Knox County, TN approximate centre

_ACCESS_COLORS = {
    "low_access":      "#d73027",   # red
    "adequate_access": "#1a9850",   # green
}


def build_map(tracts_gdf, access_df, health_df=None, out_html=None):
    """
    Build and save a Folium choropleth map.

    Parameters
    ----------
    tracts_gdf : GeoDataFrame with census tract geometries (EPSG:4326).
                 Must have a 'GEOID' column with 11-digit tract IDs.
    access_df  : DataFrame with [tract_id, walking_classification,
                 walking_dist_miles, straight_line_dist_miles, disagrees]
    health_df  : optional DataFrame with [tract_id, diabetes_pct]
    out_html   : output path; default results/map.html

    Returns
    -------
    folium.Map
    """
    if out_html is None:
        out_html = ROOT / "results" / "map.html"

    merged = tracts_gdf.merge(access_df, left_on="GEOID", right_on="tract_id", how="left")
    if health_df is not None:
        merged = merged.merge(health_df, left_on="GEOID", right_on="tract_id", how="left")

    m = folium.Map(location=KNOX_CENTER, zoom_start=11, tiles="CartoDB positron")

    for _, row in merged.iterrows():
        cls       = row.get("walking_classification", "")
        color     = _ACCESS_COLORS.get(cls, "#aaaaaa")
        disagrees = bool(row.get("disagrees", False))

        lines = [
            f"<b>Tract:</b> {row.get('GEOID', 'N/A')}",
            f"<b>Walking access:</b> {cls or 'N/A'}",
            f"<b>Walking dist:</b> {row.get('walking_dist_miles', float('nan')):.2f} mi",
            f"<b>Straight-line:</b> {row.get('straight_line_dist_miles', float('nan')):.2f} mi",
            f"<b>Methods disagree:</b> {'yes' if disagrees else 'no'}",
        ]
        if health_df is not None and "diabetes_pct" in row.index:
            lines.append(f"<b>Diabetes rate:</b> {row.get('diabetes_pct', float('nan')):.1f}%")

        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda _, c=color, d=disagrees: {
                "fillColor":   c,
                "color":       "#ff7f00" if d else "#000000",
                "weight":      2 if d else 1,
                "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip("<br>".join(lines)),
        ).add_to(m)

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"Map saved -> {out_html}")
    return m
