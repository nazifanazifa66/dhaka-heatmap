import json
from pathlib import Path

import pandas as pd

# ============================================================
# FINAL MERGE SCRIPT
# ============================================================
# Inputs:
#   1. Dhaka_Heatmap_500m_Grid.geojson
#   2. Dhaka_Heatmap_Grid_Features.csv
#   3. dhaka_heat_risk_data.csv
#
# Output:
#   Dhaka_Heatmap_Grid_Final.geojson
#
# Does:
#   1. Merge satellite/grid features
#   2. Merge predicted LST + heat-risk data
#   3. Convert geometry from EPSG:32645 (UTM Zone 45N)
#      to EPSG:4326 (longitude/latitude) for Folium
# ============================================================

geojson_file = Path("Dhaka_Heatmap_500m_Grid.geojson")
features_file = Path("Dhaka_Heatmap_Grid_Features.csv")
risk_file = Path("dhaka_heat_risk_data.csv")
output_file = Path("Dhaka_Heatmap_Grid_Final.geojson")

# ------------------------------------------------------------
# 1. Check input files
# ------------------------------------------------------------

for file in [geojson_file, features_file, risk_file]:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found: {file}\n"
            f"Make sure all three files are in the same folder as this script."
        )

# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

with geojson_file.open("r", encoding="utf-8") as f:
    geojson = json.load(f)

features = pd.read_csv(features_file)
risk = pd.read_csv(risk_file)

features.columns = features.columns.str.strip()
risk.columns = risk.columns.str.strip()

# ------------------------------------------------------------
# 3. Validate required columns
# ------------------------------------------------------------

required_features = {
    "system:index",
    "LST",
    "NDVI",
    "NDBI",
    "NDWI",
}

required_risk = {
    "predicted_LST",
    "heat_risk_score",
}

missing_features = required_features - set(features.columns)
missing_risk = required_risk - set(risk.columns)

if missing_features:
    raise ValueError(
        f"Missing columns in {features_file.name}: "
        f"{sorted(missing_features)}"
    )

if missing_risk:
    raise ValueError(
        f"Missing columns in {risk_file.name}: "
        f"{sorted(missing_risk)}"
    )

# ------------------------------------------------------------
# 4. Validate row correspondence
# ------------------------------------------------------------

if len(features) != len(risk):
    raise ValueError(
        f"Row count mismatch:\n"
        f"  Feature CSV: {len(features)} rows\n"
        f"  Risk CSV:    {len(risk)} rows"
    )

# The risk CSV does not contain system:index.
# Verify that it corresponds to the feature CSV by checking
# the environmental features row-by-row.

for col in ["NDVI", "NDBI", "NDWI"]:
    a = pd.to_numeric(features[col], errors="coerce")
    b = pd.to_numeric(risk[col], errors="coerce")

    difference = (a - b).abs()

    # Treat NaN in both files as equal.
    same = (difference < 1e-9) | (a.isna() & b.isna())

    if not same.all():
        bad_rows = same[~same].index.tolist()[:10]
        raise ValueError(
            f"{col} does not match between the two CSV files.\n"
            f"Example mismatched rows: {bad_rows}\n"
            "The files cannot safely be merged by row order."
        )

# ------------------------------------------------------------
# 5. Create risk lookup using system:index
# ------------------------------------------------------------

risk_with_id = features[["system:index"]].copy()

risk_with_id["predicted_LST"] = pd.to_numeric(
    risk["predicted_LST"], errors="coerce"
).values

risk_with_id["heat_risk_score"] = pd.to_numeric(
    risk["heat_risk_score"], errors="coerce"
).values

# Your CSV may contain the original typo risk_catagory.
if "risk_category" in risk.columns:
    risk_with_id["risk_category"] = (
        risk["risk_category"].astype(str).values
    )

elif "risk_catagory" in risk.columns:
    risk_with_id["risk_category"] = (
        risk["risk_catagory"].astype(str).values
    )

else:
    risk_with_id["risk_category"] = pd.cut(
        risk_with_id["heat_risk_score"],
        bins=[-float("inf"), 33, 66, float("inf")],
        labels=["Low", "Medium", "High"]
    ).astype(str)

# ------------------------------------------------------------
# 6. Build lookup
# ------------------------------------------------------------

risk_lookup = (
    risk_with_id
    .set_index(risk_with_id["system:index"].astype(str))
    .to_dict("index")
)

# ------------------------------------------------------------
# 7. Merge all CSV information into GeoJSON
# ------------------------------------------------------------

matched = 0

for feature in geojson["features"]:

    properties = feature.setdefault("properties", {})

    grid_id = str(properties.get("grid_id", ""))

    if grid_id not in risk_lookup:
        raise ValueError(
            f"No CSV match found for GeoJSON grid_id: {grid_id}"
        )

    # Add the original satellite features.
    feature_row = features[
        features["system:index"].astype(str) == grid_id
    ]

    if feature_row.empty:
        raise ValueError(
            f"No feature CSV match found for grid_id: {grid_id}"
        )

    feature_row = feature_row.iloc[0]

    for col in ["LST", "NDVI", "NDBI", "NDWI"]:
        value = feature_row[col]

        if pd.isna(value):
            properties[col] = None
        else:
            properties[col] = float(value)

    # Add prediction/risk information.
    risk_row = risk_lookup[grid_id]

    properties["predicted_LST"] = (
        None
        if pd.isna(risk_row["predicted_LST"])
        else float(risk_row["predicted_LST"])
    )

    properties["heat_risk_score"] = (
        None
        if pd.isna(risk_row["heat_risk_score"])
        else float(risk_row["heat_risk_score"])
    )

    properties["risk_category"] = str(
        risk_row["risk_category"]
    )

    matched += 1

# ------------------------------------------------------------
# 8. Convert geometry EPSG:32645 -> EPSG:4326
# ------------------------------------------------------------
# Folium/Leaflet expects GeoJSON coordinates as:
# [longitude, latitude]
#
# Your original geometry is UTM Zone 45N / EPSG:32645.
#
# This section uses pyproj. If it is not installed, run:
#
#     py -m pip install pyproj
#
# ------------------------------------------------------------

try:
    from pyproj import Transformer
except ImportError:
    raise ImportError(
        "pyproj is required for coordinate conversion.\n"
        "Install it with:\n"
        "py -m pip install pyproj"
    )

transformer = Transformer.from_crs(
    "EPSG:32645",
    "EPSG:4326",
    always_xy=True
)

def transform_coordinates(coords):
    """
    Recursively transform GeoJSON coordinate arrays
    from UTM x/y to longitude/latitude.
    """

    # A coordinate pair: [x, y] or [x, y, z]
    if (
        isinstance(coords, list)
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        x = coords[0]
        y = coords[1]

        lon, lat = transformer.transform(x, y)

        # Preserve Z if present.
        if len(coords) > 2:
            return [lon, lat, *coords[2:]]

        return [lon, lat]

    # Nested coordinate arrays.
    return [
        transform_coordinates(item)
        for item in coords
    ]

for feature in geojson["features"]:

    geometry = feature.get("geometry")

    if not geometry:
        continue

    if "coordinates" in geometry:
        geometry["coordinates"] = transform_coordinates(
            geometry["coordinates"]
        )

# ------------------------------------------------------------
# 9. Set correct CRS metadata
# ------------------------------------------------------------

# Standard GeoJSON uses WGS84 coordinates (EPSG:4326).
# Remove old CRS metadata if present because it may still say
# EPSG:32645 after the coordinate transformation.

geojson.pop("crs", None)

# ------------------------------------------------------------
# 10. Final validation
# ------------------------------------------------------------

required_properties = {
    "grid_id",
    "LST",
    "NDVI",
    "NDBI",
    "NDWI",
    "predicted_LST",
    "heat_risk_score",
    "risk_category",
}

if not geojson["features"]:
    raise ValueError("GeoJSON contains no features.")

first_properties = set(
    geojson["features"][0]["properties"].keys()
)

missing_final = required_properties - first_properties

if missing_final:
    raise ValueError(
        "Final GeoJSON is missing properties: "
        f"{sorted(missing_final)}"
    )

# Check transformed coordinates roughly fall in the Dhaka area.
# This is only a sanity check, not a hard geographic restriction.
first_coords = (
    geojson["features"][0]["geometry"]["coordinates"]
)

# ------------------------------------------------------------
# 11. Save
# ------------------------------------------------------------

with output_file.open("w", encoding="utf-8") as f:
    json.dump(
        geojson,
        f,
        ensure_ascii=False,
        separators=(",", ":")
    )

# ------------------------------------------------------------
# 12. Success message
# ------------------------------------------------------------

print()
print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"GeoJSON features : {len(geojson['features'])}")
print(f"Matched features : {matched}")
print("Original CRS      : EPSG:32645")
print("Final CRS         : EPSG:4326")
print(f"Output            : {output_file}")
print()
print("Final GeoJSON properties:")
for name in sorted(required_properties):
    print(f"  [OK] {name}")
print()
print("The GeoJSON is now ready for Folium/Leaflet.")
print("=" * 60)
