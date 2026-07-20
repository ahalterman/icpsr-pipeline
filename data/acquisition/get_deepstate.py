"""
get_deepstate.py -- Snapshot DeepStateMap territorial-control polygons.

Lab 3: overlaying fire detections, satellite imagery, and territorial-control
polygons around the May 2024 Vovchansk offensive.

Data source
-----------
DeepStateMap (https://deepstatemap.live) is a Ukrainian OSINT project that
maintains a frequently updated map of the frontline and areas of control in
Ukraine. Its public API returns the latest map as GeoJSON, no key required:

    https://deepstatemap.live/api/history/last

The response is a JSON object {"id": ..., "map": <FeatureCollection>}.
Features include occupied-area polygons, "liberated"/"grey zone" areas, and
frontline lines; names are bilingual ("ua name /// en name").

License / terms: DeepStateMap publishes NO formal license. We cache a single
snapshot for classroom use with attribution; do not redistribute beyond the
course.

Output
------
data/cached/deepstate_control.geojson -- simplified polygons/lines with a
`name` and `description` property, small enough to load instantly in class.
"""

import json
import os
import urllib.request

import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import transform

API_URL = "https://deepstatemap.live/api/history/last"

# Identify ourselves politely -- this is an academic course, not a scraper farm.
USER_AGENT = "ICPSR-OSINT-course/1.0 (academic teaching use; contact: instructor)"

# Simplification tolerance in degrees (~500 m at this latitude). This shrinks
# the file from several MB to well under 3 MB with no visible loss at oblast scale.
SIMPLIFY_TOL = 0.005

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "cached", "deepstate_control.geojson")

# DeepState's map also contains satirical/political polygons far outside
# Ukraine (the Kuril Islands, "occupied East Prussia", Karelia, Abkhazia...).
# We keep only the substantive control-status polygons plus Crimea and ORDLO.
# Feature names embed machine-readable keys like "geoJSON.status.occupied".
KEEP_KEYS = (
    "geoJSON.status.",              # occupied / liberated / unknown-status areas
    "geoJSON.territories.crimea",   # occupied Crimea
    "geoJSON.territories.ordlo",    # occupied Donetsk/Luhansk (pre-2022 lines)
)


def drop_z(geom):
    """DeepState coordinates are 3-D (x, y, 0); flatten to 2-D for clean GeoJSON."""
    return transform(lambda x, y, z=None: (x, y), geom)


def main() -> None:
    # 1. Fetch the latest map snapshot
    print(f"Fetching {API_URL} ...")
    req = urllib.request.Request(API_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    fc = payload["map"]  # the FeatureCollection lives under the "map" key
    print(f"Snapshot id {payload.get('id')}: {len(fc['features'])} raw features.")

    # 2. Keep control-area polygons and frontline lines; simplify geometries
    out_features = []
    for feat in fc["features"]:
        geom = feat.get("geometry")
        if geom is None:
            continue
        if geom["type"] not in (
            "Polygon", "MultiPolygon", "LineString", "MultiLineString",
        ):
            continue  # skip point markers (unit icons, etc.)

        props = feat.get("properties", {})
        name = props.get("name", "")
        if not any(key in name for key in KEEP_KEYS):
            continue  # skip the out-of-Ukraine satirical polygons

        g = drop_z(shape(geom))
        # preserve_topology avoids creating self-intersections when simplifying
        g = g.simplify(SIMPLIFY_TOL, preserve_topology=True)
        if g.is_empty:
            continue

        out_features.append(
            {
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "name": name,
                    "description": props.get("description", ""),
                },
            }
        )

    out_fc = {"type": "FeatureCollection", "features": out_features}

    # 3. Write the cache file (compact JSON: no extra whitespace)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_fc, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Wrote {os.path.abspath(OUT_PATH)} ({size_mb:.2f} MB, "
          f"{len(out_features)} features)")

    # 4. Sanity check: re-load with geopandas and confirm it sits inside Ukraine
    gdf = gpd.read_file(OUT_PATH)
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"geopandas loaded {len(gdf)} features; bounds = "
          f"({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")
    assert 22 <= minx and maxx <= 41 and 44 <= miny and maxy <= 53, (
        "Bounds fall outside Ukraine -- inspect the download!"
    )
    print("Sanity check passed: bounds are within Ukraine.")


if __name__ == "__main__":
    main()
