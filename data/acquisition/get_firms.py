"""
get_firms.py -- Download NASA FIRMS fire detections for northern Kharkiv oblast.

Lab 3: overlaying fire detections, satellite imagery, and territorial-control
polygons around the May 2024 Vovchansk offensive.

Data source
-----------
NASA FIRMS (Fire Information for Resource Management System) publishes
country-level yearly archives of VIIRS active-fire detections as plain CSV,
with no API key required:

    https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2024/viirs-snpp_2024_Ukraine.csv

If that archive is ever unavailable, the script falls back to the FIRMS area
API, which DOES require a (free) MAP_KEY read from the FIRMS_MAP_KEY
environment variable: https://firms.modaps.eosdis.nasa.gov/api/map_key/

License: NASA open data (effectively CC0). Please cite FIRMS:
    NASA FIRMS, https://firms.modaps.eosdis.nasa.gov/

Output
------
data/cached/firms_fires.csv -- one row per VIIRS detection inside the
bounding box and date window, with only the columns needed for mapping.
"""

import io
import os
import sys
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Parameters: area of interest and time window
# ---------------------------------------------------------------------------
# Bounding box around northern Kharkiv oblast (lon_min, lat_min, lon_max, lat_max)
BBOX = (36.0, 49.5, 38.0, 50.6)
DATE_START = "2024-05-01"
DATE_END = "2024-06-30"

# Columns we keep for mapping (everything else is dropped to keep the file small)
KEEP_COLS = ["latitude", "longitude", "acq_date", "frp", "confidence", "satellite"]

ARCHIVE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/"
    "viirs-snpp/2024/viirs-snpp_2024_Ukraine.csv"
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "cached", "firms_fires.csv")


def fetch_url(url: str, timeout: int = 120) -> bytes:
    """Download a URL and return the raw bytes (with a descriptive User-Agent)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "ICPSR-OSINT-course/1.0 (academic teaching use)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_country_archive() -> pd.DataFrame:
    """Primary path: the public country yearly archive (no key needed)."""
    print(f"Downloading FIRMS Ukraine 2024 archive (~7 MB):\n  {ARCHIVE_URL}")
    raw = fetch_url(ARCHIVE_URL)
    return pd.read_csv(io.BytesIO(raw))


def fetch_area_api() -> pd.DataFrame:
    """Fallback path: the FIRMS area API. Requires FIRMS_MAP_KEY in the env."""
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        sys.exit(
            "Country archive unavailable and FIRMS_MAP_KEY is not set.\n"
            "Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )
    # The area API serves at most 10 days per request, so loop over the window.
    frames = []
    w, s, e, n = BBOX[0], BBOX[1], BBOX[2], BBOX[3]
    for start in pd.date_range(DATE_START, DATE_END, freq="10D"):
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/"
            f"VIIRS_SNPP_SP/{w},{s},{e},{n}/10/{start.date()}"
        )
        print(f"  fetching {start.date()} ...")
        frames.append(pd.read_csv(io.BytesIO(fetch_url(url))))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    # 1. Download the full-country file (preferred: public, keyless, reproducible)
    try:
        df = fetch_country_archive()
    except Exception as err:  # noqa: BLE001 -- teaching code: show the error, try fallback
        print(f"Archive download failed ({err}); trying the area API instead.")
        df = fetch_area_api()

    print(f"Downloaded {len(df):,} detections for all of Ukraine / requested area.")

    # 2. Filter to our bounding box and date window
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    in_box = (
        df["longitude"].between(BBOX[0], BBOX[2])
        & df["latitude"].between(BBOX[1], BBOX[3])
        & df["acq_date"].between(DATE_START, DATE_END)
    )
    df = df.loc[in_box, KEEP_COLS].reset_index(drop=True)
    print(f"Kept {len(df):,} detections in bbox {BBOX}, {DATE_START} to {DATE_END}.")

    # 3. Write the cache file
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {os.path.abspath(OUT_PATH)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
