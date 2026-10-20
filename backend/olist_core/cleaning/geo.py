"""Zip-prefix centroids and great-circle distances.

The 61 MB geolocation table is reduced once (download step) to a committed lookup with one median
coordinate per 5-digit zip prefix, after dropping exact duplicates and points outside Brazil.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olist_core.io import write_csv_stable
from olist_core.raw import load_raw_tables
from olist_core.sorting import stable_sort

LOOKUP_FILE = "zip_prefix_centroids.csv"
BRAZIL_LAT = (-34.0, 6.0)
BRAZIL_LNG = (-74.0, -34.0)
EARTH_RADIUS_KM = 6371.0


def build_centroid_lookup(raw_dir: str | Path, lookups_dir: str | Path) -> dict:
    geo = load_raw_tables(raw_dir, ["geolocation"])["geolocation"]
    rows_in = len(geo)
    geo = geo.drop_duplicates()
    duplicates = rows_in - len(geo)
    inside = geo["geolocation_lat"].between(*BRAZIL_LAT) & geo["geolocation_lng"].between(
        *BRAZIL_LNG
    )
    out_of_bounds = int((~inside).sum())
    geo = geo[inside]
    lookup = (
        geo.groupby("geolocation_zip_code_prefix", sort=True)
        .agg(
            lat=("geolocation_lat", "median"),
            lng=("geolocation_lng", "median"),
            n_points=("geolocation_lat", "size"),
        )
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
    )
    lookup = stable_sort(lookup, "zip_prefix")
    path = write_csv_stable(lookup, Path(lookups_dir) / LOOKUP_FILE)
    return {
        "path": path.as_posix(),
        "rows": int(len(lookup)),
        "rows_in": int(rows_in),
        "dropped_duplicates": int(duplicates),
        "dropped_out_of_bounds": out_of_bounds,
    }


def load_centroids(lookups_dir: str | Path) -> pd.DataFrame:
    path = Path(lookups_dir) / LOOKUP_FILE
    return pd.read_csv(path, dtype={"zip_prefix": str}).set_index("zip_prefix")


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
