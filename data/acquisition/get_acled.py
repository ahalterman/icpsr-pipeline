"""Fetch ACLED events for Kharkiv oblast, May 2024 (Vovchansk offensive).

Output: data/cached/acled_kharkiv_may2024.csv  -- NOT cached in this repo.

WHY THERE IS NO CACHED COPY
---------------------------
ACLED's Terms of Use / EULA (sections 3.1 and 3.3) prohibit redistribution
of the data: each user must obtain the data directly from ACLED under
their own account. We therefore ship only this acquisition script, and the
output path is gitignored (see .gitignore: data/cached/acled_kharkiv_may2024.csv).

Registration at https://acleddata.com (myACLED account) is FREE for
academic use. After registering, set two environment variables and run
this script:

    export ACLED_EMAIL="you@university.edu"
    export ACLED_PASSWORD="your-myacled-password"
    python data/acquisition/get_acled.py

API mechanics (per https://acleddata.com/api-documentation/getting-started,
fetched 2026-06-10)
-------------------------------------------------------------------------
1. OAuth token request: POST https://acleddata.com/oauth/token with
   form-encoded fields username, password, grant_type="password",
   client_id="acled", scope="authenticated". Returns a Bearer
   access_token valid 24 h (and a refresh_token valid 14 days).
2. Data request: GET https://acleddata.com/api/acled/read with header
   "Authorization: Bearer <token>". Filters are query parameters; ranges
   use the `<field>_where=BETWEEN` convention with values separated by
   `|`. Default/maximum row limit per call is 5000, so we paginate with
   `limit` and `page`.

Citation: Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010).
Introducing ACLED: An Armed Conflict Location and Event Dataset.
Journal of Peace Research 47(5): 651-660. Plus ACLED's required
attribution per its citation policy.
"""

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

TOKEN_URL = "https://acleddata.com/oauth/token"
DATA_URL = "https://acleddata.com/api/acled/read"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "cached" / "acled_kharkiv_may2024.csv"

PAGE_SIZE = 5000  # ACLED's per-call maximum


def get_token(email: str, password: str) -> str:
    """Exchange myACLED credentials for a 24-hour Bearer token."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "username": email,
            "password": password,
            "grant_type": "password",   # hard-coded per ACLED docs
            "client_id": "acled",       # hard-coded per ACLED docs
            "scope": "authenticated",   # hard-coded per ACLED docs
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_events(token: str) -> pd.DataFrame:
    """Page through the ACLED endpoint for Ukraine / Kharkiv / May 2024."""
    frames = []
    page = 1
    while True:
        resp = requests.get(
            DATA_URL,
            params={
                "_format": "csv",
                "country": "Ukraine",
                "admin1": "Kharkiv",
                "event_date": "2024-05-01|2024-05-31",
                "event_date_where": "BETWEEN",
                "limit": PAGE_SIZE,
                "page": page,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=300,
        )
        resp.raise_for_status()
        if not resp.text.strip():
            break
        chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        if chunk.empty:
            break
        frames.append(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        page += 1
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    email = os.environ.get("ACLED_EMAIL", "").strip()
    password = os.environ.get("ACLED_PASSWORD", "").strip()
    if not email or not password:
        sys.exit(
            "Set ACLED_EMAIL and ACLED_PASSWORD environment variables.\n"
            "Register (free for academic use) at https://acleddata.com "
            "if you do not yet have a myACLED account."
        )

    print("Requesting OAuth token...")
    token = get_token(email, password)

    print("Fetching events (Ukraine / admin1=Kharkiv / May 2024)...")
    df = fetch_events(token)
    if df.empty:
        sys.exit("No rows returned -- check credentials/filters.")

    df = df.sort_values("event_date")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} events to {OUT_PATH}")
    print("(This file is gitignored: ACLED's EULA prohibits redistribution.)")
    if "event_type" in df.columns:
        print(df["event_type"].value_counts())


if __name__ == "__main__":
    main()
