"""
Snapshot ReliefWeb's Ukraine updates listing + a few article pages, so that
Lab 1 (web scraping) runs offline from data/cached/reliefweb_html/.

Why ReliefWeb: static server-rendered HTML, honest numeric pagination,
robots.txt allows the listing and report pages, UN OCHA terms permit
download/copy for personal, non-commercial use (reliefweb.int/terms-conditions),
and it serves plain `requests` without user-agent games. Verified 2026-06-10.

Re-run this script to refresh the snapshot. It is deliberately written in
the same style the lab teaches: polite headers, sleeps, raise_for_status.
"""

import os
import re
import time

import requests
from bs4 import BeautifulSoup

# PC241 is ReliefWeb's ID for the Ukraine country facet.
LISTING_URL = "https://reliefweb.int/updates?advanced-search=%28PC241%29&page={page}"
N_LISTING_PAGES = 3   # listing_page_0.html ... listing_page_2.html
N_ARTICLES = 8        # article pages snapshotted from listing page 0

HEADERS = {
    "User-Agent": "icpsr-pipeline-course (academic teaching; ahalterman0@gmail.com)"
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "cached", "reliefweb_html")
os.makedirs(os.path.join(OUT_DIR, "articles"), exist_ok=True)


def fetch(url):
    print(f"GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(2)  # we are guests here
    return resp.text


def main():
    # 1. The listing pages (these teach pagination)
    article_links = []
    for page in range(N_LISTING_PAGES):
        html = fetch(LISTING_URL.format(page=page))
        path = os.path.join(OUT_DIR, f"listing_page_{page}.html")
        with open(path, "w") as f:
            f.write(html)
        if page == 0:
            soup = BeautifulSoup(html, "lxml")
            # Report links look like /report/ukraine/<slug>
            for a in soup.select("a[href*='/report/']"):
                href = a["href"]
                if href.startswith("/"):
                    href = "https://reliefweb.int" + href
                if href not in article_links:
                    article_links.append(href)

    # 2. A handful of article pages (these teach detail-page parsing)
    for url in article_links[:N_ARTICLES]:
        slug = re.sub(r"[^a-z0-9-]", "", url.rstrip("/").split("/")[-1])[:80]
        html = fetch(url)
        with open(os.path.join(OUT_DIR, "articles", f"{slug}.html"), "w") as f:
            f.write(html)

    print(f"\nSnapshot written to {os.path.normpath(OUT_DIR)}")
    print("Provenance: record the date in data/cached/README.md if refreshing.")


if __name__ == "__main__":
    main()
