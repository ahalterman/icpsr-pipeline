# Cached datasets for the course labs

These files were produced by the scripts in `data/acquisition/` and cached so
the labs run offline. Each script is re-runnable if you want fresher data.

## telegram_infra.parquet (Lab 2)

- **What it is:** 632 posts from five public Ukrainian energy/infrastructure
  Telegram channels — `Ukrenergo` (national grid operator), `dtek_ua` (DTEK,
  private utility), `energoatom_ua` (state nuclear operator), `dsns_telegram`
  (State Emergency Service), and `kyivoda` (Kyiv Oblast administration) —
  collected over roughly 2024-05-17 to 2024-07-14 (the spring energy-strike
  campaign). Columns: `channel, date, text, text_en_mt, views, forwards,
  fwd_from, msg_id`. `text` is the original Ukrainian/Russian; **`text_en_mt`
  is an English machine translation (LLM, not human-verified)** — treat it as
  a rough guide, not ground truth. Note: `msg_id` is unique only *within* a
  channel, not across channels (the ranges overlap), so key on
  `(channel, msg_id)` if you join.
- **Source:** Telegram MTProto API via Telethon, public broadcast channels
  only (fetched by `data/acquisition/get_telegram.py`; translation added by a
  separate LLM pass).
- **Date fetched:** 2026-07-19
- **License/terms:** Telegram publishes no reuse license. These are public,
  institutional broadcast channels; this filtered extract is cached for
  teaching **with attribution to the channels above**. Do not redistribute
  beyond course use, and do not use it to profile or target individuals.
- **Required notice:** Attribute the originating channels (e.g.
  "t.me/Ukrenergo") when quoting posts.

## firms_fires.csv

- **What it is:** NASA FIRMS VIIRS (Suomi-NPP) active-fire detections for a
  bounding box over northern Kharkiv oblast (lon 36.0–38.0, lat 49.5–50.6),
  2024-05-01 through 2024-06-30 — the period of the Vovchansk offensive.
  Columns: `latitude, longitude, acq_date, frp, confidence, satellite`.
- **Source URL:** <https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2024/viirs-snpp_2024_Ukraine.csv>
  (filtered to the bbox/dates by `data/acquisition/get_firms.py`)
- **Date fetched:** 2026-06-10
- **License/terms:** NASA open data (effectively CC0); no restrictions.
- **Required notice:** Please cite FIRMS: "We acknowledge the use of data
  from NASA's Fire Information for Resource Management System (FIRMS),
  https://firms.modaps.eosdis.nasa.gov/, part of NASA's Earth Science Data
  and Information System (ESDIS)."

## deepstate_control.geojson

- **What it is:** Snapshot of DeepStateMap territorial-control polygons for
  Ukraine (occupied / liberated / unknown-status areas, plus Crimea and
  ORDLO outlines). Geometries simplified (tolerance 0.005°) and the
  project's satirical out-of-Ukraine polygons removed. Properties kept:
  `name` (Ukrainian /// English), `description`.
- **Source URL:** <https://deepstatemap.live/api/history/last>
  (fetched by `data/acquisition/get_deepstate.py`)
- **Date fetched:** 2026-06-10 (upstream snapshot id 1781033711)
- **License/terms:** **No published license.** Snapshot cached for teaching
  with attribution to deepstatemap.live; upstream publishes no terms;
  **do not redistribute beyond course use.**
- **Required notice:** Attribute "DeepStateMap (deepstatemap.live)" on any
  map made with these polygons.

## sentinel2_kharkiv_clip.tif

- **What it is:** One Sentinel-2 L2A scene
  (`S2A_MSIL2A_20240629T083601_R064_T37UCR_20240629T122410`, 2024-06-29,
  0% cloud) clipped to the Vovchansk area (lon 36.7–37.05, lat 50.15–50.4),
  4 bands (B04 red, B03 green, B02 blue, B08 NIR) resampled from 10 m to
  20 m, written as a Cloud-Optimized GeoTIFF (EPSG:32637, uint16 surface
  reflectance, scale 1/10000 with the post-2022 +1000 offset).
- **Source:** Microsoft Planetary Computer STAC API,
  <https://planetarycomputer.microsoft.com/api/stac/v1>, collection
  `sentinel-2-l2a` (fetched by `data/acquisition/get_sentinel.py`).
- **Date fetched:** 2026-06-10
- **License/terms:** Free and open under the Copernicus Sentinel data
  legal notice.
- **Required notices:** "Contains modified Copernicus Sentinel data 2024".
  Data accessed via the Microsoft Planetary Computer
  (<https://planetarycomputer.microsoft.com/>); please acknowledge it.

## eurosat_sample/ + eurosat_index.csv + eurosat_embeddings.npz (Lab 4)

- **What it is:** A fixed 720-image sample of EuroSAT RGB — 64x64
  Sentinel-2 land-cover patches — 120 each from 6 classes relevant to the
  course's conflict/landscape narrative: AnnualCrop, Forest, Industrial,
  Residential, River, Pasture. Layout:
  `eurosat_sample/<ClassName>/<original_filename>.jpg`.
  `eurosat_index.csv` lists `filepath` (relative to repo root), `class`,
  and `split` (stratified 80/20 train/test, seed 42).
  `eurosat_embeddings.npz` holds precomputed image embeddings as a
  fallback if the in-lab model download fails: arrays `embeddings`
  (float32, 720x768 CLS-token features), `row_index`, `filepath`,
  `label`, `split` (all aligned row-for-row with `eurosat_index.csv`),
  and `model`.
- **Embedding model:** `google/vit-base-patch16-224` (HuggingFace),
  images resized to 224x224, CLS token of the last hidden state, CPU
  inference.
- **Source URL:** <https://zenodo.org/records/7711810>
  (`EuroSAT_RGB.zip`; project page <https://github.com/phelber/EuroSAT>),
  sampled with a seeded RNG by `data/acquisition/get_eurosat.py`.
- **Date fetched:** 2026-06-10
- **License/terms:** MIT (dataset). Imagery: contains modified
  Copernicus Sentinel data 2015–2018.
- **Citation:** Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
  EuroSAT: A novel dataset and deep learning benchmark for land use and
  land cover classification. *IEEE Journal of Selected Topics in Applied
  Earth Observations and Remote Sensing*, 12(7), 2217–2226.

## ucdp_ged_kharkiv_may2024.csv (Lab 8)

- **What it is:** UCDP Georeferenced Event Dataset (GED) v26.1 events for
  Ukraine with `date_start` in May 2024, restricted to Kharkiv oblast —
  defined as `adm_1` containing "Kharkiv" OR coordinates in the bbox
  lon 35.0–38.5, lat 48.8–50.6 (the bbox rule keeps a fringe of events
  coded to adjacent Luhansk/Donetsk/Sumy oblasts; this is deliberate so
  all three Lab 8 datasets share one spatial definition). 529 events,
  30 columns (ids, dates, coordinates + `where_prec`, admin units,
  `type_of_violence`, sides, death counts incl. `best`/`high`/`low`,
  source fields).
- **Source URL:** UCDP API <https://ucdpapi.pcr.uu.se/api/gedevents/26.1>
  (now requires a free emailed access token — see
  <https://ucdp.uu.se/apidocs/>); this cache was built via the documented
  fallback, the official bulk file
  <https://ucdp.uu.se/downloads/ged/ged261-csv.zip>, filtered by
  `data/acquisition/get_ucdp.py`.
- **Date fetched:** 2026-06-10
- **License/terms:** CC BY 4.0 — caching a filtered extract with
  attribution is permitted.
- **Citations:**
  - Davies, Shawn, Therese Pettersson & Magnus Öberg (2026). Organized
    violence 1989–2025. *Journal of Peace Research* 63(4).
  - Sundberg, Ralph & Erik Melander (2013). Introducing the UCDP
    Georeferenced Event Dataset. *Journal of Peace Research* 50(4): 523–532.

## viina_kharkiv_may2024.csv (Lab 8)

- **What it is:** VIINA 2.0 event reports (news-article mentions, BERT
  machine-coded) with report dates in May 2024, restricted to Kharkiv
  oblast (`ADM1_NAME == "Kharkiv"` OR the same bbox as above). 3,619
  reports, 42 columns: `event_info` fields (date/time, GeoNames
  geocoding incl. `GEO_PRECISION`, source outlet, url, headline `text`)
  merged on `event_id` with classifier labels — raw probabilities
  `t_mil`/`t_loc` plus all binarized `*_b` event-type and actor flags
  (`t_artillery_b`, `t_uav_b`, `a_rus_b`, ...). Note: rows are *reports*,
  not deduplicated events — expect multiple rows per incident.
- **Source URL:** <https://github.com/zhukovyuri/VIINA>, files
  `Data/event_info_latest_2024.zip` and `Data/event_labels_latest_2024.zip`
  (git-lfs; fetched via media.githubusercontent.com by
  `data/acquisition/get_viina.py`).
- **Date fetched:** 2026-06-10
- **License/terms:** ODbL 1.0.
- **Required notice:** "Contains information from VIINA, which is made
  available here under the Open Database License (ODbL)."
- **Citation:** Zhukov, Yuri (2023). Near-Real Time Analysis of War and
  Economic Activity during Russia's Invasion of Ukraine. *Journal of
  Comparative Economics* 51(4): 1232–1243. (VIINA 2.0.)

## acled_kharkiv_may2024.csv (Lab 8 — NOT cached, fetch it yourself)

- **What it is:** ACLED events for Ukraine / admin1 Kharkiv / May 2024.
  **There is no cached copy in this repo**: ACLED's Terms of Use (EULA
  §3.1/3.3) prohibit redistribution — every user must pull the data under
  their own (free for academic use) myACLED account. The output path is
  gitignored.
- **How to get it:** register at <https://acleddata.com>, then
  `export ACLED_EMAIL=... ACLED_PASSWORD=...` and run
  `python data/acquisition/get_acled.py` (OAuth password grant →
  Bearer token → `https://acleddata.com/api/acled/read`, per
  <https://acleddata.com/api-documentation/getting-started>).
- **Date API docs verified:** 2026-06-10
- **License/terms:** ACLED EULA — attribution required, no redistribution.
- **Citation:** Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010).
  Introducing ACLED: An Armed Conflict Location and Event Dataset.
  *Journal of Peace Research* 47(5): 651–660.

## vlm_images/ (Lab 6)

- **What it is:** The Lab 6 (VLM labeling) image set, composed entirely from
  patches in `eurosat_sample/` above — no new downloads. Layout:
  - `task_images/task_01.png … task_16.png`: 16 single 64x64 patches
    upscaled to 256x256 (PIL LANCZOS) covering all 6 classes; three are
    hand-picked to be genuinely ambiguous (Pasture↔AnnualCrop,
    River↔Forest).
  - `hard_images/`: five PIL-composed failure-mode probes —
    `count_grid_small.png` (3x3 grid, exactly 2 River tiles),
    `count_grid_large.png` (6x6 grid, exactly 13 Industrial tiles),
    `blur_residential.png` (Residential patch, Gaussian blur radius 7 —
    confabulation bait), `montage_mixed.png` (2x2 of four different
    classes), `empty_field.png` (plain Pasture, zero buildings —
    suggestibility bait).
  - `ground_truth.csv`: per image — `filename`, `image_set` (task/hard),
    `true_class`, `true_count` (for the count/suggestibility probes),
    `source` (the originating EuroSAT patch(es)), `notes` (incl. grid
    positions of the count-target tiles).
  Built by `data/acquisition/make_vlm_images.py` (seed 42, re-runnable).
  Total ~1.8 MB.
- **Source:** derived from the EuroSAT RGB sample documented above
  (Zenodo deposit <https://zenodo.org/records/7711810>).
- **Date built:** 2026-06-10
- **License/terms:** MIT (EuroSAT dataset). Imagery: contains modified
  Copernicus Sentinel data 2015–2018.
- **Citation:** Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
  EuroSAT: A novel dataset and deep learning benchmark for land use and
  land cover classification. *IEEE Journal of Selected Topics in Applied
  Earth Observations and Remote Sensing*, 12(7), 2217–2226.
