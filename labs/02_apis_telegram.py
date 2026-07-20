# %% [markdown]
# # Lab 2: APIs and Telegram
#
# ICPSR 2026 — The Social Science Data Pipeline
# Instructor: Andy Halterman
#
# Two halves, matching the chapter. First, APIs: how you get structured data
# (usually JSON) directly from a server instead of parsing it out of HTML,
# plus authentication, pagination, and the fact that even open data now wants
# to know who's using it. Then Telegram, which is where most of the specific
# information about the Ukraine war is: channel ecology, metadata, and source
# criticism.
#
# As always: every required cell runs offline from `data/cached/`. Cells
# marked **OPTIONAL -- live** need network, and sometimes credentials.

# %%
# Setup (same block as every lab)
import os

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()
print(f"Environment detected: {'Colab' if IN_COLAB else 'Local/hosted Jupyter'}")

if IN_COLAB:
    if not os.path.exists("/content/icpsr-pipeline"):
        !git clone -q https://github.com/ahalterman/icpsr-pipeline.git /content/icpsr-pipeline
    COURSE_DIR = "/content/icpsr-pipeline"
else:
    COURSE_DIR = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) in ("labs", "solutions") else os.getcwd()

DATA_DIR = os.path.join(COURSE_DIR, "data", "cached")
OUTPUTS_DIR = os.path.join(COURSE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# %%
!pip install -q requests pandas pyarrow matplotlib telethon nest_asyncio

# %% [markdown]
# ## Part 1: APIs
#
# Part 1 is demonstration pace: read and run the worked cells, and try to
# predict each result before you run it. Your hands-on time is Part 2, Telegram.
#
# ### 1a. What an API response looks like
#
# This morning you scraped HTML that was meant for browsers to read. An API
# hands you the structured data directly. Let's call one that needs no key and
# no registration, and that's directly relevant to the course: DeepStateMap's
# territorial-control API, which serves the polygons behind the widely-used map.

# %%
# OPTIONAL -- live network. The cached fallback is two cells down.
import requests

resp = requests.get(
    "https://deepstatemap.live/api/history/last",
    headers={"User-Agent": "icpsr-pipeline-lab (academic course)"},
    timeout=30,
)
resp.raise_for_status()
raw = resp.json()          # parse the JSON into Python dicts and lists
type(raw), list(raw.keys())

# %% [markdown]
# Real-world detail #1: the GeoJSON you want is wrapped inside a `"map"`
# key. Most APIs have these undocumented quirks, and calling `resp.json()`
# and poking at `.keys()` is how you find them. Real-world detail #2: this
# API publishes no documentation and no terms of use. It works fine, but
# relying on an undocumented endpoint for a dissertation is a real risk, so
# snapshot what you depend on (the chapter's provenance lesson).

# %%
# Offline fallback: we cached a cleaned snapshot on 2026-06-10
# (see data/acquisition/get_deepstate.py for exactly what "cleaned" means).
import json

with open(os.path.join(DATA_DIR, "deepstate_control.geojson")) as f:
    control = json.load(f)

print(f"{len(control['features'])} features")
control["features"][0]["properties"]

# %% [markdown]
# ### 1b. Nested JSON → flat dataframe
#
# JSON comes back nested (dicts inside lists inside dicts), but analysis
# wants a flat table. `pd.json_normalize` does that conversion.

# %%
import pandas as pd

features = pd.json_normalize(control["features"])
features[["properties.name", "geometry.type"]].head(8)

# %%
# Worked inline (predict before you run): how many features of each
# geometry.type, and how many property names mention "occupied"?
# `.value_counts()` and `.str.contains` are the two tools.
print(features["geometry.type"].value_counts())
n_occupied = features["properties.name"].str.contains("occupied", case=False, na=False).sum()
print(f"\nFeature names containing 'occupied': {n_occupied}")

# %% [markdown]
# ### 1c. Event data APIs: registration required
#
# The major conflict-event APIs all require some kind of registration:
#
# - **ACLED**: free academic registration, then an OAuth token, then
#   paginated JSON. Their license prohibits redistributing the data, which
#   is why there's no ACLED file in `data/cached/` (an access-terms lesson
#   in itself). The full pull script is `data/acquisition/get_acled.py`;
#   run it with your own credentials tonight if you registered.
# - **UCDP**: CC-BY licensed, so we can and do cache it; the API token is
#   free by email. The cached extract below came from their bulk download.
#
# Both follow the same pattern: authenticate, request a page, append, repeat
# until the API says there's no next page, and sleep between calls. It's in
# `get_acled.py` -- about 60 lines, and close to every event-data pull you'll
# write.

# %%
# The cached UCDP extract: Kharkiv oblast, May 2024 (the Vovchansk
# offensive). 529 vetted, georeferenced events, each anchored to a fatality count.
ucdp = pd.read_csv(os.path.join(DATA_DIR, "ucdp_ged_kharkiv_may2024.csv"))
print(ucdp.shape)
ucdp[["date_start", "adm_1", "type_of_violence", "best", "source_office"]].head()

# %%
# Worked inline (predict first: does the API-vetted event stream see the
# offensive?). Events per day, with May 10 -- the offensive's start -- marked.
import matplotlib.pyplot as plt

ucdp["date_start"] = pd.to_datetime(ucdp["date_start"])
per_day = ucdp.resample("D", on="date_start").size()

fig, ax = plt.subplots(figsize=(9, 3))
per_day.plot(ax=ax)
ax.axvline(pd.Timestamp("2024-05-10"), color="red", linestyle="--", label="May 10")
ax.set_ylabel("events / day")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 2: Telegram
#
# ### 2a. How collection works (Telethon)
#
# Telegram's MTProto API gives an authenticated client the full history of
# any public channel. The `telethon` code below is the real thing: it needs
# the `api_id`/`api_hash` from my.telegram.org (see the participant guide)
# and a phone-number login the first time it runs.

# %%
# OPTIONAL -- live network + Telegram credentials.
# The five public energy/infrastructure channels this lab uses are large,
# institutional, unambiguously public broadcast channels (the chapter
# discusses why we collect channels, not chats):
CHANNELS = ["Ukrenergo", "dtek_ua", "energoatom_ua", "dsns_telegram", "kyivoda"]

if os.environ.get("TELEGRAM_API_ID"):
    import nest_asyncio
    nest_asyncio.apply()        # lets Telethon's event loop run in Jupyter
    from telethon.sync import TelegramClient

    rows = []
    with TelegramClient("lab2", os.environ["TELEGRAM_API_ID"],
                        os.environ["TELEGRAM_API_HASH"]) as client:
        for ch in CHANNELS:
            for msg in client.iter_messages(ch, limit=200):
                rows.append({"channel": ch, "date": msg.date,
                             "text": msg.text, "views": msg.views,
                             "forwards": msg.forwards})
    live_tg = pd.DataFrame(rows)
    print(f"Collected {len(live_tg)} posts")
else:
    print("No TELEGRAM_API_ID in environment — skipping live collection.")

# %% [markdown]
# ### 2b. The cached channel export
#
# This is a real Telethon export: 632 posts from five public Ukrainian
# energy/infrastructure channels, May-July 2024 (the spring energy-strike
# campaign). The five, and why each is here:
#
# - `Ukrenergo` — the national grid operator (outages, damage to the system)
# - `dtek_ua` — DTEK, the largest private utility
# - `energoatom_ua` — the state nuclear operator
# - `dsns_telegram` — the State Emergency Service (response at strike sites)
# - `kyivoda` — the Kyiv Oblast administration (regional government)
#
# `data/acquisition/get_telegram.py` shows exactly how it was pulled. Two
# caveats before you use it. First, these are real posts by real
# institutions: public broadcast, but still real words, so use them for
# source criticism, not to profile anyone. Second, the posts are in
# Ukrainian, and the `text_en_mt` column is an English **machine
# translation** done by an LLM, not checked by a human. It's usually good
# enough to work with, but wrong often enough that you should treat it as a
# rough guide, not ground truth. We'll come back to machine-generated labels
# on Day 4.

# %%
tg = pd.read_parquet(os.path.join(DATA_DIR, "telegram_infra.parquet"))
print(f"{len(tg)} posts, {tg['channel'].nunique()} channels, "
      f"{tg['date'].min():%Y-%m-%d} → {tg['date'].max():%Y-%m-%d}")
tg.sample(5, random_state=1)[["channel", "date", "text", "text_en_mt", "views", "forwards"]]

# %% [markdown]
# ### 2c. Channel ecology
#
# Before reading any individual post, profile the channels: who posts how
# much, when, and to whom? This is metadata Telethon gives you for free,
# and there's a lot you can learn from it before doing any NLP.

# %%
import matplotlib.pyplot as plt

profile = tg.groupby("channel").agg(
    posts=("text", "size"),
    median_views=("views", "median"),
    fwd_rate=("forwards", lambda s: s.sum()),
).assign(fwd_per_1k_views=lambda d: 1000 * d.fwd_rate / tg.groupby("channel")["views"].sum())
profile

# %%
# Posting tempo: posts per day per channel (3-day rolling average).
tempo = (tg.set_index("date").groupby("channel")
           .resample("D").size().unstack(0).fillna(0))
tempo.rolling(3).mean().plot(figsize=(10, 4))
plt.ylabel("posts/day (3-day avg)")
plt.title("Posting tempo by channel")
plt.show()

# %%
# Exercise: hour-of-day posting profiles per channel (df.date.dt.hour,
# groupby, unstack, plot). Which channel never sleeps, and why does that
# make sense given its role?

# try it here

# %% [markdown]
# ### 2d. The forward graph: who amplifies whom
#
# Forwards are Telegram's citation network. The `fwd_from` column records
# where a forwarded post came from. Build the channel-to-source matrix and
# look at who these official channels amplify.
#
# The chapter described *laundering loops*: channel A cites B, B cites A, and
# repetition starts to look like confirmation. You won't find a clean loop
# among these five, since they're official channels that forward upstream
# sources (the President, ministries, individual power plants) rather than
# each other. The pattern you can find here is a single source that several
# of these nominally-independent channels all relay. When the same claim
# shows up carried by five official channels, it's easy to read the
# repetition as confirmation, even though it's really one source amplified
# five times.

# %%
fwd_matrix = pd.crosstab(tg["channel"], tg["fwd_from"])
fwd_matrix

# %%
# Exercise: which single `fwd_from` source is forwarded by the most of our
# five channels? (Hint: `(fwd_matrix > 0).sum()` counts, for each source,
# how many channels forwarded it.) Pull a few of those forwarded posts and
# read them. When several official channels all carry the same source, what
# would a reader who sees it repeated conclude — and would they be right?

# try it here

# %% [markdown]
# ### 2e. Source criticism in practice
#
# Read 10 posts each (use the `text_en_mt` column) from `dtek_ua`, a private
# utility that wants customers and investors to see it as competent and in
# control, and from `dsns_telegram`, the State Emergency Service, whose posts
# foreground rescue and response. For each post, ask the chapter's question:
# what does this channel's incentive structure do to what it reports, and to
# what it leaves out? The exercise below is how you get from vibes to
# something you can actually count:

# %%
# Exercise: add a column `claim_type` to 15 posts of your choosing, coded
# by hand as: "own_report" (channel reports own side's action/observation),
# "enemy_claim" (characterizes the other side), "relay" (forwards/quotes a
# third party), "warning/admin" (alerts, logistics). What's the mix per
# channel? Thursday morning the whole class hand-codes a shared sample and
# measures how much *human* coders disagree with each other; keep your
# claim_type notes, since the hard calls you run into here are what that
# exercise is about.

sample_posts = tg.sample(15, random_state=42).copy()

# try it here

# %% [markdown]
# ### 2f. Store what you collected
#
# Back to this afternoon's storage session: keep an append-safe raw copy
# (the parquet is our "raw" here), plus a queryable SQLite copy.

# %%
import sqlite3

con = sqlite3.connect(os.path.join(OUTPUTS_DIR, "lab2.db"))
tg.astype({"date": str}).to_sql("messages", con, if_exists="replace", index=False)
pd.read_sql("""
    SELECT channel, COUNT(*) AS n, AVG(views) AS avg_views
    FROM messages GROUP BY channel ORDER BY n DESC
""", con)

# %% [markdown]
# ## Capstone variant
#
# 1. Which of this lab's two doors does your measurement target need —
#    a documented API (which one? what auth? what terms?) or platform
#    collection (which channels/accounts? public broadcast or something
#    more sensitive?)? Write down the access path and its constraints.
# 2. If Telegram is relevant to your target: list 3–5 candidate public
#    channels, and for each write ONE sentence on its incentive structure
#    (who runs it, what does it want you to believe?).
# 3. Pull or load *something* today — even 50 records — and save it with a
#    provenance note. Stand-up tomorrow: your source, your access path,
#    your first surprise.
#
# %% [markdown]
# ## If you finish early
#
# - Run `data/acquisition/get_acled.py` with your own ACLED credentials
#   and compare its Kharkiv May-2024 event count to the cached UCDP
#   extract's 529. (Thursday's lab does this comparison properly; getting
#   the raw counts tonight will make you appropriately suspicious early.)
# - Are channel views lognormal? The `views` here are real, so check
#   directly: a histogram of `np.log(tg["views"])` is a fast start. Do the
#   big institutional channels and the smaller ones differ in shape, or just
#   in scale?
# - Write the JSONL "collector" pattern from the storage chapter: a
#   function that appends each post as one JSON line, crash-safe, then a
#   reader that recovers cleanly from a truncated final line.
