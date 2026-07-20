"""get_telegram.py -- Collect public Telegram channel posts on damage to
Ukrainian infrastructure (Lab 2, Part 2).

This replaces the *synthetic* export in
``data/cached/telegram_channels.parquet`` with real messages pulled from a
handful of large, institutional, unambiguously public broadcast channels
that report on strikes against Ukrainian energy and civilian infrastructure.

Data source
-----------
Telegram's MTProto API, via Telethon. An authenticated client can read the
full public history of any public *channel* (broadcast, one-to-many) -- the
same access an ordinary reader has by opening the channel in the app, just
programmatic. We collect channels, never private chats or groups.

Channels (all public broadcast, institutional or established OSINT):

- ``Ukrenergo``      National grid operator. Official reports on damage to
                     the power system and emergency outage schedules -- the
                     primary energy-infrastructure-damage source.
- ``dsns_telegram``  State Emergency Service (DSNS). Rescue/response at
                     strike sites: fires, collapses, casualties.
- ``dtek_ua``        DTEK, the largest private energy company. Damage to and
                     repair of thermal/distribution assets.
- ``energoatom_ua``  Energoatom, the state nuclear operator. Nuclear-plant
                     safety and grid-connection damage.
- ``kyivoda``        Kyiv Oblast Military Administration. Regional-government
                     voice on strikes and local infrastructure impact.

The five span deliberately different incentive structures -- national
utility, emergency responder, private corporation, nuclear operator,
regional government -- which is exactly the contrast Lab 2's source-criticism
exercise turns on.

The list is a *starting point*, deliberately biased toward official/utility
voices so the classroom sample is defensible. Edit ``CHANNELS`` to taste;
the script resolves and validates each one before collecting and skips any
that do not resolve to a public broadcast channel.

Window
------
By default we collect the same May--July 2024 window the rest of Lab 2 uses
(the Vovchansk offensive; UCDP + DeepStateMap cells share it), so the
channel-ecology and posting-tempo plots still line up. This overlaps the
spring-2024 energy-strike campaign, so infrastructure damage is genuinely in
frame. Change ``WINDOW_START`` / ``WINDOW_END`` for a different period (e.g.
the 2024--25 winter energy war).

Credentials
-----------
Reads ``TELEGRAM_API_ID`` / ``TELEGRAM_API_HASH`` from the environment
(my.telegram.org). For convenience they may instead sit in
``data/acquisition/telegram.txt`` (gitignored) as ``KEY=value`` lines; we
fall back to parsing that. The first ever run needs an interactive phone
login, which writes ``lab2.session`` at the repo root; every run after is
non-interactive. The session file holds an auth token -- it is gitignored
(``*.session``) and must never be committed.

Ethics / privacy
----------------
These are *real posts by real people and institutions*, unlike the synthetic
file this replaces. We restrict to public broadcast channels (already
world-readable), keep only channel-level metadata plus post text, and use it
for classroom source-criticism. Do not redistribute the raw cache beyond the
course, and do not use it to profile or target individuals. Respect
Telegram's ToS and each channel's context.

Output
------
``data/cached/telegram_infra.parquet`` -- one row per post. The seven columns
below match the synthetic export exactly, so downstream Lab 2 cells bind
unchanged::

    channel  date  text  views  forwards  fwd_from  msg_id

The posts are in Ukrainian (some Russian). A *second, separate* pass adds an
eighth column, ``text_en_mt`` -- an English **machine translation** produced
by an LLM (Claude Sonnet), placed right after ``text``. It is deliberately
named ``_mt`` and documented as machine-translated and NOT human-verified, so
students never mistake it for ground truth. This script only does collection;
the translation step is run separately (see the project's translation pass /
``translate_telegram.py`` if present).

(Kept as a *separate* file for now; swapping it in for the synthetic parquet
also means updating the Lab 2 prose that calls that file synthetic.)
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from telethon.sync import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError, UsernameInvalidError
from telethon.tl.types import Channel

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "cached" / "telegram_infra.parquet"
SESSION = str(REPO_ROOT / "lab2")            # reuse the lab's login session
CREDS_TXT = Path(__file__).resolve().parent / "telegram.txt"

# Public broadcast channels reporting damage to Ukrainian infrastructure.
CHANNELS = [
    "Ukrenergo",       # national grid operator (energy damage / outages)
    "dsns_telegram",   # State Emergency Service (response at strike sites)
    "dtek_ua",         # DTEK, private energy company (damage + repair)
    "energoatom_ua",   # Energoatom, state nuclear operator
    "kyivoda",         # Kyiv Oblast Military Administration (regional govt)
]

# Collection window (UTC). Matches the rest of Lab 2's May-2024 frame.
WINDOW_START = datetime(2024, 5, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2024, 7, 15, tzinfo=timezone.utc)

MAX_PER_CHANNEL = 200      # 5 channels x 200 = 1,000-post ceiling for the class
SLEEP_BETWEEN = 2.0        # be polite between channels; also eases FloodWait


def load_credentials() -> tuple[str, str]:
    """Return (api_id, api_hash) from the environment, or telegram.txt."""
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if api_id and api_hash:
        return api_id, api_hash

    if CREDS_TXT.exists():
        kv = {}
        for line in CREDS_TXT.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
        api_id = api_id or kv.get("TELEGRAM_API_ID")
        api_hash = api_hash or kv.get("TELEGRAM_API_HASH")

    if not (api_id and api_hash):
        sys.exit(
            "No Telegram credentials. Set TELEGRAM_API_ID / TELEGRAM_API_HASH "
            f"in the environment, or put them in {CREDS_TXT}."
        )
    return api_id, api_hash


def fwd_source(msg) -> str | None:
    """Best-effort username/title of the channel a post was forwarded from."""
    fwd = getattr(msg, "forward", None)
    if fwd is None:
        return None
    chat = getattr(fwd, "chat", None)
    if chat is not None:
        return getattr(chat, "username", None) or getattr(chat, "title", None)
    # Forward from a channel we can't fully resolve, or from a user/name.
    return getattr(fwd, "from_name", None)


def validate(client, name: str):
    """Resolve a channel and confirm it's a public broadcast channel.

    Returns the entity, or None (with a printed reason) if it should be
    skipped -- unresolvable, private, a megagroup/chat, or non-public.
    """
    try:
        ent = client.get_entity(name)
    except (UsernameInvalidError, ValueError):
        print(f"  ! {name}: does not resolve -- skipping")
        return None
    except ChannelPrivateError:
        print(f"  ! {name}: private/inaccessible -- skipping")
        return None

    if not isinstance(ent, Channel):
        print(f"  ! {name}: not a channel (chat/user) -- skipping")
        return None
    if getattr(ent, "megagroup", False):
        print(f"  ! {name}: megagroup (a group chat, not broadcast) -- skipping")
        return None
    if getattr(ent, "username", None) is None:
        print(f"  ! {name}: no public username -- skipping")
        return None

    subs = getattr(ent, "participants_count", None)
    print(f"  + {name}: public broadcast '{ent.title}'"
          + (f" (~{subs:,} subscribers)" if subs else ""))
    return ent


def collect_channel(client, ent) -> list[dict]:
    """Pull posts in [WINDOW_START, WINDOW_END] from a validated channel."""
    rows = []
    # iter_messages yields newest-first; offset_date returns posts *older*
    # than WINDOW_END. We stop once we cross WINDOW_START.
    for msg in client.iter_messages(ent, offset_date=WINDOW_END,
                                    limit=MAX_PER_CHANNEL):
        if msg.date < WINDOW_START:
            break
        if not (msg.text or "").strip():
            continue  # skip pure media / service messages with no text
        rows.append({
            "channel": ent.username,
            "date": msg.date.replace(tzinfo=None),   # match synthetic (naive)
            "text": msg.text,
            "views": msg.views,
            "forwards": msg.forwards,
            "fwd_from": fwd_source(msg),
            "msg_id": msg.id,
        })
    return rows


def main() -> None:
    api_id, api_hash = load_credentials()

    all_rows = []
    with TelegramClient(SESSION, api_id, api_hash) as client:
        for name in CHANNELS:
            print(f"Channel {name} ...")
            ent = validate(client, name)
            if ent is None:
                continue
            try:
                rows = collect_channel(client, ent)
            except FloodWaitError as e:
                print(f"  ! FloodWait {e.seconds}s on {name}; sleeping then retrying once")
                time.sleep(e.seconds + 1)
                rows = collect_channel(client, ent)
            print(f"    collected {len(rows)} posts in window")
            all_rows.extend(rows)
            time.sleep(SLEEP_BETWEEN)

    if not all_rows:
        sys.exit("No posts collected -- check channels, window, and login.")

    df = pd.DataFrame(all_rows, columns=[
        "channel", "date", "text", "views", "forwards", "fwd_from", "msg_id",
    ]).sort_values(["channel", "date"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nWrote {OUT_PATH} ({len(df)} posts, "
          f"{df['channel'].nunique()} channels, "
          f"{df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d})")
    print("\nPer-channel counts:")
    print(df["channel"].value_counts().to_string())


if __name__ == "__main__":
    main()
