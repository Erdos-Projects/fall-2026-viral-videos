#!/usr/bin/env python3
"""Generate a FAKE dataset in the same format as the real shorts-data collection.

YouTube's terms of service do not let us publish the data we collected, so this
script writes a synthetic stand-in that EDA_final.ipynb and Modelling.ipynb can
run on end to end. Every video, channel and number is made up: results computed
on this data say nothing about real YouTube Shorts.

The output has the same layout, columns and quirks as the collector's output:
  data/videos.csv, data/channels.csv, data/channel_countries.csv
  data/observations/<run time>.csv   one file per (simulated) hourly collector run

Usage:
    python make_fake_data.py              # writes ../shorts-data-frozen-2026-10-31-c704032/data
    python make_fake_data.py --out DIR    # writes DIR/data instead

The script refuses to write into a folder that already exists, so it can never
overwrite a real data copy.
"""
import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shorts_features import FROZEN_DATASET

SEED = 2026
N_CHANNELS = 2500
N_VIDEOS = 6000

# Collection period of the real frozen copy, run hourly at :53.
FIRST_RUN = pd.Timestamp("2026-10-12 15:53", tz="UTC")
LAST_RUN = pd.Timestamp("2026-11-01 00:53", tz="UTC")
OUTAGE = (pd.Timestamp("2026-10-30 08:53", tz="UTC"), pd.Timestamp("2026-10-31 08:53", tz="UTC"))  # one long gap, as in the real data
DISCOVERY_HOURS = (0.1, 3.0)   # how long after publishing a video is first found
TRACK_HOURS = 80               # the collector stops tracking a video after this age
BATCH_SIZE = 50                # the API returns statistics in batches; each batch gets its own observed_at

# Self-reported channel country; "" means the channel left it blank.
COUNTRY_SHARES = {"US": 0.28, "CA": 0.05, "IN": 0.20, "ID": 0.07, "BR": 0.06, "GB": 0.03, "MX": 0.03, "": 0.28}
HIDDEN_LIKES_SHARE = 0.12

WORDS = [
    "funny", "challenge", "minecraft", "roblox", "cat", "dog", "baby", "prank", "satisfying", "cooking",
    "recipe", "football", "basketball", "dance", "music", "news", "trump", "halloween", "makeup", "car",
    "fishing", "gaming", "animation", "story", "life", "hack", "school", "family", "wait", "end",
    "try", "not", "laugh", "best", "new", "day", "pov", "real", "epic", "fail",
]
HASHTAGS = ["#shorts", "#funny", "#viral", "#fyp", "#trending", "#minecraft", "#comedy", "#news", "#gaming"]
EMOJIS = ["😂", "🔥", "😱", "❤️", "🤣", "👀"]
ID_CHARS = string.ascii_letters + string.digits + "-_"


def random_ids(rng, n, length, prefix=""):
    ids = set()
    while len(ids) < n:
        ids.add(prefix + "".join(rng.choice(list(ID_CHARS), length)))
    return np.array(sorted(ids))[rng.permutation(n)]


def round_sig(values, digits=3):
    """YouTube reports subscriber counts rounded to 3 significant figures."""
    values = np.maximum(values, 1)
    scale = 10.0 ** np.maximum(np.floor(np.log10(values)) - (digits - 1), 0)
    return (np.round(values / scale) * scale).astype(np.int64)


def iso_z(times, fractional):
    """ISO-8601 with a Z suffix; the real files mix timestamps with and without fractional seconds."""
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if fractional else "%Y-%m-%dT%H:%M:%SZ"
    return times.strftime(fmt)


def iso_duration(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return "PT" + (f"{minutes}M" if minutes else "") + (f"{secs}S" if secs or not minutes else "")


def make_title(rng):
    words = list(rng.choice(WORDS, rng.integers(2, 7)))
    title = " ".join(words)
    title = title.upper() if rng.random() < 0.07 else title.capitalize()
    if rng.random() < 0.2:
        title += rng.choice(["!", "?", "...", "!!"])
    if rng.random() < 0.4:
        title += " " + rng.choice(EMOJIS)
    tags = ["#shorts"] + list(rng.choice(HASHTAGS[1:], rng.integers(0, 3), replace=False))
    return title + " " + " ".join(tags)


def make_description(rng):
    if rng.random() < 0.4:
        return np.nan   # creator left the description empty
    words = " ".join(rng.choice(WORDS, rng.integers(5, 20)))
    return f"Fake description: {words} " + " ".join(rng.choice(HASHTAGS, rng.integers(1, 4), replace=False))


def make_tags(rng):
    if rng.random() < 0.6:
        return "[]"
    return json.dumps(list(rng.choice(WORDS, rng.integers(1, 6), replace=False)))


def collector_runs(rng):
    hours = pd.date_range(FIRST_RUN, LAST_RUN, freq="h")
    hours = hours[~((hours > OUTAGE[0]) & (hours < OUTAGE[1]))]
    started = hours + pd.to_timedelta(rng.uniform(20, 40, len(hours)), unit="s")   # each run takes a few seconds to start
    return started.floor("us")


def make_channels(rng):
    quality = rng.normal(size=N_CHANNELS)   # hidden "creator strength", drives both channel stats and video views
    avg_views = np.exp(10.5 + 1.0 * quality + rng.normal(0, 0.8, N_CHANNELS))
    n_videos = np.maximum(1, np.exp(rng.normal(5.5, 1.3, N_CHANNELS))).astype(np.int64)
    channels = pd.DataFrame({
        "channel_id": random_ids(rng, N_CHANNELS, 22, prefix="UC"),
        "channel_title": [f"Fake Channel {i:04d}" for i in range(N_CHANNELS)],
        "channel_created_at": FIRST_RUN - pd.to_timedelta(rng.uniform(30, 6000, N_CHANNELS), unit="D"),
        "total_channel_views": (avg_views * n_videos).astype(np.int64),
        "subscriber_count": round_sig(np.exp(11.5 + 1.5 * quality + rng.normal(0, 0.7, N_CHANNELS))),
        "total_channel_videos": n_videos,
        "channel_country": rng.choice(list(COUNTRY_SHARES), N_CHANNELS, p=list(COUNTRY_SHARES.values())),
    })
    return channels, quality


def make_videos(rng, channels, quality, runs):
    weights = rng.pareto(1.5, N_CHANNELS) + 1   # a few channels post many videos, most post one or two
    channel_idx = rng.choice(N_CHANNELS, N_VIDEOS, p=weights / weights.sum())
    start = FIRST_RUN - pd.Timedelta(hours=2)
    published = start + pd.to_timedelta(rng.uniform(0, (LAST_RUN - start).total_seconds(), N_VIDEOS), unit="s")
    published = published.floor("s")
    duration = np.clip(rng.gamma(2.0, 22.0, N_VIDEOS), 5, 180).astype(int)

    # Views at 72h: stronger creators and shorter videos do a bit better, plus a lot of luck.
    log_duration = np.log(duration)
    appeal = (0.6 * quality[channel_idx] - 0.25 * (log_duration - log_duration.mean()) / log_duration.std()
              + rng.normal(0, 1, N_VIDEOS))
    views_72h = np.exp(11.3 + 1.2 * appeal)
    # Growth curve views(t) = ceiling * (1 - exp(-t / tau)); bigger hits keep growing for longer.
    tau = np.exp(np.log(18) + 0.35 * (appeal - appeal.mean()) / appeal.std() + rng.normal(0, 0.25, N_VIDEOS))
    # Growth after the first hours that early views cannot see, partly explained by the creator,
    # so that models using channel features have something to add beyond 6h views.
    late_boost = np.exp(0.3 * quality[channel_idx] + rng.normal(0, 0.7, N_VIDEOS))

    discovered = published + pd.to_timedelta(rng.uniform(*DISCOVERY_HOURS, N_VIDEOS), unit="h")
    first_run = np.searchsorted(runs, discovered)   # first collector run after the video can be found
    seen = first_run < len(runs)

    videos = pd.DataFrame({
        "video_id": random_ids(rng, N_VIDEOS, 11),
        "channel_idx": channel_idx,
        "published": published,
        "first_run": first_run,
        "duration_seconds": duration,
        "ceiling": views_72h / (1 - np.exp(-72 / tau)),
        "tau": tau,
        "late_boost": late_boost,
        "like_rate": np.exp(rng.normal(np.log(0.03), 0.5, N_VIDEOS)),
        "comment_rate": np.exp(rng.normal(np.log(0.0006), 0.8, N_VIDEOS)),
        "likes_hidden": rng.random(N_VIDEOS) < HIDDEN_LIKES_SHARE,
    })[seen].reset_index(drop=True)
    videos["first_seen_at"] = runs[videos["first_run"].to_numpy()]
    return videos


def write_observations(rng, videos, runs, obs_dir):
    """Simulate every collector run: snapshot each video that is discovered and younger than TRACK_HOURS."""
    obs_dir.mkdir(parents=True)
    published = pd.DatetimeIndex(videos["published"])
    first_run = videos["first_run"].to_numpy()
    last_views = np.zeros(len(videos))
    n_rows = 0
    for k, run_at in enumerate(runs):
        age_at_run = ((run_at - published) / pd.Timedelta(hours=1)).to_numpy()
        active = np.flatnonzero((first_run <= k) & (age_at_run <= TRACK_HOURS))
        if len(active) == 0:
            continue
        observed_at = (run_at + pd.to_timedelta(np.arange(len(active)) // BATCH_SIZE * 0.4, unit="s")).floor("us")
        age = ((observed_at - published[active]) / pd.Timedelta(hours=1)).to_numpy()
        v = videos.iloc[active]

        base = (v["ceiling"].to_numpy() * (1 - np.exp(-age / v["tau"].to_numpy()))
                * v["late_boost"].to_numpy() ** np.minimum(age / 72, 1)
                * np.exp(rng.normal(0, 0.01, len(active))))
        views = np.maximum(base, last_views[active])
        dip = (rng.random(len(active)) < 0.02) & (last_views[active] > 1000)   # YouTube occasionally removes invalid views
        views[dip] = last_views[active][dip] * (1 - rng.uniform(0, 0.002, dip.sum()))
        views = np.round(views)
        last_views[active] = views

        likes = np.round(views * v["like_rate"].to_numpy() * np.exp(rng.normal(0, 0.03, len(active))))
        comments = np.round(views * v["comment_rate"].to_numpy() * np.exp(rng.normal(0, 0.1, len(active))))
        frame = pd.DataFrame({
            "video_id": v["video_id"].to_numpy(),
            "observed_at": [t.isoformat() for t in observed_at],
            "age_hours": np.round(age, 3),
            "view_count": pd.array(views, dtype="Int64"),
            "like_count": pd.array(np.where(v["likes_hidden"], np.nan, likes), dtype="Int64"),
            "comment_count": pd.array(np.where(rng.random(len(active)) < 0.007, np.nan, comments), dtype="Int64"),
        })
        frame.to_csv(obs_dir / f"{run_at.strftime('%Y-%m-%dT%H-%M-%S-%f')}Z.csv", index=False)
        n_rows += len(frame)
    return n_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / FROZEN_DATASET,
                        help="folder to create (default: the frozen-data folder the notebooks look for)")
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists. Refusing to overwrite it; delete it or pass --out with a new folder.")

    rng = np.random.default_rng(SEED)
    runs = collector_runs(rng)
    channels, quality = make_channels(rng)
    videos = make_videos(rng, channels, quality, runs)

    # Keep only channels that posted a tracked video; channel stats are recorded when the channel is first seen.
    videos = videos.join(channels, on="channel_idx")
    first_seen = videos.groupby("channel_id")["first_seen_at"].min()
    channels = channels[channels["channel_id"].isin(first_seen.index)].copy()
    channels["first_seen_at"] = channels["channel_id"].map(first_seen)

    data_dir = args.out / "data"
    data_dir.mkdir(parents=True)
    (args.out / "FAKE_DATA_README.txt").write_text(
        "This folder was generated by make_fake_data.py. All videos, channels and numbers are FAKE.\n"
        "It only exists so the notebooks can be run without the real (non-public) YouTube data.\n",
        encoding="utf-8",
    )

    pd.DataFrame({
        "video_id": videos["video_id"],
        "first_seen_at": [t.isoformat() for t in videos["first_seen_at"]],
        "published_at": iso_z(pd.DatetimeIndex(videos["published"]), fractional=False),
        "channel_id": videos["channel_id"],
        "channel_title": videos["channel_title"],
        "title": [make_title(rng) for _ in range(len(videos))],
        "description": [make_description(rng) for _ in range(len(videos))],
        "duration": videos["duration_seconds"].map(iso_duration),
        "duration_seconds": videos["duration_seconds"],
        "tags": [make_tags(rng) for _ in range(len(videos))],
    }).to_csv(data_dir / "videos.csv", index=False)

    created = pd.DatetimeIndex(channels["channel_created_at"])
    with_fraction = rng.random(len(channels)) < 0.5
    pd.DataFrame({
        "channel_id": channels["channel_id"],
        "first_seen_at": [t.isoformat() for t in channels["first_seen_at"]],
        "channel_title": channels["channel_title"],
        "channel_created_at": np.where(with_fraction, iso_z(created, True), iso_z(created, False)),
        "total_channel_views": channels["total_channel_views"],
        "subscriber_count": channels["subscriber_count"],
        "total_channel_videos": channels["total_channel_videos"],
    }).to_csv(data_dir / "channels.csv", index=False)

    pd.DataFrame({
        "channel_id": channels["channel_id"],
        "channel_country": channels["channel_country"],
        "country_checked_at": [t.isoformat() for t in channels["first_seen_at"]],
    }).to_csv(data_dir / "channel_countries.csv", index=False)

    n_rows = write_observations(rng, videos, runs, data_dir / "observations")
    print(f"Wrote FAKE data to {data_dir}")
    print(f"  {len(channels):,} channels, {len(videos):,} videos, {n_rows:,} snapshots in {len(runs)} run files")


if __name__ == "__main__":
    main()
