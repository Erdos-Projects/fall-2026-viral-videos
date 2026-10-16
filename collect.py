#!/usr/bin/env python3
"""Collect recent #shorts video candidates and track them to ~72h for the virality project.

Each run does three things:
  1. discover videos published in the last DISCOVER_WINDOW_MIN minutes
  2. fetch metadata for any video not seen before  -> videos.csv / channels.csv
  3. snapshot statistics for every video still inside the tracking window
     (0 <= age < CUTOFF_HOURS)                      -> observations/<run>.csv

All collector state lives in CSV files so it is restartable. Missed runs
create gaps that cannot be backfilled; do not infer missing 72h labels.
The workflow serializes runs and commits data to a separate private repository.
"""
import csv
import json
import os
from datetime import datetime, timezone, timedelta

import isodate
from googleapiclient.discovery import build

# ---- config (override with environment variables) ------------------------
DATA_DIR = os.environ.get("DATA_DIR", "data")
TARGET_NEW_PER_RUN = int(os.environ.get("TARGET_NEW_PER_RUN", "150"))  # cap returned search results per run
DISCOVER_WINDOW_MIN = int(os.environ.get("DISCOVER_WINDOW_MIN", "180"))  # overlapping discovery window
CUTOFF_HOURS = float(os.environ.get("CUTOFF_HOURS", "80"))             # stop tracking past this age
MAX_SEARCH_PAGES = int(os.environ.get("MAX_SEARCH_PAGES", "3"))         # 3 * 24 = 72 search calls/day normally
MAX_DURATION_SECONDS = int(os.environ.get("MAX_DURATION_SECONDS", "180"))  # candidates only; no aspect-ratio field
NUM_RETRIES = 3

VIDEOS_CSV = os.path.join(DATA_DIR, "videos.csv")
CHANNELS_CSV = os.path.join(DATA_DIR, "channels.csv")

COUNTRIES_CSV = os.path.join(DATA_DIR, "channel_countries.csv")

COUNTRY_COLS = [
    "channel_id",
    "channel_country",
    "country_checked_at"
]


OBS_DIR = os.path.join(DATA_DIR, "observations")

VIDEO_COLS = ["video_id", "first_seen_at", "published_at", "channel_id", "channel_title",
              "title", "description", "duration", "duration_seconds", "tags"]
CHANNEL_COLS = ["channel_id", "first_seen_at", "channel_title", "channel_created_at",
                "total_channel_views", "subscriber_count", "total_channel_videos"]
OBS_COLS = ["video_id", "observed_at", "age_hours", "view_count", "like_count", "comment_count"]


def now_utc():
    return datetime.now(timezone.utc)


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def known_ids(path, key):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row[key] for row in csv.DictReader(f)}


def append_rows(path, cols, rows):
    if not rows:
        return
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def batched(seq, n=50):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---- API steps ------------------------------------------------------------
def discover(youtube, published_after):
    """IDs of #shorts videos published since `published_after`, newest first, de-duped."""
    ids, token, pages = [], None, 0
    while len(ids) < TARGET_NEW_PER_RUN and pages < MAX_SEARCH_PAGES:
        resp = youtube.search().list(
            part="snippet", type="video", q="#shorts", videoDuration="short", order="date",
            publishedAfter=published_after.strftime("%Y-%m-%dT%H:%M:%SZ"),
            maxResults=50, pageToken=token,
        ).execute(num_retries=NUM_RETRIES)
        ids += [item["id"]["videoId"] for item in resp.get("items", [])]
        pages += 1
        token = resp.get("nextPageToken")
        if not token:
            break
    seen, out = set(), []
    for v in ids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def fetch_new_metadata(youtube, new_ids, seen_at):
    """Metadata rows for brand-new videos, plus the channel ids they belong to."""
    video_rows, channel_ids = [], []
    for batch in batched(new_ids):
        resp = youtube.videos().list(
            part="snippet,contentDetails", id=",".join(batch)
        ).execute(num_retries=NUM_RETRIES)
        for item in resp.get("items", []):
            snippet, content = item["snippet"], item["contentDetails"]
            try:
                duration_seconds = int(isodate.parse_duration(content["duration"]).total_seconds())
            except Exception:
                duration_seconds = None
            # YouTube can classify <=3 min square/vertical videos as Shorts.
            # The Data API does not expose a reliable aspect-ratio/Shorts flag,
            # so these remain Shorts *candidates*, never guaranteed Shorts.
            if duration_seconds is None or not (0 < duration_seconds <= MAX_DURATION_SECONDS):
                continue
            video_rows.append({
                "video_id": item["id"],
                "first_seen_at": seen_at,
                "published_at": snippet["publishedAt"],
                "channel_id": snippet["channelId"],
                "channel_title": snippet.get("channelTitle", ""),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "duration": content["duration"],
                "duration_seconds": duration_seconds,
                "tags": json.dumps(snippet.get("tags", [])),
            })
            channel_ids.append(snippet["channelId"])
    return video_rows, channel_ids


def fetch_new_channels(youtube, channel_ids, seen_at):
    """Channel rows for channels we have not stored before (stats captured once, at first sight)."""
    want = sorted(set(channel_ids) - known_ids(CHANNELS_CSV, "channel_id"))
    rows = []
    for batch in batched(want):
        resp = youtube.channels().list(
            part="snippet,statistics", id=",".join(batch)
        ).execute(num_retries=NUM_RETRIES)
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            rows.append({
                "channel_id": item["id"],
                "first_seen_at": seen_at,
                "channel_title": item["snippet"].get("title", ""),
                "channel_created_at": item["snippet"].get("publishedAt", ""),
                "total_channel_views": stats.get("viewCount", ""),
                "subscriber_count": stats.get("subscriberCount", ""),
                "total_channel_videos": stats.get("videoCount", ""),
            })
    return rows


def fetch_missing_channel_countries(youtube, checked_at):
    """Backfill country for existing and newly discovered channels."""

    if not os.path.exists(CHANNELS_CSV):
        return 0

    # All channels currently stored
    all_ids = known_ids(CHANNELS_CSV, "channel_id")

    # Channels whose country has already been checked
    checked_ids = known_ids(COUNTRIES_CSV, "channel_id")

    # Only query channels not checked previously
    missing_ids = sorted(all_ids - checked_ids)

    saved = 0

    for batch in batched(missing_ids, 50):
        response = youtube.channels().list(
            part="snippet",
            id=",".join(batch)
        ).execute(num_retries=NUM_RETRIES)

        rows = []

        for item in response.get("items", []):
            rows.append({
                "channel_id": item["id"],
                "channel_country": item["snippet"].get("country", ""),
                "country_checked_at": checked_at
            })

        append_rows(COUNTRIES_CSV, COUNTRY_COLS, rows)
        saved += len(rows)

    return saved


def missing_channel_ids():
    """Recover creator metadata missed during a prior partial collection failure."""
    if not os.path.exists(VIDEOS_CSV):
        return []
    have = known_ids(CHANNELS_CSV, "channel_id")
    with open(VIDEOS_CSV, newline="", encoding="utf-8") as f:
        return sorted({r["channel_id"] for r in csv.DictReader(f) if r["channel_id"] not in have})


def tracking_targets(now):
    """(video_id, published_at) for every stored video still inside the tracking window."""
    if not os.path.exists(VIDEOS_CSV):
        return []
    targets = []
    with open(VIDEOS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            age = (now - parse_dt(row["published_at"])).total_seconds() / 3600
            if 0 <= age < CUTOFF_HOURS:
                targets.append((row["video_id"], row["published_at"]))
    return targets


def snapshot(youtube, targets):
    """One statistics row per in-window video. Hidden likes / disabled comments stay blank."""
    published = dict(targets)
    rows = []
    for batch in batched([vid for vid, _ in targets]):
        resp = youtube.videos().list(
            part="statistics", id=",".join(batch)
        ).execute(num_retries=NUM_RETRIES)
        # Record the time of the API response, not the start of the workflow.
        batch_observed_at = now_utc()
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            age = (batch_observed_at - parse_dt(published[item["id"]])).total_seconds() / 3600
            rows.append({
                "video_id": item["id"],
                "observed_at": batch_observed_at.isoformat(),
                "age_hours": round(age, 3),
                "view_count": stats.get("viewCount", ""),
                "like_count": stats.get("likeCount", ""),
                "comment_count": stats.get("commentCount", ""),
            })
    return rows


def main():
    youtube = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])
    now = now_utc()
    observed_at = now.isoformat()
    run_stamp = now.strftime("%Y-%m-%dT%H-%M-%S-%fZ")  # distinct manual runs within a minute

    # 1 + 2: discover and store any new videos/channels. A hiccup here must not
    # cost us the snapshot below, so it is caught and logged.
    new_ids = []
    accepted_count = 0
    try:
        candidates = discover(youtube, now - timedelta(minutes=DISCOVER_WINDOW_MIN))
        known = known_ids(VIDEOS_CSV, "video_id")
        new_ids = [v for v in candidates if v not in known]

        video_rows, _ = fetch_new_metadata(youtube, new_ids, observed_at)
        accepted_count = len(video_rows)
        append_rows(VIDEOS_CSV, VIDEO_COLS, video_rows)
    except Exception as exc:
        print(f"::warning::discovery/metadata failed, continuing to snapshot: {exc!r}")

    # Retry missing creator metadata even if the first attempt failed after
    # videos.csv was already saved; never permanently skip those channels.
    try:
        append_rows(CHANNELS_CSV, CHANNEL_COLS,
                    fetch_new_channels(youtube, missing_channel_ids(), observed_at))
    except Exception as exc:
        print(f"::warning::channel metadata failed; will retry next run: {exc!r}")

    # 3: snapshot everything still in the tracking window (includes the new videos)
    targets = tracking_targets(now)
    observations = snapshot(youtube, targets)
    append_rows(os.path.join(OBS_DIR, f"{run_stamp}.csv"), OBS_COLS, observations)

    print(f"{observed_at}  search candidates not yet stored: {len(new_ids)} | accepted videos: {accepted_count} | tracking: {len(targets)} | "
          f"observations written: {len(observations)}")


if __name__ == "__main__":
    main()
