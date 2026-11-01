"""Shared data loading and feature building for the YouTube Shorts virality project.

Modelling.ipynb imports this module so that every notebook reads the data, picks
snapshots and builds features the same way. The rules match EDA_final.ipynb:
- timestamps are parsed with format="ISO8601" (the files mix two formats);
- features use the latest snapshot at or before the cutoff (no look-ahead);
- the outcome uses the snapshot nearest 72h, within ± 2h.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_COUNTRIES = ("US", "CA")
FROZEN_DATASET = "shorts-data-frozen-2026-10-31-c704032"
EMOJI_PATTERN = "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF☀-➿⬀-⯿]"


@dataclass
class ShortsData:
    videos: pd.DataFrame
    channels: pd.DataFrame
    countries: pd.DataFrame
    observations: pd.DataFrame
    snapshot_at: pd.Timestamp     # latest observation time: report it with every result
    n_snapshot_files: int


def find_data_dir():
    """Locate the frozen data copy beside this project directory."""
    candidate = Path(__file__).resolve().parent.parent / FROZEN_DATASET / "data"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Frozen data folder not found: {candidate}")


def to_utc(values):
    return pd.to_datetime(values, utc=True, format="ISO8601", errors="coerce")


def load_data(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    videos = pd.read_csv(data_dir / "videos.csv")
    channels = pd.read_csv(data_dir / "channels.csv")
    countries = pd.read_csv(data_dir / "channel_countries.csv")
    files = sorted((data_dir / "observations").glob("*.csv"))
    observations = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    for col in ["first_seen_at", "published_at"]:
        videos[col] = to_utc(videos[col])
    for col in ["first_seen_at", "channel_created_at"]:
        channels[col] = to_utc(channels[col])
    countries["country_checked_at"] = to_utc(countries["country_checked_at"])
    observations["observed_at"] = to_utc(observations["observed_at"])
    for col in ["age_hours", "view_count", "like_count", "comment_count"]:
        observations[col] = pd.to_numeric(observations[col], errors="coerce")
    observations.sort_values(["video_id", "observed_at"], inplace=True, ignore_index=True)
    return ShortsData(videos, channels, countries, observations, observations["observed_at"].max(), len(files))


def snapshot_as_of(obs, hour, window):
    """Latest snapshot with age in [hour - window, hour]: use for features (no look-ahead)."""
    age_col = "exact_age_hours" if "exact_age_hours" in obs else "age_hours"
    candidates = obs[obs[age_col].between(hour - window, hour)].dropna(subset=["view_count"])
    return candidates.sort_values(["video_id", age_col]).drop_duplicates("video_id", keep="last").set_index("video_id")


def snapshot_nearest(obs, hour, tol):
    """Snapshot closest to age `hour` within ± tol: use for outcomes."""
    age_col = "exact_age_hours" if "exact_age_hours" in obs else "age_hours"
    candidates = obs[obs[age_col].between(hour - tol, hour + tol)].dropna(subset=["view_count"])
    candidates = candidates.assign(_distance=(candidates[age_col] - hour).abs())
    return (candidates.sort_values(["video_id", "_distance", age_col])
            .drop_duplicates("video_id").drop(columns="_distance").set_index("video_id"))


def add_country(videos, countries, study_countries=STUDY_COUNTRIES):
    """Join self-reported channel country; country_group is the country code, 'Other' or 'Unknown'."""
    latest = countries.sort_values("country_checked_at").drop_duplicates("channel_id", keep="last")
    out = videos.merge(latest[["channel_id", "channel_country"]], on="channel_id", how="left", validate="many_to_one")
    out["country_group"] = out["channel_country"].where(out["channel_country"].isin(study_countries), "Other")
    out.loc[out["channel_country"].isna(), "country_group"] = "Unknown"
    out["in_study"] = out["channel_country"].isin(study_countries)
    return out


def build_model_table(data, cutoff_hour=6, mid_hour=3, window=1, outcome_hours=72, outcome_tol=2):
    """One row per video (all countries) with a 72h outcome, plus features known at `cutoff_hour`.

    Returns (table, feature_groups); feature_groups maps a group name to its column names.
    """
    obs = data.observations.merge(
        data.videos[["video_id", "published_at"]], on="video_id", how="left", validate="many_to_one"
    )
    obs["exact_age_hours"] = (obs["observed_at"] - obs["published_at"]).dt.total_seconds() / 3600
    outcome = (
        snapshot_nearest(obs, outcome_hours, outcome_tol)[["view_count", "exact_age_hours", "observed_at"]]
        .rename(columns={
            "view_count": f"views_{outcome_hours}h",
            "exact_age_hours": f"outcome_age_{outcome_hours}h",
            "observed_at": f"outcome_observed_at_{outcome_hours}h",
        })
    )
    mid_snapshot = (
        snapshot_as_of(obs, mid_hour, window)[["view_count", "exact_age_hours", "observed_at"]]
        .rename(columns={
            "view_count": f"views_{mid_hour}h",
            "exact_age_hours": f"feature_age_{mid_hour}h",
            "observed_at": f"feature_observed_at_{mid_hour}h",
        })
    )
    cutoff_snapshot = (
        snapshot_as_of(obs, cutoff_hour, window)[
            ["view_count", "like_count", "comment_count", "exact_age_hours", "observed_at"]
        ]
        .rename(columns={
            "view_count": f"views_{cutoff_hour}h",
            "like_count": f"like_count_{cutoff_hour}h",
            "comment_count": f"comment_count_{cutoff_hour}h",
            "exact_age_hours": f"feature_age_{cutoff_hour}h",
            "observed_at": f"feature_observed_at_{cutoff_hour}h",
        })
    )
    channel_data = data.channels.drop(columns=["channel_title"]).rename(
        columns={"first_seen_at": "channel_stats_observed_at"}
    )
    table = (
        add_country(data.videos, data.countries).set_index("video_id")
        .join(outcome, how="inner")
        .join([mid_snapshot, cutoff_snapshot])
        .reset_index()
        .merge(channel_data, on="channel_id", how="left", validate="many_to_one")
    )
    return add_features(table, cutoff_hour, mid_hour)


def add_features(df, cutoff_hour=6, mid_hour=3):
    """Build feature groups; every value is known by `cutoff_hour` hours after publishing."""
    c, m = cutoff_hour, mid_hour
    views = df[f"views_{c}h"]
    likes = df[f"like_count_{c}h"]
    comments = df[f"comment_count_{c}h"]
    titles = df["title"].fillna("")
    hour_et = df["published_at"].dt.tz_convert("America/New_York").dt.hour   # reference clock only
    cutoff_at = df["published_at"] + pd.Timedelta(hours=c)
    channel_stats_available = df["channel_stats_observed_at"].le(cutoff_at).fillna(False)
    subscribers = df["subscriber_count"].where(channel_stats_available)
    channel_views = df["total_channel_views"].where(channel_stats_available)
    channel_videos = df["total_channel_videos"].where(channel_stats_available)

    groups = {
        "early performance": {
            f"log_views_{c}h": np.log1p(views),
            f"view_growth_{m}_to_{c}h": np.log1p(views) - np.log1p(df[f"views_{m}h"]),   # log ratio
        },
        "engagement": {   # raw counts; intentionally not normalized by view count
            f"like_count_{c}h": likes,
            f"comment_count_{c}h": comments,
            f"likes_hidden_{c}h": likes.isna() & views.notna(),
        },
        "engagement log1p alternative": {
            f"log_like_count_{c}h": np.log1p(likes),
            f"log_comment_count_{c}h": np.log1p(comments),
        },
        "creator": {   # channel statistics as of first sighting
            "log_subscribers": np.log1p(subscribers),
            "channel_age_days": (df["published_at"] - df["channel_created_at"]).dt.total_seconds() / 86400,
            "log_channel_views": np.log1p(channel_views),
            "log_channel_avg_views": np.log1p(channel_views / channel_videos.clip(lower=1)),
        },
        "timing": {   # hour posted, as a cycle so 23:00 and 00:00 are neighbours
            "pub_hour_sin": np.sin(2 * np.pi * hour_et / 24),
            "pub_hour_cos": np.cos(2 * np.pi * hour_et / 24),
        },
        "content": {
            "duration_seconds": df["duration_seconds"],
            "n_hashtags": (titles + " " + df["description"].fillna("")).str.count(r"#\w+"),
            "title_len": titles.str.len(),
            "emoji_count": titles.str.count(EMOJI_PATTERN),
        },
    }
    columns = {name: values.astype(float) for group in groups.values() for name, values in group.items()}
    feature_groups = {group: list(cols) for group, cols in groups.items()}
    return df.assign(channel_stats_available_by_6h=channel_stats_available, **columns), feature_groups
