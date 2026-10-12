"""End-to-end test of collect.py against a fake YouTube client and a controllable clock.

usage: test_collect.py <scratch_dir>
"""
import os
import sys
import types
from datetime import datetime, timezone, timedelta

SCRATCH = sys.argv[1]
import shutil
shutil.rmtree(SCRATCH, ignore_errors=True)
os.makedirs(SCRATCH)

# config must be set before importing collect (module reads env at import time)
os.environ["DATA_DIR"] = os.path.join(SCRATCH, "data")
os.environ["CUTOFF_HOURS"] = "3"
os.environ["DISCOVER_WINDOW_MIN"] = "90"
os.environ["YOUTUBE_API_KEY"] = "test-key"

import collect

CLOCK = [datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)]
collect.now_utc = lambda: CLOCK[0]


class Req:
    def __init__(self, fn):
        self.fn = fn

    def execute(self, http=None, num_retries=0):
        return self.fn()


class Fake:
    """Holds a registry of published videos. search() returns everything published within the
    discovery window of the current clock -- so recent videos are returned on two consecutive
    hourly runs, exercising de-duplication."""

    def __init__(self):
        self.pub = {}          # video_id -> publishedAt (datetime)
        self.channel = {}      # video_id -> channel_id
        self.hidden_likes = set()
        self.long_videos = set()
        self.fail_search = False
        self.fail_channels = False

    def publish(self, vid, channel, minutes_ago=5):
        self.pub[vid] = CLOCK[0] - timedelta(minutes=minutes_ago)
        self.channel[vid] = channel

    def search(self):
        outer = self

        class S:
            def list(self, **kw):
                def run():
                    if outer.fail_search:
                        raise RuntimeError("simulated search outage")
                    assert kw["part"] == "snippet", "YouTube search.list requires part=snippet"
                    assert kw["type"] == "video"
                    after = datetime.strptime(kw["publishedAfter"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    fresh = sorted([v for v, p in outer.pub.items() if p >= after],
                                   key=lambda v: outer.pub[v], reverse=True)
                    return {"items": [{"id": {"videoId": v}} for v in fresh[:50]]}
                return Req(run)
        return S()

    def videos(self):
        outer = self

        class V:
            def list(self, part, id):
                def run():
                    ids = id.split(",")
                    assert len(ids) <= 50
                    items = []
                    for v in ids:
                        if v not in outer.pub:
                            continue
                        item = {"id": v}
                        if "snippet" in part:
                            item["snippet"] = {
                                "publishedAt": outer.pub[v].strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "channelId": outer.channel[v], "channelTitle": f"Chan {outer.channel[v]}",
                                "title": f"title {v} #shorts", "description": "line1\nline2, with comma",
                                "tags": ["alpha", "beta"],
                            }
                        if "contentDetails" in part:
                            item["contentDetails"] = {"duration": "PT4M" if v in outer.long_videos else "PT45S"}
                        if "statistics" in part:
                            age_h = (CLOCK[0] - outer.pub[v]).total_seconds() / 3600
                            st = {"viewCount": str(1000 + int(age_h * 500)),
                                  "likeCount": str(100 + int(age_h * 20)),
                                  "commentCount": str(int(age_h))}
                            if v in outer.hidden_likes:
                                del st["likeCount"]
                            item["statistics"] = st
                        items.append(item)
                    return {"items": items}
                return Req(run)
        return V()

    def channels(self):
        outer = self

        class C:
            def list(self, part, id):
                def run():
                    if outer.fail_channels:
                        raise RuntimeError("simulated channel outage")
                    return {"items": [{"id": c, "snippet": {"title": f"Chan {c}", "publishedAt": "2020-01-01T00:00:00Z"},
                                       "statistics": {"viewCount": "5000000", "subscriberCount": "120000", "videoCount": "800"}}
                                      for c in id.split(",")]}
                return Req(run)
        return C()


fake = Fake()
collect.build = lambda *a, **k: fake

import csv
import glob
import json


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def obs_files():
    return sorted(glob.glob(os.path.join(collect.OBS_DIR, "*.csv")))


results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


# ---- 5 hourly runs. 2 new videos published just before each run. ---------
# A,B share channel c0 (tests channel de-dup); the rest spread across channels.
plan = {0: [("A", "c0"), ("B", "c0")], 1: [("C", "c1"), ("D", "c2")],
        2: [("E", "c3"), ("F", "c4")], 3: [("G", "c5"), ("H", "c6")],
        4: [("I", "c7"), ("J", "c8")]}
fake.hidden_likes.add("C")
fake.long_videos.add("LONG")

for hour in range(5):
    CLOCK[0] = datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc) + timedelta(hours=hour)
    for vid, chan in plan[hour]:
        fake.publish(vid, chan)
    if hour == 0:
        fake.publish("LONG", "c99")
    collect.main()

videos = read_csv(collect.VIDEOS_CSV)
channels = read_csv(collect.CHANNELS_CSV)
vids = [r["video_id"] for r in videos]

check("videos.csv has all 10, each exactly once (re-returned videos not duplicated)",
      sorted(vids) == list("ABCDEFGHIJ"), f"{len(vids)} rows")
check("channels de-duplicated (c0 shared by A and B appears once)",
      sorted(r["channel_id"] for r in channels) == [f"c{i}" for i in range(9)],
      f"{len(channels)} channels")
check("duration parsed to seconds", {r["duration_seconds"] for r in videos} == {"45"})
check("videos longer than 180 sec excluded", "LONG" not in vids)
check("tags stored as JSON list", json.loads(videos[0]["tags"]) == ["alpha", "beta"])
check("description with newline/comma survives CSV round-trip",
      videos[0]["description"] == "line1\nline2, with comma")
check("one observations file per run", len(obs_files()) == 5, f"{len(obs_files())} files")

# window + aging: A (published at hour 0) should be snapshotted at hours 0,1,2 (age<3) but not 3,4
a_hours = []
for i, path in enumerate(obs_files()):
    ids = {r["video_id"] for r in read_csv(path)}
    if "A" in ids:
        a_hours.append(i)
check("tracking window drops videos once age reaches the 3h cutoff",
      a_hours == [0, 1, 2], f"A observed in runs {a_hours}")

# per-run counts: in-window videos at each run (age<3h), given 2 new per hour
# h0:{A,B} h1:{A,B,C,D} h2:{A,B,C,D,E,F} h3:{C,D,E,F,G,H} h4:{E,F,G,H,I,J}
expected = [2, 4, 6, 6, 6]
counts = [len(read_csv(p)) for p in obs_files()]
check("per-run observation counts match the rolling window", counts == expected, f"{counts}")

# age recorded and increasing for a tracked video
c_ages = []
for path in obs_files():
    for r in read_csv(path):
        if r["video_id"] == "C":
            c_ages.append(float(r["age_hours"]))
check("age_hours increases ~1h per run for a tracked video", c_ages == sorted(c_ages) and len(c_ages) == 3, f"{c_ages}")

# hidden likes stay blank, not zero
c_rows = [r for p in obs_files() for r in read_csv(p) if r["video_id"] == "C"]
check("hidden likes stay blank (not 0)", all(r["like_count"] == "" for r in c_rows))

# ---- discovery outage must not cost the snapshot --------------------------
CLOCK[0] = datetime(2026, 10, 12, 5, 0, tzinfo=timezone.utc)
before = len(obs_files())
fake.fail_search = True
collect.main()
fake.fail_search = False
check("snapshot still written when discovery fails", len(obs_files()) == before + 1)
# at hour 5: in-window = videos with age<3 => G,H (pub h3, age2), I,J (pub h4, age1) = 4
check("snapshot after outage used stored videos (G,H,I,J)",
      len(read_csv(obs_files()[-1])) == 4, f"{len(read_csv(obs_files()[-1]))}")

# ---- the collected CSVs load and a 72h-style label is computable ----------
import pandas as pd
obs = pd.concat([pd.read_csv(p) for p in obs_files()], ignore_index=True)
vdf = pd.read_csv(collect.VIDEOS_CSV)
for col in ["view_count", "like_count", "comment_count"]:
    obs[col] = pd.to_numeric(obs[col], errors="coerce")
check("observations load numeric; hidden likes are NaN",
      obs["view_count"].dtype.kind in "if" and obs.loc[obs.video_id == "C", "like_count"].isna().all())

# Label discipline: there must be an observation close to the target age.
# A nearest snapshot with a large gap is NOT a valid measured 72h outcome.
TARGET = 2.0  # accelerated 2h proxy for the 72h production target
with_gap = obs.assign(gap=(obs.age_hours - TARGET).abs())
nearest = with_gap.sort_values("gap").drop_duplicates("video_id")
valid = nearest.loc[nearest.gap <= 0.25].copy()
check("age tolerance excludes videos that have not reached the target age",
      len(valid) == 8 and set(valid.video_id) == set("ABCDEFGH"), f"{len(valid)} valid of {len(nearest)} nearest rows")
check("out-of-tolerance observations are not accidentally labeled",
      (valid["gap"] <= 0.25).all())
# Viral thresholds must be fit on the training split later; this is only a
# smoke test that the observations can form an age-valid outcome table.
valid["example_viral"] = valid.view_count >= valid.view_count.quantile(0.90)
check("age-valid synthetic outcome table can be formed", valid["example_viral"].any())

# Fail channel retrieval after storing video, then recover next hourly run.
CLOCK[0] = datetime(2026, 10, 12, 6, 0, tzinfo=timezone.utc)
fake.publish("K", "c9")
fake.fail_channels = True
collect.main()
check("video metadata survives a channel endpoint outage", "K" in {v["video_id"] for v in read_csv(collect.VIDEOS_CSV)})
check("failed channel request does not fabricate metadata", "c9" not in {v["channel_id"] for v in read_csv(collect.CHANNELS_CSV)})
CLOCK[0] += timedelta(hours=1)
fake.fail_channels = False
collect.main()
check("missing channel metadata is recovered in a later run", "c9" in {v["channel_id"] for v in read_csv(collect.CHANNELS_CSV)})

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
