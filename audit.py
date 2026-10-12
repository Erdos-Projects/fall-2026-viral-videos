#!/usr/bin/env python3
"""Quality check for a collected run: python audit.py [data_directory]."""
import csv
from pathlib import Path
from datetime import datetime, timezone
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
video_file = folder / "videos.csv"
if not video_file.exists():
    print("AUDIT: No videos.csv yet (possibly zero discoverable videos).")
    raise SystemExit(0)

def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

with video_file.open(encoding="utf-8", newline="") as f:
    videos = list(csv.DictReader(f))
print(f"AUDIT: {len(videos)} distinct video metadata rows")
ages = []
for v in videos:
    ages.append((parse(v["first_seen_at"]) - parse(v["published_at"])).total_seconds() / 3600)
if ages:
    print(f"AUDIT: first discovery age: min={min(ages):.2f}h median={sorted(ages)[len(ages)//2]:.2f}h max={max(ages):.2f}h")
    print(f"AUDIT: discovered >6h after publication: {sum(x>6 for x in ages)}")
now = datetime.now(timezone.utc)
matured = {v["video_id"] for v in videos if (now - parse(v["published_at"])).total_seconds() / 3600 >= 73}
obs_files = sorted((folder/"observations").glob("*.csv"))
print(f"AUDIT: snapshot files: {len(obs_files)}")
nearest = {}
total = 0
for path in obs_files:
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if not row.get("view_count"):
                continue
            try:
                age = float(row["age_hours"])
            except (TypeError, ValueError):
                continue
            gap = abs(age - 72)
            vid = row["video_id"]
            if vid not in nearest or gap < nearest[vid]:
                nearest[vid] = gap
print(f"AUDIT: observed rows: {total}; videos with views: {len(nearest)}")
print(f"AUDIT: videos old enough for a 72h label (>=73h): {len(matured)}")
print(f"AUDIT: 72h observations within +/-1h: {sum(g<=1 for g in nearest.values())}")
print(f"AUDIT: matured videos missing a 72h observation (+/-1h): {sum(v not in nearest or nearest[v]>1 for v in matured)}")
