# YouTube Shorts virality — data collection

Discovers recently published Shorts *candidates* and tracks each one's view/like/comment counts
from ~0h to ~72h after publication, so that "views at 3 days" can actually be measured.
Runs hourly on GitHub Actions; raw data is pushed to a separate private repo.

## How it works

Each hourly run of `collect.py`:

1. **Discover** — searches for `#shorts` videos published in the last 180 minutes
   (`order=date`, up to 3 search pages). Search indexing delays and
   incomplete results mean that discovery is a *sample*, not all Shorts.
2. **Store new videos/channels** — for IDs not seen before, saves metadata to
   `videos.csv` and the creator's stats to `channels.csv` (each appended once).
3. **Snapshot** — records statistics for every video still inside the tracking
   window (age < `CUTOFF_HOURS`, default 80h) to `observations/<run>.csv`.

State lives in CSVs, so a delayed run resumes, but a sufficiently long outage can
create missing 72-hour observations and must be treated as missing labels.

### Output layout (in the data repo)

```
data/
  videos.csv                     # one row per video (metadata, duration_seconds, tags)
  channels.csv                   # one row per channel (stats at first sight)
  observations/
    2026-10-12T14-17-00-123456Z.csv # one file per run: video_id, observed_at, age_hours, counts
    ...
```

Each video has many observation rows (one per hour it was in-window). The label is NOT computed by this package. For analysis, accept only videos with
a 72-hour observation within a fixed tolerance (e.g. 71–73h); otherwise leave
them unlabeled. Fit any 90th-percentile view threshold using the TRAINING split
only; keep a consistent tie policy and apply that threshold to the test split.
For early features, use only observations at/before the 6h or 24h cutoff.

## One-time setup

1. **Create two repos**
   - this **public** repo for the code, and
   - a **private** repo named e.g. `shorts-data` for the collected CSVs (keeps raw API
     data out of public view; privacy alone does not establish API-policy compliance).

2. **Get a YouTube Data API v3 key** (Google Cloud Console → APIs & Services → Credentials).

3. **Make a token so the workflow can push to the private data repo**
   - GitHub → Settings → Developer settings → **Fine-grained personal access token**
   - Repository access: only `shorts-data`; Permissions: **Contents → Read and write**.

4. **Add both as secrets on the public repo** (Settings → Secrets and variables → Actions):
   - `YOUTUBE_API_KEY` — the API key
   - `DATA_REPO_TOKEN` — the fine-grained token

5. **Edit `.github/workflows/collect.yml`**: set `repository:` to `YOUR_USERNAME/shorts-data`.

6. **Initialise the data repo** with an empty `data/` folder (e.g. commit a `.gitkeep`),
   then push. The schedule starts automatically; use the **Actions** tab → *collect* →
   *Run workflow* to trigger the first run by hand and check the logs.

> Scheduled workflows in a public repo are disabled after 60 days of no repo activity —
> not a concern during an active collection window.

## Run it locally (optional, for testing)

```bash
pip install -r requirements.txt
export YOUTUBE_API_KEY=...         # your key
export DATA_DIR=./data
export CUTOFF_HOURS=80
python collect.py
```

## Quota

`search.list` is capped by default at 100 calls/day; this version requests up to 3 pages
per hourly run (~72 calls/day, excluding retries). Snapshotting is 1
unit per 50 videos, so even ~10k in-window videos hourly stays well under the
10,000-unit/day budget. Quota is not the binding constraint — actual yield depends on search indexing, duplicate results, and pages returned.

## Tests

Install testing dependencies with `pip install -r requirements-test.txt`.
`python test_collect.py /tmp/scratch` runs the whole pipeline against a fake API and a
controllable clock (discovery, de-duplication, the aging window, blank vs zero counts,
a discovery outage, missing channel recovery, duration filtering, and age-valid
labels). On Windows PowerShell, use e.g. `python test_collect.py "$env:TEMP\shorts-test"`.
The mock tests do not prove the real YouTube API is reachable.

## Critical limitations and compliance

- The YouTube Data API has no authoritative `isShort` or aspect-ratio field.
  This collector filters to <=180 seconds but also depends on the `#shorts`
  query; call the data **Shorts candidates**, not verified Shorts.
- Video and channel API data may be non-authorized data subject to YouTube's
  30-day storage/refresh limits. A **private** git repo is not an exemption,
  and deleting a CSV in a new commit **does not remove its git history**.
  Get approval for any required derived metrics or extended statistics retention;
  do not keep raw versioned API data beyond permitted retention. For a short
  trial, plan to delete the *entire private data repository, including history*,
  within the applicable storage window unless compliance permits otherwise.
  Consult the official YouTube API Services Developer Policies and derived-metrics policy.
- Snapshot time is approximate, and some stats/likes may be unavailable.
- Channel statistics are recorded only at first successful creator lookup; treat
  them as time-stamped, potentially stale values in downstream experiments.
- Retries help with transient API errors but cannot guarantee a snapshot near 72h.
- The collector may miss newly uploaded Shorts due to YouTube search indexing or
  sampling; report this selection bias in your research report.
- The 14 original mock checks (now expanded) are engineering smoke tests, **not** validation
  of a real-world 72-hour classifier. The tests also cover duration filtering,
  channel-recovery behavior, and excluding invalid target-age snapshots.
- Before running, rotate any previously exposed/committed API keys and use
  GitHub Actions secrets. Never commit keys or private tokens.

## Daily health checks

Run `python audit.py data-repo/data` in the public repo checkout, or
`python audit.py data` if running from a locally checked-out private data repo.
It reports fresh discovery ages, observed video counts, and available 72-hour
observations. A 72-hour label won't normally be available until 3 days have passed.

## Correct ZIP extraction

Files in this corrected ZIP are at the **ZIP root**. Extract them into your
public GitHub repository root so that `collect.py` and `requirements.txt`
are at the repository root and `.github/workflows/collect.yml` exists exactly
at that path. The earlier ZIP had an extra `collector/` directory; extracting
it without moving contents would cause the supplied workflow to fail.

## Suggested first-run verification checklist

1. Confirm that `collect.py` is in the public repo root, and the workflow file is
   exactly `.github/workflows/collect.yml` on your default branch.
2. Confirm the private repo exists and contains `data/.gitkeep`.
3. Verify secrets are in the **public code repo**, under Settings -> Secrets and variables -> Actions.
4. Change `repository: YOUR_GITHUB_USERNAME/shorts-data` in the workflow to your
   actual private repo owner/name.
5. Trigger Actions -> collect -> Run workflow. If the result is green, open the private
   repo and look for `data/videos.csv`, `data/channels.csv`, `data/observations/*.csv`.
6. Check the Actions log for `new videos`, `tracking`, `observations written`, and
   the `AUDIT` counts. Empty files on first runs can reflect YouTube discovery
   sampling, but repeat failures or zero observations require troubleshooting.
7. Review your Google Cloud quotas during the first day.
8. After approximately 73h, the `AUDIT` output should begin counting valid
   72h snapshots within +/-1 hour of upload. A missing label must remain missing.
