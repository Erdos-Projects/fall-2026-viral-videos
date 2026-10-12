# Predicting YouTube Shorts Virality

**Erdos Projects | Fall 2026**

## Overview

**What makes a video go viral, and can we predict it early?**

This project investigates which content, creator, timing, and engagement factors are associated with viral YouTube Shorts, and whether machine learning can predict a video's popularity within its first few hours.

We aim to predict whether a video will be among the **top 10% by views at 72 hours**, using information available at 6 or 24 hours after publication.

## Methodology

1. **Data Collection:** Discover recently published YouTube Shorts candidates and track their views, likes, and comments over time.
2. **Exploratory Data Analysis:** Investigate patterns in content, creator characteristics, upload timing, and engagement.
3. **Feature Engineering:** Extract early engagement metrics, view growth rates, and video/channel features.
4. **Machine Learning:** Compare Logistic Regression, Random Forest, XGBoost, and simple baseline models.
5. **Evaluation:** Evaluate how accurately viral videos can be identified early.

## Data Collection

Our automated pipeline uses:

**Google Apps Script → GitHub Actions → YouTube Data API → Private CSV Repository**

The collector runs approximately hourly, discovering new videos and tracking their statistics until around 80 hours after publication.

The data is stored in three components:

- `videos.csv` — Video metadata
- `channels.csv` — Creator information
- `observations/*.csv` — Time-series engagement statistics

*Note: Videos are Shorts candidates identified through search and duration filters, not independently verified Shorts.*

## Technology Stack

Python, YouTube Data API v3, Pandas, Google Colab, GitHub Actions, Google Apps Script, Scikit-learn, XGBoost.

## Current Status

- [x] YouTube API integration
- [x] Automated hourly data collection
- [x] Private data storage
- [x] Initial collection validation
- [ ] Data cleaning and EDA
- [ ] Feature engineering and virality labeling
- [ ] ML training and evaluation

**Status:** Data collection is underway. Analysis and modeling are upcoming.

---

*Research project in progress — October 2026.*
