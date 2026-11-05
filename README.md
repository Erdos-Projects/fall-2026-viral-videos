
# Predicting YouTube Shorts Virality

**Erdős Projects · Fall 2026**

Can we predict which YouTube Shorts will go viral using only their first **six hours** of activity? We define *viral* as reaching the **top 10% of US/Canadian videos by views at approximately 72 hours**, with the threshold determined from training data.

## Dataset

An automated collector uses the **YouTube Data API v3**, GitHub Actions, and Google Apps Script to discover `#shorts` candidates and record views, likes, and comments over time. The frozen dataset analyzed here (October 12–November 1, 2026) contains **30,709 videos**, **15,687 channels**, and **2.17 million observations**. Raw data is stored separately in a private repository.

## Approach

- **EDA:** Explore view growth, engagement, creator characteristics, content, and upload timing.
- **Features:** Six-hour views and growth, likes/comments, channel statistics, duration, hashtags, title length, emojis, and publication time.
- **Models:** Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Evaluation:** Channel-grouped cross-validation and a chronological, channel-disjoint holdout. The final test contains **562 US/Canadian videos**, including **48 viral videos**.

## Results

The selected model was **XGBoost (max depth 3)** trained on worldwide videos with all feature groups. Its development cross-validation average precision was **0.752 ± 0.114** (mean ± fold standard deviation).

| Held-out metric | 6h views baseline | XGBoost |
|---|---:|---:|
| Average precision (PR-AUC) | 0.685 | **0.757** |
| ROC-AUC | 0.880 | **0.911** |
| Precision@10% | 0.561 | **0.614** |
| Recall@10% | 0.667 | **0.729** |

**Takeaway:** Early views are a strong predictor; combining them with additional features modestly improved results on the held-out set.

## Repository

| File | Purpose |
|---|---|
| `collect.py` | YouTube API collection |
| `audit.py`, `test_collect.py` | Data checks and collector tests |
| `EDA_final.ipynb` | Exploratory data analysis |
| `shorts_features.py` | Shared data loading and feature engineering |
| `Modelling.ipynb` | Cross-validation, ablations, and final evaluation |
| `.github/workflows/collect.yml` | Collection workflow |

## Setup

```bash
pip install -r requirements.txt
pip install pandas numpy matplotlib scikit-learn xgboost notebook
```

The analysis notebooks require the private frozen data copy at `../shorts-data-frozen-2026-10-31-c704032/data/`. Without access to that dataset, the published results can be inspected in the notebooks but not fully reproduced.

## Limitations

The sample consists of `#shorts` search candidates, **not independently verified Shorts**. Results are limited by the short collection period, missing or revised engagement counts, and approximate 72-hour observations. The 72-hour publication-time embargo is not sufficient by itself to guarantee availability of outcomes observed up to 74 hours after upload.


## Acknowledgements

We sincerely thank the **Erdős Institute** for providing us with the opportunity to work on this project and **Hannah Lloyd** for her continued guidance, support, and valuable feedback throughout the project.

