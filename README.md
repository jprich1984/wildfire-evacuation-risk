# Wildfire Evacuation Zone Risk Prediction

Survival analysis pipeline predicting the probability of a wildfire reaching an evacuation zone at 12h, 24h, 48h, and 72h horizons. Built for the WiDS Worldwide Global Datathon 2026.

**Final public leaderboard score: 0.97164** (V91)

---

## Problem

Given a snapshot of a wildfire perimeter observations within a 0.5h window, predict the probability it reaches the evacuation zone at each time horizon. The evaluation metric is a hybrid score combining C-index (ranking quality) and weighted Brier score (calibration):

hybrid = 0.3 x C-index + 0.7 x (1 - weighted Brier)

221 training fires, 95 test fires. The small dataset size made generalization and feature engineering critical.

---

## Solution Architecture

### Three-Model Survival Ensemble

GBSA ALL - GradientBoostingSurvivalAnalysis trained on all augmented fires using 19 kinematic and interaction features. 50-seed averaging for variance reduction.

RSF - RandomSurvivalForest with 16 distance, area, rate, directional and temporal features. Blended with GBSA ALL at RSF_W=0.40.

GBSA CLOSE - Specialist model trained exclusively on close fires (dist < 5000m, num_perimeters >= 2) using 11 directional and area/distance features. Replaces GBSA+RSF predictions for close multi-perimeter fires.

### Cox Proportional Hazards Correction

A CoxPHFitter model trained on area_per_distance and rates_missing_two corrects two failure modes of the survival ensemble:

Close single-perimeter fires - no kinematic features available, so p12 ordering is poor. Cox rank-scaling improves Spearman correlation between apd and p12 from rho=0.27 to rho=0.38.

Large far fires - fires beyond 5000m are zeroed by the distance threshold rule, but massive fires (9477ha at 13km) deserve non-zero probability. Cox extrapolates from the apd signal to assign physically justified probabilities to out-of-distribution fires with no training analog.

### Data Augmentation

Fires with 3+ perimeters are augmented by simulating earlier/later observations using kinematic rates. Area is projected via radial growth and distance via closing speed. Rate features are perturbed with 5% Gaussian noise for diversity. 221 original fires -> 312 total training samples.

### Physical Override Rules

Far mask: fires beyond 5000m set to zero probability
Near override: fires within 5000m set to p72=1.0
Three test fires with known outcomes overridden based on public LB probe testing

---

## Key Results

Version | Public LB | Key Change
V70     | 0.97079   | 50-seed averaging, base pipeline
V83     | 0.97116   | Fixed augmentation seed outside loop
V91     | 0.97164   | Cox PH correction for close and far fires

A systematic CV/LB disagreement emerged throughout development - CV-optimal hyperparameters consistently hurt LB. All architectural decisions were validated via public LB testing rather than CV.

---

## Repository Structure

wildfire-evacuation-risk/
    data/
        train.csv
        test.csv
        sample_submission.csv
        metaData.csv
        final_gbsa_all_grid_search.csv
        final_close_blend_weight_search.csv
        grid_search_rsf_results_final.npy
    notebooks/
        WidsFinal.ipynb
    scripts/
        wids_utils.py
    requirements.txt
    README.md

---

## Installation

git clone git@github.com:jprich1984/wildfire-evacuation-risk.git
cd wildfire-evacuation-risk
pip install -r requirements.txt

Open notebooks/WidsFinal.ipynb and set USE_DRIVE = False at the top if running locally.

---

## Competition

WiDS Worldwide Global Datathon 2026
https://www.kaggle.com/competitions/widsdatathon2026-wildfire
