"""
WiDS Wildfire Competition - Utility Functions
==============================================

Reusable functions for wildfire prediction modeling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_engineered_features(df):
    df = df.copy()
    
    dist = df['dist_min_ci_0_5h'].clip(lower=1)
    
    # ── Existing features ─────────────────────────────────────────────────
    df['threat_score'] = df['closing_speed_m_per_h'] / (dist + 1)
    df['directional_threat'] = df['alignment_cos'] * (1 / (dist + 1))
    df['hours_to_reach'] = dist / (df['closing_speed_abs_m_per_h'] + 0.1)
    df['fire_intensity'] = df['area_growth_rate_ha_per_h'] * df['radial_growth_rate_m_per_h']
    df['is_night'] = ((df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)).astype(int)
    df['is_peak_fire_season'] = df['event_start_month'].isin([6, 7, 8, 9]).astype(int)
    df['cross_track_abs'] = df['cross_track_component'].abs()
    df['spread_bearing_sin_abs'] = df['spread_bearing_sin'].abs()
    df['spread_bearing_cos_abs'] = df['spread_bearing_cos'].abs()
    df['along_track_speed_abs'] = df['along_track_speed'].abs()
    df['dist_slope_abs'] = df['dist_slope_ci_0_5h'].abs()
    df['dist_change_abs'] = df['dist_change_ci_0_5h'].abs()
    df['dist_accel_abs'] = df['dist_accel_m_per_h2'].abs()
    df['projected_advance_abs'] = df['projected_advance_m'].abs()
    df['closing_speed_abs'] = df['closing_speed_m_per_h'].abs()
    df['area_growth_rate_abs'] = df['area_growth_rate_ha_per_h'].abs()
    df['radial_growth_rate_abs'] = df['radial_growth_rate_m_per_h'].abs()
    df['area_growth_abs_scaled'] = np.log1p(df['area_growth_abs_0_5h'])
    df['alignment_cos_abs'] = df['alignment_cos'].abs()
    df['speed_x_alignment'] = df['centroid_speed_m_per_h'] * df['alignment_abs']
    df['growth_x_distance'] = df['area_growth_rate_ha_per_h'] * dist
    df['speed_distance_interaction'] = df['closing_speed_m_per_h'] / (dist / 1000 + 1)
    df['behavior_instability'] = (1 - df['dist_fit_r2_0_5h']) * df['dist_std_ci_0_5h']
    df['is_paradox_fire'] = (
        (df['closing_speed_m_per_h'] > 5) &
        (df['dist_min_ci_0_5h'] > 15000) &
        (df['fire_intensity'] > 100)
    ).astype(int)
    df['speed_x_alignment_v2'] = df['closing_speed_m_per_h'] * df['alignment_cos']
    df['growth_x_distance_v2'] = df['area_growth_rate_ha_per_h'] / (dist + 1)

    # ── New from Kaggle notebook ───────────────────────────────────────────
    # Distance transformations
    df['log_distance']    = np.log1p(dist)
    df['inv_distance']    = 1 / (dist / 1000 + 0.1)
    df['inv_distance_sq'] = df['inv_distance'] ** 2
    df['sqrt_distance']   = np.sqrt(dist)
    df['dist_km']         = dist / 1000
    df['dist_km_sq']      = (dist / 1000) ** 2
    df['dist_km_cb']      = (dist / 1000) ** 3
    df['dist_rank']       = dist.rank(pct=True)

    # Area-to-distance
    fire_radius = np.sqrt(df['area_first_ha'] * 10000 / np.pi)
    df['fire_radius_km']      = fire_radius / 1000
    df['radius_to_dist']      = fire_radius / dist
    df['area_to_dist_ratio']  = df['area_first_ha'] / (dist / 1000 + 0.1)
    df['log_area_dist_ratio'] = np.log1p(df['area_first_ha']) - np.log1p(dist)

    # Kinematics
    df['has_movement'] = (df['num_perimeters_0_5h'] > 1).astype(float)
    closing_pos = df['closing_speed_m_per_h'].clip(lower=0)
    df['eta_hours'] = np.where(closing_pos > 0.01, dist / closing_pos, 9999).clip(max=9999)
    df['log_eta']   = np.log1p(df['eta_hours'].clip(0, 9999))
    radial_growth   = df['radial_growth_rate_m_per_h'].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    df['effective_closing_speed'] = effective_closing
    df['eta_effective'] = np.where(effective_closing > 0.01, dist / effective_closing, 9999).clip(max=9999)
    df['fire_urgency']     = df['num_perimeters_0_5h'] * df['closing_speed_m_per_h']
    df['growth_intensity'] = df['area_growth_rate_ha_per_h'] * df['num_perimeters_0_5h']

    # Zone flags
    df['zone_near']    = (dist < 5000).astype(float)
    df['zone_warning'] = ((dist >= 5000) & (dist < 10000)).astype(float)
    df['zone_far']     = (dist >= 10000).astype(float)

    # Missing data flag
    df['rates_missing'] = (df['num_perimeters_0_5h'] == 1).astype(int)

    # Temporal
    df['is_summer']    = df['event_start_month'].isin([6, 7, 8]).astype(float)
    df['is_afternoon'] = ((df['event_start_hour'] >= 12) & 
                          (df['event_start_hour'] < 20)).astype(float)

    # Log area per distance
    df['log_area_per_distance'] = np.log1p(df['area_first_ha']) / dist

    # Late additions
    df['closing_speed_nonzero'] = df['closing_speed_m_per_h'] * (df['num_perimeters_0_5h'] > 1).astype(int)
    df['dist_slope_nonzero']    = df['dist_slope_ci_0_5h']    * (df['num_perimeters_0_5h'] > 1).astype(int)
    df['area_per_distance']     = df['area_first_ha'] / (df['dist_min_ci_0_5h'] + 1)
    df['log_area_per_distance'] = df['log1p_area_first'] / (df['dist_min_ci_0_5h'] + 1)
    df['closing_speed_nonzero'] = df['closing_speed_m_per_h'] * (df['num_perimeters_0_5h'] > 1).astype(int)
    df['dist_slope_nonzero']    = df['dist_slope_ci_0_5h']    * (df['num_perimeters_0_5h'] > 1).astype(int)
    df['area_per_distance']     = df['area_first_ha'] / (df['dist_min_ci_0_5h'] + 1)
    df['log_area_per_distance'] = df['log1p_area_first'] / (df['dist_min_ci_0_5h'] + 1)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def expand_data(df):
    """
    Create time-augmented dataset by expanding each fire into multiple
    time query points.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with one row per fire
        
    Returns:
    --------
    pd.DataFrame with multiple rows per fire (one per query time)
    """
    rows = []
    
    for ind, row in df.iterrows():
        actual_hit_time = row['time_to_hit_hours']
        did_hit = row['event'] == 1
        int_time = int(actual_hit_time)

        if did_hit:
            # Create rows BEFORE the hit (target=0)
            for t in range(max(1, int_time - 3), int_time + 1):
                if t < actual_hit_time:
                    new_row = row.copy()
                    new_row['query_time'] = t
                    new_row['target'] = 0  # Hasn't hit yet
                    rows.append(new_row)

            # Create rows AFTER the hit (target=1)
            for t in range(int_time + 1, min(int_time + 5, 73)):
                new_row = row.copy()
                new_row['query_time'] = t
                new_row['target'] = 1  # Already hit
                rows.append(new_row)
        else:
            # Fire never hit - sample at key time points
            for t in [6, 12, 24, 48, 72]:
                new_row = row.copy()
                new_row['query_time'] = t
                new_row['target'] = 0  # Never hit
                if int_time>t:
                    rows.append(new_row)

    return pd.DataFrame(rows)


# ============================================================================
# TARGET CREATION
# ============================================================================

def create_targets(df):
    """
    Create binary targets for each time horizon (12h, 24h, 48h, 72h).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'event' and 'time_to_hit_hours' columns
        
    Returns:
    --------
    pd.DataFrame with additional target columns
    """
    df = df.copy()
    
    df['target_12h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 12)).astype(int)
    df['target_24h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 24)).astype(int)
    df['target_48h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 48)).astype(int)
    df['target_72h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 72)).astype(int)
    
    return df


# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================

def get_split(df, test_size=0.2, random_state=42, stratify_col='target_12h'):
    """
    Split data into train and validation sets with stratification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    test_size : float
        Fraction for validation set
    random_state : int
        Random seed
    stratify_col : str
        Column to stratify on
        
    Returns:
    --------
    tuple of (train_set, val_set)
    """
    train_set, val_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col]
    )
    return train_set, val_set


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def calculate_hybrid_score(y_event, y_time, pred_12h, pred_24h, pred_48h, pred_72h):
    y_event = np.array(y_event)
    y_time  = np.array(y_time)
    pred_72h = np.array(pred_72h)

    # ── C-Index (correct) ─────────────────────────────────────────────────
    c_index = concordance_index(y_time, -pred_72h, y_event)

    # ── Weighted Brier Score ───────────────────────────────────────────────
    weights  = np.array([0.1, 0.2, 0.3, 0.4])
    horizons = [12, 24, 48, 72]
    preds    = [np.array(p) for p in [pred_12h, pred_24h, pred_48h, pred_72h]]

    brier_scores = []
    for h, p in zip(horizons, preds):
        target_h = ((y_event == 1) & (y_time <= h)).astype(int)
        brier_scores.append(brier_score_loss(target_h, p))

    weighted_brier = np.average(brier_scores, weights=weights)
    hybrid_score   = 0.3 * c_index + 0.7 * (1 - weighted_brier)

    return {
        'hybrid_score':   hybrid_score,
        'c_index':        c_index,
        'weighted_brier': weighted_brier,
        'brier_12h':      brier_scores[0],
        'brier_24h':      brier_scores[1],
        'brier_48h':      brier_scores[2],
        'brier_72h':      brier_scores[3],
    }
def calculate_hybrid_score(y_event, y_time, pred_12h, pred_24h, pred_48h, pred_72h):
    y_event  = np.array(y_event)
    y_time   = np.array(y_time)
    pred_12h = np.array(pred_12h)
    pred_24h = np.array(pred_24h)
    pred_48h = np.array(pred_48h)
    pred_72h = np.array(pred_72h)

    # ── C-Index: uses prob_12h ────────────────────────────────────────────
    c_index = concordance_index(y_time, -pred_12h, y_event)

    # ── Brier: uses 24h, 48h, 72h only, filters censored before horizon ──
    def brier(prob, horizon):
        valid  = ~((y_event == 0) & (y_time < horizon))
        if valid.sum() == 0:
            return 0.25
        y_true = ((y_event == 1) & (y_time <= horizon)).astype(float)[valid]
        return float(np.mean((np.clip(prob[valid], 0, 1) - y_true) ** 2))

    b24 = brier(pred_24h, 24)
    b48 = brier(pred_48h, 48)
    b72 = brier(pred_72h, 72)

    weighted_brier = 0.3 * b24 + 0.4 * b48 + 0.3 * b72
    hybrid_score   = 0.3 * c_index + 0.7 * (1 - weighted_brier)

    return {
        'hybrid_score':   hybrid_score,
        'c_index':        c_index,
        'weighted_brier': weighted_brier,
        'brier_24h':      b24,
        'brier_48h':      b48,
        'brier_72h':      b72,
    }

# ============================================================================
# FEATURE SETS
# ============================================================================

# Base temporal features
BASE_FEATURES = [
    'dt_first_last_0_5h',
    'alignment_abs',
    'num_perimeters_0_5h',
    'low_temporal_resolution_0_5h',
    'spread_bearing_cos',
    'log1p_growth',
    'dist_fit_r2_0_5h',
    'closing_speed_m_per_h',
    'alignment_cos',
    'centroid_speed_m_per_h',
]

# Extended features
EXTENDED_FEATURES = [
    'fire_intensity',
    'spread_bearing_deg',
    'centroid_displacement_m',
    'is_night',
    'event_start_hour',
]

# Absolute value symmetric features
ABS_SYMMETRIC = [
    'area_growth_rate_abs',
    'cross_track_abs',
    'spread_bearing_sin_abs',
    'dist_slope_abs',
    'along_track_speed_abs',
]

# Additional robust features
ADDITIONAL_ROBUST = [
    'radial_growth_rate_abs',
    'area_growth_abs_scaled',
    'dist_change_abs',
    'closing_speed_abs',
]

# Best configuration (25 features + query_time)
BEST_FEATURES = (BASE_FEATURES + ['dist_min_ci_0_5h'] + EXTENDED_FEATURES + 
                 ABS_SYMMETRIC + ADDITIONAL_ROBUST + ['query_time'])

# Columns to exclude from training
EXCLUDE_COLS = ['event_id', 'time_to_hit_hours', 'event',
                'target_12h', 'target_24h', 'target_48h', 'target_72h',
                'time_rank', 'time_rank_scaled', 'target', 'split', 
                'dist_bin', 'cv_strata']
