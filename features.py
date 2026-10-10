import math
import logging
import unicodedata
import warnings
import os
import random
from collections import Counter
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from penaltyblog.models import (
    PoissonGoalsModel,
    DixonColesGoalModel,
    BivariatePoissonGoalModel,
    ZeroInflatedPoissonGoalsModel,
    NegativeBinomialGoalModel,
    WeibullCopulaGoalsModel,
    BayesianGoalModel,
    HierarchicalBayesianGoalModel,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None

from config import (
    FORM_WINDOW_SHORT,
    FORM_WINDOW_MEDIUM,
    FORM_WINDOW_LONG,
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    MODEL_TRAIN_TEST_SPLIT,
    MODEL_WALK_FORWARD_FOLDS,
    MODEL_RANDOM_SEED,
    MODEL_N_JOBS,
    CALIBRATION_METHOD,
    CALIBRATION_MIN_VAL_SAMPLES,
    TEMPERATURE_SCALING_ENABLED,
    TEMPERATURE_SCALING_DEFAULT,
    TEMPERATURE_SCALING_MIN,
    TEMPERATURE_SCALING_MAX,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_REG_LAMBDA,
    XGB_REG_ALPHA,
    LGBM_N_ESTIMATORS,
    LGBM_MAX_DEPTH,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_SUBSAMPLE,
    LGBM_COLSAMPLE,
    LGBM_REG_LAMBDA,
    CATBOOST_ITERATIONS,
    CATBOOST_DEPTH,
    CATBOOST_LEARNING_RATE,
    CATBOOST_L2_LEAF_REG,
    CATBOOST_RANDOM_STRENGTH,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_SPLIT,
    RF_MAX_FEATURES_RESOLVED,
    DL_ENABLED,
    DL_HIDDEN_SIZE,
    DL_EPOCHS,
    DL_BATCH_SIZE,
    DL_LEARNING_RATE,
    DL_WEIGHT_DECAY,
    DL_DROPOUT,
    DL_EARLY_STOPPING_PATIENCE,
    DL_GRAD_CLIP,
    MIN_MATCHES_FOR_TRAINING,
    MIN_MATCHES_PER_CLASS,
    COMPETITIONS_TIER_INDEX,
    H2H_MAX_MATCHES,
    H2H_MAX_AGE_DAYS,
    H2H_MIN_MATCHES_FOR_FEATURE,
    REST_DAYS_DEFAULT,
    REST_DAYS_CAP,
    FEATURE_CLIP_ENABLED,
    FEATURE_OUTLIER_ZSCORE,
    ENSEMBLE_MIN_WEIGHT,
    ENSEMBLE_MAX_WEIGHT,
    DEFAULT_SEASON_START_MONTH,
    DEFAULT_SEASON_START_DAY,
    DEFAULT_SEASON_END_MONTH,
    DEFAULT_SEASON_END_DAY,
    BAYESIAN_ENABLED,
    BAYESIAN_N_SAMPLES,
    BAYESIAN_BURN,
    BAYESIAN_N_CHAINS,
    BAYESIAN_THIN,
    BAYESIAN_MODELS,
    TRAIN_RATIO,
    VAL_RATIO,
    CALIB_RATIO,
    TEST_RATIO,
    stage_print,
)

logger = logging.getLogger("goangel.features")

FEATURE_NAMES: List[str] = [
    "home_gf_medium",
    "home_ga_medium",
    "away_gf_medium",
    "away_ga_medium",
    "home_gf_long",
    "home_ga_long",
    "away_gf_long",
    "away_ga_long",
    "home_xg_for_5",
    "away_xg_for_5",
    "home_xg_ag_5",
    "away_xg_ag_5",
    "home_poss_5",
    "away_poss_5",
    "home_shots_5",
    "away_shots_5",
    "home_shots_target_5",
    "away_shots_target_5",
    "home_corners_5",
    "away_corners_5",
    "home_yellow_5",
    "away_yellow_5",
    "home_fouls_5",
    "away_fouls_5",
    "home_pass_acc_5",
    "away_pass_acc_5",
    "home_big_chances_5",
    "away_big_chances_5",
    "home_offsides_5",
    "away_offsides_5",
    "home_throw_ins_5",
    "away_throw_ins_5",
    "home_free_kicks_5",
    "away_free_kicks_5",
    "home_goal_kicks_5",
    "away_goal_kicks_5",
    "home_high_claims_5",
    "away_high_claims_5",
    "home_total_saves_5",
    "away_total_saves_5",
    "home_big_saves_5",
    "away_big_saves_5",
    "home_punches_5",
    "away_punches_5",
    "home_dispossessed_5",
    "away_dispossessed_5",
    "home_hit_woodwork_5",
    "away_hit_woodwork_5",
    "home_through_balls_5",
    "away_through_balls_5",
    "home_total_tackles_5",
    "away_total_tackles_5",
    "home_tackles_5",
    "away_tackles_5",
    "home_passes_5",
    "away_passes_5",
    "home_accurate_passes_5",
    "away_accurate_passes_5",
    "home_shots_off_target_5",
    "away_shots_off_target_5",
    "home_shots_outside_box_5",
    "away_shots_outside_box_5",
    "home_number_of_sprints_5",
    "away_number_of_sprints_5",
    "home_attack_5",
    "away_attack_5",
    "home_ball_safe_5",
    "away_ball_safe_5",
    "home_attack_pct_5",
    "away_attack_pct_5",
    "home_ball_safe_pct_5",
    "away_ball_safe_pct_5",
    "home_errors_lead_to_shot_5",
    "away_errors_lead_to_shot_5",
    "home_fouled_final_third_5",
    "away_fouled_final_third_5",
    "home_crosses_value_5",
    "away_crosses_value_5",
    "home_crosses_total_5",
    "away_crosses_total_5",
    "home_long_balls_value_5",
    "away_long_balls_value_5",
    "home_long_balls_total_5",
    "away_long_balls_total_5",
    "home_dribbles_value_5",
    "away_dribbles_value_5",
    "home_dribbles_total_5",
    "away_dribbles_total_5",
    "home_aerial_duels_value_5",
    "away_aerial_duels_value_5",
    "home_aerial_duels_total_5",
    "away_aerial_duels_total_5",
    "home_ground_duels_value_5",
    "away_ground_duels_value_5",
    "home_ground_duels_total_5",
    "away_ground_duels_total_5",
    "home_final_third_value_5",
    "away_final_third_value_5",
    "home_final_third_pct_5",
    "away_final_third_pct_5",
    "home_duels_5",
    "away_duels_5",
    "home_aerial_pct_5",
    "away_aerial_pct_5",
    "home_ground_pct_5",
    "away_ground_pct_5",
    "home_crosses_pct_5",
    "away_crosses_pct_5",
    "home_long_pct_5",
    "away_long_pct_5",
    "home_tackles_won_5",
    "away_tackles_won_5",
    "home_dribbles_pct_5",
    "away_dribbles_pct_5",
    "home_interceptions_5",
    "away_interceptions_5",
    "home_clearances_5",
    "away_clearances_5",
    "home_recoveries_5",
    "away_recoveries_5",
    "home_gk_saves_5",
    "away_gk_saves_5",
    "home_goals_prev_5",
    "away_goals_prev_5",
    "home_shots_inside_5",
    "away_shots_inside_5",
    "home_blocked_5",
    "away_blocked_5",
    "home_danger_pct_5",
    "away_danger_pct_5",
    "home_final_third_5",
    "away_final_third_5",
    "home_pen_touches_5",
    "away_pen_touches_5",
    "home_big_scored_5",
    "away_big_scored_5",
    "home_red_5",
    "away_red_5",
    "home_xg_target_5",
    "away_xg_target_5",
    "home_home_gf",
    "home_home_ga",
    "away_away_gf",
    "away_away_ga",
    "home_goal_diff",
    "away_goal_diff",
    "home_points_rate",
    "away_points_rate",
    "home_elo",
    "away_elo",
    "elo_diff",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_avg_home_goals",
    "h2h_avg_away_goals",
    "home_rest_days",
    "away_rest_days",
    "home_matches_played",
    "away_matches_played",
    "competition_tier_index",
    "home_ht_gf_rate",
    "home_ht_ga_rate",
    "away_ht_gf_rate",
    "away_ht_ga_rate",
    "home_1h_lead_rate",
    "away_1h_lead_rate",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "home_failed_to_score_rate",
    "away_failed_to_score_rate",
    "season_progress",
]

EXPECTED_FEATURE_COUNT: int = len(FEATURE_NAMES)

_FEATURE_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

_NON_NEGATIVE_FEATURES: Tuple[str, ...] = (
    "home_gf_medium", "home_ga_medium", "away_gf_medium", "away_ga_medium",
    "home_gf_long", "home_ga_long", "away_gf_long", "away_ga_long",
    "home_xg_for_5", "away_xg_for_5", "home_xg_ag_5", "away_xg_ag_5",
    "home_poss_5", "away_poss_5",
    "home_shots_5", "away_shots_5",
    "home_shots_target_5", "away_shots_target_5",
    "home_corners_5", "away_corners_5",
    "home_yellow_5", "away_yellow_5",
    "home_fouls_5", "away_fouls_5",
    "home_pass_acc_5", "away_pass_acc_5",
    "home_big_chances_5", "away_big_chances_5",
    "home_offsides_5", "away_offsides_5",
    "home_throw_ins_5", "away_throw_ins_5",
    "home_free_kicks_5", "away_free_kicks_5",
    "home_goal_kicks_5", "away_goal_kicks_5",
    "home_high_claims_5", "away_high_claims_5",
    "home_total_saves_5", "away_total_saves_5",
    "home_big_saves_5", "away_big_saves_5",
    "home_punches_5", "away_punches_5",
    "home_dispossessed_5", "away_dispossessed_5",
    "home_hit_woodwork_5", "away_hit_woodwork_5",
    "home_through_balls_5", "away_through_balls_5",
    "home_total_tackles_5", "away_total_tackles_5",
    "home_tackles_5", "away_tackles_5",
    "home_passes_5", "away_passes_5",
    "home_accurate_passes_5", "away_accurate_passes_5",
    "home_shots_off_target_5", "away_shots_off_target_5",
    "home_shots_outside_box_5", "away_shots_outside_box_5",
    "home_number_of_sprints_5", "away_number_of_sprints_5",
    "home_attack_5", "away_attack_5",
    "home_ball_safe_5", "away_ball_safe_5",
    "home_attack_pct_5", "away_attack_pct_5",
    "home_ball_safe_pct_5", "away_ball_safe_pct_5",
    "home_errors_lead_to_shot_5", "away_errors_lead_to_shot_5",
    "home_fouled_final_third_5", "away_fouled_final_third_5",
    "home_crosses_value_5", "away_crosses_value_5",
    "home_crosses_total_5", "away_crosses_total_5",
    "home_long_balls_value_5", "away_long_balls_value_5",
    "home_long_balls_total_5", "away_long_balls_total_5",
    "home_dribbles_value_5", "away_dribbles_value_5",
    "home_dribbles_total_5", "away_dribbles_total_5",
    "home_aerial_duels_value_5", "away_aerial_duels_value_5",
    "home_aerial_duels_total_5", "away_aerial_duels_total_5",
    "home_ground_duels_value_5", "away_ground_duels_value_5",
    "home_ground_duels_total_5", "away_ground_duels_total_5",
    "home_final_third_value_5", "away_final_third_value_5",
    "home_final_third_pct_5", "away_final_third_pct_5",
    "home_duels_5", "away_duels_5",
    "home_aerial_pct_5", "away_aerial_pct_5",
    "home_ground_pct_5", "away_ground_pct_5",
    "home_crosses_pct_5", "away_crosses_pct_5",
    "home_long_pct_5", "away_long_pct_5",
    "home_tackles_won_5", "away_tackles_won_5",
    "home_dribbles_pct_5", "away_dribbles_pct_5",
    "home_interceptions_5", "away_interceptions_5",
    "home_clearances_5", "away_clearances_5",
    "home_recoveries_5", "away_recoveries_5",
    "home_gk_saves_5", "away_gk_saves_5",
    "home_goals_prev_5", "away_goals_prev_5",
    "home_shots_inside_5", "away_shots_inside_5",
    "home_blocked_5", "away_blocked_5",
    "home_danger_pct_5", "away_danger_pct_5",
    "home_final_third_5", "away_final_third_5",
    "home_pen_touches_5", "away_pen_touches_5",
    "home_big_scored_5", "away_big_scored_5",
    "home_red_5", "away_red_5",
    "home_xg_target_5", "away_xg_target_5",
    "home_home_gf", "home_home_ga", "away_away_gf", "away_away_ga",
    "home_points_rate", "away_points_rate",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
    "h2h_avg_home_goals", "h2h_avg_away_goals",
    "home_rest_days", "away_rest_days",
    "home_matches_played", "away_matches_played",
    "home_ht_gf_rate", "home_ht_ga_rate",
    "away_ht_gf_rate", "away_ht_ga_rate",
    "home_1h_lead_rate", "away_1h_lead_rate",
    "home_clean_sheet_rate", "away_clean_sheet_rate",
    "home_failed_to_score_rate", "away_failed_to_score_rate",
    "season_progress",
)

_PB_MODEL_CLASSES: Dict[str, Any] = {
    "PoissonGoalsModel": PoissonGoalsModel,
    "DixonColesGoalModel": DixonColesGoalModel,
    "BivariatePoissonGoalModel": BivariatePoissonGoalModel,
    "ZeroInflatedPoissonGoalsModel": ZeroInflatedPoissonGoalsModel,
    "NegativeBinomialGoalModel": NegativeBinomialGoalModel,
    "WeibullCopulaGoalsModel": WeibullCopulaGoalsModel,
    "BayesianGoalModel": BayesianGoalModel,
    "HierarchicalBayesianGoalModel": HierarchicalBayesianGoalModel,
}

_PB_MODEL_ORDER: Tuple[str, ...] = (
    "PoissonGoalsModel",
    "DixonColesGoalModel",
    "BivariatePoissonGoalModel",
    "NegativeBinomialGoalModel",
    "ZeroInflatedPoissonGoalsModel",
    "WeibullCopulaGoalsModel",
    "BayesianGoalModel",
    "HierarchicalBayesianGoalModel",
)

_PB_BAYESIAN_MODELS: Tuple[str, ...] = (
    "BayesianGoalModel",
    "HierarchicalBayesianGoalModel",
)

_WARMUP_RATIO: float = 0.0

_MIN_CLASSES_FOR_TRAINING: int = 3
_MIN_TEST_SAMPLES: int = 30
_MIN_VAL_SAMPLES_FOR_DL: int = 20

_XG_MODEL_SOURCE_TAG: str = "penaltyblog_model_estimate"

_VALID_TRAINING_STATUSES: Tuple[str, ...] = ("FINISHED",)
_VALID_FUTURE_STATUSES: Tuple[str, ...] = (
    "SCHEDULED",
    "TIMED",
    "NOT_STARTED",
    "NS",
    "UPCOMING",
    "TBD",
)
_OPTIONAL_AWARDED_STATUS: str = "AWARDED"

_MIN_MATCHES_PER_COMPETITION_FOR_DEFAULTS: int = 50
_MIN_MATCHES_FOR_COMP_SPECIFIC_STATS: int = 5

_CALIBRATION_STATUS_OK: str = "calibrated"
_CALIBRATION_STATUS_NOT_CALIBRATED: str = "not_calibrated"
_CALIBRATION_STATUS_SKIPPED_SMALL: str = "skipped_small_calib_set"
_CALIBRATION_STATUS_SKIPPED_CLASSES: str = "skipped_missing_classes"
_CALIBRATION_STATUS_FAILED: str = "failed_kept_raw"
_CALIBRATION_STATUS_RAW_KEPT: str = "raw_model_kept"
_CALIBRATION_STATUS_TEMPERATURE: str = "temperature_scaled"

_H2H_SOURCE_SPECIFIC: str = "competition_specific"
_H2H_SOURCE_GLOBAL: str = "global_fallback"
_H2H_SOURCE_NEUTRAL: str = "neutral_default"

_LL_SOURCE_WALK_FORWARD: str = "walk_forward"
_LL_SOURCE_CALIBRATION_FALLBACK: str = "calibration_fallback"
_LL_SOURCE_UNAVAILABLE: str = "unavailable"

_COMPETITION_UNKNOWN_INDEX: int = -1
_MAX_ACCEPTABLE_LOGLOSS: float = 1.35

_QUALITY_RELATIVE_THRESHOLD: float = 1.25
_QUALITY_ABSOLUTE_MARGIN: float = 0.05

_GLOBAL_COMP_KEY: str = "_GLOBAL_"

_PROBA_ATOL: float = 1e-5

_PB_WF_MIN_TRAIN_SAMPLES: int = 80
_PB_WF_MIN_VAL_SAMPLES: int = 5
_PB_CALIB_MIN_VALID_SAMPLES: int = 30
_PB_CALIB_MIN_COVERAGE: float = 0.50

_MIN_WF_FOLD_COVERAGE: float = 0.50

_STRICT_COMPETITION_DEFAULT: bool = True

_DL_TEMPERATURE_GRID_MIN: float = 0.50
_DL_TEMPERATURE_GRID_MAX: float = 3.00
_DL_TEMPERATURE_GRID_STEPS: int = 26

_EPS: float = 1e-9
_EPS_WEIGHT: float = 1e-6

_DAYS_PER_SECOND: float = 86400.0
_MAX_REST_DAYS_VALIDITY: float = 365.0

_REQUIRED_CLASSES: Tuple[int, ...] = (0, 1, 2)

_WEIGHT_PROJECTION_MAX_ITER: int = 200
_WEIGHT_PROJECTION_TOL: float = 1e-10

_NAN: float = float("nan")

_TREE_MODELS_NAN_NATIVE: Tuple[str, ...] = ("XGB", "LGBM", "CatBoost")


def _configure_torch_determinism() -> None:
    if torch is None:
        return
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        random.seed(MODEL_RANDOM_SEED)
        np.random.seed(MODEL_RANDOM_SEED)
        torch.manual_seed(MODEL_RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
    except Exception:
        pass


_configure_torch_determinism()


def _safe_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            if value != value:
                return None
            if not value.is_integer():
                return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_goals_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            if value != value:
                return None
            if not value.is_integer():
                return None
        goals = int(value)
    except (TypeError, ValueError):
        return None
    if goals < 0 or goals > 30:
        return None
    return goals


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            v = float(s)
        else:
            v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _normalize_competition_code(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text else None


def _comp_key(competition_code: Optional[str]) -> str:
    normalized = _normalize_competition_code(competition_code)
    return normalized if normalized is not None else _GLOBAL_COMP_KEY


def _extract_team_name(team: Dict[str, Any]) -> Optional[str]:
    if not isinstance(team, dict):
        return None
    for field in ("name", "shortName", "tla"):
        val = team.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def normalize_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    normalized = (
        unicodedata.normalize("NFKD", name)
        .encode("ASCII", "ignore")
        .decode("ASCII")
    )
    normalized = normalized.lower().strip()
    normalized = "".join(
        ch if (ch.isalnum() or ch.isspace()) else " " for ch in normalized
    )
    normalized = " ".join(normalized.split())
    return normalized


def _team_key_raw(
    team_id: Any,
    raw_name: Any,
    competition_code: Optional[str] = None,
) -> str:
    parsed_id = _safe_int_or_none(team_id)
    if parsed_id is not None:
        return f"id:{parsed_id}"
    normalized = normalize_name(raw_name)
    comp = _normalize_competition_code(competition_code)
    if comp:
        return f"name:{comp}:{normalized}"
    return f"name:{normalized}"


def _build_name_to_id_resolver(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    votes: Dict[str, Dict[int, int]] = {}
    for r in rows:
        for name_field, id_field in (
            ("home_raw", "home_id"),
            ("away_raw", "away_id"),
        ):
            name = normalize_name(r.get(name_field))
            tid = _safe_int_or_none(r.get(id_field))
            if not name or tid is None:
                continue
            votes.setdefault(name, {})
            votes[name][tid] = votes[name].get(tid, 0) + 1
    resolver: Dict[str, int] = {}
    for name, counts in votes.items():
        resolver[name] = max(counts.items(), key=lambda kv: kv[1])[0]
    return resolver


def _extract_dates_with_log(
    dates_raw: pd.Series,
) -> Tuple[pd.Series, int]:
    parsed = pd.to_datetime(dates_raw, errors="coerce", utc=True)
    invalid_mask = parsed.isna()
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        sample = dates_raw[invalid_mask].head(5).tolist()
        logger.warning(
            "Dates invalides ignorées : %d (échantillon : %s)",
            invalid_count,
            sample,
        )
    return parsed, invalid_count


def _extract_halftime_scores(
    score: Any,
) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(score, dict):
        return None, None
    half_time = score.get("halfTime")
    if not isinstance(half_time, dict):
        return None, None
    ht_home = _safe_goals_or_none(half_time.get("home"))
    ht_away = _safe_goals_or_none(half_time.get("away"))
    if ht_home is None or ht_away is None:
        return None, None
    return ht_home, ht_away


_STAT_EXTRACT_KEYS: Tuple[Tuple[str, str], ...] = (
    ("possession", "home_ball_possession"),
    ("total_shots", "home_total_shots"),
    ("shots_on_target", "home_shots_on_target"),
    ("corner_kicks", "home_corner_kicks"),
    ("yellow_cards", "home_yellow_cards"),
    ("fouls", "home_fouls"),
    ("pass_accuracy_pct", "home_pass_accuracy_pct"),
    ("big_chances", "home_big_chances"),
    ("duels", "home_duels"),
    ("aerial_pct", "home_aerial_pct"),
    ("ground_pct", "home_ground_pct"),
    ("crosses_pct", "home_crosses_pct"),
    ("long_pct", "home_long_pct"),
    ("tackles_won", "home_tackles_won"),
    ("dribbles_pct", "home_dribbles_pct"),
    ("interceptions", "home_interceptions"),
    ("clearances", "home_clearances"),
    ("recoveries", "home_recoveries"),
    ("gk_saves", "home_goalkeeper_saves"),
    ("goals_prev", "home_goals_prevented"),
    ("shots_inside", "home_shots_inside"),
    ("blocked", "home_blocked"),
    ("danger_pct", "home_danger_pct"),
    ("final_third", "home_final_third"),
    ("pen_touches", "home_pen_touches"),
    ("big_scored", "home_big_scored"),
    ("red", "home_red_cards"),
    ("xg_target", "home_xg_target"),
    ("offsides", "home_offsides"),
    ("throw_ins", "home_throw_ins"),
    ("free_kicks", "home_free_kicks"),
    ("goal_kicks", "home_goal_kicks"),
    ("high_claims", "home_high_claims"),
    ("total_saves", "home_total_saves"),
    ("big_saves", "home_big_saves"),
    ("punches", "home_punches"),
    ("dispossessed", "home_dispossessed"),
    ("hit_woodwork", "home_hit_woodwork"),
    ("through_balls", "home_through_balls"),
    ("total_tackles", "home_total_tackles"),
    ("tackles", "home_tackles"),
    ("passes", "home_passes"),
    ("accurate_passes", "home_accurate_passes"),
    ("shots_off_target", "home_shots_off_target"),
    ("shots_outside_box", "home_shots_outside_box"),
    ("number_of_sprints", "home_number_of_sprints"),
    ("attack", "home_attack"),
    ("ball_safe", "home_ball_safe"),
    ("attack_pct", "home_attack_pct"),
    ("ball_safe_pct", "home_ball_safe_pct"),
    ("errors_lead_to_shot", "home_errors_lead_to_shot"),
    ("fouled_final_third", "home_fouled_final_third"),
    ("crosses_value", "home_crosses_value"),
    ("crosses_total", "home_crosses_total"),
    ("long_balls_value", "home_long_balls_value"),
    ("long_balls_total", "home_long_balls_total"),
    ("dribbles_value", "home_dribbles_value"),
    ("dribbles_total", "home_dribbles_total"),
    ("aerial_duels_value", "home_aerial_duels_value"),
    ("aerial_duels_total", "home_aerial_duels_total"),
    ("ground_duels_value", "home_ground_duels_value"),
    ("ground_duels_total", "home_ground_duels_total"),
    ("final_third_value", "home_final_third_value"),
    ("final_third_pct", "home_final_third_pct"),
)

_STAT_EXTRACT_KEYS_AWAY: Tuple[Tuple[str, str], ...] = tuple(
    (name, key.replace("home_", "away_", 1))
    for name, key in _STAT_EXTRACT_KEYS
)


def _process_matches_common(
    matches: List[Dict[str, Any]],
    allowed_statuses: Optional[Tuple[str, ...]],
    include_awarded: bool,
    context_label: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    skip_counters: Counter = Counter()
    seen_match_ids: set = set()
    require_scores = allowed_statuses is _VALID_TRAINING_STATUSES

    for m in matches:
        if not isinstance(m, dict):
            skip_counters["not_dict"] += 1
            continue
        status_raw = m.get("status")
        status_norm = (
            str(status_raw).strip().upper()
            if isinstance(status_raw, str)
            else ""
        )

        is_awarded = status_norm == _OPTIONAL_AWARDED_STATUS
        if is_awarded:
            if not include_awarded:
                skip_counters["awarded_excluded"] += 1
                continue
        else:
            if allowed_statuses is not None and status_norm not in allowed_statuses:
                skip_counters[f"status_{status_norm or 'UNKNOWN'}"] += 1
                continue

        match_id_raw = m.get("id")
        match_id_int = _safe_int_or_none(match_id_raw)
        if match_id_int is not None:
            if match_id_int in seen_match_ids:
                skip_counters["duplicate_match_id"] += 1
                continue
            seen_match_ids.add(match_id_int)

        home_team = m.get("homeTeam")
        away_team = m.get("awayTeam")
        if not isinstance(home_team, dict) or not isinstance(away_team, dict):
            skip_counters["missing_team_dict"] += 1
            continue
        home_name = _extract_team_name(home_team)
        away_name = _extract_team_name(away_team)
        if not home_name or not away_name:
            skip_counters["missing_team_name"] += 1
            continue

        score = m.get("score", {})
        full_time = score.get("fullTime", {}) if isinstance(score, dict) else {}
        raw_home = full_time.get("home") if isinstance(full_time, dict) else None
        raw_away = full_time.get("away") if isinstance(full_time, dict) else None
        goals_home = _safe_goals_or_none(raw_home)
        goals_away = _safe_goals_or_none(raw_away)

        if require_scores:
            if goals_home is None or goals_away is None:
                skip_counters["invalid_goals"] += 1
                continue

        ht_home, ht_away = _extract_halftime_scores(score)
        ht_available = ht_home is not None and ht_away is not None
        if not ht_available:
            skip_counters["missing_halftime"] += 1
            ht_home = None
            ht_away = None

        home_xg = _safe_float_or_none(m.get("home_xg"))
        away_xg = _safe_float_or_none(m.get("away_xg"))
        if home_xg is not None and home_xg < 0:
            home_xg = None
        if away_xg is not None and away_xg < 0:
            away_xg = None
        if (home_xg is None) != (away_xg is None):
            home_xg = None
            away_xg = None

        stat_values: Dict[str, Optional[float]] = {}
        for name, key in _STAT_EXTRACT_KEYS:
            stat_values[f"home_{name}"] = _safe_float_or_none(m.get(key))
        for name, key in _STAT_EXTRACT_KEYS_AWAY:
            stat_values[f"away_{name}"] = _safe_float_or_none(m.get(key))

        comp = (
            m.get("competition", {})
            if isinstance(m.get("competition"), dict)
            else {}
        )

        row: Dict[str, Any] = {
            "match_id": match_id_int,
            "home_raw": home_name,
            "away_raw": away_name,
            "home_id": home_team.get("id"),
            "away_id": away_team.get("id"),
            "goals_home": goals_home,
            "goals_away": goals_away,
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_available": ht_available,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "status": status_norm if status_norm else "UNKNOWN",
            "date": m.get("utcDate", ""),
            "competition": comp.get("name", ""),
            "competition_code": _normalize_competition_code(comp.get("code")),
        }
        row.update(stat_values)
        rows.append(row)

    if skip_counters:
        logger.info(
            "%s : %d matchs retenus, rejets : %s",
            context_label,
            len(rows),
            dict(skip_counters),
        )

    if not rows:
        return pd.DataFrame()

    resolver = _build_name_to_id_resolver(rows)

    name_to_resolved_id: Dict[str, int] = {}
    for r in rows:
        norm_home = normalize_name(r["home_raw"])
        norm_away = normalize_name(r["away_raw"])
        hid = _safe_int_or_none(r.get("home_id"))
        aid = _safe_int_or_none(r.get("away_id"))
        if hid is not None:
            name_to_resolved_id.setdefault(norm_home, hid)
        if aid is not None:
            name_to_resolved_id.setdefault(norm_away, aid)

    for r in rows:
        norm_home = normalize_name(r["home_raw"])
        norm_away = normalize_name(r["away_raw"])

        if _safe_int_or_none(r.get("home_id")) is None:
            if norm_home in name_to_resolved_id:
                r["home_id"] = name_to_resolved_id[norm_home]
            elif norm_home in resolver:
                r["home_id"] = resolver[norm_home]
        if _safe_int_or_none(r.get("away_id")) is None:
            if norm_away in name_to_resolved_id:
                r["away_id"] = name_to_resolved_id[norm_away]
            elif norm_away in resolver:
                r["away_id"] = resolver[norm_away]

        r["home"] = norm_home
        r["away"] = norm_away
        comp_code = r.get("competition_code")
        r["home_key"] = _team_key_raw(r.get("home_id"), r["home_raw"], comp_code)
        r["away_key"] = _team_key_raw(r.get("away_id"), r["away_raw"], comp_code)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date_parsed"], n_invalid_dates = _extract_dates_with_log(df["date"])
    if n_invalid_dates > 0:
        df = df[df["date_parsed"].notna()].copy()

    if "match_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["match_id"], keep="first")
        df = df.reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Doublons match_id supprimés : %d", dropped)

    df = df.sort_values("date_parsed", kind="mergesort").reset_index(drop=True)
    return df


def process_matches(
    matches: List[Dict[str, Any]],
    allowed_statuses: Optional[Tuple[str, ...]] = _VALID_TRAINING_STATUSES,
    include_awarded: bool = False,
) -> pd.DataFrame:
    return _process_matches_common(
        matches,
        allowed_statuses=allowed_statuses,
        include_awarded=include_awarded,
        context_label="process_matches",
    )


def process_future_matches(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    return _process_matches_common(
        matches,
        allowed_statuses=_VALID_FUTURE_STATUSES,
        include_awarded=False,
        context_label="process_future_matches",
    )


def process_matches_any_status(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    return _process_matches_common(
        matches,
        allowed_statuses=None,
        include_awarded=False,
        context_label="process_matches_any_status",
    )


class EloSystem:
    def __init__(
        self,
        initial_rating: float = ELO_INITIAL_RATING,
        k_factor: float = ELO_K_FACTOR,
        home_advantage: float = ELO_HOME_ADVANTAGE,
    ):
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.ratings: Dict[Tuple[str, str], float] = {}

    def _key(self, competition_code: Optional[str], team: str) -> Tuple[str, str]:
        comp = _comp_key(competition_code)
        return (comp, team)

    def get_rating(self, team: str, competition_code: Optional[str] = None) -> float:
        return self.ratings.get(self._key(competition_code, team), self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        home: str,
        away: str,
        goals_home: int,
        goals_away: int,
        competition_code: Optional[str] = None,
    ) -> Tuple[float, float]:
        hkey = self._key(competition_code, home)
        akey = self._key(competition_code, away)
        home_rating = self.ratings.get(hkey, self.initial_rating)
        away_rating = self.ratings.get(akey, self.initial_rating)
        pre_home = home_rating
        pre_away = away_rating
        expected_home = self.expected_score(home_rating + self.home_advantage, away_rating)
        expected_away = 1.0 - expected_home
        if goals_home > goals_away:
            actual_home, actual_away = 1.0, 0.0
        elif goals_home == goals_away:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0
        goal_diff = abs(int(goals_home) - int(goals_away))
        multiplier = 1.0 + math.log(max(goal_diff, 1))
        multiplier = min(multiplier, 2.5)
        new_home = home_rating + self.k_factor * multiplier * (actual_home - expected_home)
        new_away = away_rating + self.k_factor * multiplier * (actual_away - expected_away)
        self.ratings[hkey] = new_home
        self.ratings[akey] = new_away
        return pre_home, pre_away


class TeamHistoryTracker:
    def __init__(
        self,
        default_gf: float,
        default_ga: float,
        default_pts: float,
        strict_competition: bool = _STRICT_COMPETITION_DEFAULT,
    ):
        self.gf: Dict[Tuple[str, str], List[float]] = {}
        self.ga: Dict[Tuple[str, str], List[float]] = {}
        self.home_gf: Dict[Tuple[str, str], List[float]] = {}
        self.home_ga: Dict[Tuple[str, str], List[float]] = {}
        self.away_gf: Dict[Tuple[str, str], List[float]] = {}
        self.away_ga: Dict[Tuple[str, str], List[float]] = {}
        self.results: Dict[Tuple[str, str], List[str]] = {}
        self.last_match_date: Dict[Tuple[str, str], Optional[pd.Timestamp]] = {}
        self.matches_played: Dict[Tuple[str, str], int] = {}
        self.ht_gf: Dict[Tuple[str, str], List[float]] = {}
        self.ht_ga: Dict[Tuple[str, str], List[float]] = {}
        self.ht_lead: Dict[Tuple[str, str], List[float]] = {}
        self.clean_sheet: Dict[Tuple[str, str], List[float]] = {}
        self.failed_to_score: Dict[Tuple[str, str], List[float]] = {}
        self.xg_for: Dict[Tuple[str, str], List[float]] = {}
        self.xg_ag: Dict[Tuple[str, str], List[float]] = {}
        self.possession: Dict[Tuple[str, str], List[float]] = {}
        self.total_shots: Dict[Tuple[str, str], List[float]] = {}
        self.shots_on_target: Dict[Tuple[str, str], List[float]] = {}
        self.corners: Dict[Tuple[str, str], List[float]] = {}
        self.yellow_cards: Dict[Tuple[str, str], List[float]] = {}
        self.fouls: Dict[Tuple[str, str], List[float]] = {}
        self.pass_accuracy: Dict[Tuple[str, str], List[float]] = {}
        self.big_chances: Dict[Tuple[str, str], List[float]] = {}
        self.stat_duels: Dict[Tuple[str, str], List[float]] = {}
        self.stat_aerial_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_ground_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_crosses_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_long_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_tackles_won: Dict[Tuple[str, str], List[float]] = {}
        self.stat_dribbles_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_interceptions: Dict[Tuple[str, str], List[float]] = {}
        self.stat_clearances: Dict[Tuple[str, str], List[float]] = {}
        self.stat_recoveries: Dict[Tuple[str, str], List[float]] = {}
        self.stat_gk_saves: Dict[Tuple[str, str], List[float]] = {}
        self.stat_goals_prev: Dict[Tuple[str, str], List[float]] = {}
        self.stat_shots_inside: Dict[Tuple[str, str], List[float]] = {}
        self.stat_blocked: Dict[Tuple[str, str], List[float]] = {}
        self.stat_danger_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_final_third: Dict[Tuple[str, str], List[float]] = {}
        self.stat_pen_touches: Dict[Tuple[str, str], List[float]] = {}
        self.stat_big_scored: Dict[Tuple[str, str], List[float]] = {}
        self.stat_red: Dict[Tuple[str, str], List[float]] = {}
        self.stat_xg_target: Dict[Tuple[str, str], List[float]] = {}
        self.stat_offsides: Dict[Tuple[str, str], List[float]] = {}
        self.stat_throw_ins: Dict[Tuple[str, str], List[float]] = {}
        self.stat_free_kicks: Dict[Tuple[str, str], List[float]] = {}
        self.stat_goal_kicks: Dict[Tuple[str, str], List[float]] = {}
        self.stat_high_claims: Dict[Tuple[str, str], List[float]] = {}
        self.stat_total_saves: Dict[Tuple[str, str], List[float]] = {}
        self.stat_big_saves: Dict[Tuple[str, str], List[float]] = {}
        self.stat_punches: Dict[Tuple[str, str], List[float]] = {}
        self.stat_dispossessed: Dict[Tuple[str, str], List[float]] = {}
        self.stat_hit_woodwork: Dict[Tuple[str, str], List[float]] = {}
        self.stat_through_balls: Dict[Tuple[str, str], List[float]] = {}
        self.stat_total_tackles: Dict[Tuple[str, str], List[float]] = {}
        self.stat_tackles: Dict[Tuple[str, str], List[float]] = {}
        self.stat_passes: Dict[Tuple[str, str], List[float]] = {}
        self.stat_accurate_passes: Dict[Tuple[str, str], List[float]] = {}
        self.stat_shots_off_target: Dict[Tuple[str, str], List[float]] = {}
        self.stat_shots_outside_box: Dict[Tuple[str, str], List[float]] = {}
        self.stat_number_of_sprints: Dict[Tuple[str, str], List[float]] = {}
        self.stat_attack: Dict[Tuple[str, str], List[float]] = {}
        self.stat_ball_safe: Dict[Tuple[str, str], List[float]] = {}
        self.stat_attack_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_ball_safe_pct: Dict[Tuple[str, str], List[float]] = {}
        self.stat_errors_lead_to_shot: Dict[Tuple[str, str], List[float]] = {}
        self.stat_fouled_final_third: Dict[Tuple[str, str], List[float]] = {}
        self.stat_crosses_value: Dict[Tuple[str, str], List[float]] = {}
        self.stat_crosses_total: Dict[Tuple[str, str], List[float]] = {}
        self.stat_long_balls_value: Dict[Tuple[str, str], List[float]] = {}
        self.stat_long_balls_total: Dict[Tuple[str, str], List[float]] = {}
        self.stat_dribbles_value: Dict[Tuple[str, str], List[float]] = {}
        self.stat_dribbles_total: Dict[Tuple[str, str], List[float]] = {}
        self.stat_aerial_duels_value: Dict[Tuple[str, str], List[float]] = {}
        self.stat_aerial_duels_total: Dict[Tuple[str, str], List[float]] = {}
        self.stat_ground_duels_value: Dict[Tuple[str, str], List[float]] = {}
        self.stat_ground_duels_total: Dict[Tuple[str, str], List[float]] = {}
        self.stat_final_third_value: Dict[Tuple[str, str], List[float]] = {}
        self.stat_final_third_pct: Dict[Tuple[str, str], List[float]] = {}
        self.h2h: Dict[
            Tuple[str, str],
            List[Tuple[int, int, Optional[pd.Timestamp], Optional[str]]],
        ] = {}
        self.default_gf = float(default_gf)
        self.default_ga = float(default_ga)
        self.default_pts = float(default_pts)
        self.strict_competition = bool(strict_competition)

    def _init_key(self, key: Tuple[str, str]) -> None:
        if key in self.gf:
            return
        self.gf[key] = []
        self.ga[key] = []
        self.home_gf[key] = []
        self.home_ga[key] = []
        self.away_gf[key] = []
        self.away_ga[key] = []
        self.results[key] = []
        self.last_match_date[key] = None
        self.matches_played[key] = 0
        self.ht_gf[key] = []
        self.ht_ga[key] = []
        self.ht_lead[key] = []
        self.clean_sheet[key] = []
        self.failed_to_score[key] = []
        self.xg_for[key] = []
        self.xg_ag[key] = []
        self.possession[key] = []
        self.total_shots[key] = []
        self.shots_on_target[key] = []
        self.corners[key] = []
        self.yellow_cards[key] = []
        self.fouls[key] = []
        self.pass_accuracy[key] = []
        self.big_chances[key] = []
        self.stat_duels[key] = []
        self.stat_aerial_pct[key] = []
        self.stat_ground_pct[key] = []
        self.stat_crosses_pct[key] = []
        self.stat_long_pct[key] = []
        self.stat_tackles_won[key] = []
        self.stat_dribbles_pct[key] = []
        self.stat_interceptions[key] = []
        self.stat_clearances[key] = []
        self.stat_recoveries[key] = []
        self.stat_gk_saves[key] = []
        self.stat_goals_prev[key] = []
        self.stat_shots_inside[key] = []
        self.stat_blocked[key] = []
        self.stat_danger_pct[key] = []
        self.stat_final_third[key] = []
        self.stat_pen_touches[key] = []
        self.stat_big_scored[key] = []
        self.stat_red[key] = []
        self.stat_xg_target[key] = []
        self.stat_offsides[key] = []
        self.stat_throw_ins[key] = []
        self.stat_free_kicks[key] = []
        self.stat_goal_kicks[key] = []
        self.stat_high_claims[key] = []
        self.stat_total_saves[key] = []
        self.stat_big_saves[key] = []
        self.stat_punches[key] = []
        self.stat_dispossessed[key] = []
        self.stat_hit_woodwork[key] = []
        self.stat_through_balls[key] = []
        self.stat_total_tackles[key] = []
        self.stat_tackles[key] = []
        self.stat_passes[key] = []
        self.stat_accurate_passes[key] = []
        self.stat_shots_off_target[key] = []
        self.stat_shots_outside_box[key] = []
        self.stat_number_of_sprints[key] = []
        self.stat_attack[key] = []
        self.stat_ball_safe[key] = []
        self.stat_attack_pct[key] = []
        self.stat_ball_safe_pct[key] = []
        self.stat_errors_lead_to_shot[key] = []
        self.stat_fouled_final_third[key] = []
        self.stat_crosses_value[key] = []
        self.stat_crosses_total[key] = []
        self.stat_long_balls_value[key] = []
        self.stat_long_balls_total[key] = []
        self.stat_dribbles_value[key] = []
        self.stat_dribbles_total[key] = []
        self.stat_aerial_duels_value[key] = []
        self.stat_aerial_duels_total[key] = []
        self.stat_ground_duels_value[key] = []
        self.stat_ground_duels_total[key] = []
        self.stat_final_third_value[key] = []
        self.stat_final_third_pct[key] = []

    def _ensure(self, team: str, competition_code: Optional[str] = None) -> None:
        comp = _comp_key(competition_code)
        keys_to_init = [(comp, team)]
        if comp != _GLOBAL_COMP_KEY:
            keys_to_init.append((_GLOBAL_COMP_KEY, team))
        for key in keys_to_init:
            self._init_key(key)

    def _select_key(self, team: str, competition_code: Optional[str] = None) -> Tuple[str, str]:
        comp = _comp_key(competition_code)
        if comp == _GLOBAL_COMP_KEY:
            return (_GLOBAL_COMP_KEY, team)
        self._ensure(team, comp)
        specific_key = (comp, team)
        global_key = (_GLOBAL_COMP_KEY, team)
        specific_count = self.matches_played.get(specific_key, 0)
        global_count = self.matches_played.get(global_key, 0)
        if specific_count >= _MIN_MATCHES_FOR_COMP_SPECIFIC_STATS:
            return specific_key
        if self.strict_competition and specific_count > 0:
            return specific_key
        if global_count > 0:
            return global_key
        return specific_key

    def snapshot(self, team: str, competition_code: Optional[str] = None) -> Dict[str, Any]:
        key = self._select_key(team, competition_code)
        return {
            "gf": self.gf.get(key, []),
            "ga": self.ga.get(key, []),
            "home_gf": self.home_gf.get(key, []),
            "home_ga": self.home_ga.get(key, []),
            "away_gf": self.away_gf.get(key, []),
            "away_ga": self.away_ga.get(key, []),
            "results": self.results.get(key, []),
            "last_match_date": self.last_match_date.get(key),
            "matches_played": self.matches_played.get(key, 0),
            "ht_gf": self.ht_gf.get(key, []),
            "ht_ga": self.ht_ga.get(key, []),
            "ht_lead": self.ht_lead.get(key, []),
            "clean_sheet": self.clean_sheet.get(key, []),
            "failed_to_score": self.failed_to_score.get(key, []),
            "xg_for": self.xg_for.get(key, []),
            "xg_ag": self.xg_ag.get(key, []),
            "possession": self.possession.get(key, []),
            "total_shots": self.total_shots.get(key, []),
            "shots_on_target": self.shots_on_target.get(key, []),
            "corners": self.corners.get(key, []),
            "yellow_cards": self.yellow_cards.get(key, []),
            "fouls": self.fouls.get(key, []),
            "pass_accuracy": self.pass_accuracy.get(key, []),
            "big_chances": self.big_chances.get(key, []),
            "stat_duels": self.stat_duels.get(key, []),
            "stat_aerial_pct": self.stat_aerial_pct.get(key, []),
            "stat_ground_pct": self.stat_ground_pct.get(key, []),
            "stat_crosses_pct": self.stat_crosses_pct.get(key, []),
            "stat_long_pct": self.stat_long_pct.get(key, []),
            "stat_tackles_won": self.stat_tackles_won.get(key, []),
            "stat_dribbles_pct": self.stat_dribbles_pct.get(key, []),
            "stat_interceptions": self.stat_interceptions.get(key, []),
            "stat_clearances": self.stat_clearances.get(key, []),
            "stat_recoveries": self.stat_recoveries.get(key, []),
            "stat_gk_saves": self.stat_gk_saves.get(key, []),
            "stat_goals_prev": self.stat_goals_prev.get(key, []),
            "stat_shots_inside": self.stat_shots_inside.get(key, []),
            "stat_blocked": self.stat_blocked.get(key, []),
            "stat_danger_pct": self.stat_danger_pct.get(key, []),
            "stat_final_third": self.stat_final_third.get(key, []),
            "stat_pen_touches": self.stat_pen_touches.get(key, []),
            "stat_big_scored": self.stat_big_scored.get(key, []),
            "stat_red": self.stat_red.get(key, []),
            "stat_xg_target": self.stat_xg_target.get(key, []),
            "stat_offsides": self.stat_offsides.get(key, []),
            "stat_throw_ins": self.stat_throw_ins.get(key, []),
            "stat_free_kicks": self.stat_free_kicks.get(key, []),
            "stat_goal_kicks": self.stat_goal_kicks.get(key, []),
            "stat_high_claims": self.stat_high_claims.get(key, []),
            "stat_total_saves": self.stat_total_saves.get(key, []),
            "stat_big_saves": self.stat_big_saves.get(key, []),
            "stat_punches": self.stat_punches.get(key, []),
            "stat_dispossessed": self.stat_dispossessed.get(key, []),
            "stat_hit_woodwork": self.stat_hit_woodwork.get(key, []),
            "stat_through_balls": self.stat_through_balls.get(key, []),
            "stat_total_tackles": self.stat_total_tackles.get(key, []),
            "stat_tackles": self.stat_tackles.get(key, []),
            "stat_passes": self.stat_passes.get(key, []),
            "stat_accurate_passes": self.stat_accurate_passes.get(key, []),
            "stat_shots_off_target": self.stat_shots_off_target.get(key, []),
            "stat_shots_outside_box": self.stat_shots_outside_box.get(key, []),
            "stat_number_of_sprints": self.stat_number_of_sprints.get(key, []),
            "stat_attack": self.stat_attack.get(key, []),
            "stat_ball_safe": self.stat_ball_safe.get(key, []),
            "stat_attack_pct": self.stat_attack_pct.get(key, []),
            "stat_ball_safe_pct": self.stat_ball_safe_pct.get(key, []),
            "stat_errors_lead_to_shot": self.stat_errors_lead_to_shot.get(key, []),
            "stat_fouled_final_third": self.stat_fouled_final_third.get(key, []),
            "stat_crosses_value": self.stat_crosses_value.get(key, []),
            "stat_crosses_total": self.stat_crosses_total.get(key, []),
            "stat_long_balls_value": self.stat_long_balls_value.get(key, []),
            "stat_long_balls_total": self.stat_long_balls_total.get(key, []),
            "stat_dribbles_value": self.stat_dribbles_value.get(key, []),
            "stat_dribbles_total": self.stat_dribbles_total.get(key, []),
            "stat_aerial_duels_value": self.stat_aerial_duels_value.get(key, []),
            "stat_aerial_duels_total": self.stat_aerial_duels_total.get(key, []),
            "stat_ground_duels_value": self.stat_ground_duels_value.get(key, []),
            "stat_ground_duels_total": self.stat_ground_duels_total.get(key, []),
            "stat_final_third_value": self.stat_final_third_value.get(key, []),
            "stat_final_third_pct": self.stat_final_third_pct.get(key, []),
        }

    def head_to_head(
        self,
        home: str,
        away: str,
        limit: int = H2H_MAX_MATCHES,
        reference_date: Optional[pd.Timestamp] = None,
        max_age_days: int = H2H_MAX_AGE_DAYS,
        competition_code: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, int]], str]:
        key1 = (home, away)
        key2 = (away, home)
        raw: List[Tuple[int, int, Optional[pd.Timestamp], Optional[str]]] = []
        for gh, ga, dt, cc in self.h2h.get(key1, []):
            raw.append((gh, ga, dt, cc))
        for gh, ga, dt, cc in self.h2h.get(key2, []):
            raw.append((ga, gh, dt, cc))

        comp_norm = _normalize_competition_code(competition_code)
        source = _H2H_SOURCE_GLOBAL if raw else _H2H_SOURCE_NEUTRAL

        if comp_norm is not None and raw:
            filtered_by_comp = [
                r
                for r in raw
                if r[3] is not None and str(r[3]).strip().upper() == comp_norm
            ]
            if len(filtered_by_comp) >= H2H_MIN_MATCHES_FOR_FEATURE:
                raw = filtered_by_comp
                source = _H2H_SOURCE_SPECIFIC
            elif self.strict_competition and filtered_by_comp:
                raw = filtered_by_comp
                source = _H2H_SOURCE_SPECIFIC

        if reference_date is not None and not pd.isna(reference_date):
            strict_cutoff = reference_date
            age_cutoff = reference_date - pd.Timedelta(days=int(max_age_days))
            filtered: List[Tuple[int, int, pd.Timestamp]] = []
            for gh, ga, dt, _cc in raw:
                if dt is None or pd.isna(dt):
                    continue
                if dt < strict_cutoff and dt >= age_cutoff:
                    filtered.append((gh, ga, dt))
            filtered.sort(key=lambda r: r[2])
            tail = filtered[-limit:] if limit > 0 else filtered
            if not tail:
                source = _H2H_SOURCE_NEUTRAL
            return ([(gh, ga) for gh, ga, _ in tail], source)

        valid: List[Tuple[int, int, pd.Timestamp]] = [
            (gh, ga, dt)
            for gh, ga, dt, _cc in raw
            if dt is not None and not pd.isna(dt)
        ]
        valid.sort(key=lambda r: r[2])
        tail = valid[-limit:] if limit > 0 else valid
        if not tail:
            source = _H2H_SOURCE_NEUTRAL
        return ([(gh, ga) for gh, ga, _ in tail], source)

    def record(
        self,
        home: str,
        away: str,
        goals_home: int,
        goals_away: int,
        match_date: Optional[pd.Timestamp],
        competition_code: Optional[str] = None,
        ht_home: Optional[int] = None,
        ht_away: Optional[int] = None,
        home_xg: Optional[float] = None,
        away_xg: Optional[float] = None,
        home_possession: Optional[float] = None,
        away_possession: Optional[float] = None,
        home_shots: Optional[float] = None,
        away_shots: Optional[float] = None,
        home_shots_on_target: Optional[float] = None,
        away_shots_on_target: Optional[float] = None,
        home_corners: Optional[float] = None,
        away_corners: Optional[float] = None,
        home_yellow: Optional[float] = None,
        away_yellow: Optional[float] = None,
        home_fouls: Optional[float] = None,
        away_fouls: Optional[float] = None,
        home_pass_acc: Optional[float] = None,
        away_pass_acc: Optional[float] = None,
        home_big_chances: Optional[float] = None,
        away_big_chances: Optional[float] = None,
        home_duels: Optional[float] = None,
        away_duels: Optional[float] = None,
        home_aerial_pct: Optional[float] = None,
        away_aerial_pct: Optional[float] = None,
        home_ground_pct: Optional[float] = None,
        away_ground_pct: Optional[float] = None,
        home_crosses_pct: Optional[float] = None,
        away_crosses_pct: Optional[float] = None,
        home_long_pct: Optional[float] = None,
        away_long_pct: Optional[float] = None,
        home_tackles_won: Optional[float] = None,
        away_tackles_won: Optional[float] = None,
        home_dribbles_pct: Optional[float] = None,
        away_dribbles_pct: Optional[float] = None,
        home_interceptions: Optional[float] = None,
        away_interceptions: Optional[float] = None,
        home_clearances: Optional[float] = None,
        away_clearances: Optional[float] = None,
        home_recoveries: Optional[float] = None,
        away_recoveries: Optional[float] = None,
        home_gk_saves: Optional[float] = None,
        away_gk_saves: Optional[float] = None,
        home_goals_prev: Optional[float] = None,
        away_goals_prev: Optional[float] = None,
        home_shots_inside: Optional[float] = None,
        away_shots_inside: Optional[float] = None,
        home_blocked: Optional[float] = None,
        away_blocked: Optional[float] = None,
        home_danger_pct: Optional[float] = None,
        away_danger_pct: Optional[float] = None,
        home_final_third: Optional[float] = None,
        away_final_third: Optional[float] = None,
        home_pen_touches: Optional[float] = None,
        away_pen_touches: Optional[float] = None,
        home_big_scored: Optional[float] = None,
        away_big_scored: Optional[float] = None,
        home_red: Optional[float] = None,
        away_red: Optional[float] = None,
        home_xg_target: Optional[float] = None,
        away_xg_target: Optional[float] = None,
        home_offsides: Optional[float] = None,
        away_offsides: Optional[float] = None,
        home_throw_ins: Optional[float] = None,
        away_throw_ins: Optional[float] = None,
        home_free_kicks: Optional[float] = None,
        away_free_kicks: Optional[float] = None,
        home_goal_kicks: Optional[float] = None,
        away_goal_kicks: Optional[float] = None,
        home_high_claims: Optional[float] = None,
        away_high_claims: Optional[float] = None,
        home_total_saves: Optional[float] = None,
        away_total_saves: Optional[float] = None,
        home_big_saves: Optional[float] = None,
        away_big_saves: Optional[float] = None,
        home_punches: Optional[float] = None,
        away_punches: Optional[float] = None,
        home_dispossessed: Optional[float] = None,
        away_dispossessed: Optional[float] = None,
        home_hit_woodwork: Optional[float] = None,
        away_hit_woodwork: Optional[float] = None,
        home_through_balls: Optional[float] = None,
        away_through_balls: Optional[float] = None,
        home_total_tackles: Optional[float] = None,
        away_total_tackles: Optional[float] = None,
        home_tackles: Optional[float] = None,
        away_tackles: Optional[float] = None,
        home_passes: Optional[float] = None,
        away_passes: Optional[float] = None,
        home_accurate_passes: Optional[float] = None,
        away_accurate_passes: Optional[float] = None,
        home_shots_off_target: Optional[float] = None,
        away_shots_off_target: Optional[float] = None,
        home_shots_outside_box: Optional[float] = None,
        away_shots_outside_box: Optional[float] = None,
        home_number_of_sprints: Optional[float] = None,
        away_number_of_sprints: Optional[float] = None,
        home_attack: Optional[float] = None,
        away_attack: Optional[float] = None,
        home_ball_safe: Optional[float] = None,
        away_ball_safe: Optional[float] = None,
        home_attack_pct: Optional[float] = None,
        away_attack_pct: Optional[float] = None,
        home_ball_safe_pct: Optional[float] = None,
        away_ball_safe_pct: Optional[float] = None,
        home_errors_lead_to_shot: Optional[float] = None,
        away_errors_lead_to_shot: Optional[float] = None,
        home_fouled_final_third: Optional[float] = None,
        away_fouled_final_third: Optional[float] = None,
        home_crosses_value: Optional[float] = None,
        away_crosses_value: Optional[float] = None,
        home_crosses_total: Optional[float] = None,
        away_crosses_total: Optional[float] = None,
        home_long_balls_value: Optional[float] = None,
        away_long_balls_value: Optional[float] = None,
        home_long_balls_total: Optional[float] = None,
        away_long_balls_total: Optional[float] = None,
        home_dribbles_value: Optional[float] = None,
        away_dribbles_value: Optional[float] = None,
        home_dribbles_total: Optional[float] = None,
        away_dribbles_total: Optional[float] = None,
        home_aerial_duels_value: Optional[float] = None,
        away_aerial_duels_value: Optional[float] = None,
        home_aerial_duels_total: Optional[float] = None,
        away_aerial_duels_total: Optional[float] = None,
        home_ground_duels_value: Optional[float] = None,
        away_ground_duels_value: Optional[float] = None,
        home_ground_duels_total: Optional[float] = None,
        away_ground_duels_total: Optional[float] = None,
        home_final_third_value: Optional[float] = None,
        away_final_third_value: Optional[float] = None,
        home_final_third_pct: Optional[float] = None,
        away_final_third_pct: Optional[float] = None,
    ) -> None:
        comp = _comp_key(competition_code)
        home_key = (comp, home)
        away_key = (comp, away)
        target_keys: List[Tuple[Tuple[str, str], bool]] = [
            (home_key, True),
            (away_key, False),
        ]
        if comp != _GLOBAL_COMP_KEY:
            target_keys.append(((_GLOBAL_COMP_KEY, home), True))
            target_keys.append(((_GLOBAL_COMP_KEY, away), False))

        for key, _is_home in target_keys:
            self._init_key(key)

        if goals_home > goals_away:
            res_home, res_away = "W", "L"
        elif goals_home == goals_away:
            res_home, res_away = "D", "D"
        else:
            res_home, res_away = "L", "W"

        home_clean_sheet = 1.0 if goals_away == 0 else 0.0
        home_failed_to_score = 1.0 if goals_home == 0 else 0.0
        away_clean_sheet = 1.0 if goals_home == 0 else 0.0
        away_failed_to_score = 1.0 if goals_away == 0 else 0.0

        ht_available = ht_home is not None and ht_away is not None
        if ht_available:
            home_ht_lead = 1.0 if ht_home > ht_away else 0.0
            away_ht_lead = 1.0 if ht_away > ht_home else 0.0
        else:
            home_ht_lead = None
            away_ht_lead = None

        xg_available = home_xg is not None and away_xg is not None
        try:
            hxg = float(home_xg) if xg_available else None
            axg = float(away_xg) if xg_available else None
            if hxg is not None and (not np.isfinite(hxg) or hxg < 0):
                hxg = None
            if axg is not None and (not np.isfinite(axg) or axg < 0):
                axg = None
            if hxg is None or axg is None:
                xg_available = False
                hxg = None
                axg = None
        except (TypeError, ValueError):
            xg_available = False
            hxg = None
            axg = None

        home_stats_map: List[Tuple[Optional[float], List[float]]] = [
            (home_possession, self.possession),
            (home_shots, self.total_shots),
            (home_shots_on_target, self.shots_on_target),
            (home_corners, self.corners),
            (home_yellow, self.yellow_cards),
            (home_fouls, self.fouls),
            (home_pass_acc, self.pass_accuracy),
            (home_big_chances, self.big_chances),
            (home_duels, self.stat_duels),
            (home_aerial_pct, self.stat_aerial_pct),
            (home_ground_pct, self.stat_ground_pct),
            (home_crosses_pct, self.stat_crosses_pct),
            (home_long_pct, self.stat_long_pct),
            (home_tackles_won, self.stat_tackles_won),
            (home_dribbles_pct, self.stat_dribbles_pct),
            (home_interceptions, self.stat_interceptions),
            (home_clearances, self.stat_clearances),
            (home_recoveries, self.stat_recoveries),
            (home_gk_saves, self.stat_gk_saves),
            (home_goals_prev, self.stat_goals_prev),
            (home_shots_inside, self.stat_shots_inside),
            (home_blocked, self.stat_blocked),
            (home_danger_pct, self.stat_danger_pct),
            (home_final_third, self.stat_final_third),
            (home_pen_touches, self.stat_pen_touches),
            (home_big_scored, self.stat_big_scored),
            (home_red, self.stat_red),
            (home_xg_target, self.stat_xg_target),
            (home_offsides, self.stat_offsides),
            (home_throw_ins, self.stat_throw_ins),
            (home_free_kicks, self.stat_free_kicks),
            (home_goal_kicks, self.stat_goal_kicks),
            (home_high_claims, self.stat_high_claims),
            (home_total_saves, self.stat_total_saves),
            (home_big_saves, self.stat_big_saves),
            (home_punches, self.stat_punches),
            (home_dispossessed, self.stat_dispossessed),
            (home_hit_woodwork, self.stat_hit_woodwork),
            (home_through_balls, self.stat_through_balls),
            (home_total_tackles, self.stat_total_tackles),
            (home_tackles, self.stat_tackles),
            (home_passes, self.stat_passes),
            (home_accurate_passes, self.stat_accurate_passes),
            (home_shots_off_target, self.stat_shots_off_target),
            (home_shots_outside_box, self.stat_shots_outside_box),
            (home_number_of_sprints, self.stat_number_of_sprints),
            (home_attack, self.stat_attack),
            (home_ball_safe, self.stat_ball_safe),
            (home_attack_pct, self.stat_attack_pct),
            (home_ball_safe_pct, self.stat_ball_safe_pct),
            (home_errors_lead_to_shot, self.stat_errors_lead_to_shot),
            (home_fouled_final_third, self.stat_fouled_final_third),
            (home_crosses_value, self.stat_crosses_value),
            (home_crosses_total, self.stat_crosses_total),
            (home_long_balls_value, self.stat_long_balls_value),
            (home_long_balls_total, self.stat_long_balls_total),
            (home_dribbles_value, self.stat_dribbles_value),
            (home_dribbles_total, self.stat_dribbles_total),
            (home_aerial_duels_value, self.stat_aerial_duels_value),
            (home_aerial_duels_total, self.stat_aerial_duels_total),
            (home_ground_duels_value, self.stat_ground_duels_value),
            (home_ground_duels_total, self.stat_ground_duels_total),
            (home_final_third_value, self.stat_final_third_value),
            (home_final_third_pct, self.stat_final_third_pct),
        ]

        away_stats_map: List[Tuple[Optional[float], List[float]]] = [
            (away_possession, self.possession),
            (away_shots, self.total_shots),
            (away_shots_on_target, self.shots_on_target),
            (away_corners, self.corners),
            (away_yellow, self.yellow_cards),
            (away_fouls, self.fouls),
            (away_pass_acc, self.pass_accuracy),
            (away_big_chances, self.big_chances),
            (away_duels, self.stat_duels),
            (away_aerial_pct, self.stat_aerial_pct),
            (away_ground_pct, self.stat_ground_pct),
            (away_crosses_pct, self.stat_crosses_pct),
            (away_long_pct, self.stat_long_pct),
            (away_tackles_won, self.stat_tackles_won),
            (away_dribbles_pct, self.stat_dribbles_pct),
            (away_interceptions, self.stat_interceptions),
            (away_clearances, self.stat_clearances),
            (away_recoveries, self.stat_recoveries),
            (away_gk_saves, self.stat_gk_saves),
            (away_goals_prev, self.stat_goals_prev),
            (away_shots_inside, self.stat_shots_inside),
            (away_blocked, self.stat_blocked),
            (away_danger_pct, self.stat_danger_pct),
            (away_final_third, self.stat_final_third),
            (away_pen_touches, self.stat_pen_touches),
            (away_big_scored, self.stat_big_scored),
            (away_red, self.stat_red),
            (away_xg_target, self.stat_xg_target),
            (away_offsides, self.stat_offsides),
            (away_throw_ins, self.stat_throw_ins),
            (away_free_kicks, self.stat_free_kicks),
            (away_goal_kicks, self.stat_goal_kicks),
            (away_high_claims, self.stat_high_claims),
            (away_total_saves, self.stat_total_saves),
            (away_big_saves, self.stat_big_saves),
            (away_punches, self.stat_punches),
            (away_dispossessed, self.stat_dispossessed),
            (away_hit_woodwork, self.stat_hit_woodwork),
            (away_through_balls, self.stat_through_balls),
            (away_total_tackles, self.stat_total_tackles),
            (away_tackles, self.stat_tackles),
            (away_passes, self.stat_passes),
            (away_accurate_passes, self.stat_accurate_passes),
            (away_shots_off_target, self.stat_shots_off_target),
            (away_shots_outside_box, self.stat_shots_outside_box),
            (away_number_of_sprints, self.stat_number_of_sprints),
            (away_attack, self.stat_attack),
            (away_ball_safe, self.stat_ball_safe),
            (away_attack_pct, self.stat_attack_pct),
            (away_ball_safe_pct, self.stat_ball_safe_pct),
            (away_errors_lead_to_shot, self.stat_errors_lead_to_shot),
            (away_fouled_final_third, self.stat_fouled_final_third),
            (away_crosses_value, self.stat_crosses_value),
            (away_crosses_total, self.stat_crosses_total),
            (away_long_balls_value, self.stat_long_balls_value),
            (away_long_balls_total, self.stat_long_balls_total),
            (away_dribbles_value, self.stat_dribbles_value),
            (away_dribbles_total, self.stat_dribbles_total),
            (away_aerial_duels_value, self.stat_aerial_duels_value),
            (away_aerial_duels_total, self.stat_aerial_duels_total),
            (away_ground_duels_value, self.stat_ground_duels_value),
            (away_ground_duels_total, self.stat_ground_duels_total),
            (away_final_third_value, self.stat_final_third_value),
            (away_final_third_pct, self.stat_final_third_pct),
        ]

        for key, is_home_slot in target_keys:
            if is_home_slot:
                self.gf[key].append(goals_home)
                self.ga[key].append(goals_away)
                self.home_gf[key].append(goals_home)
                self.home_ga[key].append(goals_away)
                self.results[key].append(res_home)
                self.clean_sheet[key].append(home_clean_sheet)
                self.failed_to_score[key].append(home_failed_to_score)
                if ht_available:
                    self.ht_gf[key].append(float(ht_home))
                    self.ht_ga[key].append(float(ht_away))
                    self.ht_lead[key].append(float(home_ht_lead))
                if xg_available:
                    self.xg_for[key].append(float(hxg))
                    self.xg_ag[key].append(float(axg))
                for val, store in home_stats_map:
                    if val is not None:
                        try:
                            store[key].append(float(val))
                        except (TypeError, ValueError):
                            pass
            else:
                self.gf[key].append(goals_away)
                self.ga[key].append(goals_home)
                self.away_gf[key].append(goals_away)
                self.away_ga[key].append(goals_home)
                self.results[key].append(res_away)
                self.clean_sheet[key].append(away_clean_sheet)
                self.failed_to_score[key].append(away_failed_to_score)
                if ht_available:
                    self.ht_gf[key].append(float(ht_away))
                    self.ht_ga[key].append(float(ht_home))
                    self.ht_lead[key].append(float(away_ht_lead))
                if xg_available:
                    self.xg_for[key].append(float(axg))
                    self.xg_ag[key].append(float(hxg))
                for val, store in away_stats_map:
                    if val is not None:
                        try:
                            store[key].append(float(val))
                        except (TypeError, ValueError):
                            pass
            self.last_match_date[key] = match_date
            self.matches_played[key] += 1

        comp_norm = _normalize_competition_code(competition_code)
        key_h2h = (home, away)
        if key_h2h not in self.h2h:
            self.h2h[key_h2h] = []
        self.h2h[key_h2h].append(
            (int(goals_home), int(goals_away), match_date, comp_norm)
        )


def _safe_mean(values: List[float], window: int, default: float = _NAN) -> float:
    if not values:
        return float(default)
    if window <= 0:
        tail = values
    else:
        tail = values[-window:]
    if not tail:
        return float(default)
    arr = np.asarray(tail, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(arr.mean())


def _points_rate(results: List[str], window: int, default: float = _NAN) -> float:
    if not results:
        return float(default)
    if window <= 0:
        tail = results
    else:
        tail = results[-window:]
    if not tail:
        return float(default)
    points = sum(3 if r == "W" else 1 if r == "D" else 0 for r in tail)
    return points / (3.0 * len(tail))


def _rest_days(
    last_date: Optional[pd.Timestamp],
    current_date: Optional[pd.Timestamp],
    default_days: float = REST_DAYS_DEFAULT,
    cap_days: float = REST_DAYS_CAP,
) -> float:
    if (
        last_date is None
        or current_date is None
        or pd.isna(last_date)
        or pd.isna(current_date)
    ):
        return float(default_days)
    delta = (current_date - last_date).total_seconds() / _DAYS_PER_SECOND
    if delta < 0 or delta > _MAX_REST_DAYS_VALIDITY:
        return float(default_days)
    if delta > cap_days:
        return float(cap_days)
    return float(delta)


def _season_progress(match_date: Optional[pd.Timestamp]) -> float:
    if match_date is None:
        return _NAN
    try:
        if pd.isna(match_date):
            return _NAN
    except (TypeError, ValueError):
        return _NAN
    try:
        d = match_date.date() if hasattr(match_date, "date") else match_date
    except Exception:
        return _NAN
    if not isinstance(d, date):
        return _NAN
    try:
        if d.month >= DEFAULT_SEASON_START_MONTH:
            start_year = d.year
        else:
            start_year = d.year - 1
        season_start = date(
            start_year, DEFAULT_SEASON_START_MONTH, DEFAULT_SEASON_START_DAY
        )
        season_end = date(
            start_year + 1, DEFAULT_SEASON_END_MONTH, DEFAULT_SEASON_END_DAY
        )
    except (TypeError, ValueError):
        return _NAN
    total_days = (season_end - season_start).days
    if total_days <= 0:
        return _NAN
    elapsed_days = (d - season_start).days
    progress = float(elapsed_days) / float(total_days)
    return float(np.clip(progress, 0.0, 1.0))


def _validate_goals_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.dropna()
    values = values[np.isfinite(values)]
    values = values[(values >= 0) & (values <= 30)]
    return values


def compute_league_defaults(
    df_past: pd.DataFrame,
    competition_code: Optional[str] = None,
) -> Tuple[float, float, float]:
    comp_norm = _normalize_competition_code(competition_code)
    ref = df_past
    if (
        comp_norm is not None
        and df_past is not None
        and not df_past.empty
        and "competition_code" in df_past.columns
    ):
        specific = df_past[df_past["competition_code"] == comp_norm]
        if len(specific) >= _MIN_MATCHES_PER_COMPETITION_FOR_DEFAULTS:
            ref = specific
    if ref is None or ref.empty:
        return _NAN, _NAN, _NAN
    if "goals_home" not in ref.columns or "goals_away" not in ref.columns:
        return _NAN, _NAN, _NAN
    home_goals = _validate_goals_series(ref["goals_home"])
    away_goals = _validate_goals_series(ref["goals_away"])
    if home_goals.empty or away_goals.empty:
        return _NAN, _NAN, _NAN
    default_home_gf = float(home_goals.mean())
    all_goals = pd.concat([home_goals, away_goals], ignore_index=True)
    default_gf = float(all_goals.mean())
    paired = pd.concat(
        [home_goals.reset_index(drop=True), away_goals.reset_index(drop=True)],
        axis=1,
    )
    if paired.shape[0] > 0:
        draw_rate = float((paired.iloc[:, 0] == paired.iloc[:, 1]).mean())
    else:
        draw_rate = _NAN
    if np.isfinite(draw_rate):
        default_pts = (3.0 - draw_rate) / 6.0
    else:
        default_pts = _NAN
    return default_gf, default_pts, default_home_gf


def build_feature_vector(
    home: str,
    away: str,
    tracker: TeamHistoryTracker,
    elo_system: EloSystem,
    match_date: Optional[pd.Timestamp],
    competition_code: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    home_hist = tracker.snapshot(home, competition_code)
    away_hist = tracker.snapshot(away, competition_code)
    dgf = tracker.default_gf
    dga = tracker.default_ga
    dpts = tracker.default_pts

    home_gf_medium = _safe_mean(home_hist["gf"], FORM_WINDOW_MEDIUM, dgf)
    home_ga_medium = _safe_mean(home_hist["ga"], FORM_WINDOW_MEDIUM, dga)
    away_gf_medium = _safe_mean(away_hist["gf"], FORM_WINDOW_MEDIUM, dgf)
    away_ga_medium = _safe_mean(away_hist["ga"], FORM_WINDOW_MEDIUM, dga)

    home_gf_long = _safe_mean(home_hist["gf"], FORM_WINDOW_LONG, dgf)
    home_ga_long = _safe_mean(home_hist["ga"], FORM_WINDOW_LONG, dga)
    away_gf_long = _safe_mean(away_hist["gf"], FORM_WINDOW_LONG, dgf)
    away_ga_long = _safe_mean(away_hist["ga"], FORM_WINDOW_LONG, dga)

    home_xg_for_5 = _safe_mean(home_hist.get("xg_for", []), FORM_WINDOW_SHORT, _NAN)
    away_xg_for_5 = _safe_mean(away_hist.get("xg_for", []), FORM_WINDOW_SHORT, _NAN)
    home_xg_ag_5 = _safe_mean(home_hist.get("xg_ag", []), FORM_WINDOW_SHORT, _NAN)
    away_xg_ag_5 = _safe_mean(away_hist.get("xg_ag", []), FORM_WINDOW_SHORT, _NAN)

    home_poss_5 = _safe_mean(home_hist.get("possession", []), FORM_WINDOW_SHORT, _NAN)
    away_poss_5 = _safe_mean(away_hist.get("possession", []), FORM_WINDOW_SHORT, _NAN)
    home_shots_5 = _safe_mean(home_hist.get("total_shots", []), FORM_WINDOW_SHORT, _NAN)
    away_shots_5 = _safe_mean(away_hist.get("total_shots", []), FORM_WINDOW_SHORT, _NAN)
    home_shots_target_5 = _safe_mean(home_hist.get("shots_on_target", []), FORM_WINDOW_SHORT, _NAN)
    away_shots_target_5 = _safe_mean(away_hist.get("shots_on_target", []), FORM_WINDOW_SHORT, _NAN)
    home_corners_5 = _safe_mean(home_hist.get("corners", []), FORM_WINDOW_SHORT, _NAN)
    away_corners_5 = _safe_mean(away_hist.get("corners", []), FORM_WINDOW_SHORT, _NAN)
    home_yellow_5 = _safe_mean(home_hist.get("yellow_cards", []), FORM_WINDOW_SHORT, _NAN)
    away_yellow_5 = _safe_mean(away_hist.get("yellow_cards", []), FORM_WINDOW_SHORT, _NAN)
    home_fouls_5 = _safe_mean(home_hist.get("fouls", []), FORM_WINDOW_SHORT, _NAN)
    away_fouls_5 = _safe_mean(away_hist.get("fouls", []), FORM_WINDOW_SHORT, _NAN)
    home_pass_acc_5 = _safe_mean(home_hist.get("pass_accuracy", []), FORM_WINDOW_SHORT, _NAN)
    away_pass_acc_5 = _safe_mean(away_hist.get("pass_accuracy", []), FORM_WINDOW_SHORT, _NAN)
    home_big_chances_5 = _safe_mean(home_hist.get("big_chances", []), FORM_WINDOW_SHORT, _NAN)
    away_big_chances_5 = _safe_mean(away_hist.get("big_chances", []), FORM_WINDOW_SHORT, _NAN)

    home_duels_5 = _safe_mean(home_hist.get("stat_duels", []), FORM_WINDOW_SHORT, _NAN)
    away_duels_5 = _safe_mean(away_hist.get("stat_duels", []), FORM_WINDOW_SHORT, _NAN)
    home_aerial_pct_5 = _safe_mean(home_hist.get("stat_aerial_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_aerial_pct_5 = _safe_mean(away_hist.get("stat_aerial_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_ground_pct_5 = _safe_mean(home_hist.get("stat_ground_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_ground_pct_5 = _safe_mean(away_hist.get("stat_ground_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_crosses_pct_5 = _safe_mean(home_hist.get("stat_crosses_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_crosses_pct_5 = _safe_mean(away_hist.get("stat_crosses_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_long_pct_5 = _safe_mean(home_hist.get("stat_long_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_long_pct_5 = _safe_mean(away_hist.get("stat_long_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_tackles_won_5 = _safe_mean(home_hist.get("stat_tackles_won", []), FORM_WINDOW_SHORT, _NAN)
    away_tackles_won_5 = _safe_mean(away_hist.get("stat_tackles_won", []), FORM_WINDOW_SHORT, _NAN)
    home_dribbles_pct_5 = _safe_mean(home_hist.get("stat_dribbles_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_dribbles_pct_5 = _safe_mean(away_hist.get("stat_dribbles_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_interceptions_5 = _safe_mean(home_hist.get("stat_interceptions", []), FORM_WINDOW_SHORT, _NAN)
    away_interceptions_5 = _safe_mean(away_hist.get("stat_interceptions", []), FORM_WINDOW_SHORT, _NAN)
    home_clearances_5 = _safe_mean(home_hist.get("stat_clearances", []), FORM_WINDOW_SHORT, _NAN)
    away_clearances_5 = _safe_mean(away_hist.get("stat_clearances", []), FORM_WINDOW_SHORT, _NAN)
    home_recoveries_5 = _safe_mean(home_hist.get("stat_recoveries", []), FORM_WINDOW_SHORT, _NAN)
    away_recoveries_5 = _safe_mean(away_hist.get("stat_recoveries", []), FORM_WINDOW_SHORT, _NAN)
    home_gk_saves_5 = _safe_mean(home_hist.get("stat_gk_saves", []), FORM_WINDOW_SHORT, _NAN)
    away_gk_saves_5 = _safe_mean(away_hist.get("stat_gk_saves", []), FORM_WINDOW_SHORT, _NAN)
    home_goals_prev_5 = _safe_mean(home_hist.get("stat_goals_prev", []), FORM_WINDOW_SHORT, _NAN)
    away_goals_prev_5 = _safe_mean(away_hist.get("stat_goals_prev", []), FORM_WINDOW_SHORT, _NAN)
    home_shots_inside_5 = _safe_mean(home_hist.get("stat_shots_inside", []), FORM_WINDOW_SHORT, _NAN)
    away_shots_inside_5 = _safe_mean(away_hist.get("stat_shots_inside", []), FORM_WINDOW_SHORT, _NAN)
    home_blocked_5 = _safe_mean(home_hist.get("stat_blocked", []), FORM_WINDOW_SHORT, _NAN)
    away_blocked_5 = _safe_mean(away_hist.get("stat_blocked", []), FORM_WINDOW_SHORT, _NAN)
    home_danger_pct_5 = _safe_mean(home_hist.get("stat_danger_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_danger_pct_5 = _safe_mean(away_hist.get("stat_danger_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_final_third_5 = _safe_mean(home_hist.get("stat_final_third", []), FORM_WINDOW_SHORT, _NAN)
    away_final_third_5 = _safe_mean(away_hist.get("stat_final_third", []), FORM_WINDOW_SHORT, _NAN)
    home_pen_touches_5 = _safe_mean(home_hist.get("stat_pen_touches", []), FORM_WINDOW_SHORT, _NAN)
    away_pen_touches_5 = _safe_mean(away_hist.get("stat_pen_touches", []), FORM_WINDOW_SHORT, _NAN)
    home_big_scored_5 = _safe_mean(home_hist.get("stat_big_scored", []), FORM_WINDOW_SHORT, _NAN)
    away_big_scored_5 = _safe_mean(away_hist.get("stat_big_scored", []), FORM_WINDOW_SHORT, _NAN)
    home_red_5 = _safe_mean(home_hist.get("stat_red", []), FORM_WINDOW_SHORT, _NAN)
    away_red_5 = _safe_mean(away_hist.get("stat_red", []), FORM_WINDOW_SHORT, _NAN)
    home_xg_target_5 = _safe_mean(home_hist.get("stat_xg_target", []), FORM_WINDOW_SHORT, _NAN)
    away_xg_target_5 = _safe_mean(away_hist.get("stat_xg_target", []), FORM_WINDOW_SHORT, _NAN)

    home_offsides_5 = _safe_mean(home_hist.get("stat_offsides", []), FORM_WINDOW_SHORT, _NAN)
    away_offsides_5 = _safe_mean(away_hist.get("stat_offsides", []), FORM_WINDOW_SHORT, _NAN)
    home_throw_ins_5 = _safe_mean(home_hist.get("stat_throw_ins", []), FORM_WINDOW_SHORT, _NAN)
    away_throw_ins_5 = _safe_mean(away_hist.get("stat_throw_ins", []), FORM_WINDOW_SHORT, _NAN)
    home_free_kicks_5 = _safe_mean(home_hist.get("stat_free_kicks", []), FORM_WINDOW_SHORT, _NAN)
    away_free_kicks_5 = _safe_mean(away_hist.get("stat_free_kicks", []), FORM_WINDOW_SHORT, _NAN)
    home_goal_kicks_5 = _safe_mean(home_hist.get("stat_goal_kicks", []), FORM_WINDOW_SHORT, _NAN)
    away_goal_kicks_5 = _safe_mean(away_hist.get("stat_goal_kicks", []), FORM_WINDOW_SHORT, _NAN)
    home_high_claims_5 = _safe_mean(home_hist.get("stat_high_claims", []), FORM_WINDOW_SHORT, _NAN)
    away_high_claims_5 = _safe_mean(away_hist.get("stat_high_claims", []), FORM_WINDOW_SHORT, _NAN)
    home_total_saves_5 = _safe_mean(home_hist.get("stat_total_saves", []), FORM_WINDOW_SHORT, _NAN)
    away_total_saves_5 = _safe_mean(away_hist.get("stat_total_saves", []), FORM_WINDOW_SHORT, _NAN)
    home_big_saves_5 = _safe_mean(home_hist.get("stat_big_saves", []), FORM_WINDOW_SHORT, _NAN)
    away_big_saves_5 = _safe_mean(away_hist.get("stat_big_saves", []), FORM_WINDOW_SHORT, _NAN)
    home_punches_5 = _safe_mean(home_hist.get("stat_punches", []), FORM_WINDOW_SHORT, _NAN)
    away_punches_5 = _safe_mean(away_hist.get("stat_punches", []), FORM_WINDOW_SHORT, _NAN)
    home_dispossessed_5 = _safe_mean(home_hist.get("stat_dispossessed", []), FORM_WINDOW_SHORT, _NAN)
    away_dispossessed_5 = _safe_mean(away_hist.get("stat_dispossessed", []), FORM_WINDOW_SHORT, _NAN)
    home_hit_woodwork_5 = _safe_mean(home_hist.get("stat_hit_woodwork", []), FORM_WINDOW_SHORT, _NAN)
    away_hit_woodwork_5 = _safe_mean(away_hist.get("stat_hit_woodwork", []), FORM_WINDOW_SHORT, _NAN)
    home_through_balls_5 = _safe_mean(home_hist.get("stat_through_balls", []), FORM_WINDOW_SHORT, _NAN)
    away_through_balls_5 = _safe_mean(away_hist.get("stat_through_balls", []), FORM_WINDOW_SHORT, _NAN)
    home_total_tackles_5 = _safe_mean(home_hist.get("stat_total_tackles", []), FORM_WINDOW_SHORT, _NAN)
    away_total_tackles_5 = _safe_mean(away_hist.get("stat_total_tackles", []), FORM_WINDOW_SHORT, _NAN)
    home_tackles_5 = _safe_mean(home_hist.get("stat_tackles", []), FORM_WINDOW_SHORT, _NAN)
    away_tackles_5 = _safe_mean(away_hist.get("stat_tackles", []), FORM_WINDOW_SHORT, _NAN)
    home_passes_5 = _safe_mean(home_hist.get("stat_passes", []), FORM_WINDOW_SHORT, _NAN)
    away_passes_5 = _safe_mean(away_hist.get("stat_passes", []), FORM_WINDOW_SHORT, _NAN)
    home_accurate_passes_5 = _safe_mean(home_hist.get("stat_accurate_passes", []), FORM_WINDOW_SHORT, _NAN)
    away_accurate_passes_5 = _safe_mean(away_hist.get("stat_accurate_passes", []), FORM_WINDOW_SHORT, _NAN)
    home_shots_off_target_5 = _safe_mean(home_hist.get("stat_shots_off_target", []), FORM_WINDOW_SHORT, _NAN)
    away_shots_off_target_5 = _safe_mean(away_hist.get("stat_shots_off_target", []), FORM_WINDOW_SHORT, _NAN)
    home_shots_outside_box_5 = _safe_mean(home_hist.get("stat_shots_outside_box", []), FORM_WINDOW_SHORT, _NAN)
    away_shots_outside_box_5 = _safe_mean(away_hist.get("stat_shots_outside_box", []), FORM_WINDOW_SHORT, _NAN)
    home_number_of_sprints_5 = _safe_mean(home_hist.get("stat_number_of_sprints", []), FORM_WINDOW_SHORT, _NAN)
    away_number_of_sprints_5 = _safe_mean(away_hist.get("stat_number_of_sprints", []), FORM_WINDOW_SHORT, _NAN)
    home_attack_5 = _safe_mean(home_hist.get("stat_attack", []), FORM_WINDOW_SHORT, _NAN)
    away_attack_5 = _safe_mean(away_hist.get("stat_attack", []), FORM_WINDOW_SHORT, _NAN)
    home_ball_safe_5 = _safe_mean(home_hist.get("stat_ball_safe", []), FORM_WINDOW_SHORT, _NAN)
    away_ball_safe_5 = _safe_mean(away_hist.get("stat_ball_safe", []), FORM_WINDOW_SHORT, _NAN)
    home_attack_pct_5 = _safe_mean(home_hist.get("stat_attack_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_attack_pct_5 = _safe_mean(away_hist.get("stat_attack_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_ball_safe_pct_5 = _safe_mean(home_hist.get("stat_ball_safe_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_ball_safe_pct_5 = _safe_mean(away_hist.get("stat_ball_safe_pct", []), FORM_WINDOW_SHORT, _NAN)
    home_errors_lead_to_shot_5 = _safe_mean(home_hist.get("stat_errors_lead_to_shot", []), FORM_WINDOW_SHORT, _NAN)
    away_errors_lead_to_shot_5 = _safe_mean(away_hist.get("stat_errors_lead_to_shot", []), FORM_WINDOW_SHORT, _NAN)
    home_fouled_final_third_5 = _safe_mean(home_hist.get("stat_fouled_final_third", []), FORM_WINDOW_SHORT, _NAN)
    away_fouled_final_third_5 = _safe_mean(away_hist.get("stat_fouled_final_third", []), FORM_WINDOW_SHORT, _NAN)
    home_crosses_value_5 = _safe_mean(home_hist.get("stat_crosses_value", []), FORM_WINDOW_SHORT, _NAN)
    away_crosses_value_5 = _safe_mean(away_hist.get("stat_crosses_value", []), FORM_WINDOW_SHORT, _NAN)
    home_crosses_total_5 = _safe_mean(home_hist.get("stat_crosses_total", []), FORM_WINDOW_SHORT, _NAN)
    away_crosses_total_5 = _safe_mean(away_hist.get("stat_crosses_total", []), FORM_WINDOW_SHORT, _NAN)
    home_long_balls_value_5 = _safe_mean(home_hist.get("stat_long_balls_value", []), FORM_WINDOW_SHORT, _NAN)
    away_long_balls_value_5 = _safe_mean(away_hist.get("stat_long_balls_value", []), FORM_WINDOW_SHORT, _NAN)
    home_long_balls_total_5 = _safe_mean(home_hist.get("stat_long_balls_total", []), FORM_WINDOW_SHORT, _NAN)
    away_long_balls_total_5 = _safe_mean(away_hist.get("stat_long_balls_total", []), FORM_WINDOW_SHORT, _NAN)
    home_dribbles_value_5 = _safe_mean(home_hist.get("stat_dribbles_value", []), FORM_WINDOW_SHORT, _NAN)
    away_dribbles_value_5 = _safe_mean(away_hist.get("stat_dribbles_value", []), FORM_WINDOW_SHORT, _NAN)
    home_dribbles_total_5 = _safe_mean(home_hist.get("stat_dribbles_total", []), FORM_WINDOW_SHORT, _NAN)
    away_dribbles_total_5 = _safe_mean(away_hist.get("stat_dribbles_total", []), FORM_WINDOW_SHORT, _NAN)
    home_aerial_duels_value_5 = _safe_mean(home_hist.get("stat_aerial_duels_value", []), FORM_WINDOW_SHORT, _NAN)
    away_aerial_duels_value_5 = _safe_mean(away_hist.get("stat_aerial_duels_value", []), FORM_WINDOW_SHORT, _NAN)
    home_aerial_duels_total_5 = _safe_mean(home_hist.get("stat_aerial_duels_total", []), FORM_WINDOW_SHORT, _NAN)
    away_aerial_duels_total_5 = _safe_mean(away_hist.get("stat_aerial_duels_total", []), FORM_WINDOW_SHORT, _NAN)
    home_ground_duels_value_5 = _safe_mean(home_hist.get("stat_ground_duels_value", []), FORM_WINDOW_SHORT, _NAN)
    away_ground_duels_value_5 = _safe_mean(away_hist.get("stat_ground_duels_value", []), FORM_WINDOW_SHORT, _NAN)
    home_ground_duels_total_5 = _safe_mean(home_hist.get("stat_ground_duels_total", []), FORM_WINDOW_SHORT, _NAN)
    away_ground_duels_total_5 = _safe_mean(away_hist.get("stat_ground_duels_total", []), FORM_WINDOW_SHORT, _NAN)
    home_final_third_value_5 = _safe_mean(home_hist.get("stat_final_third_value", []), FORM_WINDOW_SHORT, _NAN)
    away_final_third_value_5 = _safe_mean(away_hist.get("stat_final_third_value", []), FORM_WINDOW_SHORT, _NAN)
    home_final_third_pct_5 = _safe_mean(home_hist.get("stat_final_third_pct", []), FORM_WINDOW_SHORT, _NAN)
    away_final_third_pct_5 = _safe_mean(away_hist.get("stat_final_third_pct", []), FORM_WINDOW_SHORT, _NAN)

    home_home_gf = _safe_mean(home_hist["home_gf"], FORM_WINDOW_MEDIUM, dgf)
    home_home_ga = _safe_mean(home_hist["home_ga"], FORM_WINDOW_MEDIUM, dga)
    away_away_gf = _safe_mean(away_hist["away_gf"], FORM_WINDOW_MEDIUM, dgf)
    away_away_ga = _safe_mean(away_hist["away_ga"], FORM_WINDOW_MEDIUM, dga)

    home_goal_diff = home_gf_medium - home_ga_medium
    away_goal_diff = away_gf_medium - away_ga_medium

    home_points_rate = _points_rate(home_hist["results"], FORM_WINDOW_MEDIUM, dpts)
    away_points_rate = _points_rate(away_hist["results"], FORM_WINDOW_MEDIUM, dpts)

    home_elo = elo_system.get_rating(home, competition_code)
    away_elo = elo_system.get_rating(away, competition_code)
    elo_diff = home_elo - away_elo

    h2h_matches, h2h_source = tracker.head_to_head(
        home,
        away,
        limit=H2H_MAX_MATCHES,
        reference_date=match_date,
        competition_code=competition_code,
    )
    if len(h2h_matches) >= H2H_MIN_MATCHES_FOR_FEATURE:
        total = len(h2h_matches)
        h2h_home_wins = sum(1 for gh, ga in h2h_matches if gh > ga)
        h2h_draws = sum(1 for gh, ga in h2h_matches if gh == ga)
        h2h_away_wins = sum(1 for gh, ga in h2h_matches if gh < ga)
        h2h_home_win_rate = h2h_home_wins / total
        h2h_draw_rate = h2h_draws / total
        h2h_away_win_rate = h2h_away_wins / total
        h2h_avg_home_goals = float(np.mean([gh for gh, _ in h2h_matches]))
        h2h_avg_away_goals = float(np.mean([ga for _, ga in h2h_matches]))
    else:
        h2h_home_win_rate = _NAN
        h2h_draw_rate = _NAN
        h2h_away_win_rate = _NAN
        h2h_avg_home_goals = _NAN
        h2h_avg_away_goals = _NAN
        h2h_source = _H2H_SOURCE_NEUTRAL

    home_rest_days = _rest_days(home_hist["last_match_date"], match_date)
    away_rest_days = _rest_days(away_hist["last_match_date"], match_date)

    home_matches_played = float(home_hist["matches_played"])
    away_matches_played = float(away_hist["matches_played"])

    comp_code = _normalize_competition_code(competition_code)
    if comp_code is None:
        competition_tier_index = float(_COMPETITION_UNKNOWN_INDEX)
    else:
        competition_tier_index = float(
            COMPETITIONS_TIER_INDEX.get(comp_code, _COMPETITION_UNKNOWN_INDEX)
        )

    home_ht_gf_rate = _safe_mean(home_hist["ht_gf"], FORM_WINDOW_MEDIUM, _NAN)
    home_ht_ga_rate = _safe_mean(home_hist["ht_ga"], FORM_WINDOW_MEDIUM, _NAN)
    away_ht_gf_rate = _safe_mean(away_hist["ht_gf"], FORM_WINDOW_MEDIUM, _NAN)
    away_ht_ga_rate = _safe_mean(away_hist["ht_ga"], FORM_WINDOW_MEDIUM, _NAN)

    home_1h_lead_rate = _safe_mean(home_hist["ht_lead"], FORM_WINDOW_MEDIUM, _NAN)
    away_1h_lead_rate = _safe_mean(away_hist["ht_lead"], FORM_WINDOW_MEDIUM, _NAN)

    home_clean_sheet_rate = _safe_mean(home_hist["clean_sheet"], FORM_WINDOW_MEDIUM, _NAN)
    away_clean_sheet_rate = _safe_mean(away_hist["clean_sheet"], FORM_WINDOW_MEDIUM, _NAN)

    home_failed_to_score_rate = _safe_mean(home_hist["failed_to_score"], FORM_WINDOW_MEDIUM, _NAN)
    away_failed_to_score_rate = _safe_mean(away_hist["failed_to_score"], FORM_WINDOW_MEDIUM, _NAN)

    season_progress = _season_progress(match_date)

    feat = np.array([
        home_gf_medium, home_ga_medium, away_gf_medium, away_ga_medium,
        home_gf_long, home_ga_long, away_gf_long, away_ga_long,
        home_xg_for_5, away_xg_for_5, home_xg_ag_5, away_xg_ag_5,
        home_poss_5, away_poss_5,
        home_shots_5, away_shots_5,
        home_shots_target_5, away_shots_target_5,
        home_corners_5, away_corners_5,
        home_yellow_5, away_yellow_5,
        home_fouls_5, away_fouls_5,
        home_pass_acc_5, away_pass_acc_5,
        home_big_chances_5, away_big_chances_5,
        home_offsides_5, away_offsides_5,
        home_throw_ins_5, away_throw_ins_5,
        home_free_kicks_5, away_free_kicks_5,
        home_goal_kicks_5, away_goal_kicks_5,
        home_high_claims_5, away_high_claims_5,
        home_total_saves_5, away_total_saves_5,
        home_big_saves_5, away_big_saves_5,
        home_punches_5, away_punches_5,
        home_dispossessed_5, away_dispossessed_5,
        home_hit_woodwork_5, away_hit_woodwork_5,
        home_through_balls_5, away_through_balls_5,
        home_total_tackles_5, away_total_tackles_5,
        home_tackles_5, away_tackles_5,
        home_passes_5, away_passes_5,
        home_accurate_passes_5, away_accurate_passes_5,
        home_shots_off_target_5, away_shots_off_target_5,
        home_shots_outside_box_5, away_shots_outside_box_5,
        home_number_of_sprints_5, away_number_of_sprints_5,
        home_attack_5, away_attack_5,
        home_ball_safe_5, away_ball_safe_5,
        home_attack_pct_5, away_attack_pct_5,
        home_ball_safe_pct_5, away_ball_safe_pct_5,
        home_errors_lead_to_shot_5, away_errors_lead_to_shot_5,
        home_fouled_final_third_5, away_fouled_final_third_5,
        home_crosses_value_5, away_crosses_value_5,
        home_crosses_total_5, away_crosses_total_5,
        home_long_balls_value_5, away_long_balls_value_5,
        home_long_balls_total_5, away_long_balls_total_5,
        home_dribbles_value_5, away_dribbles_value_5,
        home_dribbles_total_5, away_dribbles_total_5,
        home_aerial_duels_value_5, away_aerial_duels_value_5,
        home_aerial_duels_total_5, away_aerial_duels_total_5,
        home_ground_duels_value_5, away_ground_duels_value_5,
        home_ground_duels_total_5, away_ground_duels_total_5,
        home_final_third_value_5, away_final_third_value_5,
        home_final_third_pct_5, away_final_third_pct_5,
        home_duels_5, away_duels_5,
        home_aerial_pct_5, away_aerial_pct_5,
        home_ground_pct_5, away_ground_pct_5,
        home_crosses_pct_5, away_crosses_pct_5,
        home_long_pct_5, away_long_pct_5,
        home_tackles_won_5, away_tackles_won_5,
        home_dribbles_pct_5, away_dribbles_pct_5,
        home_interceptions_5, away_interceptions_5,
        home_clearances_5, away_clearances_5,
        home_recoveries_5, away_recoveries_5,
        home_gk_saves_5, away_gk_saves_5,
        home_goals_prev_5, away_goals_prev_5,
        home_shots_inside_5, away_shots_inside_5,
        home_blocked_5, away_blocked_5,
        home_danger_pct_5, away_danger_pct_5,
        home_final_third_5, away_final_third_5,
        home_pen_touches_5, away_pen_touches_5,
        home_big_scored_5, away_big_scored_5,
        home_red_5, away_red_5,
        home_xg_target_5, away_xg_target_5,
        home_home_gf, home_home_ga, away_away_gf, away_away_ga,
        home_goal_diff, away_goal_diff,
        home_points_rate, away_points_rate,
        home_elo, away_elo, elo_diff,
        h2h_home_win_rate, h2h_draw_rate, h2h_away_win_rate,
        h2h_avg_home_goals, h2h_avg_away_goals,
        home_rest_days, away_rest_days,
        home_matches_played, away_matches_played,
        competition_tier_index,
        home_ht_gf_rate, home_ht_ga_rate,
        away_ht_gf_rate, away_ht_ga_rate,
        home_1h_lead_rate, away_1h_lead_rate,
        home_clean_sheet_rate, away_clean_sheet_rate,
        home_failed_to_score_rate, away_failed_to_score_rate,
        season_progress,
    ], dtype=np.float64)

    if feat.shape[0] != EXPECTED_FEATURE_COUNT:
        logger.error(
            "Vecteur de features incohérent : %d != %d",
            feat.shape[0], EXPECTED_FEATURE_COUNT,
        )

    if FEATURE_CLIP_ENABLED:
        feat = np.where(np.isinf(feat), _NAN, feat)
        for name in _NON_NEGATIVE_FEATURES:
            idx = _FEATURE_INDEX.get(name)
            if idx is None:
                continue
            val = feat[idx]
            if np.isfinite(val) and val < 0.0:
                feat[idx] = 0.0
        season_idx = _FEATURE_INDEX.get("season_progress")
        if season_idx is not None:
            val = feat[season_idx]
            if np.isfinite(val):
                feat[season_idx] = float(np.clip(val, 0.0, 1.0))

    return feat, h2h_source


def build_tracking_state(
    df_past: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
    competition_code: Optional[str] = None,
    strict_competition: bool = _STRICT_COMPETITION_DEFAULT,
) -> Tuple[TeamHistoryTracker, EloSystem]:
    if df_past is None or df_past.empty:
        tracker = TeamHistoryTracker(
            default_gf=_NAN, default_ga=_NAN, default_pts=_NAN,
            strict_competition=strict_competition,
        )
        return tracker, EloSystem()

    if "date_parsed" not in df_past.columns:
        logger.error("build_tracking_state : colonne 'date_parsed' manquante.")
        tracker = TeamHistoryTracker(
            default_gf=_NAN, default_ga=_NAN, default_pts=_NAN,
            strict_competition=strict_competition,
        )
        return tracker, EloSystem()

    comp_norm = _normalize_competition_code(competition_code)
    df_source = df_past
    if comp_norm is not None and "competition_code" in df_past.columns:
        specific = df_past[df_past["competition_code"] == comp_norm]
        if not specific.empty:
            df_source = specific

    df_source = df_source.sort_values("date_parsed", kind="mergesort")

    if cutoff_date is not None:
        df_for_defaults = df_source[df_source["date_parsed"] < cutoff_date]
    else:
        df_for_defaults = df_source

    dgf, dpts, _ = compute_league_defaults(df_for_defaults, comp_norm)

    tracker = TeamHistoryTracker(
        default_gf=dgf, default_ga=dgf, default_pts=dpts,
        strict_competition=strict_competition,
    )
    elo_system = EloSystem()

    df_sorted = df_source
    if cutoff_date is not None:
        df_sorted = df_sorted[df_sorted["date_parsed"] < cutoff_date]

    def _f(row: Any, name: str) -> Optional[float]:
        return _safe_float_or_none(getattr(row, name, None))

    for row in df_sorted.itertuples(index=False):
        home = getattr(row, "home_key", None)
        away = getattr(row, "away_key", None)
        gh_i = _safe_goals_or_none(getattr(row, "goals_home", None))
        ga_i = _safe_goals_or_none(getattr(row, "goals_away", None))
        if home is None or away is None or gh_i is None or ga_i is None:
            continue
        match_date = getattr(row, "date_parsed", None)
        comp_code = _normalize_competition_code(getattr(row, "competition_code", None))
        ht_home = _safe_goals_or_none(getattr(row, "ht_home", None))
        ht_away = _safe_goals_or_none(getattr(row, "ht_away", None))
        if ht_home is None or ht_away is None:
            ht_home = None
            ht_away = None

        tracker.record(
            home, away, gh_i, ga_i, match_date, comp_code,
            ht_home=ht_home, ht_away=ht_away,
            home_xg=_f(row, "home_xg"), away_xg=_f(row, "away_xg"),
            home_possession=_f(row, "home_possession"),
            away_possession=_f(row, "away_possession"),
            home_shots=_f(row, "home_total_shots"),
            away_shots=_f(row, "away_total_shots"),
            home_shots_on_target=_f(row, "home_shots_on_target"),
            away_shots_on_target=_f(row, "away_shots_on_target"),
            home_corners=_f(row, "home_corner_kicks"),
            away_corners=_f(row, "away_corner_kicks"),
            home_yellow=_f(row, "home_yellow_cards"),
            away_yellow=_f(row, "away_yellow_cards"),
            home_fouls=_f(row, "home_fouls"), away_fouls=_f(row, "away_fouls"),
            home_pass_acc=_f(row, "home_pass_accuracy_pct"),
            away_pass_acc=_f(row, "away_pass_accuracy_pct"),
            home_big_chances=_f(row, "home_big_chances"),
            away_big_chances=_f(row, "away_big_chances"),
            home_offsides=_f(row, "home_offsides"), away_offsides=_f(row, "away_offsides"),
            home_throw_ins=_f(row, "home_throw_ins"), away_throw_ins=_f(row, "away_throw_ins"),
            home_free_kicks=_f(row, "home_free_kicks"), away_free_kicks=_f(row, "away_free_kicks"),
            home_goal_kicks=_f(row, "home_goal_kicks"), away_goal_kicks=_f(row, "away_goal_kicks"),
            home_high_claims=_f(row, "home_high_claims"), away_high_claims=_f(row, "away_high_claims"),
            home_total_saves=_f(row, "home_total_saves"), away_total_saves=_f(row, "away_total_saves"),
            home_big_saves=_f(row, "home_big_saves"), away_big_saves=_f(row, "away_big_saves"),
            home_punches=_f(row, "home_punches"), away_punches=_f(row, "away_punches"),
            home_dispossessed=_f(row, "home_dispossessed"), away_dispossessed=_f(row, "away_dispossessed"),
            home_hit_woodwork=_f(row, "home_hit_woodwork"), away_hit_woodwork=_f(row, "away_hit_woodwork"),
            home_through_balls=_f(row, "home_through_balls"), away_through_balls=_f(row, "away_through_balls"),
            home_total_tackles=_f(row, "home_total_tackles"), away_total_tackles=_f(row, "away_total_tackles"),
            home_tackles=_f(row, "home_tackles"), away_tackles=_f(row, "away_tackles"),
            home_passes=_f(row, "home_passes"), away_passes=_f(row, "away_passes"),
            home_accurate_passes=_f(row, "home_accurate_passes"), away_accurate_passes=_f(row, "away_accurate_passes"),
            home_shots_off_target=_f(row, "home_shots_off_target"), away_shots_off_target=_f(row, "away_shots_off_target"),
            home_shots_outside_box=_f(row, "home_shots_outside_box"), away_shots_outside_box=_f(row, "away_shots_outside_box"),
            home_number_of_sprints=_f(row, "home_number_of_sprints"), away_number_of_sprints=_f(row, "away_number_of_sprints"),
            home_attack=_f(row, "home_attack"), away_attack=_f(row, "away_attack"),
            home_ball_safe=_f(row, "home_ball_safe"), away_ball_safe=_f(row, "away_ball_safe"),
            home_attack_pct=_f(row, "home_attack_pct"), away_attack_pct=_f(row, "away_attack_pct"),
            home_ball_safe_pct=_f(row, "home_ball_safe_pct"), away_ball_safe_pct=_f(row, "away_ball_safe_pct"),
            home_errors_lead_to_shot=_f(row, "home_errors_lead_to_shot"), away_errors_lead_to_shot=_f(row, "away_errors_lead_to_shot"),
            home_fouled_final_third=_f(row, "home_fouled_final_third"), away_fouled_final_third=_f(row, "away_fouled_final_third"),
            home_crosses_value=_f(row, "home_crosses_value"), away_crosses_value=_f(row, "away_crosses_value"),
            home_crosses_total=_f(row, "home_crosses_total"), away_crosses_total=_f(row, "away_crosses_total"),
            home_long_balls_value=_f(row, "home_long_balls_value"), away_long_balls_value=_f(row, "away_long_balls_value"),
            home_long_balls_total=_f(row, "home_long_balls_total"), away_long_balls_total=_f(row, "away_long_balls_total"),
            home_dribbles_value=_f(row, "home_dribbles_value"), away_dribbles_value=_f(row, "away_dribbles_value"),
            home_dribbles_total=_f(row, "home_dribbles_total"), away_dribbles_total=_f(row, "away_dribbles_total"),
            home_aerial_duels_value=_f(row, "home_aerial_duels_value"), away_aerial_duels_value=_f(row, "away_aerial_duels_value"),
            home_aerial_duels_total=_f(row, "home_aerial_duels_total"), away_aerial_duels_total=_f(row, "away_aerial_duels_total"),
            home_ground_duels_value=_f(row, "home_ground_duels_value"), away_ground_duels_value=_f(row, "away_ground_duels_value"),
            home_ground_duels_total=_f(row, "home_ground_duels_total"), away_ground_duels_total=_f(row, "away_ground_duels_total"),
            home_final_third_value=_f(row, "home_final_third_value"), away_final_third_value=_f(row, "away_final_third_value"),
            home_final_third_pct=_f(row, "home_final_third_pct"), away_final_third_pct=_f(row, "away_final_third_pct"),
            home_duels=_f(row, "home_duels"), away_duels=_f(row, "away_duels"),
            home_aerial_pct=_f(row, "home_aerial_pct"),
            away_aerial_pct=_f(row, "away_aerial_pct"),
            home_ground_pct=_f(row, "home_ground_pct"),
            away_ground_pct=_f(row, "away_ground_pct"),
            home_crosses_pct=_f(row, "home_crosses_pct"),
            away_crosses_pct=_f(row, "away_crosses_pct"),
            home_long_pct=_f(row, "home_long_pct"),
            away_long_pct=_f(row, "away_long_pct"),
            home_tackles_won=_f(row, "home_tackles_won"),
            away_tackles_won=_f(row, "away_tackles_won"),
            home_dribbles_pct=_f(row, "home_dribbles_pct"),
            away_dribbles_pct=_f(row, "away_dribbles_pct"),
            home_interceptions=_f(row, "home_interceptions"),
            away_interceptions=_f(row, "away_interceptions"),
            home_clearances=_f(row, "home_clearances"),
            away_clearances=_f(row, "away_clearances"),
            home_recoveries=_f(row, "home_recoveries"),
            away_recoveries=_f(row, "away_recoveries"),
            home_gk_saves=_f(row, "home_gk_saves"),
            away_gk_saves=_f(row, "away_gk_saves"),
            home_goals_prev=_f(row, "home_goals_prev"),
            away_goals_prev=_f(row, "away_goals_prev"),
            home_shots_inside=_f(row, "home_shots_inside"),
            away_shots_inside=_f(row, "away_shots_inside"),
            home_blocked=_f(row, "home_blocked"),
            away_blocked=_f(row, "away_blocked"),
            home_danger_pct=_f(row, "home_danger_pct"),
            away_danger_pct=_f(row, "away_danger_pct"),
            home_final_third=_f(row, "home_final_third"),
            away_final_third=_f(row, "away_final_third"),
            home_pen_touches=_f(row, "home_pen_touches"),
            away_pen_touches=_f(row, "away_pen_touches"),
            home_big_scored=_f(row, "home_big_scored"),
            away_big_scored=_f(row, "away_big_scored"),
            home_red=_f(row, "home_red"), away_red=_f(row, "away_red"),
            home_xg_target=_f(row, "home_xg_target"),
            away_xg_target=_f(row, "away_xg_target"),
        )
        elo_system.update(home, away, gh_i, ga_i, comp_code)

    return tracker, elo_system


def _verify_chronological_order(df_sorted: pd.DataFrame) -> bool:
    if df_sorted is None or df_sorted.empty:
        return True
    if "date_parsed" not in df_sorted.columns:
        return True
    try:
        dates = pd.to_datetime(df_sorted["date_parsed"], errors="coerce")
        if dates.isna().any():
            logger.warning("Dates non parsables dans l'ordre chronologique")
            return False
        diffs = dates.diff().dropna()
        if (diffs < pd.Timedelta(0)).any():
            logger.error("Ordre chronologique non respecté")
            return False
        return True
    except Exception as exc:
        logger.warning("Impossible de vérifier l'ordre chronologique : %s", exc)
        return False


def create_features_with_context(
    df_past: pd.DataFrame,
    strict_competition: bool = _STRICT_COMPETITION_DEFAULT,
) -> Tuple[
    np.ndarray, np.ndarray, List[str], List[str], List[int], List[int],
    List[Optional[str]], List[str],
]:
    empty_return = (
        np.zeros((0, EXPECTED_FEATURE_COUNT)),
        np.zeros((0,), dtype=np.int64),
        [], [], [], [], [], [],
    )
    if df_past is None or df_past.empty:
        return empty_return
    if "date_parsed" not in df_past.columns:
        logger.error("create_features_with_context : 'date_parsed' manquante.")
        return empty_return

    df_sorted = df_past.sort_values("date_parsed", kind="mergesort").reset_index(drop=True)

    if not _verify_chronological_order(df_sorted):
        logger.error("Ordre chronologique non vérifiable : abandon.")
        return empty_return

    n_total = len(df_sorted)
    if n_total == 0:
        return empty_return

    if _WARMUP_RATIO <= 0.0:
        warmup_count = 0
    else:
        warmup_count = max(1, int(n_total * _WARMUP_RATIO))
        warmup_count = min(warmup_count, max(0, n_total - 1))

    if warmup_count > 0:
        df_warmup = df_sorted.iloc[:warmup_count]
        dgf, dpts, _ = compute_league_defaults(df_warmup)
    else:
        dgf, dpts, _ = compute_league_defaults(df_sorted)

    tracker = TeamHistoryTracker(
        default_gf=dgf, default_ga=dgf, default_pts=dpts,
        strict_competition=strict_competition,
    )
    elo_system = EloSystem()

    def _f(row: Any, name: str) -> Optional[float]:
        return _safe_float_or_none(getattr(row, name, None))

    def _record_from_row(row: Any, home: str, away: str, gh_i: int, ga_i: int,
                         match_date: Any, comp_code: Optional[str]) -> None:
        ht_home = _safe_goals_or_none(getattr(row, "ht_home", None))
        ht_away = _safe_goals_or_none(getattr(row, "ht_away", None))
        if ht_home is None or ht_away is None:
            ht_home = None
            ht_away = None
        tracker.record(
            home, away, gh_i, ga_i, match_date, comp_code,
            ht_home=ht_home, ht_away=ht_away,
            home_xg=_f(row, "home_xg"), away_xg=_f(row, "away_xg"),
            home_possession=_f(row, "home_possession"),
            away_possession=_f(row, "away_possession"),
            home_shots=_f(row, "home_total_shots"),
            away_shots=_f(row, "away_total_shots"),
            home_shots_on_target=_f(row, "home_shots_on_target"),
            away_shots_on_target=_f(row, "away_shots_on_target"),
            home_corners=_f(row, "home_corner_kicks"),
            away_corners=_f(row, "away_corner_kicks"),
            home_yellow=_f(row, "home_yellow_cards"),
            away_yellow=_f(row, "away_yellow_cards"),
            home_fouls=_f(row, "home_fouls"), away_fouls=_f(row, "away_fouls"),
            home_pass_acc=_f(row, "home_pass_accuracy_pct"),
            away_pass_acc=_f(row, "away_pass_accuracy_pct"),
            home_big_chances=_f(row, "home_big_chances"),
            away_big_chances=_f(row, "away_big_chances"),
            home_offsides=_f(row, "home_offsides"), away_offsides=_f(row, "away_offsides"),
            home_throw_ins=_f(row, "home_throw_ins"), away_throw_ins=_f(row, "away_throw_ins"),
            home_free_kicks=_f(row, "home_free_kicks"), away_free_kicks=_f(row, "away_free_kicks"),
            home_goal_kicks=_f(row, "home_goal_kicks"), away_goal_kicks=_f(row, "away_goal_kicks"),
            home_high_claims=_f(row, "home_high_claims"), away_high_claims=_f(row, "away_high_claims"),
            home_total_saves=_f(row, "home_total_saves"), away_total_saves=_f(row, "away_total_saves"),
            home_big_saves=_f(row, "home_big_saves"), away_big_saves=_f(row, "away_big_saves"),
            home_punches=_f(row, "home_punches"), away_punches=_f(row, "away_punches"),
            home_dispossessed=_f(row, "home_dispossessed"), away_dispossessed=_f(row, "away_dispossessed"),
            home_hit_woodwork=_f(row, "home_hit_woodwork"), away_hit_woodwork=_f(row, "away_hit_woodwork"),
            home_through_balls=_f(row, "home_through_balls"), away_through_balls=_f(row, "away_through_balls"),
            home_total_tackles=_f(row, "home_total_tackles"), away_total_tackles=_f(row, "away_total_tackles"),
            home_tackles=_f(row, "home_tackles"), away_tackles=_f(row, "away_tackles"),
            home_passes=_f(row, "home_passes"), away_passes=_f(row, "away_passes"),
            home_accurate_passes=_f(row, "home_accurate_passes"), away_accurate_passes=_f(row, "away_accurate_passes"),
            home_shots_off_target=_f(row, "home_shots_off_target"), away_shots_off_target=_f(row, "away_shots_off_target"),
            home_shots_outside_box=_f(row, "home_shots_outside_box"), away_shots_outside_box=_f(row, "away_shots_outside_box"),
            home_number_of_sprints=_f(row, "home_number_of_sprints"), away_number_of_sprints=_f(row, "away_number_of_sprints"),
            home_attack=_f(row, "home_attack"), away_attack=_f(row, "away_attack"),
            home_ball_safe=_f(row, "home_ball_safe"), away_ball_safe=_f(row, "away_ball_safe"),
            home_attack_pct=_f(row, "home_attack_pct"), away_attack_pct=_f(row, "away_attack_pct"),
            home_ball_safe_pct=_f(row, "home_ball_safe_pct"), away_ball_safe_pct=_f(row, "away_ball_safe_pct"),
            home_errors_lead_to_shot=_f(row, "home_errors_lead_to_shot"), away_errors_lead_to_shot=_f(row, "away_errors_lead_to_shot"),
            home_fouled_final_third=_f(row, "home_fouled_final_third"), away_fouled_final_third=_f(row, "away_fouled_final_third"),
            home_crosses_value=_f(row, "home_crosses_value"), away_crosses_value=_f(row, "away_crosses_value"),
            home_crosses_total=_f(row, "home_crosses_total"), away_crosses_total=_f(row, "away_crosses_total"),
            home_long_balls_value=_f(row, "home_long_balls_value"), away_long_balls_value=_f(row, "away_long_balls_value"),
            home_long_balls_total=_f(row, "home_long_balls_total"), away_long_balls_total=_f(row, "away_long_balls_total"),
            home_dribbles_value=_f(row, "home_dribbles_value"), away_dribbles_value=_f(row, "away_dribbles_value"),
            home_dribbles_total=_f(row, "home_dribbles_total"), away_dribbles_total=_f(row, "away_dribbles_total"),
            home_aerial_duels_value=_f(row, "home_aerial_duels_value"), away_aerial_duels_value=_f(row, "away_aerial_duels_value"),
            home_aerial_duels_total=_f(row, "home_aerial_duels_total"), away_aerial_duels_total=_f(row, "away_aerial_duels_total"),
            home_ground_duels_value=_f(row, "home_ground_duels_value"), away_ground_duels_value=_f(row, "away_ground_duels_value"),
            home_ground_duels_total=_f(row, "home_ground_duels_total"), away_ground_duels_total=_f(row, "away_ground_duels_total"),
            home_final_third_value=_f(row, "home_final_third_value"), away_final_third_value=_f(row, "away_final_third_value"),
            home_final_third_pct=_f(row, "home_final_third_pct"), away_final_third_pct=_f(row, "away_final_third_pct"),
            home_duels=_f(row, "home_duels"), away_duels=_f(row, "away_duels"),
            home_aerial_pct=_f(row, "home_aerial_pct"),
            away_aerial_pct=_f(row, "away_aerial_pct"),
            home_ground_pct=_f(row, "home_ground_pct"),
            away_ground_pct=_f(row, "away_ground_pct"),
            home_crosses_pct=_f(row, "home_crosses_pct"),
            away_crosses_pct=_f(row, "away_crosses_pct"),
            home_long_pct=_f(row, "home_long_pct"),
            away_long_pct=_f(row, "away_long_pct"),
            home_tackles_won=_f(row, "home_tackles_won"),
            away_tackles_won=_f(row, "away_tackles_won"),
            home_dribbles_pct=_f(row, "home_dribbles_pct"),
            away_dribbles_pct=_f(row, "away_dribbles_pct"),
            home_interceptions=_f(row, "home_interceptions"),
            away_interceptions=_f(row, "away_interceptions"),
            home_clearances=_f(row, "home_clearances"),
            away_clearances=_f(row, "away_clearances"),
            home_recoveries=_f(row, "home_recoveries"),
            away_recoveries=_f(row, "away_recoveries"),
            home_gk_saves=_f(row, "home_gk_saves"),
            away_gk_saves=_f(row, "away_gk_saves"),
            home_goals_prev=_f(row, "home_goals_prev"),
            away_goals_prev=_f(row, "away_goals_prev"),
            home_shots_inside=_f(row, "home_shots_inside"),
            away_shots_inside=_f(row, "away_shots_inside"),
            home_blocked=_f(row, "home_blocked"),
            away_blocked=_f(row, "away_blocked"),
            home_danger_pct=_f(row, "home_danger_pct"),
            away_danger_pct=_f(row, "away_danger_pct"),
            home_final_third=_f(row, "home_final_third"),
            away_final_third=_f(row, "away_final_third"),
            home_pen_touches=_f(row, "home_pen_touches"),
            away_pen_touches=_f(row, "away_pen_touches"),
            home_big_scored=_f(row, "home_big_scored"),
            away_big_scored=_f(row, "away_big_scored"),
            home_red=_f(row, "home_red"), away_red=_f(row, "away_red"),
            home_xg_target=_f(row, "home_xg_target"),
            away_xg_target=_f(row, "away_xg_target"),
        )

    if warmup_count > 0:
        for row in df_sorted.iloc[:warmup_count].itertuples(index=False):
            home = getattr(row, "home_key", None)
            away = getattr(row, "away_key", None)
            gh_i = _safe_goals_or_none(getattr(row, "goals_home", None))
            ga_i = _safe_goals_or_none(getattr(row, "goals_away", None))
            if home is None or away is None or gh_i is None or ga_i is None:
                continue
            match_date = getattr(row, "date_parsed", None)
            comp_code = _normalize_competition_code(getattr(row, "competition_code", None))
            _record_from_row(row, home, away, gh_i, ga_i, match_date, comp_code)
            elo_system.update(home, away, gh_i, ga_i, comp_code)

    if warmup_count == 0:
        logger.info("Warmup désactivé : 100%% des matchs contribuent à l'entraînement")
        stage_print(f"🔥 Warmup désactivé : {n_total} matchs utilisés pour entraînement")

    feature_rows: List[np.ndarray] = []
    labels: List[int] = []
    home_keys: List[str] = []
    away_keys: List[str] = []
    goals_home_list: List[int] = []
    goals_away_list: List[int] = []
    comp_codes: List[Optional[str]] = []
    h2h_sources: List[str] = []
    skip_counters: Counter = Counter()

    for row in df_sorted.iloc[warmup_count:].itertuples(index=False):
        home = getattr(row, "home_key", None)
        away = getattr(row, "away_key", None)
        gh = getattr(row, "goals_home", None)
        ga = getattr(row, "goals_away", None)
        if home is None or away is None:
            skip_counters["missing_team_key"] += 1
            continue
        gh_i = _safe_goals_or_none(gh)
        ga_i = _safe_goals_or_none(ga)
        if gh_i is None or ga_i is None:
            skip_counters["invalid_goals"] += 1
            continue
        match_date = getattr(row, "date_parsed", None)
        if match_date is None or pd.isna(match_date):
            skip_counters["invalid_date"] += 1
            continue
        comp_code = _normalize_competition_code(getattr(row, "competition_code", None))

        feat, h2h_source = build_feature_vector(
            home, away, tracker, elo_system, match_date, comp_code
        )
        if feat.shape[0] != EXPECTED_FEATURE_COUNT:
            skip_counters["wrong_feature_length"] += 1
            continue

        feature_rows.append(feat)
        if gh_i > ga_i:
            labels.append(0)
        elif gh_i == ga_i:
            labels.append(1)
        else:
            labels.append(2)
        home_keys.append(home)
        away_keys.append(away)
        goals_home_list.append(gh_i)
        goals_away_list.append(ga_i)
        comp_codes.append(comp_code)
        h2h_sources.append(h2h_source)

        _record_from_row(row, home, away, gh_i, ga_i, match_date, comp_code)
        elo_system.update(home, away, gh_i, ga_i, comp_code)

    if skip_counters:
        logger.info(
            "create_features_with_context : %d exemples générés, rejets : %s",
            len(feature_rows), dict(skip_counters),
        )

    if not feature_rows:
        return empty_return

    X = np.array(feature_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)

    if X.shape[0] != len(y) or X.shape[0] != len(home_keys) or X.shape[0] != len(away_keys):
        logger.error("Incohérence de longueurs X/y/keys")
        return empty_return
    if X.shape[0] != len(goals_home_list) or X.shape[0] != len(goals_away_list):
        logger.error("Incohérence de longueurs X/gh/ga")
        return empty_return
    if X.shape[0] != len(comp_codes) or X.shape[0] != len(h2h_sources):
        logger.error("Incohérence de longueurs X/comp/h2h")
        return empty_return
    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        logger.error("Nombre de features incohérent : %d", X.shape[1])
        return empty_return

    return (X, y, home_keys, away_keys, goals_home_list, goals_away_list, comp_codes, h2h_sources)


if nn is not None:
    class _TorchNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_size: int = DL_HIDDEN_SIZE,
            output_dim: int = 3,
            dropout: float = DL_DROPOUT,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            h = max(8, int(hidden_size))
            h2 = max(4, h // 2)
            self.net = nn.Sequential(
                nn.Linear(input_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h2),
                nn.LayerNorm(h2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, output_dim),
            )

        def forward(self, x):
            return self.net(x)
else:
    _TorchNet = None


if torch is not None and nn is not None:
    class _TorchProbaWrapper:
        def __init__(self, model: Any):
            self.model = model
            self.classes_ = np.array([0, 1, 2])
            self._estimator_type = "classifier"
            n_features: Optional[int] = None
            try:
                first_linear = None
                for module in model.net:
                    if isinstance(module, nn.Linear):
                        first_linear = module
                        break
                if first_linear is not None:
                    n_features = int(first_linear.in_features)
            except Exception:
                n_features = None
            if n_features is None:
                n_features = int(getattr(model, "input_dim", 1))
            self.n_features_in_ = n_features

        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            return {"model": self.model}

        def set_params(self, **params: Any) -> "_TorchProbaWrapper":
            if "model" in params:
                self.model = params["model"]
            return self

        def fit(self, X: Any, y: Any) -> "_TorchProbaWrapper":
            return self

        def __sklearn_is_fitted__(self) -> bool:
            return True

        def predict_proba(self, X: Any) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32))
                logits = self.model(tensor)
                return torch.softmax(logits, dim=1).numpy()

        def predict(self, X: Any) -> np.ndarray:
            return np.argmax(self.predict_proba(X), axis=1)

    class _TorchTemperatureWrapper:
        def __init__(self, model: Any, temperature: float = 1.0):
            self.model = model
            if not np.isfinite(temperature) or temperature <= 0:
                temperature = 1.0
            self.temperature = float(temperature)
            self.classes_ = np.array([0, 1, 2])
            self._estimator_type = "classifier"
            n_features: Optional[int] = None
            try:
                first_linear = None
                for module in model.net:
                    if isinstance(module, nn.Linear):
                        first_linear = module
                        break
                if first_linear is not None:
                    n_features = int(first_linear.in_features)
            except Exception:
                n_features = None
            if n_features is None:
                n_features = int(getattr(model, "input_dim", 1))
            self.n_features_in_ = n_features

        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            return {"model": self.model, "temperature": self.temperature}

        def set_params(self, **params: Any) -> "_TorchTemperatureWrapper":
            if "model" in params:
                self.model = params["model"]
            if "temperature" in params:
                self.temperature = params["temperature"]
            return self

        def fit(self, X: Any, y: Any) -> "_TorchTemperatureWrapper":
            return self

        def __sklearn_is_fitted__(self) -> bool:
            return True

        def _raw_logits(self, X: Any) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32))
                return self.model(tensor).numpy()

        def predict_proba(self, X: Any) -> np.ndarray:
            logits = self._raw_logits(X)
            if abs(self.temperature - 1.0) > 1e-6:
                logits = logits / self.temperature
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

        def predict(self, X: Any) -> np.ndarray:
            return np.argmax(self.predict_proba(X), axis=1)
else:
    _TorchProbaWrapper = None
    _TorchTemperatureWrapper = None


def _fit_pb_model(
    class_name: str,
    gh: List[int],
    ga: List[int],
    th: List[str],
    ta: List[str],
    weights: Optional[np.ndarray] = None,
) -> Tuple[str, Any]:
    cls = _PB_MODEL_CLASSES.get(class_name)
    if cls is None:
        return "error", f"Modèle penaltyblog inconnu : {class_name}"

    def _new_model():
        if weights is not None:
            try:
                return cls(gh, ga, th, ta, weights=weights)
            except TypeError:
                pass
            except Exception as exc:
                logger.debug(
                    "Poids rejetés pour %s : %s", class_name, exc,
                )
        return cls(gh, ga, th, ta)

    if class_name in _PB_BAYESIAN_MODELS:
        try:
            m = _new_model()
            fit_kwargs: Dict[str, Any] = {
                "n_samples": BAYESIAN_N_SAMPLES,
                "burn": BAYESIAN_BURN,
                "n_chains": BAYESIAN_N_CHAINS,
                "thin": BAYESIAN_THIN,
            }
            try:
                import inspect as _inspect
                sig = _inspect.signature(m.fit)
                if "n_cores" in sig.parameters:
                    fit_kwargs["n_cores"] = 1
            except Exception:
                pass
            m.fit(**fit_kwargs)
            return "ok", m
        except Exception as exc:
            return (
                "error",
                f"Bayesian {class_name} : {type(exc).__name__}: {exc}",
            )

    try:
        m = _new_model()
        m.fit(minimizer_options={"maxiter": 5000})
        return "ok", m
    except Exception as exc:
        return (
            "error",
            f"{class_name} : {type(exc).__name__}: {exc}",
        )


def _calibrate_or_keep(
    name: str,
    base_model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
) -> Tuple[Any, str]:
    if X_calib is None or X_calib.shape[0] == 0:
        logger.warning("%s : set de calibration vide", name)
        return base_model, _CALIBRATION_STATUS_SKIPPED_SMALL

    if name == "DL" and _TorchProbaWrapper is not None:
        wrapped: Any = _TorchProbaWrapper(base_model)
    else:
        wrapped = base_model

    if X_calib.shape[0] < CALIBRATION_MIN_VAL_SAMPLES:
        logger.warning("%s : calibration trop petit (%d)", name, X_calib.shape[0])
        return wrapped, _CALIBRATION_STATUS_SKIPPED_SMALL

    unique_classes = np.unique(y_calib)
    if len(unique_classes) < _MIN_CLASSES_FOR_TRAINING:
        logger.warning("%s : calibration %s classes", name, unique_classes.tolist())
        return wrapped, _CALIBRATION_STATUS_SKIPPED_CLASSES

    if name == "DL" and _TorchTemperatureWrapper is not None:
        try:
            probe = _TorchTemperatureWrapper(base_model)
            raw_logits = probe._raw_logits(X_calib)
            if raw_logits.ndim != 2 or raw_logits.shape[1] != 3:
                raise ValueError("DL logits invalide")
            if not np.all(np.isfinite(raw_logits)):
                raise ValueError("DL logits non finis")
            best_t = 1.0
            best_ll = float("inf")
            for t in np.linspace(_DL_TEMPERATURE_GRID_MIN, _DL_TEMPERATURE_GRID_MAX, _DL_TEMPERATURE_GRID_STEPS):
                if t <= 0:
                    continue
                scaled = raw_logits / float(t)
                exp_logits = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                try:
                    ll = float(log_loss(y_calib, proba, labels=[0, 1, 2]))
                except Exception:
                    continue
                if ll < best_ll:
                    best_ll = ll
                    best_t = float(t)
            if not np.isfinite(best_ll):
                raise ValueError("Température : aucune valeur valide")
            return _TorchTemperatureWrapper(base_model, best_t), _CALIBRATION_STATUS_TEMPERATURE
        except Exception as exc:
            logger.warning("DL calibration température échouée : %s", exc)
            return wrapped, _CALIBRATION_STATUS_FAILED

    if FrozenEstimator is not None:
        try:
            calibrated = CalibratedClassifierCV(FrozenEstimator(wrapped), method=CALIBRATION_METHOD)
            calibrated.fit(X_calib, y_calib)
            return calibrated, _CALIBRATION_STATUS_OK
        except Exception as exc:
            logger.warning("%s : calibration FrozenEstimator échouée : %s", name, exc)

    try:
        calibrated = CalibratedClassifierCV(wrapped, method=CALIBRATION_METHOD, cv="prefit")
        calibrated.fit(X_calib, y_calib)
        return calibrated, _CALIBRATION_STATUS_OK
    except Exception as exc:
        logger.error("%s : calibration échouée : %s", name, exc)
        return wrapped, _CALIBRATION_STATUS_FAILED


def _extract_model_probabilities(grid: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        hda = np.array(grid.home_draw_away, dtype=np.float64)
        if hda.shape == (3,) and np.all(np.isfinite(hda)) and hda.sum() > 0:
            result["home_draw_away"] = (hda / hda.sum()).tolist()
    except Exception:
        pass
    for line, key in ((1.5, "over15"), (2.5, "over25")):
        val = None
        try:
            val = float(grid.total_goals("over", line))
        except Exception:
            try:
                _, _, over_val = grid.totals(line)
                val = float(over_val)
            except Exception:
                try:
                    attr_name = f"over_{str(line).replace('.', '')}"
                    val = float(getattr(grid, attr_name))
                except Exception:
                    val = None
        if val is not None and np.isfinite(val):
            result[key] = float(np.clip(val, 0.0, 1.0))
    try:
        btts = float(grid.btts_yes)
        if np.isfinite(btts):
            result["btts_yes"] = float(np.clip(btts, 0.0, 1.0))
    except Exception:
        pass
    try:
        home_expected = float(np.ravel(grid.home_goal_expectation)[0])
        away_expected = float(np.ravel(grid.away_goal_expectation)[0])
        if np.isfinite(home_expected) and np.isfinite(away_expected):
            result["expected_home_goals"] = home_expected
            result["expected_away_goals"] = away_expected
            result["home_xg"] = home_expected
            result["away_xg"] = away_expected
            result["xg_source"] = _XG_MODEL_SOURCE_TAG
            result["xg_is_official"] = False
    except Exception:
        pass
    return result


_extract_market_probabilities = _extract_model_probabilities


def _walk_forward_splits(
    n_samples: int,
    n_folds: int,
    initial_train_ratio: float = 0.40,
) -> List[Tuple[int, int]]:
    if n_samples < 200 or n_folds < 2:
        cut = max(1, int(n_samples * (1.0 - MODEL_TRAIN_TEST_SPLIT)))
        cut = min(cut, n_samples - 1)
        return [(cut, n_samples)]
    initial_train = max(int(n_samples * initial_train_ratio), 100)
    initial_train = min(initial_train, n_samples - n_folds * 10)
    if initial_train < 50:
        initial_train = max(50, n_samples // 3)
    remaining = n_samples - initial_train
    fold_size = max(remaining // n_folds, 10)
    splits: List[Tuple[int, int]] = []
    for i in range(n_folds):
        train_end = initial_train + i * fold_size
        val_end = train_end + fold_size
        if i == n_folds - 1:
            val_end = n_samples
        train_end = min(train_end, n_samples - 1)
        val_end = min(val_end, n_samples)
        if val_end - train_end < 5:
            continue
        splits.append((train_end, val_end))
    if not splits:
        cut = max(1, int(n_samples * 0.80))
        cut = min(cut, n_samples - 1)
        splits.append((cut, n_samples))
    return splits


def _proba_array_is_valid(proba: Any) -> bool:
    if proba is None:
        return False
    try:
        arr = np.asarray(proba, dtype=np.float64)
    except Exception:
        return False
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if np.any(arr < 0.0) or np.any(arr > 1.0 + _PROBA_ATOL):
        return False
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=_PROBA_ATOL):
        return False
    return True


def _validate_model_classes(model: Any, name: str) -> bool:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return True
    try:
        classes_list = list(classes)
    except Exception:
        return True
    if classes_list != list(_REQUIRED_CLASSES):
        logger.error("Modèle %s : ordre classes invalide %s", name, classes_list)
        return False
    return True


def _model_produces_three_classes(model: Any, X_probe: np.ndarray, name: str = "") -> bool:
    if X_probe is None or X_probe.shape[0] == 0:
        return False
    if not _validate_model_classes(model, name):
        return False
    try:
        proba = model.predict_proba(X_probe[:1])
    except Exception:
        return False
    return _proba_array_is_valid(proba)


def _evaluate_ml_model_walk_forward(
    factory: Callable[[], Any],
    X_raw: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    model_name: str = "",
) -> Tuple[float, int, int]:
    splits = _walk_forward_splits(len(X_raw), n_folds)
    log_losses: List[float] = []
    evaluated = 0
    attempted = 0
    for train_end, val_end in splits:
        attempted += 1
        try:
            fold_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            X_tr_imp = fold_pipeline.fit_transform(X_raw[:train_end])
            X_va_imp = fold_pipeline.transform(X_raw[train_end:val_end])
            y_tr = y[:train_end]
            y_va = y[train_end:val_end]
            if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
                continue
            m = factory()
            if model_name in _TREE_MODELS_NAN_NATIVE:
                m.fit(X_raw[:train_end], y_tr)
                proba = m.predict_proba(X_raw[train_end:val_end])
            else:
                m.fit(X_tr_imp, y_tr)
                proba = m.predict_proba(X_va_imp)
            if not _proba_array_is_valid(proba):
                continue
            if proba.shape[0] != y_va.shape[0]:
                continue
            ll = log_loss(y_va, proba, labels=[0, 1, 2])
            log_losses.append(float(ll))
            evaluated += 1
        except Exception as exc:
            logger.debug("Walk-forward fold échoué : %s", exc)
            continue
    if not log_losses:
        return float("inf"), evaluated, attempted
    return float(np.mean(log_losses)), evaluated, attempted


def _evaluate_dl_walk_forward(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    input_dim: int,
) -> Tuple[float, int, int]:
    if torch is None or nn is None or _TorchNet is None:
        return float("inf"), 0, 0
    if X_raw.shape[0] < 200 or n_folds < 2:
        return float("inf"), 0, 0
    splits = _walk_forward_splits(len(X_raw), n_folds)
    log_losses: List[float] = []
    evaluated = 0
    attempted = 0

    for train_end, val_end in splits:
        attempted += 1
        try:
            fold_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            X_tr = fold_pipeline.fit_transform(X_raw[:train_end])
            X_va = fold_pipeline.transform(X_raw[train_end:val_end])
            y_tr = y[:train_end]
            y_va = y[train_end:val_end]

            if len(np.unique(y_tr)) < _MIN_CLASSES_FOR_TRAINING:
                continue
            if X_va.shape[0] < 5:
                continue

            torch.manual_seed(MODEL_RANDOM_SEED)
            np.random.seed(MODEL_RANDOM_SEED)
            random.seed(MODEL_RANDOM_SEED)

            fold_model = _TorchNet(input_dim)
            opt = torch.optim.AdamW(
                fold_model.parameters(),
                lr=DL_LEARNING_RATE,
                weight_decay=DL_WEIGHT_DECAY,
            )
            crit = nn.CrossEntropyLoss()
            Xt = torch.FloatTensor(X_tr)
            yt = torch.LongTensor(y_tr)
            Xv = torch.FloatTensor(X_va)
            yv = torch.LongTensor(y_va)

            use_drop_last = len(Xt) > DL_BATCH_SIZE
            generator = torch.Generator().manual_seed(MODEL_RANDOM_SEED)
            loader = DataLoader(
                TensorDataset(Xt, yt),
                batch_size=DL_BATCH_SIZE,
                shuffle=True,
                drop_last=use_drop_last,
                generator=generator,
            )

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0
            epochs_no_early_stop = max(5, DL_EPOCHS // 4)

            for _ in range(epochs_no_early_stop):
                fold_model.train()
                for bx, by in loader:
                    opt.zero_grad()
                    loss = crit(fold_model(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), DL_GRAD_CLIP)
                    opt.step()
                fold_model.eval()
                with torch.no_grad():
                    val_logits = fold_model(Xv)
                    val_loss = crit(val_logits, yv).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in fold_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= DL_EARLY_STOPPING_PATIENCE:
                        break

            if best_state is not None:
                fold_model.load_state_dict(best_state)

            fold_model.eval()
            with torch.no_grad():
                proba = torch.softmax(fold_model(Xv), dim=1).numpy()

            if not _proba_array_is_valid(proba):
                continue
            if proba.shape[0] != y_va.shape[0]:
                continue
            ll = log_loss(y_va, proba, labels=[0, 1, 2])
            log_losses.append(float(ll))
            evaluated += 1
        except Exception as exc:
            logger.debug("DL walk-forward fold échoué : %s", exc)
            continue

    if not log_losses:
        return float("inf"), evaluated, attempted
    return float(np.mean(log_losses)), evaluated, attempted


def _evaluate_pb_model_walk_forward(
    class_name: str,
    gh: List[int],
    ga: List[int],
    th: List[str],
    ta: List[str],
    n_folds: int,
) -> Tuple[float, int, int]:
    n_samples = len(gh)
    if n_samples < max(200, _PB_WF_MIN_TRAIN_SAMPLES) or n_folds < 2:
        return float("inf"), 0, 0
    splits = _walk_forward_splits(n_samples, n_folds)
    log_losses: List[float] = []
    evaluated = 0
    attempted = 0

    for train_end, val_end in splits:
        attempted += 1
        if train_end < _PB_WF_MIN_TRAIN_SAMPLES:
            continue
        if (val_end - train_end) < _PB_WF_MIN_VAL_SAMPLES:
            continue
        try:
            gh_tr = gh[:train_end]
            ga_tr = ga[:train_end]
            th_tr = th[:train_end]
            ta_tr = ta[:train_end]
            gh_va = gh[train_end:val_end]
            ga_va = ga[train_end:val_end]
            th_va = th[train_end:val_end]
            ta_va = ta[train_end:val_end]

            y_va = []
            for h_g, a_g in zip(gh_va, ga_va):
                if h_g > a_g:
                    y_va.append(0)
                elif h_g == a_g:
                    y_va.append(1)
                else:
                    y_va.append(2)

            if len(np.unique(y_va)) < 2:
                continue

            status, m = _fit_pb_model(class_name, gh_tr, ga_tr, th_tr, ta_tr)
            if status != "ok":
                continue

            probas: List[List[float]] = []
            y_kept: List[int] = []
            for h, a, yv in zip(th_va, ta_va, y_va):
                try:
                    grid = m.predict(h, a)
                    markets = _extract_model_probabilities(grid)
                    hda = markets.get("home_draw_away")
                    if hda is None or len(hda) != 3:
                        continue
                    probas.append(hda)
                    y_kept.append(yv)
                except Exception:
                    continue

            if len(probas) < _PB_WF_MIN_VAL_SAMPLES:
                continue
            coverage = len(probas) / float(len(y_va)) if len(y_va) > 0 else 0.0
            if coverage < _PB_CALIB_MIN_COVERAGE:
                continue

            probas_arr = np.array(probas, dtype=np.float64)
            y_arr = np.array(y_kept, dtype=np.int64)
            if not _proba_array_is_valid(probas_arr):
                continue
            if len(np.unique(y_arr)) < 2:
                continue
            if probas_arr.shape[0] != y_arr.shape[0]:
                continue

            ll = log_loss(y_arr, probas_arr, labels=[0, 1, 2])
            log_losses.append(float(ll))
            evaluated += 1
        except Exception as exc:
            logger.debug("PB walk-forward fold échoué : %s", exc)
            continue

    if not log_losses:
        return float("inf"), evaluated, attempted
    return float(np.mean(log_losses)), evaluated, attempted


def _make_xgb_factory() -> Callable[[], Any]:
    n_jobs = MODEL_N_JOBS if MODEL_N_JOBS != 0 else 1

    def factory() -> Any:
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=MODEL_RANDOM_SEED,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            n_jobs=n_jobs,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            reg_lambda=XGB_REG_LAMBDA,
            reg_alpha=XGB_REG_ALPHA,
            tree_method="hist",
        )

    return factory


def _make_lgbm_factory() -> Callable[[], Any]:
    n_jobs = MODEL_N_JOBS if MODEL_N_JOBS != 0 else 1

    def factory() -> Any:
        return LGBMClassifier(
            n_estimators=LGBM_N_ESTIMATORS,
            max_depth=LGBM_MAX_DEPTH,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            subsample=LGBM_SUBSAMPLE,
            colsample_bytree=LGBM_COLSAMPLE,
            reg_lambda=LGBM_REG_LAMBDA,
            random_state=MODEL_RANDOM_SEED,
            verbose=-1,
            objective="multiclass",
            num_class=3,
            n_jobs=n_jobs,
        )

    return factory


def _make_catboost_factory() -> Callable[[], Any]:
    thread_count = MODEL_N_JOBS if MODEL_N_JOBS > 0 else -1

    def factory() -> Any:
        return CatBoostClassifier(
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LEARNING_RATE,
            l2_leaf_reg=CATBOOST_L2_LEAF_REG,
            random_strength=CATBOOST_RANDOM_STRENGTH,
            random_state=MODEL_RANDOM_SEED,
            verbose=False,
            loss_function="MultiClass",
            classes_count=3,
            thread_count=thread_count,
            allow_writing_files=False,
        )

    return factory


def _make_rf_factory() -> Callable[[], Any]:
    n_jobs = MODEL_N_JOBS if MODEL_N_JOBS != 0 else 1
    max_features: Any = RF_MAX_FEATURES_RESOLVED

    def factory() -> Any:
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=max_features,
            random_state=MODEL_RANDOM_SEED,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        )

    return factory


def _compute_optimal_temperature(
    probas: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, bool]:
    if probas.shape[0] < 50:
        return float(TEMPERATURE_SCALING_DEFAULT), False
    best_t = float(TEMPERATURE_SCALING_DEFAULT)
    best_ll = float("inf")
    lo = float(TEMPERATURE_SCALING_MIN)
    hi = float(TEMPERATURE_SCALING_MAX)
    for t in np.linspace(lo, hi, 17):
        if t <= 0:
            continue
        logits = np.log(np.clip(probas, _EPS, 1.0)) / t
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        try:
            ll = float(log_loss(y_true, softmax, labels=[0, 1, 2]))
        except Exception:
            continue
        if ll < best_ll:
            best_ll = ll
            best_t = float(t)
    if not np.isfinite(best_ll):
        return float(TEMPERATURE_SCALING_DEFAULT), False
    return best_t, True


def _compute_ensemble_probas(
    ml_models: Dict[str, Any],
    pb_models: List[Tuple[str, Any]],
    Xs_set: np.ndarray,
    home_keys_set: List[str],
    away_keys_set: List[str],
    weights: Dict[str, float],
    Xs_set_raw: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if Xs_set.shape[0] == 0:
        return None
    n_rows = Xs_set.shape[0]
    hda_list: List[np.ndarray] = []
    w_list: List[float] = []

    for name, model in ml_models.items():
        try:
            if name in _TREE_MODELS_NAN_NATIVE and Xs_set_raw is not None:
                X_input = Xs_set_raw
            else:
                X_input = Xs_set
            proba = model.predict_proba(X_input)
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue
            arr = np.asarray(proba, dtype=np.float64)
            if arr.shape != (n_rows, 3):
                continue
            if not _proba_array_is_valid(arr):
                continue
            hda_list.append(arr)
            w_list.append(w)
        except Exception:
            continue

    for i, (class_name, model) in enumerate(pb_models):
        w = float(weights.get(f"pb_{i}", 0.0))
        if w <= 0:
            continue
        probas: List[List[float]] = []
        try:
            for h, a in zip(home_keys_set, away_keys_set):
                grid = model.predict(h, a)
                markets = _extract_model_probabilities(grid)
                hda = markets.get("home_draw_away")
                if hda is None:
                    raise ValueError("home_draw_away indisponible")
                probas.append(hda)
            probas_arr = np.array(probas, dtype=np.float64)
            if probas_arr.shape != (n_rows, 3):
                continue
            if not _proba_array_is_valid(probas_arr):
                continue
            hda_list.append(probas_arr)
            w_list.append(w)
        except Exception:
            continue

    if not hda_list:
        return None

    w_arr = np.array(w_list, dtype=np.float64)
    if w_arr.sum() <= 0:
        w_arr = np.ones_like(w_arr)
    w_arr = w_arr / w_arr.sum()

    stacked = np.stack(hda_list, axis=0)
    blended = np.tensordot(w_arr, stacked, axes=(0, 0))
    blended = np.clip(blended, _EPS, 1.0)
    blended = blended / blended.sum(axis=1, keepdims=True)
    return blended


def _evaluate_ensemble_on_set(
    ml_models: Dict[str, Any],
    pb_models: List[Tuple[str, Any]],
    Xs_set: np.ndarray,
    y_set: np.ndarray,
    home_keys_set: List[str],
    away_keys_set: List[str],
    weights: Dict[str, float],
    temperature: float,
    Xs_set_raw: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    if Xs_set.shape[0] == 0 or y_set.shape[0] == 0:
        return None
    blended = _compute_ensemble_probas(
        ml_models, pb_models, Xs_set, home_keys_set, away_keys_set, weights, Xs_set_raw,
    )
    if blended is None:
        return None
    if blended.shape[0] != y_set.shape[0]:
        return None
    if (
        temperature is not None
        and np.isfinite(temperature)
        and temperature > 0
        and abs(temperature - 1.0) > 1e-6
    ):
        logits = np.log(np.clip(blended, _EPS, 1.0)) / float(temperature)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        blended = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    try:
        ll = log_loss(y_set, blended, labels=[0, 1, 2])
    except Exception:
        return None
    acc = float((blended.argmax(axis=1) == y_set).mean())
    return (float(ll), acc)


def _project_weights_to_bounds(
    vals: Dict[str, float],
    eff_min: float,
    eff_max: float,
) -> Dict[str, float]:
    keys = list(vals.keys())
    if not keys:
        return vals
    n = len(keys)
    arr = np.array([max(float(vals[k]), 0.0) for k in keys], dtype=np.float64)
    if eff_min * n > 1.0 + 1e-12:
        eff_min = 1.0 / n
    if eff_max < eff_min:
        eff_max = max(eff_min, 1.0 / n)
    if eff_max * n < 1.0 - 1e-12:
        eff_max = 1.0 / n
    total = arr.sum()
    if total <= 0:
        arr = np.ones(n, dtype=np.float64) / n
    else:
        arr = arr / total
    for _ in range(_WEIGHT_PROJECTION_MAX_ITER):
        arr = np.clip(arr, eff_min, eff_max)
        total = arr.sum()
        if total <= 0:
            arr = np.ones(n, dtype=np.float64) / n
            continue
        arr = arr / total
        if abs(total - 1.0) < _WEIGHT_PROJECTION_TOL and np.all(
            (arr >= eff_min - 1e-12) & (arr <= eff_max + 1e-12)
        ):
            break
    arr = np.clip(arr, eff_min, eff_max)
    total = arr.sum()
    if total <= 0:
        arr = np.ones(n, dtype=np.float64) / n
    else:
        arr = arr / total
    for k, v in zip(keys, arr.tolist()):
        vals[k] = float(v)
    return vals


def _final_validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    keys = [k for k in weights.keys() if not k.startswith("__")]
    if not keys:
        return weights
    cleaned: Dict[str, float] = {}
    for k in keys:
        try:
            v = float(weights[k])
        except (TypeError, ValueError):
            v = 0.0
        if not np.isfinite(v) or v < 0.0:
            v = 0.0
        cleaned[k] = v
    positive = {k: cleaned[k] for k in keys if cleaned[k] > 0.0}
    if not positive:
        for k in keys:
            weights[k] = 0.0
        return weights
    n = len(positive)
    eff_min = float(ENSEMBLE_MIN_WEIGHT)
    if eff_min * n > 1.0:
        eff_min = 1.0 / n
    eff_max = float(ENSEMBLE_MAX_WEIGHT)
    if eff_max < eff_min:
        eff_max = max(eff_min, 1.0 / n)
    projected = _project_weights_to_bounds(positive, eff_min, eff_max)
    for k in keys:
        weights[k] = float(projected.get(k, 0.0))
    return weights


def _normalize_weights(
    weights: Dict[str, float],
    min_w: float,
    max_w: float,
) -> Dict[str, float]:
    keys = [k for k in weights.keys() if not k.startswith("__")]
    if not keys:
        return weights
    zero_keys = set()
    for k in keys:
        try:
            v = float(weights[k])
        except (TypeError, ValueError):
            v = 0.0
        if not np.isfinite(v) or v <= 0.0:
            zero_keys.add(k)
    positive_keys = [k for k in keys if k not in zero_keys]
    if not positive_keys:
        return weights
    n_positive = len(positive_keys)
    effective_min_w = float(min_w)
    if effective_min_w * n_positive > 1.0:
        logger.warning("ENSEMBLE_MIN_WEIGHT ajusté à %.4f", 1.0 / n_positive)
        effective_min_w = 1.0 / n_positive
    effective_max_w = float(max_w)
    if effective_max_w < effective_min_w:
        effective_max_w = max(effective_min_w, 1.0 / n_positive)
    vals: Dict[str, float] = {}
    for k in positive_keys:
        try:
            v = float(weights[k])
        except (TypeError, ValueError):
            v = 0.0
        if not np.isfinite(v) or v <= 0:
            v = 0.0
        vals[k] = v
    total = sum(vals.values())
    if total <= 0:
        uniform = 1.0 / n_positive
        for k in positive_keys:
            vals[k] = uniform
    else:
        for k in positive_keys:
            vals[k] = vals[k] / total
    vals = _project_weights_to_bounds(vals, effective_min_w, effective_max_w)
    for k in keys:
        if k in zero_keys:
            weights[k] = 0.0
        else:
            weights[k] = float(vals.get(k, 0.0))
    weights = _final_validate_weights(weights)
    return weights


def _clip_outliers_signed(
    X_ref: np.ndarray,
    X_apply_list: List[np.ndarray],
    z_limit: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if z_limit <= 0 or X_ref.size == 0:
        return X_ref, X_apply_list
    mean = np.nanmean(X_ref, axis=0)
    std = np.nanstd(X_ref, axis=0)
    std = np.where((std == 0) | (~np.isfinite(std)), 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    lower = mean - z_limit * std
    upper = mean + z_limit * std
    clipped_ref = np.clip(X_ref, lower, upper)
    clipped_apply = [np.clip(X, lower, upper) for X in X_apply_list]
    return clipped_ref, clipped_apply


def _filter_valid_weights(weights: Dict[str, float]) -> Dict[str, float]:
    filtered: Dict[str, float] = {}
    for k, v in weights.items():
        if k.startswith("__"):
            filtered[k] = v
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv) and fv > 0.0:
            filtered[k] = fv
    return filtered


def _select_fallback_uniform_models(
    ml_models: Dict[str, Any],
    pb_models: List[Tuple[str, Any]],
    wf_ll_by_key: Dict[str, float],
) -> List[str]:
    valid_keys: List[str] = []
    for k, ll in wf_ll_by_key.items():
        try:
            ll_f = float(ll)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ll_f) or ll_f <= 0:
            continue
        if ll_f > _MAX_ACCEPTABLE_LOGLOSS:
            continue
        if k in ml_models:
            model = ml_models[k]
            if not _validate_model_classes(model, k):
                continue
            valid_keys.append(k)
        elif k.startswith("pb_"):
            try:
                idx = int(k.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if 0 <= idx < len(pb_models):
                valid_keys.append(k)
    return valid_keys


def _apply_quality_filter_on_weights(
    weights: Dict[str, float],
    wf_ll_by_key: Dict[str, float],
) -> Dict[str, float]:
    valid_lls: List[float] = []
    for k in weights.keys():
        if k.startswith("__"):
            continue
        ll = wf_ll_by_key.get(k, None)
        if ll is None:
            continue
        try:
            ll_f = float(ll)
        except (TypeError, ValueError):
            continue
        if np.isfinite(ll_f) and ll_f > 0:
            valid_lls.append(ll_f)
    if not valid_lls:
        return weights
    best_ll = min(valid_lls)
    relative_threshold = best_ll * _QUALITY_RELATIVE_THRESHOLD
    absolute_threshold = best_ll + _QUALITY_ABSOLUTE_MARGIN
    threshold = max(relative_threshold, absolute_threshold)
    filtered: Dict[str, float] = {}
    for k, v in weights.items():
        if k.startswith("__"):
            filtered[k] = v
            continue
        ll = wf_ll_by_key.get(k, None)
        if ll is None:
            continue
        try:
            ll_f = float(ll)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ll_f) or ll_f <= 0:
            continue
        if ll_f > threshold:
            logger.info("Modèle %s exclu (WF %.4f > %.4f)", k, ll_f, threshold)
            continue
        filtered[k] = v
    if not any(not k.startswith("__") for k in filtered.keys()):
        return weights
    return filtered


def _validate_splits(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_calib: np.ndarray,
    y_test: np.ndarray,
) -> bool:
    ok = True
    required = set(_REQUIRED_CLASSES)
    for label, arr in (
        ("train", y_train),
        ("val", y_val),
        ("calib", y_calib),
        ("test", y_test),
    ):
        if arr is None or arr.shape[0] == 0:
            logger.error("Split %s vide", label)
            ok = False
            continue
        present = set(int(c) for c in np.unique(arr))
        missing = required - present
        if missing:
            logger.error("Split %s manque classes %s", label, sorted(missing))
            ok = False
    if y_test is not None and y_test.shape[0] < _MIN_TEST_SAMPLES:
        logger.warning("Test trop petit (%d)", y_test.shape[0])
    return ok


def _compute_calibration_log_loss(
    model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_calib_raw: Optional[np.ndarray] = None,
    model_name: str = "",
) -> float:
    if X_calib.shape[0] == 0:
        return float("inf")
    try:
        if model_name in _TREE_MODELS_NAN_NATIVE and X_calib_raw is not None:
            proba = model.predict_proba(X_calib_raw)
        else:
            proba = model.predict_proba(X_calib)
        if not _proba_array_is_valid(proba):
            return float("inf")
        if proba.shape[0] != y_calib.shape[0]:
            return float("inf")
        return float(log_loss(y_calib, proba, labels=[0, 1, 2]))
    except Exception as exc:
        logger.debug("Calibration LogLoss échouée : %s", exc)
        return float("inf")


def train_models(
    df_past: pd.DataFrame,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[List[Tuple[str, Any]]],
    Optional[Any],
    Dict[str, float],
]:
    if df_past is None or len(df_past) < MIN_MATCHES_FOR_TRAINING:
        logger.warning(
            "Données insuffisantes : %d < %d",
            len(df_past) if df_past is not None else 0,
            MIN_MATCHES_FOR_TRAINING,
        )
        return None, None, None, {}

    (
        X, y, home_keys, away_keys, gh_all, ga_all, _, h2h_sources_all,
    ) = create_features_with_context(df_past, strict_competition=_STRICT_COMPETITION_DEFAULT)

    if X.shape[0] < MIN_MATCHES_FOR_TRAINING or X.shape[0] != len(y):
        logger.warning("Échec construction features")
        return None, None, None, {}

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        logger.error("X.shape[1] = %d ≠ %d", X.shape[1], EXPECTED_FEATURE_COUNT)
        return None, None, None, {}

    if not (
        X.shape[0] == len(home_keys) == len(away_keys) == len(gh_all)
        == len(ga_all) == len(h2h_sources_all) == len(y)
    ):
        logger.error("Longueurs incohérentes après features")
        return None, None, None, {}

    class_counts = np.bincount(y, minlength=3)
    if np.count_nonzero(class_counts) < _MIN_CLASSES_FOR_TRAINING:
        logger.error("Moins de %d classes : %s", _MIN_CLASSES_FOR_TRAINING, class_counts.tolist())
        return None, None, None, {}

    n = len(X)
    cut_train = int(n * TRAIN_RATIO)
    cut_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    cut_calib = int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO))

    cut_train = max(1, min(cut_train, n - 4))
    cut_val = max(cut_train + 1, min(cut_val, n - 3))
    cut_calib = max(cut_val + 1, min(cut_calib, n - 2))

    X_train_raw = X[:cut_train].astype(np.float64, copy=True)
    X_val_raw = X[cut_train:cut_val].astype(np.float64, copy=True)
    X_calib_raw = X[cut_val:cut_calib].astype(np.float64, copy=True)
    X_test_raw = X[cut_calib:].astype(np.float64, copy=True)

    y_train = y[:cut_train]
    y_val = y[cut_train:cut_val]
    y_calib = y[cut_val:cut_calib]
    y_test = y[cut_calib:]

    if not _validate_splits(y_train, y_val, y_calib, y_test):
        logger.error("Splits invalides ; entraînement annulé.")
        stage_print("❌ Splits d'entraînement invalides ; entraînement annulé.")
        return None, None, None, {}

    home_keys_train = home_keys[:cut_train]
    away_keys_train = away_keys[:cut_train]
    home_keys_calib = home_keys[cut_val:cut_calib]
    away_keys_calib = away_keys[cut_val:cut_calib]
    home_keys_test = home_keys[cut_calib:]
    away_keys_test = away_keys[cut_calib:]
    gh_train = gh_all[:cut_train]
    ga_train = ga_all[:cut_train]

    stage_print("🧠 ENTRAÎNEMENT")

    nan_count_before = int(np.isnan(X_train_raw).sum())
    inf_count_before = int(np.isinf(X_train_raw).sum())
    if nan_count_before > 0 or inf_count_before > 0:
        stage_print(
            f"🔧 Features brutes : {nan_count_before} NaN, "
            f"{inf_count_before} inf — traitement hybride (trees NaN natif, autres imputés)"
        )

    X_train_raw = np.where(np.isfinite(X_train_raw), X_train_raw, _NAN)
    X_val_raw = np.where(np.isfinite(X_val_raw), X_val_raw, _NAN)
    X_calib_raw = np.where(np.isfinite(X_calib_raw), X_calib_raw, _NAN)
    X_test_raw = np.where(np.isfinite(X_test_raw), X_test_raw, _NAN)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_calib_imp = imputer.transform(X_calib_raw)
    X_test_imp = imputer.transform(X_test_raw)

    if FEATURE_OUTLIER_ZSCORE > 0:
        (
            X_train_imp,
            [X_val_imp, X_calib_imp, X_test_imp],
        ) = _clip_outliers_signed(
            X_train_imp,
            [X_val_imp, X_calib_imp, X_test_imp],
            float(FEATURE_OUTLIER_ZSCORE),
        )

    scaler_only = StandardScaler()
    Xs_train = scaler_only.fit_transform(X_train_imp)
    Xs_val = scaler_only.transform(X_val_imp)
    Xs_calib = scaler_only.transform(X_calib_imp)
    Xs_test = scaler_only.transform(X_test_imp)

    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler_only),
    ])

    nan_count_after = int(np.isnan(Xs_train).sum())
    stage_print(
        f"✅ Preprocessing terminé — NaN restants (imputés) : {nan_count_after}"
    )

    ml_models: Dict[str, Any] = {}
    weights: Dict[str, float] = {}
    calib_ll_by_key: Dict[str, float] = {}
    wf_ll_by_key: Dict[str, float] = {}
    ll_source_by_key: Dict[str, str] = {}
    calibration_status_by_key: Dict[str, str] = {}
    wf_fold_coverage_by_key: Dict[str, float] = {}

    factories: List[Tuple[str, Callable[[], Any], str]] = []
    if XGBClassifier is not None:
        factories.append(("XGB", _make_xgb_factory(), "XGBOOST"))
    if LGBMClassifier is not None:
        factories.append(("LGBM", _make_lgbm_factory(), "LIGHTGBM"))
    if CatBoostClassifier is not None:
        factories.append(("CatBoost", _make_catboost_factory(), "CATBOOST"))
    if RandomForestClassifier is not None:
        factories.append(("RF", _make_rf_factory(), "RANDOM FOREST"))

    for name, factory, display in factories:
        try:
            m = factory()
            if name in _TREE_MODELS_NAN_NATIVE:
                X_fit = X_train_raw
                X_calib_fit = X_calib_raw
            else:
                X_fit = Xs_train
                X_calib_fit = Xs_calib

            m.fit(X_fit, y_train)
            if not _model_produces_three_classes(m, X_fit, name):
                logger.warning("%s ignoré : probabilités invalides", name)
                stage_print(f"❌ {display} : probabilités invalides")
                continue

            calibrated, calib_status = _calibrate_or_keep(name, m, X_calib_fit, y_calib)
            if not _model_produces_three_classes(calibrated, X_calib_fit, name):
                calibrated = m
                calib_status = _CALIBRATION_STATUS_RAW_KEPT
            ml_models[name] = calibrated
            calibration_status_by_key[name] = calib_status

            ll_calib = _compute_calibration_log_loss(
                calibrated, X_calib_fit, y_calib, X_calib_fit, name,
            )
            calib_ll_by_key[name] = float(ll_calib)

            ll_wf, wf_eval, wf_attempt = _evaluate_ml_model_walk_forward(
                factory, X_train_raw, y_train, MODEL_WALK_FORWARD_FOLDS, name,
            )
            wf_ll_by_key[name] = float(ll_wf)
            ll_source_by_key[name] = _LL_SOURCE_WALK_FORWARD
            coverage = wf_eval / wf_attempt if wf_attempt > 0 else 0.0
            wf_fold_coverage_by_key[name] = float(coverage)

            if coverage < _MIN_WF_FOLD_COVERAGE:
                logger.warning("%s : couverture WF faible (%d/%d)", name, wf_eval, wf_attempt)

            if np.isfinite(ll_calib):
                stage_print(
                    f"✅ {display} : LogLoss {float(ll_calib):.4f} "
                    f"(WF {wf_eval}/{wf_attempt} folds)"
                )
            else:
                stage_print(f"⚠️ {display} : LogLoss invalide")
        except Exception as exc:
            logger.error("%s : %s", display, exc)
            stage_print(f"❌ {display} : échec")

    if DL_ENABLED and torch is not None and nn is not None and _TorchNet is not None:
        if len(Xs_val) < _MIN_VAL_SAMPLES_FOR_DL or len(np.unique(y_val)) < 2:
            stage_print("⚠️ DEEP LEARNING : données validation insuffisantes, ignoré")
        else:
            try:
                torch.manual_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                random.seed(MODEL_RANDOM_SEED)
                model = _TorchNet(Xs_train.shape[1])
                opt = torch.optim.AdamW(
                    model.parameters(), lr=DL_LEARNING_RATE, weight_decay=DL_WEIGHT_DECAY,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=5
                )
                crit = nn.CrossEntropyLoss()
                Xt = torch.FloatTensor(Xs_train)
                yt = torch.LongTensor(y_train)
                Xv = torch.FloatTensor(Xs_val)
                yv = torch.LongTensor(y_val)
                use_drop_last = len(Xt) > DL_BATCH_SIZE
                generator = torch.Generator().manual_seed(MODEL_RANDOM_SEED)
                loader = DataLoader(
                    TensorDataset(Xt, yt),
                    batch_size=DL_BATCH_SIZE,
                    shuffle=True,
                    drop_last=use_drop_last,
                    generator=generator,
                )
                best_val_loss = float("inf")
                best_state = None
                patience_counter = 0
                for _ in range(DL_EPOCHS):
                    model.train()
                    for bx, by in loader:
                        opt.zero_grad()
                        loss = crit(model(bx), by)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), DL_GRAD_CLIP)
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(Xv)
                        val_loss = crit(val_logits, yv).item()
                    scheduler.step(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= DL_EARLY_STOPPING_PATIENCE:
                            break
                if best_state is not None:
                    model.load_state_dict(best_state)

                calibrated, calib_status = _calibrate_or_keep("DL", model, Xs_calib, y_calib)
                if not _model_produces_three_classes(calibrated, Xs_calib, "DL"):
                    calibrated = _TorchProbaWrapper(model)
                    calib_status = _CALIBRATION_STATUS_RAW_KEPT
                ml_models["DL"] = calibrated
                calibration_status_by_key["DL"] = calib_status

                ll_dl_calib = _compute_calibration_log_loss(
                    calibrated, Xs_calib, y_calib, None, "DL",
                )
                calib_ll_by_key["DL"] = float(ll_dl_calib)

                ll_dl_wf, wf_eval, wf_attempt = _evaluate_dl_walk_forward(
                    X_train_raw, y_train, MODEL_WALK_FORWARD_FOLDS, Xs_train.shape[1],
                )
                wf_ll_by_key["DL"] = float(ll_dl_wf)
                ll_source_by_key["DL"] = _LL_SOURCE_WALK_FORWARD
                coverage = wf_eval / wf_attempt if wf_attempt > 0 else 0.0
                wf_fold_coverage_by_key["DL"] = float(coverage)

                if coverage < _MIN_WF_FOLD_COVERAGE:
                    logger.warning("DL : couverture WF faible (%d/%d)", wf_eval, wf_attempt)

                if np.isfinite(ll_dl_calib):
                    stage_print(
                        f"✅ DEEP LEARNING : LogLoss {float(ll_dl_calib):.4f} "
                        f"(WF {wf_eval}/{wf_attempt} folds)"
                    )
                else:
                    stage_print("⚠️ DEEP LEARNING : LogLoss invalide")
            except Exception as exc:
                logger.error("Deep Learning : %s", exc)
                stage_print("❌ DEEP LEARNING : échec")

    stage_print("🤖 PENALTY BLOG MODELS")
    pb_candidates: List[Tuple[str, Any]] = []
    for class_name in _PB_MODEL_ORDER:
        is_bayesian = class_name in _PB_BAYESIAN_MODELS
        display_name = class_name.upper()
        if is_bayesian and not BAYESIAN_ENABLED:
            stage_print(f"⚠️ {display_name} : désactivé")
            continue
        if is_bayesian:
            stage_print(f"🧠 {display_name} : échantillonnage MCMC en cours...")
        status, payload = _fit_pb_model(class_name, gh_train, ga_train, home_keys_train, away_keys_train)
        if status == "ok":
            pb_candidates.append((class_name, payload))
            logger.info("penaltyblog %s entraîné", class_name)
            stage_print(f"✅ {display_name} : entraîné")
        else:
            logger.error("penaltyblog %s : %s", class_name, payload)
            stage_print(f"❌ {display_name} : échec")

    pb_models: List[Tuple[str, Any]] = []
    for class_name, model in pb_candidates:
        try:
            probas: List[List[float]] = []
            y_kept: List[int] = []
            total_calib = len(home_keys_calib)
            for h, a, yv in zip(home_keys_calib, away_keys_calib, y_calib):
                try:
                    grid = model.predict(h, a)
                    markets = _extract_model_probabilities(grid)
                    hda = markets.get("home_draw_away")
                    if hda is None or len(hda) != 3:
                        continue
                    hda_arr = np.asarray(hda, dtype=np.float64)
                    if hda_arr.shape != (3,) or not np.all(np.isfinite(hda_arr)):
                        continue
                    if hda_arr.sum() <= 0:
                        continue
                    hda_arr = hda_arr / hda_arr.sum()
                    probas.append(hda_arr.tolist())
                    y_kept.append(int(yv))
                except Exception:
                    continue

            if len(probas) < _PB_CALIB_MIN_VALID_SAMPLES:
                logger.warning("penaltyblog %s : peu de prédictions valides (%d)", class_name, len(probas))
                continue
            coverage = len(probas) / float(total_calib) if total_calib > 0 else 0.0
            if coverage < _PB_CALIB_MIN_COVERAGE:
                logger.warning("penaltyblog %s : couverture faible (%.2f)", class_name, coverage)
                continue

            probas_arr = np.array(probas, dtype=np.float64)
            y_arr = np.array(y_kept, dtype=np.int64)
            if not _proba_array_is_valid(probas_arr):
                logger.warning("penaltyblog %s : probabilités invalides", class_name)
                continue
            if len(np.unique(y_arr)) < 2:
                logger.warning("penaltyblog %s : une seule classe", class_name)
                continue

            ll_calib = float(log_loss(y_arr, probas_arr, labels=[0, 1, 2]))
            if not np.isfinite(ll_calib) or ll_calib <= 0:
                logger.warning("penaltyblog %s : LogLoss calibration invalide", class_name)
                continue

            ll_wf_pb, wf_eval, wf_attempt = _evaluate_pb_model_walk_forward(
                class_name, gh_train, ga_train, home_keys_train, away_keys_train,
                MODEL_WALK_FORWARD_FOLDS,
            )
            ll_source_pb = _LL_SOURCE_WALK_FORWARD
            if not np.isfinite(ll_wf_pb) or ll_wf_pb <= 0:
                logger.warning("penaltyblog %s : WF indisponible, repli calibration", class_name)
                ll_wf_pb = ll_calib
                ll_source_pb = _LL_SOURCE_CALIBRATION_FALLBACK

            if ll_wf_pb > _MAX_ACCEPTABLE_LOGLOSS:
                logger.warning("penaltyblog %s : WF %.4f > %.4f, poids réduit", class_name, ll_wf_pb, _MAX_ACCEPTABLE_LOGLOSS)
                ll_wf_pb = _MAX_ACCEPTABLE_LOGLOSS + 0.20

            pb_index = len(pb_models)
            pb_models.append((class_name, model))
            calib_ll_by_key[f"pb_{pb_index}"] = ll_calib
            wf_ll_by_key[f"pb_{pb_index}"] = float(ll_wf_pb)
            ll_source_by_key[f"pb_{pb_index}"] = ll_source_pb
            calibration_status_by_key[f"pb_{pb_index}"] = _CALIBRATION_STATUS_NOT_CALIBRATED
            coverage_wf = wf_eval / wf_attempt if wf_attempt > 0 else 0.0
            wf_fold_coverage_by_key[f"pb_{pb_index}"] = float(coverage_wf)
            source_label = "WF" if ll_source_pb == _LL_SOURCE_WALK_FORWARD else "CALIB-FALLBACK"
            stage_print(
                f"✅ {class_name.upper()} — CALIB: {ll_calib:.4f} | "
                f"{source_label}: {ll_wf_pb:.4f} "
                f"(calib cov {coverage * 100:.0f}% ; wf {wf_eval}/{wf_attempt})"
            )
        except Exception as exc:
            logger.error("Pondération penaltyblog %s : %s", class_name, exc)

    for k, ll in wf_ll_by_key.items():
        try:
            ll_f = float(ll)
        except (TypeError, ValueError):
            ll_f = float("inf")
        if np.isfinite(ll_f) and ll_f > 0 and ll_f <= _MAX_ACCEPTABLE_LOGLOSS:
            weights[k] = 1.0 / (ll_f + _EPS_WEIGHT)
        else:
            weights[k] = 0.0

    if not ml_models and not pb_models:
        logger.error("Aucun modèle n'a pu être entraîné")
        stage_print("❌ Aucun modèle n'a pu être entraîné")
        return None, None, None, {}

    filtered_weights = _filter_valid_weights(weights)
    valid_model_keys = [k for k in filtered_weights.keys() if not k.startswith("__")]
    if not valid_model_keys:
        fallback_keys = _select_fallback_uniform_models(ml_models, pb_models, wf_ll_by_key)
        if not fallback_keys:
            logger.error("Aucun modèle valide pour l'ensemble")
            stage_print("❌ Aucun modèle valide disponible pour l'ensemble")
            return None, None, None, {}
        uniform = 1.0 / len(fallback_keys)
        for k in fallback_keys:
            filtered_weights[k] = uniform
    else:
        filtered_weights = _apply_quality_filter_on_weights(filtered_weights, wf_ll_by_key)

    weights = _normalize_weights(filtered_weights, ENSEMBLE_MIN_WEIGHT, ENSEMBLE_MAX_WEIGHT)

    temperature_validated = False
    if TEMPERATURE_SCALING_ENABLED:
        try:
            calib_blend = _compute_ensemble_probas(
                ml_models, pb_models, Xs_calib, home_keys_calib, away_keys_calib,
                weights, X_calib_raw,
            )
            if calib_blend is not None:
                temp, temp_ok = _compute_optimal_temperature(calib_blend, y_calib)
                weights["__temperature__"] = float(temp)
                temperature_validated = bool(temp_ok)
            else:
                weights["__temperature__"] = float(TEMPERATURE_SCALING_DEFAULT)
        except Exception:
            weights["__temperature__"] = float(TEMPERATURE_SCALING_DEFAULT)

    weights["__temperature_validated__"] = bool(temperature_validated)
    weights["__ll_source_map__"] = ll_source_by_key
    weights["__calibration_status_map__"] = calibration_status_by_key
    weights["__walk_forward_ll_map__"] = wf_ll_by_key
    weights["__calibration_ll_map__"] = calib_ll_by_key
    weights["__walk_forward_fold_coverage_map__"] = wf_fold_coverage_by_key
    weights["__strict_competition__"] = bool(_STRICT_COMPETITION_DEFAULT)
    weights["__feature_names__"] = list(FEATURE_NAMES)
    weights["__feature_count__"] = int(EXPECTED_FEATURE_COUNT)

    if calib_ll_by_key:
        try:
            finite_lls = [
                float(v) for v in calib_ll_by_key.values()
                if np.isfinite(float(v)) and float(v) > 0
            ]
            if finite_lls:
                weights["__calibration_logloss__"] = float(np.median(finite_lls))
        except Exception:
            pass

    final_temperature = float(weights.get("__temperature__", 1.0))
    test_metrics = _evaluate_ensemble_on_set(
        ml_models, pb_models, Xs_test, y_test, home_keys_test, away_keys_test,
        weights, final_temperature, X_test_raw,
    )
    if test_metrics is not None:
        ll_test, acc_test = test_metrics
        stage_print(
            f"📊 TEST FINAL : LogLoss {ll_test:.4f} | "
            f"Accuracy {acc_test * 100:.1f}%"
        )
        weights["__test_logloss__"] = float(ll_test)
        weights["__test_accuracy__"] = float(acc_test)

    return ml_models, pb_models, pipeline, weights
