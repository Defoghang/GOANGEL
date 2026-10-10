import os
import sys
import pickle
import logging
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    import torch
except ImportError:
    torch = None

from features import (
    process_matches,
    process_future_matches,
    train_models,
    build_feature_vector,
    build_tracking_state,
    TeamHistoryTracker,
    EloSystem,
    EXPECTED_FEATURE_COUNT,
    FEATURE_NAMES,
    _extract_model_probabilities,
)
from api import (
    get_all_matches_for_date,
    get_historical_matches,
    get_event_odds,
    CheckpointManager,
)
from config import (
    CACHE_DIR,
    HISTORICAL_FILE,
    MODELS_FILE,
    SCALER_FILE,
    WEIGHTS_FILE,
    CHECKPOINT_DIR,
    CHECKPOINT_STATE_FILE,
    CHECKPOINT_RAW_FILE,
    CHECKPOINT_META_FILE,
    CHECKPOINT_SAVE_INTERVAL_SECONDS,
    CHECKPOINT_PUSH_INTERVAL_SECONDS,
    GOANGEL_GITHUB_PUSH,
    GOANGEL_GITHUB_REPO,
    GOANGEL_GITHUB_USER,
    GOANGEL_GITHUB_BRANCH,
    GH_PAT,
    CACHE_NEVER_EXPIRES,
    CACHE_TTL_HOURS,
    CACHE_ATOMIC_WRITE,
    MIN_MATCHES_FOR_TRAINING,
    DIXON_COLES_RHO,
    MAX_GOALS_GRID,
    TEMPERATURE_SCALING_ENABLED,
    TEMPERATURE_SCALING_DEFAULT,
    TEMPERATURE_SCALING_MIN,
    TEMPERATURE_SCALING_MAX,
    REALISTIC_HOME_LAMBDA_BY_GROUP,
    REALISTIC_AWAY_LAMBDA_BY_GROUP,
    get_competition_group,
    stage_print,
)

logger = logging.getLogger("goangel.predict")

_SCORE_GRID_SIZE: int = MAX_GOALS_GRID
_HT_SCORE_GRID_SIZE: int = 8
_TOP_EXACT_SCORES: int = 5
_TOP_SCORE_BANDS: int = 3

_PLAYABLE_STATUSES: frozenset = frozenset({
    "SCHEDULED",
    "TIMED",
    "NOT_STARTED",
    "NS",
    "UPCOMING",
    "TBD",
})

_REQUIRED_MATCH_COLUMNS: Tuple[str, ...] = (
    "home_raw",
    "away_raw",
    "home_key",
    "away_key",
)

_XG_SOURCE_TAG: str = "ensemble_model_estimate"

_FT_EDGE_MIN: float = 0.05
_HT_EDGE_MIN: float = 0.08

_THRESHOLD_RELIABLE: float = 0.55
_THRESHOLD_STRONG: float = 0.65
_THRESHOLD_VERY_STRONG: float = 0.75

_WINNER_MARGIN_MIN: float = 0.05
_EXACT_SCORE_MIN_PROBA: float = 0.05

_MIN_ACTIVE_ML_MODELS: int = 3
_MIN_TOTAL_ACTIVE_MODELS: int = 4

_MAX_ACCEPTED_ENSEMBLE_STD: float = 0.25

_PROBA_SUM_TOLERANCE: float = 0.005
_EPS: float = 1e-9
_EPS_WEIGHT: float = 1e-6

_MIN_EV_THRESHOLD: float = 0.03
_KELLY_FRACTION: float = 0.25
_MAX_KELLY_STAKE: float = 0.05

_ODDS_CACHE: Dict[str, Dict[str, Any]] = {}
_ODDS_CACHE_TIMESTAMP: Dict[str, float] = {}
_ODDS_CACHE_TTL: float = 1800.0

_HT_LAMBDA_RATIO_MIN: float = 0.30
_HT_LAMBDA_RATIO_MAX: float = 0.60
_HT_LAMBDA_RATIO_DEFAULT: float = 0.45

_NAN: float = float("nan")

_LAMBDA_MIN: float = 0.05
_LAMBDA_MAX: float = 6.0


def _cache_is_fresh(path: Path) -> bool:
    if CACHE_NEVER_EXPIRES:
        return path.exists()
    if not path.exists():
        return False
    try:
        mtime = dt.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    age_hours = (dt.now(timezone.utc) - mtime).total_seconds() / 3600.0
    return age_hours < CACHE_TTL_HOURS


def _atomic_write_pickle(path: Path, obj: Any) -> None:
    if not CACHE_ATOMIC_WRITE:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.error("Échec écriture atomique %s : %s", path, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        text = str(value).strip()
    except Exception:
        return default
    if not text or text.lower() in ("nan", "none", "null", "nat"):
        return default
    return text


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_probability(value: Any, fallback: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(fallback)
    if not np.isfinite(v):
        v = float(fallback)
    return float(np.clip(v, 0.0, 1.0))


def _safe_xg(value: Any, fallback: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(fallback)
    if not np.isfinite(v) or v <= 0.0:
        v = float(fallback)
    return float(np.clip(v, _LAMBDA_MIN, _LAMBDA_MAX))


def _safe_goals_mean(df: pd.DataFrame, column: str) -> Optional[float]:
    if df is None or df.empty or column not in df.columns:
        return None
    try:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
    except Exception:
        return None
    series = series[np.isfinite(series)]
    if series.empty:
        return None
    m = float(series.mean())
    if not np.isfinite(m) or m <= 0.0:
        return None
    return m


def _validate_hda(proba: Any) -> Optional[np.ndarray]:
    if proba is None:
        return None
    try:
        arr = np.asarray(proba, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim == 2 and arr.shape[0] >= 1:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] != 3:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if np.any(arr < 0.0):
        return None
    if np.any(arr > 1.0 + 1e-6):
        return None
    total = float(arr.sum())
    if total <= 0.0:
        return None
    return arr / total


def _compute_expected_value(model_prob: float, bookmaker_odds: float) -> float:
    if bookmaker_odds <= 1.0:
        return -1.0
    return float(model_prob * bookmaker_odds - 1.0)


def _compute_kelly_stake(
    model_prob: float,
    bookmaker_odds: float,
    fraction: float = _KELLY_FRACTION,
) -> float:
    if bookmaker_odds <= 1.0:
        return 0.0
    b = bookmaker_odds - 1.0
    p = model_prob
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0.0
    stake = kelly * fraction
    return float(np.clip(stake, 0.0, _MAX_KELLY_STAKE))


def _fetch_bookmaker_odds(
    competition_code: Optional[str],
    home_team: str,
    away_team: str,
    event_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(event_id, int) or event_id <= 0:
        return None
    cache_key = f"bzz:{event_id}"
    now_ts = dt.now(timezone.utc).timestamp()
    if cache_key in _ODDS_CACHE:
        if now_ts - _ODDS_CACHE_TIMESTAMP.get(cache_key, 0.0) < _ODDS_CACHE_TTL:
            return _ODDS_CACHE[cache_key]
    try:
        raw = get_event_odds(event_id)
    except Exception as exc:
        logger.debug("Bzzoiro odds échec pour %s : %s", event_id, exc)
        return None
    if not isinstance(raw, dict):
        return None
    od = raw.get("odds")
    if not isinstance(od, dict) or not od:
        return None

    def _pick(*keys: str) -> Optional[float]:
        for k in keys:
            v = od.get(k)
            if isinstance(v, dict):
                v = v.get("value") or v.get("odds") or v.get("price")
            fv = _safe_float(v)
            if fv is not None and fv > 1.0:
                return fv
        return None

    h2h = {
        "odds_home": _pick("home_win", "1", "home"),
        "odds_draw": _pick("draw", "X", "x"),
        "odds_away": _pick("away_win", "2", "away"),
    }
    if not all(v and v > 1.0 for v in h2h.values()):
        return None
    totals: Dict[float, Dict[str, float]] = {}
    for over_key, under_key, line in (
        ("over_15_goals", "under_15_goals", 1.5),
        ("over_25_goals", "under_25_goals", 2.5),
        ("over_35_goals", "under_35_goals", 3.5),
    ):
        o = _pick(over_key)
        u = _pick(under_key)
        if o and u:
            totals[line] = {"over_odds": o, "under_odds": u}
    btts = None
    by = _pick("btts_yes")
    bn = _pick("btts_no")
    if by and bn:
        btts = {"yes_odds": by, "no_odds": bn}
    dnb = None
    hw = _pick("home_win_or_draw", "1x")
    aw = _pick("away_win_or_draw", "x2")
    if hw and aw:
        dnb = {"home_odds": hw, "away_odds": aw}
    result = {
        "source": "bzzoiro",
        "h2h": h2h,
        "totals": totals if totals else None,
        "btts": btts,
        "draw_no_bet": dnb,
    }
    _ODDS_CACHE[cache_key] = result
    _ODDS_CACHE_TIMESTAMP[cache_key] = now_ts
    return result


def _load_full_history(checkpoint_mgr: Optional[CheckpointManager] = None) -> pd.DataFrame:
    stage_print("♻️ RÉCUPÉRATION DE L'HISTORIQUE COMPLET DES MATCHS")
    logger.info("Récupération de l'historique complet via Bzzoiro...")
    try:
        historical = get_historical_matches(checkpoint_manager=checkpoint_mgr)
    except Exception as exc:
        logger.error("Erreur lors de la récupération historique : %s", exc)
        return pd.DataFrame()
    if not historical:
        logger.error("Aucun match historique récupéré")
        stage_print("🏆 TOTAL MATCHS HISTORIQUES RÉCUPÉRÉS : 0")
        return pd.DataFrame()
    df_hist = process_matches(
        historical,
        allowed_statuses=("FINISHED",),
        include_awarded=False,
    )
    if df_hist.empty:
        logger.error("Le traitement a produit un DataFrame vide")
        return df_hist
    df_past = df_hist.dropna(subset=["goals_home", "goals_away"]).copy()
    if "match_id" in df_past.columns:
        df_past = df_past.drop_duplicates(subset=["match_id"], keep="first")
    if "date_parsed" in df_past.columns:
        df_past = (
            df_past.dropna(subset=["date_parsed"])
            .sort_values("date_parsed", kind="mergesort")
            .reset_index(drop=True)
        )
    logger.info(
        "%d matchs terminés valides retenus sur %d matchs bruts",
        len(df_past), len(df_hist),
    )
    return df_past


def _validate_cached_payload(
    df_past: Any,
    ml_models: Any,
    pb_models: Any,
    pipeline: Any,
    weights: Any,
) -> bool:
    if not isinstance(df_past, pd.DataFrame) or df_past.empty:
        return False
    if "home_key" not in df_past.columns or "away_key" not in df_past.columns:
        return False
    if pipeline is None or not hasattr(pipeline, "transform"):
        return False
    n_features_in = getattr(pipeline, "n_features_in_", None)
    if n_features_in is None:
        try:
            scaler_step = pipeline.named_steps.get("scaler")
            if scaler_step is not None:
                n_features_in = getattr(scaler_step, "n_features_in_", None)
        except Exception:
            n_features_in = None
    if n_features_in is not None:
        try:
            if int(n_features_in) != EXPECTED_FEATURE_COUNT:
                logger.warning("Pipeline obsolète : %d features", int(n_features_in))
                return False
        except Exception:
            return False
    if ml_models is not None and not isinstance(ml_models, dict):
        return False
    if pb_models is not None and not isinstance(pb_models, list):
        return False
    if weights is not None and not isinstance(weights, dict):
        return False
    n_ml = len(ml_models) if isinstance(ml_models, dict) else 0
    n_pb = len(pb_models) if isinstance(pb_models, list) else 0
    if n_ml == 0 and n_pb == 0:
        logger.warning("Cache : aucun modèle utilisable")
        return False
    if isinstance(weights, dict):
        cached_feature_count = weights.get("__feature_count__")
        if cached_feature_count is not None:
            try:
                if int(cached_feature_count) != EXPECTED_FEATURE_COUNT:
                    logger.warning("Cache : features %s != %d", cached_feature_count, EXPECTED_FEATURE_COUNT)
                    return False
            except (TypeError, ValueError):
                return False
        cached_features = weights.get("__feature_names__")
        if isinstance(cached_features, list):
            try:
                if list(cached_features) != list(FEATURE_NAMES):
                    logger.warning("Cache : liste features différente")
                    return False
            except Exception:
                pass
        cached_schema = weights.get("__score_grid_size__")
        if cached_schema is not None:
            try:
                if int(cached_schema) != _SCORE_GRID_SIZE:
                    logger.warning("Cache : score_grid différent")
                    return False
            except (TypeError, ValueError):
                pass
        cached_version = weights.get("__cache_version__")
        if cached_version is not None:
            try:
                if str(cached_version) != "v10":
                    logger.warning("Cache obsolète (%s)", cached_version)
                    return False
            except Exception:
                return False
        cached_last_date = weights.get("__last_match_date__")
        if cached_last_date is not None and "date_parsed" in df_past.columns:
            try:
                last_series = pd.to_datetime(df_past["date_parsed"], errors="coerce").dropna()
                if not last_series.empty:
                    actual_last = last_series.max()
                    cached_dt = pd.to_datetime(cached_last_date, errors="coerce")
                    if pd.notna(cached_dt) and actual_last < cached_dt:
                        logger.warning("Cache antérieur")
                        return False
            except Exception:
                pass
    return True


def load_or_build_data_and_models(
    force_refresh: bool = False,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[List[Tuple[str, Any]]],
    Any,
    pd.DataFrame,
    Dict[str, float],
]:
    cache_files = [HISTORICAL_FILE, MODELS_FILE, SCALER_FILE, WEIGHTS_FILE]
    files_present = all(f.exists() for f in cache_files)
    files_fresh = files_present and all(_cache_is_fresh(f) for f in cache_files)
    cache_available = not force_refresh and files_fresh

    if cache_available:
        try:
            with open(HISTORICAL_FILE, "rb") as f:
                df_past = pickle.load(f)
            with open(MODELS_FILE, "rb") as f:
                ml_models, pb_models = pickle.load(f)
            with open(SCALER_FILE, "rb") as f:
                pipeline = pickle.load(f)
            with open(WEIGHTS_FILE, "rb") as f:
                weights = pickle.load(f)
            if _validate_cached_payload(df_past, ml_models, pb_models, pipeline, weights):
                n_ml = len(ml_models) if isinstance(ml_models, dict) else 0
                n_pb = len(pb_models) if isinstance(pb_models, list) else 0
                logger.info("Cache chargé : %d matchs, %d ML, %d PB", len(df_past), n_ml, n_pb)
                stage_print(f"✅ Historique chargé : {len(df_past)} matchs")
                stage_print(f"✅ Modèles actifs : {n_ml} ML + {n_pb} Penaltyblog")
                return ml_models, pb_models, pipeline, df_past, weights
            logger.warning("Cache invalide. Reconstruction forcée.")
        except Exception as exc:
            logger.error("Erreur chargement cache : %s. Reconstruction.", exc)

    stage_print("🛠️ CONSTRUCTION INITIALE DES DONNÉES & MODÈLES")
    logger.info("Construction initiale...")

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR,
        save_interval=CHECKPOINT_SAVE_INTERVAL_SECONDS,
        push_interval=CHECKPOINT_PUSH_INTERVAL_SECONDS,
        github_enabled=GOANGEL_GITHUB_PUSH,
        github_repo=GOANGEL_GITHUB_REPO,
        github_user=GOANGEL_GITHUB_USER,
        github_token=GH_PAT,
        github_branch=GOANGEL_GITHUB_BRANCH,
    )

    df_past = _load_full_history(checkpoint_mgr=ckpt_mgr)
    ml_models: Optional[Dict[str, Any]] = None
    pb_models: Optional[List[Tuple[str, Any]]] = None
    pipeline: Any = None
    weights: Dict[str, float] = {}

    if not df_past.empty and len(df_past) >= MIN_MATCHES_FOR_TRAINING:
        try:
            ml_models, pb_models, pipeline, weights = train_models(df_past)
        except Exception:
            logger.exception("Échec de l'entraînement")
            ml_models, pb_models, pipeline, weights = None, None, None, {}
    else:
        logger.error(
            "Historique insuffisant : %d matchs",
            len(df_past) if df_past is not None else 0,
        )
        stage_print("❌ Historique insuffisant pour entraîner les modèles")

    if isinstance(weights, dict):
        weights["__feature_count__"] = EXPECTED_FEATURE_COUNT
        weights["__feature_names__"] = list(FEATURE_NAMES)
        weights["__score_grid_size__"] = _SCORE_GRID_SIZE
        weights["__data_source__"] = "bzzoiro"
        weights["__cache_version__"] = "v10"
        if not df_past.empty and "date_parsed" in df_past.columns:
            try:
                last_series = pd.to_datetime(df_past["date_parsed"], errors="coerce").dropna()
                if not last_series.empty:
                    weights["__last_match_date__"] = str(last_series.max())
            except Exception:
                pass

    stage_print("💾 FIN DE CHARGEMENT DES MODÈLES & DE L'ENTRAÎNEMENT")

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_pickle(HISTORICAL_FILE, df_past)
        _atomic_write_pickle(MODELS_FILE, (ml_models, pb_models))
        _atomic_write_pickle(SCALER_FILE, pipeline)
        _atomic_write_pickle(WEIGHTS_FILE, weights)
        logger.info("Cache sauvegardé.")
        stage_print(
            f"✅ MODÈLES ET DONNÉES SAUVEGARDÉS DANS ({CACHE_DIR}) "
            f"POUR ÉVITER LE RÉENTRAÎNEMENT COMPLET À CHAQUE LANCEMENT."
        )
    except OSError as exc:
        logger.error("Erreur sauvegarde cache : %s", exc)
        stage_print("❌ Erreur lors de la sauvegarde du cache")

    try:
        ckpt_mgr.finalize()
    except Exception:
        pass

    return ml_models, pb_models, pipeline, df_past, weights


def _dixon_coles_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    if rho == 0.0:
        return 1.0
    if i == 0 and j == 0:
        tau = 1.0 - lh * la * rho
    elif i == 0 and j == 1:
        tau = 1.0 + lh * rho
    elif i == 1 and j == 0:
        tau = 1.0 + la * rho
    elif i == 1 and j == 1:
        tau = 1.0 - rho
    else:
        tau = 1.0
    if tau < 0.01:
        tau = 0.01
    return tau


def _poisson_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = DIXON_COLES_RHO,
    grid_size: int = _SCORE_GRID_SIZE,
) -> np.ndarray:
    lh = float(np.clip(lambda_home, _LAMBDA_MIN, _LAMBDA_MAX))
    la = float(np.clip(lambda_away, _LAMBDA_MIN, _LAMBDA_MAX))
    grid = max(5, int(grid_size))
    mat = np.zeros((grid, grid), dtype=np.float64)
    for i in range(grid):
        pi = poisson.pmf(i, lh)
        for j in range(grid):
            pj = poisson.pmf(j, la)
            tau = _dixon_coles_tau(i, j, lh, la, rho)
            mat[i, j] = pi * pj * tau
    total = mat.sum()
    if total > 0:
        mat = mat / total
    else:
        mat[0, 0] = 1.0
    return mat


def _markets_from_matrix(mat: np.ndarray) -> Dict[str, Any]:
    n = mat.shape[0]
    prob_home = float(np.sum(np.tril(mat, k=-1)))
    prob_draw = float(np.trace(mat))
    prob_away = float(np.sum(np.triu(mat, k=1)))
    total = prob_home + prob_draw + prob_away
    if total > 0:
        prob_home /= total
        prob_draw /= total
        prob_away /= total
    over15 = float(sum(mat[i, j] for i in range(n) for j in range(n) if i + j > 1))
    over25 = float(sum(mat[i, j] for i in range(n) for j in range(n) if i + j > 2))
    over35 = float(sum(mat[i, j] for i in range(n) for j in range(n) if i + j > 3))
    btts_yes = float(sum(mat[i, j] for i in range(n) for j in range(n) if i > 0 and j > 0))
    return {
        "home_draw_away": [prob_home, prob_draw, prob_away],
        "over15": float(np.clip(over15, 0.0, 1.0)),
        "over25": float(np.clip(over25, 0.0, 1.0)),
        "over35": float(np.clip(over35, 0.0, 1.0)),
        "btts_yes": float(np.clip(btts_yes, 0.0, 1.0)),
    }


def _ht_poisson_matrix(
    lambda_home_ht: float,
    lambda_away_ht: float,
    grid_size: int = _HT_SCORE_GRID_SIZE,
) -> np.ndarray:
    lh = float(np.clip(lambda_home_ht, 0.02, 4.0))
    la = float(np.clip(lambda_away_ht, 0.02, 4.0))
    grid = max(4, int(grid_size))
    mat = np.zeros((grid, grid), dtype=np.float64)
    for i in range(grid):
        pi = poisson.pmf(i, lh)
        for j in range(grid):
            pj = poisson.pmf(j, la)
            mat[i, j] = pi * pj
    total = mat.sum()
    if total > 0:
        mat = mat / total
    else:
        mat[0, 0] = 1.0
    return mat


def _ht_markets_from_matrix(mat: np.ndarray) -> Dict[str, float]:
    n = mat.shape[0]
    prob_home = float(np.sum(np.tril(mat, k=-1)))
    prob_draw = float(np.trace(mat))
    prob_away = float(np.sum(np.triu(mat, k=1)))
    total = prob_home + prob_draw + prob_away
    if total > 0:
        prob_home /= total
        prob_draw /= total
        prob_away /= total
    over05 = 1.0 - float(mat[0, 0])
    over15 = float(sum(mat[i, j] for i in range(n) for j in range(n) if i + j > 1))
    over25 = float(sum(mat[i, j] for i in range(n) for j in range(n) if i + j > 2))
    return {
        "ht_home_draw_away": [prob_home, prob_draw, prob_away],
        "ht_over05": float(np.clip(over05, 0.0, 1.0)),
        "ht_over15": float(np.clip(over15, 0.0, 1.0)),
        "ht_over25": float(np.clip(over25, 0.0, 1.0)),
    }


def _compute_ht_markets(
    lambda_home: float,
    lambda_away: float,
    ht_ratio: float = _HT_LAMBDA_RATIO_DEFAULT,
) -> Dict[str, Any]:
    lh = float(np.clip(lambda_home, _LAMBDA_MIN, _LAMBDA_MAX))
    la = float(np.clip(lambda_away, _LAMBDA_MIN, _LAMBDA_MAX))
    ratio = float(np.clip(ht_ratio, _HT_LAMBDA_RATIO_MIN, _HT_LAMBDA_RATIO_MAX))
    lh_ht = float(np.clip(lh * ratio, 0.02, 4.0))
    la_ht = float(np.clip(la * ratio, 0.02, 4.0))
    mat = _ht_poisson_matrix(lh_ht, la_ht)
    markets = _ht_markets_from_matrix(mat)
    markets["ht_home_xg"] = lh_ht
    markets["ht_away_xg"] = la_ht
    return markets


def _estimate_ht_lambda_ratio(df_past: pd.DataFrame) -> float:
    if df_past is None or df_past.empty:
        return _HT_LAMBDA_RATIO_DEFAULT
    required = {"ht_home", "ht_away", "goals_home", "goals_away"}
    if not required.issubset(df_past.columns):
        return _HT_LAMBDA_RATIO_DEFAULT
    try:
        valid = df_past.dropna(subset=list(required))
        if len(valid) < 50:
            return _HT_LAMBDA_RATIO_DEFAULT
        ht_total = float(valid["ht_home"].sum() + valid["ht_away"].sum())
        ft_total = float(valid["goals_home"].sum() + valid["goals_away"].sum())
        if ft_total <= 0:
            return _HT_LAMBDA_RATIO_DEFAULT
        ratio = ht_total / ft_total
        if not np.isfinite(ratio) or ratio <= 0:
            return _HT_LAMBDA_RATIO_DEFAULT
        return float(np.clip(ratio, _HT_LAMBDA_RATIO_MIN, _HT_LAMBDA_RATIO_MAX))
    except Exception:
        return _HT_LAMBDA_RATIO_DEFAULT


def _top_exact_scores(
    mat: np.ndarray,
    n: int = _TOP_EXACT_SCORES,
) -> List[Tuple[str, float]]:
    scores: List[Tuple[float, int, int]] = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            scores.append((float(mat[i, j]), i, j))
    scores.sort(reverse=True)
    top = scores[: max(1, n)]
    return [(f"{i}-{j}", prob) for prob, i, j in top]


def _score_bands(mat: np.ndarray) -> List[Tuple[str, float]]:
    bands = [
        ("0-0", 0, 0),
        ("1-0", 1, 0),
        ("0-1", 0, 1),
        ("1-1", 1, 1),
        ("2-0", 2, 0),
        ("0-2", 0, 2),
        ("2-1", 2, 1),
        ("1-2", 1, 2),
        ("2-2", 2, 2),
    ]
    result: List[Tuple[str, float]] = []
    for label, i, j in bands:
        if i < mat.shape[0] and j < mat.shape[1]:
            result.append((label, float(mat[i, j])))
    result.sort(key=lambda x: -x[1])
    return result[:_TOP_SCORE_BANDS]


def _resolve_prior_lambdas(
    tracker: TeamHistoryTracker,
    avg_h: Optional[float],
    avg_a: Optional[float],
) -> Optional[Tuple[float, float]]:
    prior_home = tracker.default_gf
    prior_away = tracker.default_ga
    if not np.isfinite(prior_home) or prior_home <= 0:
        if avg_h is not None and np.isfinite(avg_h) and avg_h > 0:
            prior_home = float(avg_h)
    if not np.isfinite(prior_away) or prior_away <= 0:
        if avg_a is not None and np.isfinite(avg_a) and avg_a > 0:
            prior_away = float(avg_a)
    if (
        not np.isfinite(prior_home) or prior_home <= 0
        or not np.isfinite(prior_away) or prior_away <= 0
    ):
        return None
    return float(prior_home), float(prior_away)


def _estimate_lambdas_from_history(
    home_key: str,
    away_key: str,
    tracker: TeamHistoryTracker,
    elo_system: EloSystem,
    competition_code: Optional[str],
    avg_h: Optional[float],
    avg_a: Optional[float],
) -> Optional[Tuple[float, float]]:
    priors = _resolve_prior_lambdas(tracker, avg_h, avg_a)
    if priors is None:
        return None
    prior_home, prior_away = priors
    group = get_competition_group(competition_code)
    group_home = REALISTIC_HOME_LAMBDA_BY_GROUP.get(group, prior_home)
    group_away = REALISTIC_AWAY_LAMBDA_BY_GROUP.get(group, prior_away)
    if not np.isfinite(group_home) or group_home <= 0:
        group_home = prior_home
    if not np.isfinite(group_away) or group_away <= 0:
        group_away = prior_away
    prior_home = 0.5 * float(prior_home) + 0.5 * float(group_home)
    prior_away = 0.5 * float(prior_away) + 0.5 * float(group_away)
    league_avg = max(0.5, (prior_home + prior_away) / 2.0)
    home_hist = tracker.snapshot(home_key, competition_code)
    away_hist = tracker.snapshot(away_key, competition_code)

    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr.mean())

    home_gf = _mean(home_hist["gf"])
    home_ga = _mean(home_hist["ga"])
    away_gf = _mean(away_hist["gf"])
    away_ga = _mean(away_hist["ga"])
    home_home_gf = _mean(home_hist["home_gf"])
    home_home_ga = _mean(home_hist["home_ga"])
    away_away_gf = _mean(away_hist["away_gf"])
    away_away_ga = _mean(away_hist["away_ga"])

    home_attack_values = [v for v in (home_gf, home_home_gf) if v is not None]
    home_attack = float(np.mean(home_attack_values)) if home_attack_values else league_avg
    home_defense_values = [v for v in (home_ga, home_home_ga) if v is not None]
    home_defense = float(np.mean(home_defense_values)) if home_defense_values else league_avg
    away_attack_values = [v for v in (away_gf, away_away_gf) if v is not None]
    away_attack = float(np.mean(away_attack_values)) if away_attack_values else league_avg
    away_defense_values = [v for v in (away_ga, away_away_ga) if v is not None]
    away_defense = float(np.mean(away_defense_values)) if away_defense_values else league_avg

    home_attack_ratio = float(np.clip(home_attack / league_avg if league_avg > 0 else 1.0, 0.3, 3.0))
    home_defense_ratio = float(np.clip(home_defense / league_avg if league_avg > 0 else 1.0, 0.3, 3.0))
    away_attack_ratio = float(np.clip(away_attack / league_avg if league_avg > 0 else 1.0, 0.3, 3.0))
    away_defense_ratio = float(np.clip(away_defense / league_avg if league_avg > 0 else 1.0, 0.3, 3.0))

    base_home = prior_home * home_attack_ratio * away_defense_ratio
    base_away = prior_away * away_attack_ratio * home_defense_ratio
    base_home = 0.6 * base_home + 0.4 * prior_home
    base_away = 0.6 * base_away + 0.4 * prior_away

    home_elo = elo_system.get_rating(home_key, competition_code)
    away_elo = elo_system.get_rating(away_key, competition_code)
    elo_diff = home_elo - away_elo
    elo_scale = float(np.clip(elo_diff / 400.0, -1.5, 1.5))
    elo_factor_home = float(np.exp(0.20 * elo_scale))
    elo_factor_away = float(np.exp(-0.20 * elo_scale))

    lambda_home = base_home * elo_factor_home
    lambda_away = base_away * elo_factor_away

    matches_home = home_hist["matches_played"]
    matches_away = away_hist["matches_played"]
    reliability_home = float(np.clip(matches_home / 20.0, 0.0, 1.0))
    reliability_away = float(np.clip(matches_away / 20.0, 0.0, 1.0))
    lambda_home = reliability_home * lambda_home + (1.0 - reliability_home) * prior_home
    lambda_away = reliability_away * lambda_away + (1.0 - reliability_away) * prior_away

    lambda_home = _safe_xg(lambda_home, prior_home)
    lambda_away = _safe_xg(lambda_away, prior_away)
    return lambda_home, lambda_away


def _sanitize_weights(weights: List[float]) -> np.ndarray:
    if not weights:
        return np.array([], dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.where(arr < 0.0, 0.0, arr)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / float(arr.size)
    return arr / total


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    if temperature is None or temperature <= 0:
        return proba
    if abs(temperature - 1.0) < 1e-6:
        return proba
    clipped = np.clip(proba, _EPS, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max()
    exp_logits = np.exp(logits)
    total = exp_logits.sum()
    if total <= 0:
        return proba
    return exp_logits / total


def _finalize_probabilities(proba: np.ndarray) -> np.ndarray:
    arr = np.array(proba, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    arr = np.clip(arr, 0.0, 1.0)
    total = arr.sum()
    if total <= 0:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return arr / total


def _check_model_classes(model: Any, name: str) -> bool:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return True
    try:
        classes_list = list(classes)
    except Exception:
        return True
    if classes_list != [0, 1, 2]:
        logger.error("Modèle %s : ordre classes invalide %s", name, classes_list)
        return False
    return True


def _model_predict_proba(
    model: Any,
    name: str,
    feat_scaled: np.ndarray,
    feat_raw: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if model is None:
        return None
    if not _check_model_classes(model, name):
        return None
    if name in ("XGB", "LGBM", "CatBoost") and feat_raw is not None:
        X_input = feat_raw
    else:
        X_input = feat_scaled
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        try:
            proba = predict_proba(X_input)
            validated = _validate_hda(proba)
            if validated is not None:
                return validated
        except Exception as exc:
            logger.debug("predict_proba échoué pour %s : %s", name, exc)
    if torch is not None and isinstance(model, torch.nn.Module):
        try:
            model.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(feat_scaled, dtype=torch.float32)
                logits = model(tensor)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                validated = _validate_hda(proba)
                if validated is not None:
                    return validated
        except Exception as exc:
            logger.debug("forward PyTorch échoué pour %s : %s", name, exc)
    return None


def _resolve_pb_weight(
    weights: Dict[str, float],
    index: int,
    class_name: str,
    n_pb: int,
) -> float:
    if n_pb <= 0 or not isinstance(weights, dict):
        return 0.0
    for key in (f"pb_{index}", class_name):
        raw = weights.get(key)
        if raw is None:
            continue
        try:
            w = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(w) and w >= 0.0:
            return w
    return 0.0


def _build_insufficient_output(reason: str) -> Dict[str, Any]:
    return {
        "home_draw_away": None,
        "matrix_home_draw_away": None,
        "over15": None,
        "over25": None,
        "over35": None,
        "btts_yes": None,
        "home_xg": None,
        "away_xg": None,
        "xg_source": _XG_SOURCE_TAG,
        "xg_is_official": False,
        "uncertainty": None,
        "temperature": 1.0,
        "fallback": True,
        "insufficient_data": True,
        "insufficient_reason": str(reason),
        "score_matrix": None,
        "matrix_markets": None,
        "model_count_ml": 0,
        "model_count_pb": 0,
        "model_count_total": 0,
        "bookmaker_odds": None,
        "calibration_validated": False,
        "top_exact_scores": [],
        "top_score_bands": [],
        "winner": None,
        "winner_label": None,
        "winner_probability": None,
        "winner_margin": None,
        "most_likely_score": None,
        "most_likely_score_proba": None,
    }


def _build_output_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    ht_ratio: float,
) -> Dict[str, Any]:
    lh = float(np.clip(lambda_home, _LAMBDA_MIN, _LAMBDA_MAX))
    la = float(np.clip(lambda_away, _LAMBDA_MIN, _LAMBDA_MAX))
    mat = _poisson_matrix(lh, la)
    markets = _markets_from_matrix(mat)
    base_proba = np.asarray(markets["home_draw_away"], dtype=np.float64)
    base_proba = _finalize_probabilities(base_proba)
    markets["home_draw_away"] = base_proba.tolist()
    markets["home_xg"] = float(lh)
    markets["away_xg"] = float(la)
    markets["xg_source"] = _XG_SOURCE_TAG
    markets["xg_is_official"] = False
    markets["uncertainty"] = None
    markets["temperature"] = 1.0
    markets["fallback"] = True
    markets["insufficient_data"] = False
    markets["score_matrix"] = mat
    markets["matrix_markets"] = markets["home_draw_away"]
    markets["model_count_ml"] = 0
    markets["model_count_pb"] = 0
    markets["model_count_total"] = 0
    markets["bookmaker_odds"] = None
    markets["calibration_validated"] = False

    top_scores = _top_exact_scores(mat, n=_TOP_EXACT_SCORES)
    markets["top_exact_scores"] = top_scores
    if top_scores:
        markets["most_likely_score"] = top_scores[0][0]
        markets["most_likely_score_proba"] = float(top_scores[0][1])
    else:
        markets["most_likely_score"] = None
        markets["most_likely_score_proba"] = None
    markets["top_score_bands"] = _score_bands(mat)

    winner_idx = int(np.argmax(base_proba))
    if winner_idx == 0:
        winner_label = "HOME"
    elif winner_idx == 1:
        winner_label = "DRAW"
    else:
        winner_label = "AWAY"
    sorted_proba = np.sort(base_proba)
    margin = float(sorted_proba[-1] - sorted_proba[-2]) if len(sorted_proba) >= 2 else 0.0
    markets["winner"] = winner_idx
    markets["winner_label"] = winner_label
    markets["winner_probability"] = float(base_proba[winner_idx])
    markets["winner_margin"] = margin

    ht_markets = _compute_ht_markets(lh, la, ht_ratio)
    markets.update(ht_markets)
    return markets


def _reconcile_hda_with_matrix(
    ensemble_hda: np.ndarray,
    matrix_hda: np.ndarray,
    blend_weight: float = 0.5,
) -> np.ndarray:
    ensemble_hda = _finalize_probabilities(ensemble_hda)
    matrix_hda = np.asarray(matrix_hda, dtype=np.float64)
    if matrix_hda.shape != (3,) or not np.all(np.isfinite(matrix_hda)):
        return ensemble_hda
    matrix_hda = _finalize_probabilities(matrix_hda)
    blended = blend_weight * ensemble_hda + (1.0 - blend_weight) * matrix_hda
    return _finalize_probabilities(blended)


def _validate_odds_structure(odds: Dict[str, Any]) -> bool:
    if not isinstance(odds, dict):
        return False
    h2h = odds.get("h2h")
    if not isinstance(h2h, dict):
        return False
    for key in ("odds_home", "odds_draw", "odds_away"):
        v = _safe_float(h2h.get(key))
        if v is None or v <= 1.0:
            return False
    totals = odds.get("totals")
    if totals is not None:
        if not isinstance(totals, dict):
            return False
        for line, t in totals.items():
            if not isinstance(t, dict):
                return False
            if _safe_float(t.get("over_odds")) is None:
                return False
            if _safe_float(t.get("under_odds")) is None:
                return False
    btts = odds.get("btts")
    if btts is not None:
        if not isinstance(btts, dict):
            return False
        if _safe_float(btts.get("yes_odds")) is None:
            return False
        if _safe_float(btts.get("no_odds")) is None:
            return False
    dnb = odds.get("draw_no_bet")
    if dnb is not None:
        if not isinstance(dnb, dict):
            return False
        if _safe_float(dnb.get("home_odds")) is None:
            return False
        if _safe_float(dnb.get("away_odds")) is None:
            return False
    return True


def _build_market_candidates(
    proba: np.ndarray,
    home_name: str,
    away_name: str,
    over15: float,
    over25: float,
    btts_yes: float,
    ht_markets: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    p_home = float(proba[0])
    p_draw = float(proba[1])
    p_away = float(proba[2])
    dc_1x = p_home + p_draw
    dc_x2 = p_away + p_draw
    candidates.append({
        "name": f"Victoire {home_name}",
        "market_key": "home_win",
        "probability": p_home,
        "is_ht": False,
        "odds_key": ("h2h", "odds_home"),
    })
    candidates.append({
        "name": f"Victoire {away_name}",
        "market_key": "away_win",
        "probability": p_away,
        "is_ht": False,
        "odds_key": ("h2h", "odds_away"),
    })
    candidates.append({
        "name": "Match Nul",
        "market_key": "draw",
        "probability": p_draw,
        "is_ht": False,
        "odds_key": ("h2h", "odds_draw"),
    })
    candidates.append({
        "name": f"Double chance 1X ({home_name} ou Nul)",
        "market_key": "dc_1x",
        "probability": float(np.clip(dc_1x, 0.0, 1.0)),
        "is_ht": False,
        "odds_key": ("draw_no_bet", "home_odds"),
    })
    candidates.append({
        "name": f"Double chance X2 (Nul ou {away_name})",
        "market_key": "dc_x2",
        "probability": float(np.clip(dc_x2, 0.0, 1.0)),
        "is_ht": False,
        "odds_key": ("draw_no_bet", "away_odds"),
    })
    candidates.append({
        "name": "+2.5 buts",
        "market_key": "over_2_5",
        "probability": float(over25),
        "is_ht": False,
        "odds_key": ("totals", "over_odds", 2.5),
    })
    candidates.append({
        "name": "-2.5 buts",
        "market_key": "under_2_5",
        "probability": float(1.0 - over25),
        "is_ht": False,
        "odds_key": ("totals", "under_odds", 2.5),
    })
    candidates.append({
        "name": "+1.5 buts",
        "market_key": "over_1_5",
        "probability": float(over15),
        "is_ht": False,
        "odds_key": ("totals", "over_odds", 1.5),
    })
    candidates.append({
        "name": "-1.5 buts",
        "market_key": "under_1_5",
        "probability": float(1.0 - over15),
        "is_ht": False,
        "odds_key": ("totals", "under_odds", 1.5),
    })
    candidates.append({
        "name": "BTTS — Les deux marquent (Oui)",
        "market_key": "btts_yes",
        "probability": float(btts_yes),
        "is_ht": False,
        "odds_key": ("btts", "yes_odds"),
    })
    candidates.append({
        "name": "BTTS — Les deux marquent (Non)",
        "market_key": "btts_no",
        "probability": float(1.0 - btts_yes),
        "is_ht": False,
        "odds_key": ("btts", "no_odds"),
    })
    ht_hda = ht_markets.get("ht_home_draw_away")
    if isinstance(ht_hda, list) and len(ht_hda) == 3:
        ht_home = float(ht_hda[0])
        ht_draw = float(ht_hda[1])
        ht_away = float(ht_hda[2])
        candidates.append({
            "name": f"Victoire {home_name} (1ère MT)",
            "market_key": "ht_home_win",
            "probability": ht_home,
            "is_ht": True,
            "odds_key": None,
        })
        candidates.append({
            "name": f"Victoire {away_name} (1ère MT)",
            "market_key": "ht_away_win",
            "probability": ht_away,
            "is_ht": True,
            "odds_key": None,
        })
        candidates.append({
            "name": "Match Nul (1ère MT)",
            "market_key": "ht_draw",
            "probability": ht_draw,
            "is_ht": True,
            "odds_key": None,
        })
        ht_dc_1x = ht_home + ht_draw
        ht_dc_x2 = ht_away + ht_draw
        candidates.append({
            "name": f"Double chance 1X ({home_name} ou Nul) (1ère MT)",
            "market_key": "ht_dc_1x",
            "probability": float(np.clip(ht_dc_1x, 0.0, 1.0)),
            "is_ht": True,
            "odds_key": None,
        })
        candidates.append({
            "name": f"Double chance X2 (Nul ou {away_name}) (1ère MT)",
            "market_key": "ht_dc_x2",
            "probability": float(np.clip(ht_dc_x2, 0.0, 1.0)),
            "is_ht": True,
            "odds_key": None,
        })
        ht_over05 = _safe_probability(ht_markets.get("ht_over05", 0.5), 0.5)
        ht_over15 = _safe_probability(ht_markets.get("ht_over15", 0.5), 0.5)
        candidates.append({
            "name": "+0.5 buts (1ère MT)",
            "market_key": "ht_over_0_5",
            "probability": ht_over05,
            "is_ht": True,
            "odds_key": None,
        })
        candidates.append({
            "name": "-0.5 buts (1ère MT)",
            "market_key": "ht_under_0_5",
            "probability": float(1.0 - ht_over05),
            "is_ht": True,
            "odds_key": None,
        })
        candidates.append({
            "name": "+1.5 buts (1ère MT)",
            "market_key": "ht_over_1_5",
            "probability": ht_over15,
            "is_ht": True,
            "odds_key": None,
        })
        candidates.append({
            "name": "-1.5 buts (1ère MT)",
            "market_key": "ht_under_1_5",
            "probability": float(1.0 - ht_over15),
            "is_ht": True,
            "odds_key": None,
        })
    return candidates


def _resolve_market_odds(
    odds: Optional[Dict[str, Any]],
    odds_key: Optional[Tuple],
) -> Optional[float]:
    if odds is None or odds_key is None:
        return None
    if not isinstance(odds_key, tuple) or len(odds_key) < 2:
        return None
    market_name = odds_key[0]
    field_name = odds_key[1]
    if market_name == "h2h":
        h2h = odds.get("h2h")
        if not isinstance(h2h, dict):
            return None
        return _safe_float(h2h.get(field_name))
    if market_name == "draw_no_bet":
        dnb = odds.get("draw_no_bet")
        if not isinstance(dnb, dict):
            return None
        return _safe_float(dnb.get(field_name))
    if market_name == "btts":
        btts = odds.get("btts")
        if not isinstance(btts, dict):
            return None
        return _safe_float(btts.get(field_name))
    if market_name == "totals":
        totals = odds.get("totals")
        if not isinstance(totals, dict):
            return None
        if len(odds_key) < 3:
            return None
        line = odds_key[2]
        try:
            line_key = float(line)
        except (TypeError, ValueError):
            return None
        best_line = None
        best_diff = float("inf")
        for k in totals.keys():
            try:
                kf = float(k)
            except (TypeError, ValueError):
                continue
            diff = abs(kf - line_key)
            if diff < best_diff:
                best_diff = diff
                best_line = k
        if best_line is None or best_diff > 0.51:
            return None
        entry = totals.get(best_line)
        if not isinstance(entry, dict):
            return None
        return _safe_float(entry.get(field_name))
    return None


def _select_best_ev_bet(
    candidates: List[Dict[str, Any]],
    bookmaker_odds: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if bookmaker_odds is None:
        return None
    best: Optional[Dict[str, Any]] = None
    best_ev = _MIN_EV_THRESHOLD
    for cand in candidates:
        odds_value = _resolve_market_odds(bookmaker_odds, cand.get("odds_key"))
        if odds_value is None or odds_value <= 1.0:
            continue
        prob = float(cand.get("probability", 0.0))
        if not np.isfinite(prob) or prob <= 0.0 or prob >= 1.0:
            continue
        ev = _compute_expected_value(prob, odds_value)
        if ev > best_ev:
            best_ev = ev
            best = {
                "name": cand["name"],
                "probability": prob,
                "is_ht": bool(cand.get("is_ht", False)),
                "odds": odds_value,
                "ev": ev,
                "kelly": _compute_kelly_stake(prob, odds_value),
            }
    return best


def _select_best_no_odds_bet(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    ft_eligible: List[Dict[str, Any]] = []
    ht_eligible: List[Dict[str, Any]] = []
    for cand in candidates:
        prob = float(cand.get("probability", 0.0))
        if not np.isfinite(prob):
            continue
        is_ht = bool(cand.get("is_ht", False))
        edge = prob - 0.5
        if is_ht:
            edge *= 0.35
        if is_ht and edge >= _HT_EDGE_MIN:
            ht_eligible.append({**cand, "edge": edge})
        elif not is_ht and edge >= _FT_EDGE_MIN:
            ft_eligible.append({**cand, "edge": edge})
    if ft_eligible:
        ft_eligible.sort(key=lambda c: c["edge"], reverse=True)
        top = ft_eligible[0]
        return {
            "name": top["name"],
            "probability": float(top["probability"]),
            "is_ht": False,
            "odds": None,
            "ev": None,
            "kelly": None,
        }
    if ht_eligible:
        ht_eligible.sort(key=lambda c: c["edge"], reverse=True)
        top = ht_eligible[0]
        return {
            "name": top["name"],
            "probability": float(top["probability"]),
            "is_ht": True,
            "odds": None,
            "ev": None,
            "kelly": None,
        }
    return None


def _check_calibration_validity(
    weights: Dict[str, float],
    uncertainty: Optional[List[float]],
) -> bool:
    calibration_ll = weights.get("__calibration_logloss__")
    if calibration_ll is not None:
        try:
            ll = float(calibration_ll)
            if not np.isfinite(ll) or ll > 1.20:
                return False
        except (TypeError, ValueError):
            return False
    if uncertainty is not None:
        max_std = float(max(uncertainty))
        if max_std > _MAX_ACCEPTED_ENSEMBLE_STD:
            return False
    return True


def predict_match(
    home_key: str,
    away_key: str,
    home_raw: str,
    away_raw: str,
    competition_code: Optional[str],
    ml_models: Optional[Dict[str, Any]],
    pb_models: Optional[List[Tuple[str, Any]]],
    pipeline: Any,
    df_past: pd.DataFrame,
    tracker: TeamHistoryTracker,
    elo_system: EloSystem,
    weights: Dict[str, float],
    match_date: Optional[pd.Timestamp] = None,
    ht_ratio: float = _HT_LAMBDA_RATIO_DEFAULT,
    event_id: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(weights, dict):
        weights = {}
    if not isinstance(competition_code, str):
        competition_code = None
    elif competition_code:
        competition_code = competition_code.strip().upper()
    if not competition_code:
        competition_code = None

    avg_h = _safe_goals_mean(df_past, "goals_home")
    avg_a = _safe_goals_mean(df_past, "goals_away")

    history_lambdas = _estimate_lambdas_from_history(
        home_key, away_key, tracker, elo_system, competition_code, avg_h, avg_a,
    )
    if history_lambdas is None:
        return _build_insufficient_output(
            "Aucune donnée historique exploitable pour estimer un baseline lambda"
        )
    history_lambda_home, history_lambda_away = history_lambdas

    feat, _h2h_source = build_feature_vector(
        home_key, away_key, tracker, elo_system, match_date, competition_code,
    )
    feat_arr = np.asarray(feat, dtype=np.float64)
    if feat_arr.ndim == 0:
        feat_arr = feat_arr.reshape(1, -1)
    elif feat_arr.ndim == 1:
        feat_arr = feat_arr.reshape(1, -1)

    if feat_arr.ndim != 2 or feat_arr.shape[1] != EXPECTED_FEATURE_COUNT:
        logger.warning(
            "Feature shape inattendu : %s (attendu 1x%d)",
            feat_arr.shape, EXPECTED_FEATURE_COUNT,
        )
        return _build_output_from_lambdas(
            history_lambda_home, history_lambda_away, ht_ratio,
        )

    if pipeline is None or (not ml_models and not pb_models):
        return _build_output_from_lambdas(
            history_lambda_home, history_lambda_away, ht_ratio,
        )

    try:
        feat_scaled = pipeline.transform(feat_arr)
    except Exception as exc:
        logger.error("Erreur pipeline pour %s vs %s : %s", home_raw, away_raw, exc)
        return _build_output_from_lambdas(
            history_lambda_home, history_lambda_away, ht_ratio,
        )

    hda_probas: List[np.ndarray] = []
    hda_weights: List[float] = []
    over15_list: List[float] = []
    over15_weights: List[float] = []
    over25_list: List[float] = []
    over25_weights: List[float] = []
    btts_list: List[float] = []
    btts_weights: List[float] = []
    xg_home_list: List[float] = []
    xg_away_list: List[float] = []
    xg_weights: List[float] = []
    ml_model_count: int = 0
    pb_model_count: int = 0

    if pb_models:
        n_pb = len(pb_models)
        for i, (class_name, model) in enumerate(pb_models):
            try:
                grid = model.predict(home_key, away_key)
                markets_pb = _extract_model_probabilities(grid)
                w = _resolve_pb_weight(weights, i, class_name, n_pb)
                hda = markets_pb.get("home_draw_away")
                if hda is not None:
                    validated = _validate_hda(hda)
                    if validated is not None:
                        hda_probas.append(validated)
                        hda_weights.append(w)
                        pb_model_count += 1
                if "over15" in markets_pb:
                    over15_list.append(_safe_probability(markets_pb["over15"], 0.5))
                    over15_weights.append(w)
                if "over25" in markets_pb:
                    over25_list.append(_safe_probability(markets_pb["over25"], 0.5))
                    over25_weights.append(w)
                if "btts_yes" in markets_pb:
                    btts_list.append(_safe_probability(markets_pb["btts_yes"], 0.5))
                    btts_weights.append(w)
                expected_home = markets_pb.get("expected_home_goals", markets_pb.get("home_xg"))
                expected_away = markets_pb.get("expected_away_goals", markets_pb.get("away_xg"))
                if expected_home is not None and expected_away is not None:
                    eh = _safe_xg(expected_home, history_lambda_home)
                    ea = _safe_xg(expected_away, history_lambda_away)
                    xg_home_list.append(eh)
                    xg_away_list.append(ea)
                    xg_weights.append(w)
            except Exception as exc:
                logger.debug("PB[%d=%s] échec : %s", i, class_name, exc)

    if ml_models:
        for name, model in ml_models.items():
            proba = _model_predict_proba(model, name, feat_scaled, feat_arr)
            if proba is None:
                continue
            raw = weights.get(name, 0.0)
            try:
                w = float(raw)
            except (TypeError, ValueError):
                w = 0.0
            if not np.isfinite(w) or w < 0.0:
                w = 0.0
            hda_probas.append(proba)
            hda_weights.append(w)
            ml_model_count += 1

    total_active = ml_model_count + pb_model_count
    if not hda_probas:
        result = _build_output_from_lambdas(
            history_lambda_home, history_lambda_away, ht_ratio,
        )
        result["model_count_ml"] = ml_model_count
        result["model_count_pb"] = pb_model_count
        result["model_count_total"] = total_active
        return result

    probas_matrix = np.array(hda_probas, dtype=np.float64)
    uncertainty: Optional[List[float]] = None
    if probas_matrix.shape[0] > 1:
        uncertainty = probas_matrix.std(axis=0).tolist()

    if hda_weights and all(w == 0.0 for w in hda_weights):
        logger.warning("Tous les poids des modèles sont nuls pour %s vs %s", home_raw, away_raw)

    w_arr = _sanitize_weights(hda_weights)
    if w_arr.size == 0 or w_arr.size != probas_matrix.shape[0]:
        w_arr = np.ones(probas_matrix.shape[0]) / float(probas_matrix.shape[0])

    avg_hda = np.average(probas_matrix, axis=0, weights=w_arr)
    avg_hda = _finalize_probabilities(avg_hda)

    raw_temperature = weights.get("__temperature__", TEMPERATURE_SCALING_DEFAULT)
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError):
        temperature = float(TEMPERATURE_SCALING_DEFAULT)
    if not np.isfinite(temperature):
        temperature = float(TEMPERATURE_SCALING_DEFAULT)
    if TEMPERATURE_SCALING_ENABLED:
        temperature = float(np.clip(temperature, TEMPERATURE_SCALING_MIN, TEMPERATURE_SCALING_MAX))
    else:
        temperature = 1.0
    avg_hda = _apply_temperature(avg_hda, temperature)
    avg_hda = _finalize_probabilities(avg_hda)

    if xg_home_list and xg_away_list:
        xg_weight_arr = _sanitize_weights(xg_weights)
        if xg_weight_arr.size != len(xg_home_list):
            xg_weight_arr = np.ones(len(xg_home_list)) / float(len(xg_home_list))
        lambda_home = float(np.average(xg_home_list, weights=xg_weight_arr))
        lambda_away = float(np.average(xg_away_list, weights=xg_weight_arr))
        lambda_home = 0.5 * lambda_home + 0.5 * history_lambda_home
        lambda_away = 0.5 * lambda_away + 0.5 * history_lambda_away
    else:
        lambda_home = history_lambda_home
        lambda_away = history_lambda_away

    lambda_home = _safe_xg(lambda_home, history_lambda_home)
    lambda_away = _safe_xg(lambda_away, history_lambda_away)

    unified_matrix = _poisson_matrix(lambda_home, lambda_away)
    matrix_markets = _markets_from_matrix(unified_matrix)

    avg_over15 = matrix_markets["over15"]
    avg_over25 = matrix_markets["over25"]
    avg_btts = matrix_markets["btts_yes"]

    if over15_list:
        over15_weight_arr = _sanitize_weights(over15_weights)
        if over15_weight_arr.size != len(over15_list):
            over15_weight_arr = np.ones(len(over15_list)) / float(len(over15_list))
        pb_over15 = float(np.average(over15_list, weights=over15_weight_arr))
        avg_over15 = _safe_probability(
            0.5 * pb_over15 + 0.5 * matrix_markets["over15"],
            matrix_markets["over15"],
        )
    if over25_list:
        over25_weight_arr = _sanitize_weights(over25_weights)
        if over25_weight_arr.size != len(over25_list):
            over25_weight_arr = np.ones(len(over25_list)) / float(len(over25_list))
        pb_over25 = float(np.average(over25_list, weights=over25_weight_arr))
        avg_over25 = _safe_probability(
            0.5 * pb_over25 + 0.5 * matrix_markets["over25"],
            matrix_markets["over25"],
        )
    if btts_list:
        btts_weight_arr = _sanitize_weights(btts_weights)
        if btts_weight_arr.size != len(btts_list):
            btts_weight_arr = np.ones(len(btts_list)) / float(len(btts_list))
        pb_btts = float(np.average(btts_list, weights=btts_weight_arr))
        avg_btts = _safe_probability(
            0.5 * pb_btts + 0.5 * matrix_markets["btts_yes"],
            matrix_markets["btts_yes"],
        )
    if avg_over25 > avg_over15:
        avg_over25 = avg_over15

    hda_from_matrix = np.asarray(matrix_markets["home_draw_away"], dtype=np.float64)
    avg_hda = _reconcile_hda_with_matrix(avg_hda, hda_from_matrix)

    bookmaker_odds = _fetch_bookmaker_odds(competition_code, home_raw, away_raw, event_id=event_id)
    calibration_validated = _check_calibration_validity(weights, uncertainty)

    top_scores = _top_exact_scores(unified_matrix, n=_TOP_EXACT_SCORES)
    top_bands = _score_bands(unified_matrix)
    winner_idx = int(np.argmax(avg_hda))
    if winner_idx == 0:
        winner_label = "HOME"
    elif winner_idx == 1:
        winner_label = "DRAW"
    else:
        winner_label = "AWAY"
    sorted_proba = np.sort(avg_hda)
    winner_margin = float(sorted_proba[-1] - sorted_proba[-2]) if len(sorted_proba) >= 2 else 0.0

    markets: Dict[str, Any] = {
        "home_draw_away": avg_hda.tolist(),
        "matrix_home_draw_away": hda_from_matrix.tolist(),
        "over15": float(np.clip(avg_over15, 0.0, 1.0)),
        "over25": float(np.clip(avg_over25, 0.0, 1.0)),
        "over35": float(np.clip(matrix_markets["over35"], 0.0, 1.0)),
        "btts_yes": float(np.clip(avg_btts, 0.0, 1.0)),
        "home_xg": lambda_home,
        "away_xg": lambda_away,
        "xg_source": _XG_SOURCE_TAG,
        "xg_is_official": False,
        "uncertainty": uncertainty,
        "temperature": temperature,
        "fallback": False,
        "insufficient_data": False,
        "score_matrix": unified_matrix,
        "matrix_markets": matrix_markets,
        "model_count_ml": ml_model_count,
        "model_count_pb": pb_model_count,
        "model_count_total": total_active,
        "bookmaker_odds": bookmaker_odds,
        "calibration_validated": calibration_validated,
        "top_exact_scores": top_scores,
        "top_score_bands": top_bands,
        "most_likely_score": top_scores[0][0] if top_scores else None,
        "most_likely_score_proba": float(top_scores[0][1]) if top_scores else None,
        "winner": winner_idx,
        "winner_label": winner_label,
        "winner_probability": float(avg_hda[winner_idx]),
        "winner_margin": winner_margin,
    }

    ht_markets = _compute_ht_markets(lambda_home, lambda_away, ht_ratio)
    markets.update(ht_markets)
    return markets


def _markets_are_valid(markets: Dict[str, Any]) -> bool:
    if markets.get("insufficient_data"):
        return False
    try:
        proba = markets.get("home_draw_away")
        if not isinstance(proba, list) or len(proba) != 3:
            return False
        values = [float(x) for x in proba]
        if not all(np.isfinite(v) for v in values):
            return False
        if any(v < 0.0 or v > 1.0 for v in values):
            return False
        total = sum(values)
        if not np.isfinite(total) or total <= 0:
            return False
        if abs(total - 1.0) > _PROBA_SUM_TOLERANCE:
            return False
        for key in ("over15", "over25", "btts_yes"):
            v = markets.get(key)
            if v is None:
                return False
            fv = float(v)
            if not np.isfinite(fv) or fv < 0.0 or fv > 1.0:
                return False
        for key in ("home_xg", "away_xg"):
            v = markets.get(key)
            if v is None:
                return False
            fv = float(v)
            if not np.isfinite(fv) or fv <= 0.0:
                return False
        score_matrix = markets.get("score_matrix")
        if score_matrix is not None:
            if not isinstance(score_matrix, np.ndarray):
                return False
            if score_matrix.ndim != 2:
                return False
            if score_matrix.shape[0] != score_matrix.shape[1]:
                return False
            if score_matrix.shape[0] < 5:
                return False
            if not np.all(np.isfinite(score_matrix)):
                return False
            if np.any(score_matrix < 0.0):
                return False
            mat_sum = float(score_matrix.sum())
            if not np.isfinite(mat_sum) or mat_sum <= 0:
                return False
            if abs(mat_sum - 1.0) > 0.01:
                return False
        ht_markets = markets.get("ht_home_draw_away")
        if ht_markets is not None:
            if not isinstance(ht_markets, list) or len(ht_markets) != 3:
                return False
            ht_vals = [float(x) for x in ht_markets]
            if not all(np.isfinite(v) for v in ht_vals):
                return False
            ht_total = sum(ht_vals)
            if ht_total <= 0 or abs(ht_total - 1.0) > _PROBA_SUM_TOLERANCE:
                return False
        return True
    except Exception:
        return False


def _reliability_indicator(reference_prob: Any, is_ht: bool = False) -> str:
    if is_ht:
        return "⚠️ Pari peu fiable"
    try:
        value = float(reference_prob)
    except (TypeError, ValueError):
        return "⚠️ Pari peu fiable"
    if not np.isfinite(value):
        return "⚠️ Pari peu fiable"
    if value >= _THRESHOLD_VERY_STRONG:
        return "🔥 Pari très fiable"
    if value >= _THRESHOLD_STRONG:
        return "✅ Pari fiable"
    if value >= _THRESHOLD_RELIABLE:
        return "🟡 Pari moyennement fiable"
    return "⚠️ Pari peu fiable"


def _winner_confidence_label(prob: float, margin: float) -> str:
    if prob >= 0.60 and margin >= 0.25:
        return "🔥 Vainqueur très probable"
    if prob >= 0.50 and margin >= 0.15:
        return "✅ Vainqueur probable"
    if prob >= 0.42 and margin >= 0.08:
        return "🟡 Vainqueur léger"
    if prob >= 0.38:
        return "🟠 Issue incertaine"
    return "⚠️ Issue très incertaine"


def _print_match_prediction(
    row: pd.Series,
    markets: Dict[str, Any],
    ml_models: Optional[Dict[str, Any]] = None,
    pb_models: Optional[List[Tuple[str, Any]]] = None,
) -> None:
    home_display = _safe_str(row.get("home_raw", ""), default="Home")
    away_display = _safe_str(row.get("away_raw", ""), default="Away")

    print()
    print("═" * 70)
    print(f"⚽ {home_display}  vs  {away_display}")
    print("═" * 70)

    if markets.get("insufficient_data"):
        reason = markets.get("insufficient_reason") or "données manquantes"
        print(" 🚨 Conseiller : Aucun pari conseillé")
        print(" ⚠️ Pari peu fiable")
        print(f" ℹ️  Données insuffisantes : {reason}")
        return

    if not _markets_are_valid(markets):
        print(" 🚨 Conseiller : Aucun pari conseillé")
        print(" ⚠️ Pari peu fiable")
        print(" ℹ️  Données invalides ou incohérentes")
        return

    proba = markets["home_draw_away"]
    lambda_home = float(markets["home_xg"])
    lambda_away = float(markets["away_xg"])
    over15 = float(markets["over15"])
    over25 = float(markets["over25"])
    btts_yes = float(markets["btts_yes"])

    winner_label = markets.get("winner_label", "UNKNOWN")
    winner_prob = markets.get("winner_probability", 0.0)
    winner_margin = markets.get("winner_margin", 0.0)
    most_likely_score = markets.get("most_likely_score")
    most_likely_score_proba = markets.get("most_likely_score_proba")
    top_exact_scores = markets.get("top_exact_scores", [])
    top_score_bands = markets.get("top_score_bands", [])

    print(" 🎯 PRÉDICTION GAGNANT SEC :")
    if winner_label == "HOME":
        winner_display = f"🏠 {home_display}"
    elif winner_label == "AWAY":
        winner_display = f"✈️ {away_display}"
    else:
        winner_display = "🤝 MATCH NUL"

    winner_confidence = _winner_confidence_label(winner_prob, winner_margin)
    print(f"    → {winner_display}")
    print(f"    Confiance : {winner_confidence} "
          f"({winner_prob * 100:.1f}% / marge {winner_margin * 100:+.1f}%)")
    print()

    print(" 🎲 SCORE EXACT LE PLUS PROBABLE :")
    if most_likely_score is not None:
        print(f"    → {most_likely_score}  ({most_likely_score_proba * 100:.1f}%)")
    else:
        print("    → Indisponible")
    print()

    if top_exact_scores:
        print(" 📊 TOP SCORES EXACTS :")
        for i, (score_label, prob) in enumerate(top_exact_scores, 1):
            bar = "█" * int(prob * 100)
            print(f"    {i}. {score_label:<6} {prob * 100:5.1f}%  {bar}")
        print()

    if top_score_bands:
        print(" 🎯 SCORES GROUPÉS (top 3) :")
        for i, (band_label, prob) in enumerate(top_score_bands, 1):
            print(f"    {i}. {band_label:<6} {prob * 100:5.1f}%")
        print()

    print(f" 🤔 xG : {lambda_home:.2f} - {lambda_away:.2f}")
    print()

    print(" 📈 MARCHÉS 1X2 :")
    print(f"    ⚽ {home_display:<28} : {proba[0] * 100:5.1f}%")
    print(f"    🤝 Nul{'':<25} : {proba[1] * 100:5.1f}%")
    print(f"    ⚽ {away_display:<28} : {proba[2] * 100:5.1f}%")
    print()

    print(" 📈 MARCHÉS BUTS :")
    print(f"    🏆 +1.5 buts : {over15 * 100:5.1f}%  |  -1.5 buts : {(1 - over15) * 100:5.1f}%")
    print(f"    🏆 +2.5 buts : {over25 * 100:5.1f}%  |  -2.5 buts : {(1 - over25) * 100:5.1f}%")
    print(f"    🎯 BTTS Oui  : {btts_yes * 100:5.1f}%  |  BTTS Non  : {(1 - btts_yes) * 100:5.1f}%")
    print()

    ht_hda = markets.get("ht_home_draw_away")
    if isinstance(ht_hda, list) and len(ht_hda) == 3:
        print(" 📈 MARCHÉS 1ÈRE MI-TEMPS :")
        print(f"    ⚽ {home_display:<28} : {ht_hda[0] * 100:5.1f}%")
        print(f"    🤝 Nul{'':<25} : {ht_hda[1] * 100:5.1f}%")
        print(f"    ⚽ {away_display:<28} : {ht_hda[2] * 100:5.1f}%")
        ht_over05 = markets.get("ht_over05", 0.0)
        ht_over15 = markets.get("ht_over15", 0.0)
        print(f"    🏆 +0.5 buts MT : {ht_over05 * 100:5.1f}%  |  +1.5 buts MT : {ht_over15 * 100:5.1f}%")
        print()

    ht_markets = {
        "ht_home_draw_away": markets.get("ht_home_draw_away"),
        "ht_over05": markets.get("ht_over05"),
        "ht_over15": markets.get("ht_over15"),
        "ht_over25": markets.get("ht_over25"),
    }

    n_ml = int(markets.get("model_count_ml", 0))
    n_pb = int(markets.get("model_count_pb", 0))
    n_total = int(markets.get("model_count_total", 0))
    print(" 🤖 MODÈLES :")
    print(f"    ML : {n_ml}  |  Penaltyblog : {n_pb}  |  Total : {n_total}")
    if markets.get("calibration_validated") is False:
        print("    ⚠️ Calibration non validée")
    if markets.get("fallback"):
        print("    ⚠️ Mode dégradé (fallback)")
    print()

    if n_ml < _MIN_ACTIVE_ML_MODELS or n_total < _MIN_TOTAL_ACTIVE_MODELS:
        print(" 🚨 Conseiller : Aucun pari conseillé")
        print(" ⚠️ Pari peu fiable")
        print(f" ℹ️  Modèles opérationnels insuffisants ({n_ml} ML + {n_pb} PB)")
        print()
        return

    if markets.get("fallback"):
        print(" 🚨 Conseiller : Aucun pari conseillé")
        print(" ⚠️ Pari peu fiable")
        print(" ℹ️  Prédiction en mode dégradé — aucune recommandation émise")
        print()
        return

    if markets.get("calibration_validated") is False:
        print(" 🚨 Conseiller : Aucun pari conseillé")
        print(" ⚠️ Pari peu fiable")
        print(" ℹ️  Calibration non validée — recommandation bloquée")
        print()
        return

    uncertainty = markets.get("uncertainty")
    if uncertainty is not None and len(uncertainty) == 3:
        max_std = float(max(uncertainty))
        if max_std > _MAX_ACCEPTED_ENSEMBLE_STD:
            print(" 🚨 Conseiller : Aucun pari conseillé")
            print(" ⚠️ Pari peu fiable")
            print(f" ℹ️  Fort désaccord entre modèles (std max {max_std:.3f})")
            print()
            return

    bookmaker_odds = markets.get("bookmaker_odds")
    candidates = _build_market_candidates(
        np.asarray(proba, dtype=np.float64),
        home_display, away_display,
        over15, over25, btts_yes, ht_markets,
    )
    best_bet: Optional[Dict[str, Any]] = None
    if bookmaker_odds is not None and _validate_odds_structure(bookmaker_odds):
        best_bet = _select_best_ev_bet(candidates, bookmaker_odds)
    if best_bet is None:
        if bookmaker_odds is None:
            best_bet = _select_best_no_odds_bet(candidates)
        else:
            print(" 🚨 Conseiller : Aucun pari conseillé")
            print(" ⚠️ Pari peu fiable")
            print(f" ℹ️  Aucun marché n'atteint le seuil d'EV minimal ({_MIN_EV_THRESHOLD * 100:.1f}%)")
            print()
            return

    if best_bet is None:
        print(" 🚨 Conseiller : Aucun pari conseillé")
        print(" ⚠️ Pari peu fiable")
        print(" ℹ️  Aucun marché ne dépasse les seuils stricts")
        print()
        return

    bet_name = best_bet["name"]
    bet_prob = best_bet["probability"]
    bet_is_ht = best_bet["is_ht"]
    bet_odds = best_bet.get("odds")
    bet_ev = best_bet.get("ev")
    bet_kelly = best_bet.get("kelly")
    if bet_odds is not None:
        print(f" 💰 Cote du marché recommandé : {bet_odds:.2f}")
    reliability = _reliability_indicator(bet_prob, bet_is_ht)
    print(f" 🚨 Conseiller : {bet_name}")
    print(f" {reliability}")
    if bet_ev is not None:
        print(f" 💎 EV : {bet_ev * 100:.2f}%")
    if bet_kelly is not None and bet_kelly > 0:
        print(f" 💼 Mise Kelly fractionnée : {bet_kelly * 100:.2f}%")
    print()


def _check_required_columns(df: pd.DataFrame) -> bool:
    missing = [c for c in _REQUIRED_MATCH_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Colonnes obligatoires manquantes : %s", missing)
        return False
    return True


def _filter_history_before_date(
    df_past: pd.DataFrame,
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    if df_past is None or df_past.empty:
        return df_past
    if "date_parsed" not in df_past.columns:
        return df_past
    try:
        mask = df_past["date_parsed"] < match_date
        return df_past[mask].copy()
    except Exception:
        return df_past


def predict_for_date(
    date_str: str,
    ml_models: Optional[Dict[str, Any]],
    pb_models: Optional[List[Tuple[str, Any]]],
    pipeline: Any,
    df_past: pd.DataFrame,
    weights: Dict[str, float],
) -> None:
    if not isinstance(weights, dict):
        weights = {}
    try:
        matches, source = get_all_matches_for_date(date_str)
    except Exception as exc:
        logger.error("Erreur récupération matchs du %s : %s", date_str, exc)
        print("❌ Erreur lors de la récupération des matchs.")
        return
    if not matches:
        print("❌ Aucun match trouvé.")
        return
    df_today = process_future_matches(matches)
    if df_today.empty:
        print("❌ Aucun match exploitable trouvé.")
        return
    if not _check_required_columns(df_today):
        print("❌ Colonnes obligatoires manquantes dans la réponse API.")
        return
    if "status" not in df_today.columns:
        logger.warning("Colonne 'status' absente — statut UNKNOWN")
        df_today["status"] = "UNKNOWN"
    df_today["status"] = df_today["status"].astype(str).str.strip().str.upper()
    df_future = df_today[
        df_today["status"].isin(_PLAYABLE_STATUSES)
        & df_today["home_raw"].notna()
        & df_today["away_raw"].notna()
    ].copy()
    if df_future.empty:
        print("Aucun match à venir.")
        return
    df_future = df_future[
        df_future["home_key"].notna()
        & df_future["away_key"].notna()
        & (df_future["home_key"].astype(str).str.strip() != "")
        & (df_future["away_key"].astype(str).str.strip() != "")
    ].copy()
    if df_future.empty:
        print("Aucun match à venir.")
        return

    print(f"✅ {len(df_future)} matchs à venir (SOURCE: @ANGELO)")

    avg_h = _safe_goals_mean(df_past, "goals_home")
    avg_a = _safe_goals_mean(df_past, "goals_away")
    ht_ratio = _estimate_ht_lambda_ratio(df_past)

    if df_past is None or df_past.empty or len(df_past) < MIN_MATCHES_FOR_TRAINING:
        logger.warning("Historique insuffisant, estimation dégradée.")
        tracker_empty = TeamHistoryTracker(
            default_gf=avg_h if avg_h is not None else _NAN,
            default_ga=avg_a if avg_a is not None else _NAN,
            default_pts=_NAN,
        )
        elo_empty = EloSystem()
        if df_past is not None and not df_past.empty:
            try:
                tracker_empty, elo_empty = build_tracking_state(df_past)
            except Exception as exc:
                logger.error("Erreur build_tracking_state dégradé : %s", exc)
        print("\n🔮 PRÉDICTIONS (mode dégradé) :")
        for _, row in df_future.iterrows():
            home_key = _safe_str(row.get("home_key", ""))
            away_key = _safe_str(row.get("away_key", ""))
            if not home_key or not away_key:
                continue
            comp_code = _safe_str(row.get("competition_code", ""), default="").upper()
            if not comp_code:
                comp_code = None
            try:
                lambdas = _estimate_lambdas_from_history(
                    home_key, away_key, tracker_empty, elo_empty,
                    comp_code, avg_h, avg_a,
                )
                if lambdas is None:
                    markets = _build_insufficient_output(
                        "Aucun baseline disponible pour ce match"
                    )
                else:
                    lh, la = lambdas
                    markets = _build_output_from_lambdas(lh, la, ht_ratio)
            except Exception as exc:
                logger.error("Erreur fallback %s vs %s : %s", home_key, away_key, exc)
                continue
            _print_match_prediction(row, markets, ml_models, pb_models)
        return

    print("\n🔮 PRÉDICTIONS :")
    for _, row in df_future.iterrows():
        home_display = _safe_str(row.get("home_raw", ""), default="Home")
        away_display = _safe_str(row.get("away_raw", ""), default="Away")
        home_key = _safe_str(row.get("home_key", ""))
        away_key = _safe_str(row.get("away_key", ""))
        if not home_key or not away_key:
            continue
        comp = _safe_str(row.get("competition", ""))
        comp_code = _safe_str(row.get("competition_code", ""), default="").upper()
        if not comp_code:
            comp_code = None
        match_date = row.get("date_parsed")
        event_id = _safe_int(row.get("match_id"))
        df_past_match = df_past
        if match_date is not None and not pd.isna(match_date):
            try:
                df_past_match = _filter_history_before_date(df_past, match_date)
            except Exception:
                df_past_match = df_past
        if df_past_match is None or df_past_match.empty:
            df_past_match = df_past
        try:
            tracker, elo_system = build_tracking_state(df_past_match)
        except Exception as exc:
            logger.error("Erreur build_tracking_state %s vs %s : %s", home_display, away_display, exc)
            print(f"\n{home_display} vs {away_display} ({comp})")
            print(" \033[1;31mErreur de construction de l'état de suivi.\033[0m")
            continue
        try:
            markets = predict_match(
                home_key, away_key, home_display, away_display,
                comp_code, ml_models, pb_models, pipeline,
                df_past_match, tracker, elo_system, weights,
                match_date, ht_ratio, event_id,
            )
        except Exception:
            logger.exception("Erreur de prédiction %s vs %s", home_display, away_display)
            print(f"\n{home_display} vs {away_display} ({comp})")
            print(" \033[1;31mErreur lors du calcul de la prédiction.\033[0m")
            continue
        _print_match_prediction(row, markets, ml_models, pb_models)
