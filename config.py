import os
import re
import sys
import json
import hashlib
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Mapping

from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

logger = logging.getLogger("goangel.config")

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

_LOGGING_CONFIGURED = False

MIN_TOKEN_LENGTH: int = 10
MAX_TOKEN_LENGTH: int = 512
MIN_TOKENS_REQUIRED: int = 1

_FEATURES_SCHEMA_COUNT: int = 172
_ARTIFACT_SCHEMA_VERSION: str = "goangel-bzzoiro-2026.10-v2"
_EXPECTED_CLASS_COUNT: int = 3

_TOKEN_SEPARATOR_RE = re.compile(r"[,;\s]+")


class ConfigError(Exception):
    pass


class ConfigValidationError(ConfigError):
    pass


class ConfigMissingError(ConfigError):
    pass


def _load_environment() -> None:
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    else:
        load_dotenv(override=False)


_load_environment()


def _split_token_string(raw: str) -> List[str]:
    if not raw:
        return []
    parts = _TOKEN_SEPARATOR_RE.split(str(raw))
    cleaned: List[str] = []
    for part in parts:
        value = part.strip().strip('"').strip("'").strip()
        if value:
            cleaned.append(value)
    return cleaned


def _validate_token_format(token: str) -> Tuple[bool, str]:
    if not isinstance(token, str):
        return False, "type invalide"
    if len(token) < MIN_TOKEN_LENGTH:
        return False, f"trop court ({len(token)} < {MIN_TOKEN_LENGTH})"
    if len(token) > MAX_TOKEN_LENGTH:
        return False, f"trop long ({len(token)} > {MAX_TOKEN_LENGTH})"
    if any(ch.isspace() for ch in token):
        return False, "contient des espaces"
    return True, ""


def _resolve_strict_tokens() -> bool:
    raw = os.getenv("GOANGEL_STRICT_TOKENS")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y", "o")


TOKENS_STRICT: bool = _resolve_strict_tokens()


def _collect_tokens() -> List[str]:
    raw_pool = os.getenv("BZZOIRO_TOKENS", "").strip()
    raw_single = os.getenv("BZZOIRO_TOKEN", "").strip()
    raw_key = os.getenv("BZZOIRO_API_KEY", "").strip()

    candidates: List[str] = []
    candidates.extend(_split_token_string(raw_pool))
    candidates.extend(_split_token_string(raw_single))
    candidates.extend(_split_token_string(raw_key))

    seen: set = set()
    unique: List[str] = []
    for token in candidates:
        ok, reason = _validate_token_format(token)
        if not ok:
            logger.warning(
                "Token ignoré (%s) : %s***",
                reason,
                token[:4] if len(token) >= 4 else "?",
            )
            continue
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)

    if len(unique) < MIN_TOKENS_REQUIRED:
        message = (
            "Aucun token Bzzoiro valide trouvé. "
            "Renseignez BZZOIRO_TOKENS dans le fichier .env."
        )
        if TOKENS_STRICT:
            raise ConfigMissingError(message)
        logger.warning("%s (mode non strict)", message)
        return []
    return unique


tokens: List[str] = _collect_tokens()
TOKEN_COUNT: int = len(tokens)


def get_token_pool_metadata() -> Dict[str, Any]:
    return {
        "count": TOKEN_COUNT,
        "min_length": min((len(t) for t in tokens), default=0),
        "max_length": max((len(t) for t in tokens), default=0),
        "strict": bool(TOKENS_STRICT),
    }


@dataclass(frozen=True)
class Competition:
    code: str
    league_id: int
    name: str
    country: str = ""
    flag: str = ""
    tier: int = 2
    group: str = "OTHER"


@dataclass(frozen=True)
class ModelHyperparams:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


TOP_60_LEAGUES: Dict[int, str] = {
    1: "Premier League",
    3: "La Liga",
    4: "Serie A",
    5: "Bundesliga",
    6: "Ligue 1",
    10: "Eredivisie",
    9: "Brasileirão Serie A",
    2: "Liga Portugal Betclic",
    11: "Trendyol Super Lig",
    13: "Scottish Premiership",
    14: "Pro League (BEL)",
    49: "J1 League",
    85: "Liga Profesional (ARG)",
    84: "Danish Superliga",
    15: "Super League (SUI)",
    12: "Championship (ENG)",
    89: "Ligue 2",
    38: "Segunda División",
    34: "Brasileirão Serie B",
    18: "MLS",
    19: "Liga MX Apertura",
    20: "Liga MX Clausura",
    17: "Saudi Pro League",
    53: "Botola Pro",
    47: "Tunisian Ligue 1",
    86: "League One (ENG)",
    87: "League Two (ENG)",
    8: "Europa League",
    83: "Conference League",
}

_LEAGUE_GROUPS: Dict[str, List[int]] = {
    "EUROPE_TOP5": [1, 3, 4, 5, 6],
    "EUROPE_CUP": [8, 83],
    "EUROPE_SECOND": [12, 86, 87, 89, 38],
    "EUROPE_OTHER": [10, 2, 11, 13, 14, 84, 15],
    "AMERICAS": [9, 34, 18, 19, 20, 85],
    "ASIA_AFRICA": [49, 17, 53, 47],
}

_GROUPED_LEAGUE_IDS: set = {
    lid for ids in _LEAGUE_GROUPS.values() for lid in ids
}

_OTHER_LEAGUE_IDS: List[int] = [
    lid for lid in TOP_60_LEAGUES.keys() if lid not in _GROUPED_LEAGUE_IDS
]

_LEAGUE_GROUPS["OTHER"] = _OTHER_LEAGUE_IDS

COMPETITION_GROUP_ORDER: Tuple[str, ...] = (
    "EUROPE_TOP5",
    "EUROPE_CUP",
    "EUROPE_SECOND",
    "EUROPE_OTHER",
    "AMERICAS",
    "ASIA_AFRICA",
    "OTHER",
)

COMPETITION_GROUP_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(COMPETITION_GROUP_ORDER)
}

_CODE_TO_GROUP: Dict[str, str] = {}
for _group_name, _ids in _LEAGUE_GROUPS.items():
    for _lid in _ids:
        _CODE_TO_GROUP[str(_lid)] = _group_name

_TIER_MAP: Dict[int, int] = {}
for _i, _lid in enumerate(TOP_60_LEAGUES.keys()):
    if _lid in (1, 3, 4, 5, 6, 10, 9, 2, 11):
        _TIER_MAP[_lid] = 1
    elif _lid in (13, 14, 49, 85, 96, 84, 99, 15, 12, 17, 18):
        _TIER_MAP[_lid] = 2
    else:
        _TIER_MAP[_lid] = 3

_COMPETITIONS_DEFINITIONS: List[Competition] = []
for _lid, _name in TOP_60_LEAGUES.items():
    _COMPETITIONS_DEFINITIONS.append(
        Competition(
            code=str(_lid),
            league_id=_lid,
            name=_name,
            country="",
            flag="",
            tier=_TIER_MAP.get(_lid, 2),
            group=_CODE_TO_GROUP.get(str(_lid), "OTHER"),
        )
    )

COMPETITIONS: List[str] = [c.code for c in _COMPETITIONS_DEFINITIONS]
COMPETITIONS_DETAILS: List[Competition] = _COMPETITIONS_DEFINITIONS
COMPETITIONS_BY_CODE: Dict[str, Competition] = {
    c.code: c for c in _COMPETITIONS_DEFINITIONS
}

COMPETITIONS_CATEGORICAL_INDEX: Dict[str, int] = {
    c.code: i for i, c in enumerate(_COMPETITIONS_DEFINITIONS)
}

COMPETITIONS_TIER_LEVEL: Dict[str, int] = {
    c.code: c.tier for c in _COMPETITIONS_DEFINITIONS
}

COMPETITIONS_TIER_INDEX: Dict[str, int] = dict(COMPETITIONS_TIER_LEVEL)

COMPETITION_GROUPS: Dict[str, List[str]] = {
    group_name: [str(lid) for lid in ids]
    for group_name, ids in _LEAGUE_GROUPS.items()
}

CODE_TO_GROUP: Dict[str, str] = dict(_CODE_TO_GROUP)

_VALID_COMPETITION_CODES_SET: frozenset = frozenset(COMPETITIONS)
_VALID_GROUPS_SET: frozenset = frozenset(COMPETITION_GROUPS.keys())


def _validate_competition_definitions() -> List[str]:
    errors: List[str] = []
    seen_codes: set = set()
    for c in _COMPETITIONS_DEFINITIONS:
        if c.code in seen_codes:
            errors.append(f"Code compétition dupliqué : {c.code}")
        seen_codes.add(c.code)
        if c.tier < 1:
            errors.append(f"Compétition {c.code} : tier invalide ({c.tier})")
        if c.group not in _VALID_GROUPS_SET:
            errors.append(f"Compétition {c.code} : groupe inconnu ({c.group})")
    for group, codes in COMPETITION_GROUPS.items():
        for code in codes:
            if code not in _VALID_COMPETITION_CODES_SET:
                errors.append(f"Groupe {group} référence un code inconnu : {code}")
    all_grouped = set()
    for codes in COMPETITION_GROUPS.values():
        all_grouped.update(codes)
    missing = _VALID_COMPETITION_CODES_SET - all_grouped
    if missing:
        errors.append(f"Codes absents des groupes : {sorted(missing)}")
    return errors


_COMPETITION_DEFINITION_ERRORS = _validate_competition_definitions()
if _COMPETITION_DEFINITION_ERRORS:
    for _err_msg in _COMPETITION_DEFINITION_ERRORS:
        logger.critical("Configuration compétitions invalide : %s", _err_msg)
    raise ConfigValidationError(
        "Configuration compétitions invalide : "
        + " | ".join(_COMPETITION_DEFINITION_ERRORS)
    )


def get_competition_group(code: Optional[str]) -> str:
    if not code:
        return "OTHER"
    return CODE_TO_GROUP.get(str(code), "OTHER")


_FALLBACK_HOME_LAMBDA: float = 1.40
_FALLBACK_AWAY_LAMBDA: float = 1.15
_FALLBACK_DRAW_RATE: float = 0.25
_FALLBACK_HOME_ADVANTAGE: float = 60.0

REALISTIC_HOME_LAMBDA_BY_GROUP: Dict[str, float] = {
    "EUROPE_CUP": 1.42,
    "EUROPE_TOP5": 1.55,
    "EUROPE_SECOND": 1.40,
    "EUROPE_OTHER": 1.45,
    "AMERICAS": 1.35,
    "ASIA_AFRICA": 1.30,
    "OTHER": 1.40,
}

REALISTIC_AWAY_LAMBDA_BY_GROUP: Dict[str, float] = {
    "EUROPE_CUP": 1.18,
    "EUROPE_TOP5": 1.25,
    "EUROPE_SECOND": 1.15,
    "EUROPE_OTHER": 1.20,
    "AMERICAS": 1.05,
    "ASIA_AFRICA": 1.00,
    "OTHER": 1.15,
}

REALISTIC_DRAW_RATE_BY_GROUP: Dict[str, float] = {
    "EUROPE_CUP": 0.25,
    "EUROPE_TOP5": 0.24,
    "EUROPE_SECOND": 0.26,
    "EUROPE_OTHER": 0.25,
    "AMERICAS": 0.28,
    "ASIA_AFRICA": 0.29,
    "OTHER": 0.25,
}

REALISTIC_HOME_ADVANTAGE_BY_GROUP: Dict[str, float] = {
    "EUROPE_CUP": 55.0,
    "EUROPE_TOP5": 60.0,
    "EUROPE_SECOND": 58.0,
    "EUROPE_OTHER": 60.0,
    "AMERICAS": 70.0,
    "ASIA_AFRICA": 72.0,
    "OTHER": 60.0,
}


HEADERS: Dict[str, str] = (
    {"Authorization": f"Token {tokens[0]}"} if tokens else {}
)


def get_auth_headers(token_index: int = 0) -> Dict[str, str]:
    if not tokens:
        return {}
    try:
        idx = int(token_index)
    except (TypeError, ValueError):
        idx = 0
    idx = idx % len(tokens)
    return {"Authorization": f"Token {tokens[idx]}"}


API_BASE_URL: str = "https://sports.bzzoiro.com/api/v2"
API_TIMEOUT_SECONDS: float = 20.0
API_MAX_RETRIES: int = 4
API_RETRY_BACKOFF_SECONDS: float = 0.5
API_RATE_LIMIT_SLEEP_SECONDS: float = 6.0
API_COMPETITION_SLEEP_SECONDS: float = 0.15
API_MAX_CONCURRENT_REQUESTS: int = 1
API_CHUNK_DAYS: int = 365
API_CIRCUIT_BREAKER_FAILURES: int = 5
API_CIRCUIT_BREAKER_COOLDOWN: float = 60.0

CACHE_DIR: Path = (BASE_DIR / "cache").resolve()
_CACHE_DIR_ERROR: Optional[str] = None
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    _CACHE_DIR_ERROR = f"Impossible de créer {CACHE_DIR} : {exc}"

HISTORICAL_FILE: Path = CACHE_DIR / "historical.pkl"
MODELS_FILE: Path = CACHE_DIR / "models.pkl"
SCALER_FILE: Path = CACHE_DIR / "scaler.pkl"
WEIGHTS_FILE: Path = CACHE_DIR / "weights.pkl"
NAME_TO_ID_FILE: Path = CACHE_DIR / "name_to_id.pkl"
CALIBRATION_STATE_FILE: Path = CACHE_DIR / "calibration.pkl"
TEMPERATURE_STATE_FILE: Path = CACHE_DIR / "temperature.pkl"
BACKTEST_STATE_FILE: Path = CACHE_DIR / "backtest.pkl"
MANIFEST_FILE: Path = CACHE_DIR / "manifest.json"

CHECKPOINT_DIR: Path = CACHE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_STATE_FILE: Path = CHECKPOINT_DIR / "state.json"
CHECKPOINT_RAW_FILE: Path = CHECKPOINT_DIR / "raw_matches.jsonl"
CHECKPOINT_META_FILE: Path = CHECKPOINT_DIR / "meta.json"
CHECKPOINT_SAVE_INTERVAL_SECONDS: float = 30.0
CHECKPOINT_PUSH_INTERVAL_SECONDS: float = 300.0
CHECKPOINT_MAX_FILE_MB: float = 200.0

CACHE_TTL_HOURS: int = 0
CACHE_NEVER_EXPIRES: bool = True
CACHE_ATOMIC_WRITE: bool = True


def cache_is_fresh(path: Path) -> bool:
    if CACHE_NEVER_EXPIRES:
        return path.exists()
    if not path.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0
    return age_hours < CACHE_TTL_HOURS


def _resolve_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


def _resolve_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


def _resolve_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y", "o")


HISTORICAL_SEASONS_LOOKBACK: int = _resolve_int(
    "BZZOIRO_HISTORICAL_SEASONS_LOOKBACK", 3, 1, 30
)
HISTORICAL_MIN_MATCHES_PER_TEAM: int = 3
MIN_MATCHES_FOR_TRAINING: int = _resolve_int(
    "GOANGEL_MIN_MATCHES_FOR_TRAINING", 500, 50, 1_000_000
)
MIN_MATCHES_PER_CLASS: int = 50

BZZOIRO_WORKERS_PER_KEY: int = _resolve_int(
    "BZZOIRO_WORKERS_PER_KEY", 2, 1, 32
)
BZZOIRO_MAX_WORKERS: int = _resolve_int(
    "BZZOIRO_MAX_WORKERS", 64, 1, 512
)
BZZOIRO_RATE_LIMIT_PER_MINUTE: int = _resolve_int(
    "BZZOIRO_RATE_LIMIT_PER_MINUTE", 10, 1, 600
)
BZZOIRO_RATE_LIMIT_SAFETY_MARGIN: int = _resolve_int(
    "BZZOIRO_RATE_LIMIT_SAFETY_MARGIN", 1, 0, 100
)
BZZOIRO_ENRICH_XG: bool = _resolve_bool("BZZOIRO_ENRICH_XG", True)

GOANGEL_GITHUB_PUSH: bool = _resolve_bool("GOANGEL_GITHUB_PUSH", False)
GOANGEL_GITHUB_REPO: str = os.getenv(
    "GOANGEL_GITHUB_REPO", "Angeloda444/GOANGELCLOUD"
).strip()
GOANGEL_GITHUB_USER: str = os.getenv(
    "GOANGEL_GITHUB_USER", "Angeloda444"
).strip()
GOANGEL_GITHUB_BRANCH: str = os.getenv(
    "GOANGEL_GITHUB_BRANCH", "cache-auto"
).strip()
GH_PAT: str = os.getenv("GH_PAT", "").strip() or os.getenv(
    "GITHUB_TOKEN", ""
).strip()

FORM_WINDOW_SHORT: int = 5
FORM_WINDOW_MEDIUM: int = 10
FORM_WINDOW_LONG: int = 20

ELO_INITIAL_RATING: float = 1500.0
ELO_K_FACTOR: float = 20.0
ELO_HOME_ADVANTAGE: float = 65.0
ELO_SEASON_REGRESSION: float = 0.15

TRAIN_RATIO: float = 0.65
VAL_RATIO: float = 0.10
CALIB_RATIO: float = 0.10
TEST_RATIO: float = 0.15

MODEL_TRAIN_TEST_SPLIT: float = 0.20
MODEL_WALK_FORWARD_FOLDS: int = _resolve_int(
    "GOANGEL_WALK_FORWARD_FOLDS", 5, 2, 20
)
MODEL_RANDOM_SEED: int = _resolve_int("GOANGEL_RANDOM_SEED", 42, 0, 2**31 - 1)
MODEL_RETRAIN_INTERVAL_HOURS: int = 24
MODEL_N_JOBS: int = -1

CALIBRATION_METHOD: str = os.getenv(
    "GOANGEL_CALIBRATION_METHOD", "isotonic"
).strip().lower()
if CALIBRATION_METHOD not in ("isotonic", "sigmoid"):
    CALIBRATION_METHOD = "isotonic"
CALIBRATION_MIN_VAL_SAMPLES: int = 300
CALIBRATION_CV_FOLDS: int = 5

TEMPERATURE_SCALING_ENABLED: bool = True
TEMPERATURE_SCALING_DEFAULT: float = _resolve_float(
    "GOANGEL_TEMPERATURE_DEFAULT", 1.15, 0.50, 5.00
)
TEMPERATURE_SCALING_MIN: float = 0.70
TEMPERATURE_SCALING_MAX: float = 2.50

ODDS_BLEND_WEIGHT: float = 0.0
MODEL_BLEND_WEIGHT: float = 1.0
ODDS_ENABLED_IN_TRAINING: bool = False

MAX_CONFIDENT_PROBABILITY: float = 0.78
MIN_CONFIDENT_PROBABILITY: float = 0.03
DRAW_FLOOR_PROBABILITY: float = 0.20

PROBABILITY_CLIP_EPS: float = 0.005

DIXON_COLES_RHO: float = -0.05
MAX_GOALS_GRID: int = 12

XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 5
XGB_LEARNING_RATE: float = 0.05
XGB_SUBSAMPLE: float = 0.80
XGB_COLSAMPLE: float = 0.80
XGB_MIN_CHILD_WEIGHT: float = 5.0
XGB_REG_LAMBDA: float = 1.5
XGB_REG_ALPHA: float = 0.5

LGBM_N_ESTIMATORS: int = 300
LGBM_MAX_DEPTH: int = -1
LGBM_LEARNING_RATE: float = 0.05
LGBM_NUM_LEAVES: int = 31
LGBM_MIN_CHILD_SAMPLES: int = 25
LGBM_SUBSAMPLE: float = 0.80
LGBM_COLSAMPLE: float = 0.80
LGBM_REG_LAMBDA: float = 1.5

CATBOOST_ITERATIONS: int = 400
CATBOOST_DEPTH: int = 5
CATBOOST_LEARNING_RATE: float = 0.05
CATBOOST_L2_LEAF_REG: float = 3.0
CATBOOST_RANDOM_STRENGTH: float = 1.0

RF_N_ESTIMATORS: int = 200
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_LEAF: int = 5
RF_MIN_SAMPLES_SPLIT: int = 10
RF_MAX_FEATURES: str = "sqrt"
RF_MAX_FEATURES_RESOLVED: Optional[str] = "sqrt"

DL_ENABLED: bool = True
DL_HIDDEN_SIZE: int = 96
DL_EPOCHS: int = 100
DL_BATCH_SIZE: int = 32
DL_LEARNING_RATE: float = 3e-4
DL_WEIGHT_DECAY: float = 1e-4
DL_DROPOUT: float = 0.20
DL_EARLY_STOPPING_PATIENCE: int = 15
DL_GRAD_CLIP: float = 0.5

BAYESIAN_ENABLED: bool = True
BAYESIAN_N_SAMPLES: int = 1000
BAYESIAN_BURN: int = 500
BAYESIAN_N_CHAINS: int = 2
BAYESIAN_THIN: int = 1
BAYESIAN_MODELS: Tuple[str, ...] = (
    "BayesianGoalModel",
    "HierarchicalBayesianGoalModel",
)

MODEL_HYPERPARAMS: Dict[str, ModelHyperparams] = {
    "XGB": ModelHyperparams(
        name="XGB",
        params={
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE,
            "min_child_weight": XGB_MIN_CHILD_WEIGHT,
            "reg_lambda": XGB_REG_LAMBDA,
            "reg_alpha": XGB_REG_ALPHA,
            "random_state": MODEL_RANDOM_SEED,
            "n_jobs": MODEL_N_JOBS,
        },
        enabled=True,
    ),
    "LGBM": ModelHyperparams(
        name="LGBM",
        params={
            "n_estimators": LGBM_N_ESTIMATORS,
            "max_depth": LGBM_MAX_DEPTH,
            "learning_rate": LGBM_LEARNING_RATE,
            "num_leaves": LGBM_NUM_LEAVES,
            "min_child_samples": LGBM_MIN_CHILD_SAMPLES,
            "subsample": LGBM_SUBSAMPLE,
            "colsample_bytree": LGBM_COLSAMPLE,
            "reg_lambda": LGBM_REG_LAMBDA,
            "random_state": MODEL_RANDOM_SEED,
            "n_jobs": MODEL_N_JOBS,
        },
        enabled=True,
    ),
    "CatBoost": ModelHyperparams(
        name="CatBoost",
        params={
            "iterations": CATBOOST_ITERATIONS,
            "depth": CATBOOST_DEPTH,
            "learning_rate": CATBOOST_LEARNING_RATE,
            "l2_leaf_reg": CATBOOST_L2_LEAF_REG,
            "random_strength": CATBOOST_RANDOM_STRENGTH,
            "random_state": MODEL_RANDOM_SEED,
        },
        enabled=True,
    ),
    "RF": ModelHyperparams(
        name="RF",
        params={
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "min_samples_split": RF_MIN_SAMPLES_SPLIT,
            "max_features": RF_MAX_FEATURES_RESOLVED,
            "random_state": MODEL_RANDOM_SEED,
            "n_jobs": MODEL_N_JOBS,
        },
        enabled=True,
    ),
    "DL": ModelHyperparams(
        name="DL",
        params={
            "hidden_size": DL_HIDDEN_SIZE,
            "epochs": DL_EPOCHS,
            "batch_size": DL_BATCH_SIZE,
            "learning_rate": DL_LEARNING_RATE,
            "weight_decay": DL_WEIGHT_DECAY,
            "dropout": DL_DROPOUT,
            "early_stopping_patience": DL_EARLY_STOPPING_PATIENCE,
            "grad_clip": DL_GRAD_CLIP,
        },
        enabled=DL_ENABLED,
    ),
    "Bayesian": ModelHyperparams(
        name="Bayesian",
        params={
            "n_samples": BAYESIAN_N_SAMPLES,
            "burn": BAYESIAN_BURN,
            "n_chains": BAYESIAN_N_CHAINS,
            "thin": BAYESIAN_THIN,
        },
        enabled=BAYESIAN_ENABLED,
    ),
}

H2H_MAX_MATCHES: int = 8
H2H_MAX_AGE_DAYS: int = 1825
H2H_MIN_MATCHES_FOR_FEATURE: int = 2

REST_DAYS_DEFAULT: float = 7.0
REST_DAYS_CAP: float = 21.0

BACKTEST_LAST_N_MATCHES: int = 500
BACKTEST_MIN_LOG_LOSS: float = 1.05
BACKTEST_MIN_ACCURACY: float = 0.45
BACKTEST_MIN_BRIER: float = 0.20
BACKTEST_WALK_FORWARD_STEPS: int = 5
BACKTEST_SAVE_REPORT: bool = True

ENSEMBLE_MIN_WEIGHT: float = 0.0
ENSEMBLE_MAX_WEIGHT: float = 0.60
ENSEMBLE_DISAGREEMENT_THRESHOLD: float = 0.10

FEATURE_MIN_COVERAGE: float = 0.60
FEATURE_OUTLIER_ZSCORE: float = 5.0
FEATURE_CLIP_ENABLED: bool = True

LOG_LEVEL: int = getattr(
    logging, os.getenv("GOANGEL_LOG_LEVEL", "INFO").strip().upper() or "INFO",
    logging.INFO,
)
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE: str = str(CACHE_DIR / "goangel.log")
LOG_MAX_BYTES: int = 10 * 1024 * 1024
LOG_BACKUP_COUNT: int = 5
CONSOLE_VERBOSE: bool = False

TIMEZONE: str = os.getenv("GOANGEL_TIMEZONE", "UTC").strip() or "UTC"
_TZ_CACHE: Dict[str, Any] = {}


def _resolve_timezone() -> Any:
    if TIMEZONE in _TZ_CACHE:
        return _TZ_CACHE[TIMEZONE]
    tz: Any = timezone.utc
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(TIMEZONE)
        except Exception as exc:
            logger.warning(
                "Timezone %r invalide, repli sur UTC : %s", TIMEZONE, exc
            )
            tz = timezone.utc
    _TZ_CACHE[TIMEZONE] = tz
    return tz


def get_timezone() -> Any:
    return _resolve_timezone()


def now_local() -> datetime:
    return datetime.now(get_timezone())


DEFAULT_SEASON_START_MONTH: int = 7
DEFAULT_SEASON_START_DAY: int = 1
DEFAULT_SEASON_END_MONTH: int = 6
DEFAULT_SEASON_END_DAY: int = 30


def _build_log_handlers() -> List[logging.Handler]:
    handlers: List[logging.Handler] = []
    if LOG_FILE:
        try:
            from logging.handlers import RotatingFileHandler
            log_path = Path(LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(LOG_LEVEL)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            handlers.append(file_handler)
        except Exception as exc:
            print(
                f"Impossible d'initialiser le fichier de log : {exc}",
                file=sys.stderr,
            )
    if CONSOLE_VERBOSE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(console_handler)
    if not handlers:
        handlers.append(logging.NullHandler())
    return handlers


def configure_logging(force: bool = False) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return
    handlers = _build_log_handlers()
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(LOG_LEVEL)
    warnings_logger.propagate = False
    for h in list(warnings_logger.handlers):
        warnings_logger.removeHandler(h)
    for h in handlers:
        warnings_logger.addHandler(h)
    _LOGGING_CONFIGURED = True


configure_logging()


def stage_print(message: str) -> None:
    try:
        print(message, flush=True)
    except Exception:
        pass


def _validate_ratios() -> List[str]:
    errors: List[str] = []
    total = TRAIN_RATIO + VAL_RATIO + CALIB_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        errors.append(
            f"TRAIN_RATIO + VAL_RATIO + CALIB_RATIO + TEST_RATIO = {total:.6f} "
            f"(attendu 1.0)"
        )
    for name, value in (
        ("TRAIN_RATIO", TRAIN_RATIO),
        ("VAL_RATIO", VAL_RATIO),
        ("CALIB_RATIO", CALIB_RATIO),
        ("TEST_RATIO", TEST_RATIO),
    ):
        if value <= 0 or value >= 1:
            errors.append(f"{name} doit être dans (0, 1) (reçu {value})")
    return errors


def _validate_api_parameters() -> List[str]:
    errors: List[str] = []
    if API_TIMEOUT_SECONDS <= 0 or API_TIMEOUT_SECONDS > 300:
        errors.append(
            f"API_TIMEOUT_SECONDS hors bornes (reçu {API_TIMEOUT_SECONDS})"
        )
    if API_MAX_RETRIES < 1 or API_MAX_RETRIES > 20:
        errors.append(f"API_MAX_RETRIES hors bornes (reçu {API_MAX_RETRIES})")
    if API_RETRY_BACKOFF_SECONDS < 0:
        errors.append("API_RETRY_BACKOFF_SECONDS doit être >= 0")
    if API_RATE_LIMIT_SLEEP_SECONDS < 0:
        errors.append("API_RATE_LIMIT_SLEEP_SECONDS doit être >= 0")
    if API_COMPETITION_SLEEP_SECONDS < 0:
        errors.append("API_COMPETITION_SLEEP_SECONDS doit être >= 0")
    if API_MAX_CONCURRENT_REQUESTS < 1 or API_MAX_CONCURRENT_REQUESTS > 64:
        errors.append(
            f"API_MAX_CONCURRENT_REQUESTS hors bornes "
            f"(reçu {API_MAX_CONCURRENT_REQUESTS})"
        )
    if API_CHUNK_DAYS < 1 or API_CHUNK_DAYS > 365:
        errors.append(f"API_CHUNK_DAYS hors bornes (reçu {API_CHUNK_DAYS})")
    if API_CIRCUIT_BREAKER_FAILURES < 1:
        errors.append("API_CIRCUIT_BREAKER_FAILURES doit être >= 1")
    if API_CIRCUIT_BREAKER_COOLDOWN < 1:
        errors.append("API_CIRCUIT_BREAKER_COOLDOWN doit être >= 1")
    return errors


def _validate_bzzoiro_params() -> List[str]:
    errors: List[str] = []
    if BZZOIRO_WORKERS_PER_KEY < 1 or BZZOIRO_WORKERS_PER_KEY > 32:
        errors.append(
            f"BZZOIRO_WORKERS_PER_KEY hors bornes "
            f"(reçu {BZZOIRO_WORKERS_PER_KEY})"
        )
    if BZZOIRO_MAX_WORKERS < 1 or BZZOIRO_MAX_WORKERS > 512:
        errors.append(
            f"BZZOIRO_MAX_WORKERS hors bornes "
            f"(reçu {BZZOIRO_MAX_WORKERS})"
        )
    if BZZOIRO_RATE_LIMIT_PER_MINUTE < 1 or BZZOIRO_RATE_LIMIT_PER_MINUTE > 600:
        errors.append(
            f"BZZOIRO_RATE_LIMIT_PER_MINUTE hors bornes "
            f"(reçu {BZZOIRO_RATE_LIMIT_PER_MINUTE})"
        )
    if BZZOIRO_RATE_LIMIT_SAFETY_MARGIN < 0:
        errors.append(
            "BZZOIRO_RATE_LIMIT_SAFETY_MARGIN doit être >= 0"
        )
    return errors


def _validate_xgb_parameters() -> List[str]:
    errors: List[str] = []
    if XGB_N_ESTIMATORS < 10 or XGB_N_ESTIMATORS > 20000:
        errors.append(
            f"XGB_N_ESTIMATORS hors bornes (reçu {XGB_N_ESTIMATORS})"
        )
    if XGB_MAX_DEPTH < 1 or XGB_MAX_DEPTH > 32:
        errors.append(f"XGB_MAX_DEPTH hors bornes (reçu {XGB_MAX_DEPTH})")
    if XGB_LEARNING_RATE <= 0 or XGB_LEARNING_RATE > 1:
        errors.append(
            f"XGB_LEARNING_RATE hors bornes (reçu {XGB_LEARNING_RATE})"
        )
    if XGB_SUBSAMPLE <= 0 or XGB_SUBSAMPLE > 1:
        errors.append(f"XGB_SUBSAMPLE hors bornes (reçu {XGB_SUBSAMPLE})")
    if XGB_COLSAMPLE <= 0 or XGB_COLSAMPLE > 1:
        errors.append(f"XGB_COLSAMPLE hors bornes (reçu {XGB_COLSAMPLE})")
    if XGB_MIN_CHILD_WEIGHT < 0:
        errors.append("XGB_MIN_CHILD_WEIGHT doit être >= 0")
    if XGB_REG_LAMBDA < 0:
        errors.append("XGB_REG_LAMBDA doit être >= 0")
    if XGB_REG_ALPHA < 0:
        errors.append("XGB_REG_ALPHA doit être >= 0")
    return errors


def _validate_lgbm_parameters() -> List[str]:
    errors: List[str] = []
    if LGBM_N_ESTIMATORS < 10 or LGBM_N_ESTIMATORS > 20000:
        errors.append(
            f"LGBM_N_ESTIMATORS hors bornes (reçu {LGBM_N_ESTIMATORS})"
        )
    if LGBM_MAX_DEPTH < -1 or LGBM_MAX_DEPTH > 64:
        errors.append(f"LGBM_MAX_DEPTH hors bornes (reçu {LGBM_MAX_DEPTH})")
    if LGBM_LEARNING_RATE <= 0 or LGBM_LEARNING_RATE > 1:
        errors.append(
            f"LGBM_LEARNING_RATE hors bornes (reçu {LGBM_LEARNING_RATE})"
        )
    if LGBM_NUM_LEAVES < 4 or LGBM_NUM_LEAVES > 4096:
        errors.append(f"LGBM_NUM_LEAVES hors bornes (reçu {LGBM_NUM_LEAVES})")
    if LGBM_MIN_CHILD_SAMPLES < 1:
        errors.append("LGBM_MIN_CHILD_SAMPLES doit être >= 1")
    if LGBM_SUBSAMPLE <= 0 or LGBM_SUBSAMPLE > 1:
        errors.append(f"LGBM_SUBSAMPLE hors bornes (reçu {LGBM_SUBSAMPLE})")
    if LGBM_COLSAMPLE <= 0 or LGBM_COLSAMPLE > 1:
        errors.append(f"LGBM_COLSAMPLE hors bornes (reçu {LGBM_COLSAMPLE})")
    if LGBM_REG_LAMBDA < 0:
        errors.append("LGBM_REG_LAMBDA doit être >= 0")
    return errors


def _validate_catboost_parameters() -> List[str]:
    errors: List[str] = []
    if CATBOOST_ITERATIONS < 10 or CATBOOST_ITERATIONS > 20000:
        errors.append(
            f"CATBOOST_ITERATIONS hors bornes (reçu {CATBOOST_ITERATIONS})"
        )
    if CATBOOST_DEPTH < 1 or CATBOOST_DEPTH > 16:
        errors.append(f"CATBOOST_DEPTH hors bornes (reçu {CATBOOST_DEPTH})")
    if CATBOOST_LEARNING_RATE <= 0 or CATBOOST_LEARNING_RATE > 1:
        errors.append(
            f"CATBOOST_LEARNING_RATE hors bornes (reçu {CATBOOST_LEARNING_RATE})"
        )
    if CATBOOST_L2_LEAF_REG < 0:
        errors.append("CATBOOST_L2_LEAF_REG doit être >= 0")
    if CATBOOST_RANDOM_STRENGTH < 0:
        errors.append("CATBOOST_RANDOM_STRENGTH doit être >= 0")
    return errors


def _validate_rf_parameters() -> List[str]:
    errors: List[str] = []
    if RF_N_ESTIMATORS < 10 or RF_N_ESTIMATORS > 10000:
        errors.append(
            f"RF_N_ESTIMATORS hors bornes (reçu {RF_N_ESTIMATORS})"
        )
    if RF_MAX_DEPTH < 1 or RF_MAX_DEPTH > 200:
        errors.append(f"RF_MAX_DEPTH hors bornes (reçu {RF_MAX_DEPTH})")
    if RF_MIN_SAMPLES_LEAF < 1:
        errors.append("RF_MIN_SAMPLES_LEAF doit être >= 1")
    if RF_MIN_SAMPLES_SPLIT < 2:
        errors.append("RF_MIN_SAMPLES_SPLIT doit être >= 2")
    if RF_MAX_FEATURES not in ("sqrt", "log2", "None"):
        errors.append(f"RF_MAX_FEATURES invalide (reçu {RF_MAX_FEATURES})")
    return errors


def _validate_dl_parameters() -> List[str]:
    errors: List[str] = []
    if DL_HIDDEN_SIZE < 8 or DL_HIDDEN_SIZE > 4096:
        errors.append(f"DL_HIDDEN_SIZE hors bornes (reçu {DL_HIDDEN_SIZE})")
    if DL_EPOCHS < 1 or DL_EPOCHS > 5000:
        errors.append(f"DL_EPOCHS hors bornes (reçu {DL_EPOCHS})")
    if DL_BATCH_SIZE < 1 or DL_BATCH_SIZE > 8192:
        errors.append(f"DL_BATCH_SIZE hors bornes (reçu {DL_BATCH_SIZE})")
    if DL_LEARNING_RATE <= 0 or DL_LEARNING_RATE > 1:
        errors.append(
            f"DL_LEARNING_RATE hors bornes (reçu {DL_LEARNING_RATE})"
        )
    if DL_WEIGHT_DECAY < 0 or DL_WEIGHT_DECAY > 1:
        errors.append(f"DL_WEIGHT_DECAY hors bornes (reçu {DL_WEIGHT_DECAY})")
    if DL_DROPOUT < 0 or DL_DROPOUT >= 1:
        errors.append(f"DL_DROPOUT hors bornes (reçu {DL_DROPOUT})")
    if DL_EARLY_STOPPING_PATIENCE < 1:
        errors.append("DL_EARLY_STOPPING_PATIENCE doit être >= 1")
    if DL_GRAD_CLIP <= 0:
        errors.append("DL_GRAD_CLIP doit être > 0")
    return errors


def _validate_bayesian_parameters() -> List[str]:
    errors: List[str] = []
    if BAYESIAN_N_SAMPLES < 50 or BAYESIAN_N_SAMPLES > 50000:
        errors.append(
            f"BAYESIAN_N_SAMPLES hors bornes (reçu {BAYESIAN_N_SAMPLES})"
        )
    if BAYESIAN_BURN < 50 or BAYESIAN_BURN > 50000:
        errors.append(
            f"BAYESIAN_BURN hors bornes (reçu {BAYESIAN_BURN})"
        )
    if BAYESIAN_N_CHAINS < 1 or BAYESIAN_N_CHAINS > 8:
        errors.append(
            f"BAYESIAN_N_CHAINS hors bornes (reçu {BAYESIAN_N_CHAINS})"
        )
    if BAYESIAN_THIN < 1 or BAYESIAN_THIN > 50:
        errors.append(
            f"BAYESIAN_THIN hors bornes (reçu {BAYESIAN_THIN})"
        )
    if not isinstance(BAYESIAN_MODELS, tuple) or len(BAYESIAN_MODELS) == 0:
        errors.append("BAYESIAN_MODELS doit être un tuple non vide")
    return errors


def _validate_goal_grid() -> List[str]:
    errors: List[str] = []
    if MAX_GOALS_GRID < 6 or MAX_GOALS_GRID > 30:
        errors.append(f"MAX_GOALS_GRID hors bornes (reçu {MAX_GOALS_GRID})")
    if DIXON_COLES_RHO < -1.0 or DIXON_COLES_RHO > 1.0:
        errors.append(
            f"DIXON_COLES_RHO hors bornes (reçu {DIXON_COLES_RHO})"
        )
    return errors


def _validate_calibration_parameters() -> List[str]:
    errors: List[str] = []
    if CALIBRATION_METHOD not in ("isotonic", "sigmoid"):
        errors.append(
            f"CALIBRATION_METHOD invalide (reçu {CALIBRATION_METHOD})"
        )
    if CALIBRATION_MIN_VAL_SAMPLES < 30:
        errors.append("CALIBRATION_MIN_VAL_SAMPLES doit être >= 30")
    if CALIBRATION_CV_FOLDS < 2 or CALIBRATION_CV_FOLDS > 20:
        errors.append(
            f"CALIBRATION_CV_FOLDS hors bornes (reçu {CALIBRATION_CV_FOLDS})"
        )
    if TEMPERATURE_SCALING_MIN >= TEMPERATURE_SCALING_MAX:
        errors.append(
            "TEMPERATURE_SCALING_MIN doit être < TEMPERATURE_SCALING_MAX"
        )
    if not (
        TEMPERATURE_SCALING_MIN
        <= TEMPERATURE_SCALING_DEFAULT
        <= TEMPERATURE_SCALING_MAX
    ):
        errors.append(
            "TEMPERATURE_SCALING_DEFAULT doit être dans "
            "[TEMPERATURE_SCALING_MIN, TEMPERATURE_SCALING_MAX]"
        )
    return errors


def _validate_blend_parameters() -> List[str]:
    errors: List[str] = []
    total = ODDS_BLEND_WEIGHT + MODEL_BLEND_WEIGHT
    if total <= 0:
        errors.append("ODDS_BLEND_WEIGHT + MODEL_BLEND_WEIGHT doit être > 0")
    if ODDS_BLEND_WEIGHT < 0 or ODDS_BLEND_WEIGHT > 1:
        errors.append(
            f"ODDS_BLEND_WEIGHT hors bornes (reçu {ODDS_BLEND_WEIGHT})"
        )
    if MODEL_BLEND_WEIGHT < 0 or MODEL_BLEND_WEIGHT > 1:
        errors.append(
            f"MODEL_BLEND_WEIGHT hors bornes (reçu {MODEL_BLEND_WEIGHT})"
        )
    return errors


def _validate_probability_parameters() -> List[str]:
    errors: List[str] = []
    if MAX_CONFIDENT_PROBABILITY <= MIN_CONFIDENT_PROBABILITY:
        errors.append(
            "MAX_CONFIDENT_PROBABILITY doit être > MIN_CONFIDENT_PROBABILITY"
        )
    if MAX_CONFIDENT_PROBABILITY > 1.0:
        errors.append("MAX_CONFIDENT_PROBABILITY doit être <= 1.0")
    if MIN_CONFIDENT_PROBABILITY < 0.0:
        errors.append("MIN_CONFIDENT_PROBABILITY doit être >= 0.0")
    if DRAW_FLOOR_PROBABILITY < 0.0 or DRAW_FLOOR_PROBABILITY > 0.5:
        errors.append(
            f"DRAW_FLOOR_PROBABILITY hors bornes (reçu {DRAW_FLOOR_PROBABILITY})"
        )
    if PROBABILITY_CLIP_EPS <= 0 or PROBABILITY_CLIP_EPS > 0.5:
        errors.append(
            f"PROBABILITY_CLIP_EPS hors bornes (reçu {PROBABILITY_CLIP_EPS})"
        )
    return errors


def _validate_ensemble_parameters() -> List[str]:
    errors: List[str] = []
    if ENSEMBLE_MIN_WEIGHT < 0 or ENSEMBLE_MIN_WEIGHT > 1:
        errors.append(
            f"ENSEMBLE_MIN_WEIGHT hors bornes (reçu {ENSEMBLE_MIN_WEIGHT})"
        )
    if ENSEMBLE_MAX_WEIGHT <= 0 or ENSEMBLE_MAX_WEIGHT > 1:
        errors.append(
            f"ENSEMBLE_MAX_WEIGHT hors bornes (reçu {ENSEMBLE_MAX_WEIGHT})"
        )
    if ENSEMBLE_MAX_WEIGHT < ENSEMBLE_MIN_WEIGHT:
        errors.append("ENSEMBLE_MAX_WEIGHT doit être >= ENSEMBLE_MIN_WEIGHT")
    if ENSEMBLE_DISAGREEMENT_THRESHOLD <= 0 or ENSEMBLE_DISAGREEMENT_THRESHOLD > 1:
        errors.append(
            f"ENSEMBLE_DISAGREEMENT_THRESHOLD hors bornes "
            f"(reçu {ENSEMBLE_DISAGREEMENT_THRESHOLD})"
        )
    return errors


def _validate_backtest_parameters() -> List[str]:
    errors: List[str] = []
    if BACKTEST_LAST_N_MATCHES < 10:
        errors.append("BACKTEST_LAST_N_MATCHES doit être >= 10")
    if BACKTEST_MIN_LOG_LOSS <= 0:
        errors.append("BACKTEST_MIN_LOG_LOSS doit être > 0")
    if BACKTEST_MIN_ACCURACY <= 0 or BACKTEST_MIN_ACCURACY >= 1:
        errors.append("BACKTEST_MIN_ACCURACY doit être dans (0, 1)")
    if BACKTEST_MIN_BRIER <= 0 or BACKTEST_MIN_BRIER >= 1:
        errors.append("BACKTEST_MIN_BRIER doit être dans (0, 1)")
    if BACKTEST_WALK_FORWARD_STEPS < 2:
        errors.append("BACKTEST_WALK_FORWARD_STEPS doit être >= 2")
    return errors


def _validate_feature_parameters() -> List[str]:
    errors: List[str] = []
    if FEATURE_MIN_COVERAGE <= 0 or FEATURE_MIN_COVERAGE > 1:
        errors.append(
            f"FEATURE_MIN_COVERAGE hors bornes (reçu {FEATURE_MIN_COVERAGE})"
        )
    if FEATURE_OUTLIER_ZSCORE <= 0:
        errors.append("FEATURE_OUTLIER_ZSCORE doit être > 0")
    return errors


def _validate_form_windows() -> List[str]:
    errors: List[str] = []
    if FORM_WINDOW_SHORT < 1:
        errors.append("FORM_WINDOW_SHORT doit être >= 1")
    if FORM_WINDOW_MEDIUM < FORM_WINDOW_SHORT:
        errors.append("FORM_WINDOW_MEDIUM doit être >= FORM_WINDOW_SHORT")
    if FORM_WINDOW_LONG < FORM_WINDOW_MEDIUM:
        errors.append("FORM_WINDOW_LONG doit être >= FORM_WINDOW_MEDIUM")
    return errors


def _validate_elo_parameters() -> List[str]:
    errors: List[str] = []
    if ELO_INITIAL_RATING <= 0:
        errors.append("ELO_INITIAL_RATING doit être > 0")
    if ELO_K_FACTOR <= 0 or ELO_K_FACTOR > 200:
        errors.append(f"ELO_K_FACTOR hors bornes (reçu {ELO_K_FACTOR})")
    if ELO_HOME_ADVANTAGE < 0:
        errors.append("ELO_HOME_ADVANTAGE doit être >= 0")
    if ELO_SEASON_REGRESSION < 0 or ELO_SEASON_REGRESSION > 1:
        errors.append(
            f"ELO_SEASON_REGRESSION hors bornes "
            f"(reçu {ELO_SEASON_REGRESSION})"
        )
    return errors


def _validate_training_parameters() -> List[str]:
    errors: List[str] = []
    if MIN_MATCHES_FOR_TRAINING < 50:
        errors.append("MIN_MATCHES_FOR_TRAINING doit être >= 50")
    if MIN_MATCHES_PER_CLASS < 10:
        errors.append("MIN_MATCHES_PER_CLASS doit être >= 10")
    if HISTORICAL_SEASONS_LOOKBACK < 1 or HISTORICAL_SEASONS_LOOKBACK > 30:
        errors.append(
            f"HISTORICAL_SEASONS_LOOKBACK hors bornes "
            f"(reçu {HISTORICAL_SEASONS_LOOKBACK})"
        )
    if MODEL_TRAIN_TEST_SPLIT <= 0 or MODEL_TRAIN_TEST_SPLIT >= 1:
        errors.append(
            f"MODEL_TRAIN_TEST_SPLIT hors bornes "
            f"(reçu {MODEL_TRAIN_TEST_SPLIT})"
        )
    if MODEL_WALK_FORWARD_FOLDS < 2 or MODEL_WALK_FORWARD_FOLDS > 30:
        errors.append(
            f"MODEL_WALK_FORWARD_FOLDS hors bornes "
            f"(reçu {MODEL_WALK_FORWARD_FOLDS})"
        )
    if MODEL_RANDOM_SEED < 0:
        errors.append("MODEL_RANDOM_SEED doit être >= 0")
    if MODEL_N_JOBS < -1:
        errors.append("MODEL_N_JOBS doit être >= -1")
    return errors


def _validate_cache_parameters() -> List[str]:
    errors: List[str] = []
    if _CACHE_DIR_ERROR is not None:
        errors.append(_CACHE_DIR_ERROR)
    elif not CACHE_DIR.exists():
        errors.append(f"Le répertoire de cache {CACHE_DIR} n'existe pas")
    elif not os.access(CACHE_DIR, os.W_OK):
        errors.append(
            f"Le répertoire de cache {CACHE_DIR} n'est pas accessible en écriture"
        )
    if not CACHE_NEVER_EXPIRES:
        errors.append(
            "CACHE_NEVER_EXPIRES doit rester True pour garantir la persistance"
        )
    return errors


def _validate_checkpoint_parameters() -> List[str]:
    errors: List[str] = []
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"Impossible de créer {CHECKPOINT_DIR} : {exc}")
    if not os.access(CHECKPOINT_DIR, os.W_OK):
        errors.append(
            f"Le répertoire checkpoint {CHECKPOINT_DIR} n'est pas accessible en écriture"
        )
    if CHECKPOINT_SAVE_INTERVAL_SECONDS < 5.0:
        errors.append("CHECKPOINT_SAVE_INTERVAL_SECONDS doit être >= 5.0")
    if CHECKPOINT_PUSH_INTERVAL_SECONDS < 60.0:
        errors.append("CHECKPOINT_PUSH_INTERVAL_SECONDS doit être >= 60.0")
    if CHECKPOINT_MAX_FILE_MB <= 0:
        errors.append("CHECKPOINT_MAX_FILE_MB doit être > 0")
    if GOANGEL_GITHUB_PUSH and not GH_PAT:
        errors.append(
            "GOANGEL_GITHUB_PUSH=true mais GH_PAT/GITHUB_TOKEN n'est pas défini"
        )
    return errors


def _validate_h2h_parameters() -> List[str]:
    errors: List[str] = []
    if H2H_MAX_MATCHES < 1 or H2H_MAX_MATCHES > 100:
        errors.append(f"H2H_MAX_MATCHES hors bornes (reçu {H2H_MAX_MATCHES})")
    if H2H_MAX_AGE_DAYS < 1:
        errors.append("H2H_MAX_AGE_DAYS doit être >= 1")
    if H2H_MIN_MATCHES_FOR_FEATURE < 1:
        errors.append("H2H_MIN_MATCHES_FOR_FEATURE doit être >= 1")
    return errors


def _validate_rest_days_parameters() -> List[str]:
    errors: List[str] = []
    if REST_DAYS_DEFAULT <= 0:
        errors.append("REST_DAYS_DEFAULT doit être > 0")
    if REST_DAYS_CAP <= REST_DAYS_DEFAULT:
        errors.append("REST_DAYS_CAP doit être > REST_DAYS_DEFAULT")
    return errors


def _validate_ensemble_and_hyperparams_consistency() -> List[str]:
    errors: List[str] = []
    if not MODEL_HYPERPARAMS:
        errors.append("MODEL_HYPERPARAMS est vide")
    for name, hp in MODEL_HYPERPARAMS.items():
        if not isinstance(hp, ModelHyperparams):
            errors.append(
                f"MODEL_HYPERPARAMS[{name}] n'est pas un ModelHyperparams"
            )
    if TOKENS_STRICT and TOKEN_COUNT < MIN_TOKENS_REQUIRED:
        errors.append(
            f"Au moins {MIN_TOKENS_REQUIRED} token(s) requis "
            f"(reçu {TOKEN_COUNT})"
        )
    return errors


def _aggregate_validation() -> List[str]:
    errors: List[str] = []
    errors.extend(_validate_ratios())
    errors.extend(_validate_api_parameters())
    errors.extend(_validate_bzzoiro_params())
    errors.extend(_validate_xgb_parameters())
    errors.extend(_validate_lgbm_parameters())
    errors.extend(_validate_catboost_parameters())
    errors.extend(_validate_rf_parameters())
    errors.extend(_validate_dl_parameters())
    errors.extend(_validate_bayesian_parameters())
    errors.extend(_validate_goal_grid())
    errors.extend(_validate_calibration_parameters())
    errors.extend(_validate_blend_parameters())
    errors.extend(_validate_probability_parameters())
    errors.extend(_validate_ensemble_parameters())
    errors.extend(_validate_backtest_parameters())
    errors.extend(_validate_feature_parameters())
    errors.extend(_validate_form_windows())
    errors.extend(_validate_elo_parameters())
    errors.extend(_validate_training_parameters())
    errors.extend(_validate_cache_parameters())
    errors.extend(_validate_checkpoint_parameters())
    errors.extend(_validate_h2h_parameters())
    errors.extend(_validate_rest_days_parameters())
    errors.extend(_validate_ensemble_and_hyperparams_consistency())
    return errors


def _compute_config_fingerprint() -> str:
    hasher = hashlib.sha256()
    parts: List[str] = [
        f"schema={_ARTIFACT_SCHEMA_VERSION}",
        f"features={_FEATURES_SCHEMA_COUNT}",
        f"classes={_EXPECTED_CLASS_COUNT}",
        f"max_goals={MAX_GOALS_GRID}",
        f"rho={DIXON_COLES_RHO}",
        f"calibration={CALIBRATION_METHOD}",
        f"temperature={TEMPERATURE_SCALING_DEFAULT}",
        f"ensemble_min={ENSEMBLE_MIN_WEIGHT}",
        f"ensemble_max={ENSEMBLE_MAX_WEIGHT}",
        "competitions=" + ",".join(
            sorted(COMPETITIONS, key=lambda c: int(c) if c.isdigit() else 0)
        ),
    ]
    for group_name in COMPETITION_GROUP_ORDER:
        codes = COMPETITION_GROUPS.get(group_name, [])
        parts.append(f"group:{group_name}={','.join(codes)}")
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _compute_feature_schema_fingerprint(feature_names: List[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"count={len(feature_names)}".encode("utf-8"))
    hasher.update(b"\x00")
    for name in feature_names:
        hasher.update(str(name).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


CONFIG_FINGERPRINT: str = _compute_config_fingerprint()
ARTIFACT_SCHEMA_VERSION: str = _ARTIFACT_SCHEMA_VERSION
EXPECTED_FEATURE_COUNT_REFERENCE: int = _FEATURES_SCHEMA_COUNT
EXPECTED_CLASS_COUNT: int = _EXPECTED_CLASS_COUNT


def compute_feature_schema_hash(feature_names: List[str]) -> str:
    return _compute_feature_schema_fingerprint(feature_names)


def write_cache_manifest(
    extra: Optional[Mapping[str, Any]] = None
) -> Optional[Path]:
    payload: Dict[str, Any] = {
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "feature_schema_count": _FEATURES_SCHEMA_COUNT,
        "class_count": _EXPECTED_CLASS_COUNT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "token_metadata": get_token_pool_metadata(),
        "competitions": list(COMPETITIONS),
        "source": "bzzoiro",
        "seasons_lookback": HISTORICAL_SEASONS_LOOKBACK,
        "enrich_xg": BZZOIRO_ENRICH_XG,
    }
    if extra:
        for key, value in extra.items():
            payload[str(key)] = value
    try:
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = MANIFEST_FILE.with_suffix(MANIFEST_FILE.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_path, MANIFEST_FILE)
        return MANIFEST_FILE
    except Exception as exc:
        logger.warning("Impossible d'écrire le manifeste de cache : %s", exc)
        return None


def read_cache_manifest() -> Optional[Dict[str, Any]]:
    if not MANIFEST_FILE.exists():
        return None
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("Manifeste de cache illisible : %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def validate_runtime_environment() -> None:
    errors = _aggregate_validation()
    if errors:
        for err in errors:
            logger.critical("Configuration invalide : %s", err)
        raise ConfigValidationError(
            "Configuration invalide : " + " | ".join(errors)
        )


validate_runtime_environment()

logger.info(
    "Configuration chargée : %d token(s), %d ligues Bzzoiro, cache=%s, "
    "ratios=%.2f/%.2f/%.2f/%.2f, calibration=%s, temperature=%.2f, "
    "cache_never_expires=%s, modèles=%d, checkpoint=%s, fingerprint=%s",
    TOKEN_COUNT,
    len(COMPETITIONS),
    CACHE_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    CALIB_RATIO,
    TEST_RATIO,
    CALIBRATION_METHOD,
    TEMPERATURE_SCALING_DEFAULT,
    CACHE_NEVER_EXPIRES,
    len(MODEL_HYPERPARAMS),
    CHECKPOINT_DIR,
    CONFIG_FINGERPRINT[:12],
)

PB_MAX_ATTEMPTS: int = _resolve_int("GOANGEL_PB_MAX_ATTEMPTS", 15, 1, 100)
PB_MAX_GOALS_FILTER: int = _resolve_int("GOANGEL_PB_MAX_GOALS_FILTER", 8, 1, 30)
PB_HALF_LIFE_DEFAULT: float = _resolve_float("GOANGEL_PB_HALF_LIFE_DEFAULT", 180.0, 7.0, 3650.0)
PB_HALF_LIFE_SOFT: float = _resolve_float("GOANGEL_PB_HALF_LIFE_SOFT", 365.0, 7.0, 3650.0)
PB_MIN_MATCHES_HARD_FLOOR: int = _resolve_int("GOANGEL_PB_MIN_MATCHES_HARD_FLOOR", 100, 10, 100000)

PB_BAYESIAN_N_SAMPLES: int = _resolve_int("GOANGEL_PB_BAYESIAN_N_SAMPLES", 500, 100, 100000)
PB_BAYESIAN_BURN: int = _resolve_int("GOANGEL_PB_BAYESIAN_BURN", 250, 50, 50000)
PB_BAYESIAN_N_CHAINS: int = _resolve_int("GOANGEL_PB_BAYESIAN_N_CHAINS", 2, 1, 16)
PB_BAYESIAN_THIN: int = _resolve_int("GOANGEL_PB_BAYESIAN_THIN", 2, 1, 100)
PB_BAYESIAN_N_CORES: int = _resolve_int("GOANGEL_PB_BAYESIAN_N_CORES", 2, 1, 128)

PB_TRUST_CONSTR_MAX_ITER: int = _resolve_int("GOANGEL_PB_TRUST_CONSTR_MAX_ITER", 5000, 100, 1000000)
PB_TRUST_CONSTR_GTOL: float = _resolve_float("GOANGEL_PB_TRUST_CONSTR_GTOL", 1e-6, 1e-12, 1.0)
PB_TRUST_CONSTR_XTOL: float = _resolve_float("GOANGEL_PB_TRUST_CONSTR_XTOL", 1e-8, 1e-12, 1.0)

PB_SLSQP_MAX_ITER: int = _resolve_int("GOANGEL_PB_SLSQP_MAX_ITER", 5000, 100, 1000000)
PB_SLSQP_FTOL: float = _resolve_float("GOANGEL_PB_SLSQP_FTOL", 1e-6, 1e-12, 1.0)

PB_WEIGHT_MIN: float = _resolve_float("GOANGEL_PB_WEIGHT_MIN", 1e-8, 1e-15, 1.0)
PB_WEIGHT_MAX: float = _resolve_float("GOANGEL_PB_WEIGHT_MAX", 1e6, 1.0, 1e12)

PB_HISTORY_MAX_ENTRIES: int = _resolve_int("GOANGEL_PB_HISTORY_MAX_ENTRIES", 50, 1, 1000)
PB_CHECKPOINTS_KEPT: int = _resolve_int("GOANGEL_PB_CHECKPOINTS_KEPT", 3, 1, 50)

PB_FALLBACK_CHAIN: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    (
        "trust-constr",
        {
            "maxiter": PB_TRUST_CONSTR_MAX_ITER,
            "gtol": PB_TRUST_CONSTR_GTOL,
            "xtol": PB_TRUST_CONSTR_XTOL,
            "barrier_tol": 1e-8,
            "initial_tr_radius": 1.0,
            "verbose": 0,
        },
    ),
    (
        "Powell",
        {
            "maxiter": PB_TRUST_CONSTR_MAX_ITER,
            "xtol": PB_TRUST_CONSTR_XTOL,
            "ftol": PB_TRUST_CONSTR_GTOL,
        },
    ),
    (
        "L-BFGS-B",
        {
            "maxiter": PB_TRUST_CONSTR_MAX_ITER,
            "ftol": PB_TRUST_CONSTR_XTOL,
            "gtol": PB_TRUST_CONSTR_GTOL,
            "maxls": 50,
        },
    ),
)

PB_FILTER_PLAN: Dict[int, Dict[str, Any]] = {
    1:  {"min_matches": 0,  "max_goals": 100,                   "half_life": PB_HALF_LIFE_DEFAULT},
    2:  {"min_matches": 0,  "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_DEFAULT},
    3:  {"min_matches": 4,  "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
    4:  {"min_matches": 6,  "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
    5:  {"min_matches": 8,  "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
    6:  {"min_matches": 10, "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
    7:  {"min_matches": 12, "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
    8:  {"min_matches": 15, "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
    15: {"min_matches": 20, "max_goals": PB_MAX_GOALS_FILTER,   "half_life": PB_HALF_LIFE_SOFT},
}


def _validate_pb_parameters() -> List[str]:
    errors: List[str] = []
    if PB_MAX_ATTEMPTS < 1 or PB_MAX_ATTEMPTS > 100:
        errors.append(f"PB_MAX_ATTEMPTS hors bornes (reçu {PB_MAX_ATTEMPTS})")
    if PB_MAX_GOALS_FILTER < 1 or PB_MAX_GOALS_FILTER > 30:
        errors.append(f"PB_MAX_GOALS_FILTER hors bornes (reçu {PB_MAX_GOALS_FILTER})")
    if PB_HALF_LIFE_DEFAULT <= 0 or PB_HALF_LIFE_DEFAULT > 3650:
        errors.append(f"PB_HALF_LIFE_DEFAULT hors bornes (reçu {PB_HALF_LIFE_DEFAULT})")
    if PB_HALF_LIFE_SOFT <= 0 or PB_HALF_LIFE_SOFT > 3650:
        errors.append(f"PB_HALF_LIFE_SOFT hors bornes (reçu {PB_HALF_LIFE_SOFT})")
    if PB_HALF_LIFE_SOFT < PB_HALF_LIFE_DEFAULT:
        errors.append("PB_HALF_LIFE_SOFT doit être >= PB_HALF_LIFE_DEFAULT")
    if PB_MIN_MATCHES_HARD_FLOOR < 10 or PB_MIN_MATCHES_HARD_FLOOR > 100000:
        errors.append(f"PB_MIN_MATCHES_HARD_FLOOR hors bornes (reçu {PB_MIN_MATCHES_HARD_FLOOR})")
    if PB_BAYESIAN_N_SAMPLES < 100 or PB_BAYESIAN_N_SAMPLES > 100000:
        errors.append(f"PB_BAYESIAN_N_SAMPLES hors bornes (reçu {PB_BAYESIAN_N_SAMPLES})")
    if PB_BAYESIAN_BURN < 50 or PB_BAYESIAN_BURN > 50000:
        errors.append(f"PB_BAYESIAN_BURN hors bornes (reçu {PB_BAYESIAN_BURN})")
    if PB_BAYESIAN_BURN >= PB_BAYESIAN_N_SAMPLES:
        errors.append("PB_BAYESIAN_BURN doit être < PB_BAYESIAN_N_SAMPLES")
    if PB_BAYESIAN_N_CHAINS < 1 or PB_BAYESIAN_N_CHAINS > 16:
        errors.append(f"PB_BAYESIAN_N_CHAINS hors bornes (reçu {PB_BAYESIAN_N_CHAINS})")
    if PB_BAYESIAN_THIN < 1 or PB_BAYESIAN_THIN > 100:
        errors.append(f"PB_BAYESIAN_THIN hors bornes (reçu {PB_BAYESIAN_THIN})")
    if PB_BAYESIAN_N_CORES < 1 or PB_BAYESIAN_N_CORES > 128:
        errors.append(f"PB_BAYESIAN_N_CORES hors bornes (reçu {PB_BAYESIAN_N_CORES})")
    if PB_TRUST_CONSTR_MAX_ITER < 100:
        errors.append("PB_TRUST_CONSTR_MAX_ITER doit être >= 100")
    if PB_TRUST_CONSTR_GTOL <= 0 or PB_TRUST_CONSTR_GTOL > 1.0:
        errors.append(f"PB_TRUST_CONSTR_GTOL hors bornes (reçu {PB_TRUST_CONSTR_GTOL})")
    if PB_TRUST_CONSTR_XTOL <= 0 or PB_TRUST_CONSTR_XTOL > 1.0:
        errors.append(f"PB_TRUST_CONSTR_XTOL hors bornes (reçu {PB_TRUST_CONSTR_XTOL})")
    if PB_SLSQP_MAX_ITER < 100:
        errors.append("PB_SLSQP_MAX_ITER doit être >= 100")
    if PB_SLSQP_FTOL <= 0 or PB_SLSQP_FTOL > 1.0:
        errors.append(f"PB_SLSQP_FTOL hors bornes (reçu {PB_SLSQP_FTOL})")
    if PB_WEIGHT_MIN <= 0:
        errors.append("PB_WEIGHT_MIN doit être > 0")
    if PB_WEIGHT_MAX <= PB_WEIGHT_MIN:
        errors.append("PB_WEIGHT_MAX doit être > PB_WEIGHT_MIN")
    if PB_HISTORY_MAX_ENTRIES < 1 or PB_HISTORY_MAX_ENTRIES > 1000:
        errors.append(f"PB_HISTORY_MAX_ENTRIES hors bornes (reçu {PB_HISTORY_MAX_ENTRIES})")
    if PB_CHECKPOINTS_KEPT < 1 or PB_CHECKPOINTS_KEPT > 50:
        errors.append(f"PB_CHECKPOINTS_KEPT hors bornes (reçu {PB_CHECKPOINTS_KEPT})")
    if not isinstance(PB_FALLBACK_CHAIN, tuple) or len(PB_FALLBACK_CHAIN) == 0:
        errors.append("PB_FALLBACK_CHAIN doit être un tuple non vide")
    else:
        for _entry in PB_FALLBACK_CHAIN:
            if not isinstance(_entry, tuple) or len(_entry) != 2:
                errors.append("PB_FALLBACK_CHAIN : chaque entrée doit être (method, options)")
                break
            _m, _o = _entry
            if not isinstance(_m, str) or not _m:
                errors.append("PB_FALLBACK_CHAIN : méthode invalide")
                break
            if not isinstance(_o, dict):
                errors.append("PB_FALLBACK_CHAIN : options invalides")
                break
    if not isinstance(PB_FILTER_PLAN, dict) or len(PB_FILTER_PLAN) == 0:
        errors.append("PB_FILTER_PLAN doit être un dict non vide")
    else:
        for _k, _v in PB_FILTER_PLAN.items():
            if not isinstance(_k, int) or _k < 1 or _k > PB_MAX_ATTEMPTS:
                errors.append(f"PB_FILTER_PLAN : clé invalide {_k}")
                break
            if not isinstance(_v, dict):
                errors.append(f"PB_FILTER_PLAN[{_k}] doit être un dict")
                break
            for _field in ("min_matches", "max_goals", "half_life"):
                if _field not in _v:
                    errors.append(f"PB_FILTER_PLAN[{_k}] : champ {_field} manquant")
                    break
    return errors


_PB_PATCH_ERRORS: List[str] = _validate_pb_parameters()
if _PB_PATCH_ERRORS:
    for _err_pb in _PB_PATCH_ERRORS:
        logger.critical("Configuration PB invalide : %s", _err_pb)
    raise ConfigValidationError(
        "Configuration PB invalide : " + " | ".join(_PB_PATCH_ERRORS)
    )


def get_pb_runtime_config() -> Dict[str, Any]:
    return {
        "max_attempts": int(PB_MAX_ATTEMPTS),
        "max_goals_filter": int(PB_MAX_GOALS_FILTER),
        "half_life_default": float(PB_HALF_LIFE_DEFAULT),
        "half_life_soft": float(PB_HALF_LIFE_SOFT),
        "min_matches_hard_floor": int(PB_MIN_MATCHES_HARD_FLOOR),
        "bayesian": {
            "n_samples": int(PB_BAYESIAN_N_SAMPLES),
            "burn": int(PB_BAYESIAN_BURN),
            "n_chains": int(PB_BAYESIAN_N_CHAINS),
            "thin": int(PB_BAYESIAN_THIN),
            "n_cores": int(PB_BAYESIAN_N_CORES),
        },
        "trust_constr": {
            "max_iter": int(PB_TRUST_CONSTR_MAX_ITER),
            "gtol": float(PB_TRUST_CONSTR_GTOL),
            "xtol": float(PB_TRUST_CONSTR_XTOL),
        },
        "slsqp": {
            "max_iter": int(PB_SLSQP_MAX_ITER),
            "ftol": float(PB_SLSQP_FTOL),
        },
        "weights": {
            "min": float(PB_WEIGHT_MIN),
            "max": float(PB_WEIGHT_MAX),
        },
        "history_max_entries": int(PB_HISTORY_MAX_ENTRIES),
        "checkpoints_kept": int(PB_CHECKPOINTS_KEPT),
        "fallback_chain": [m for m, _ in PB_FALLBACK_CHAIN],
    }


def get_pb_filter_for_attempt(attempt: int) -> Dict[str, Any]:
    try:
        a = int(attempt)
    except (TypeError, ValueError):
        a = 1
    if a < 1:
        a = 1
    if a in PB_FILTER_PLAN:
        spec = PB_FILTER_PLAN[a]
        return {
            "min_matches": int(spec["min_matches"]),
            "max_goals": int(spec["max_goals"]),
            "half_life": float(spec["half_life"]),
        }
    last_key = max(PB_FILTER_PLAN.keys())
    spec = PB_FILTER_PLAN[last_key]
    return {
        "min_matches": int(spec["min_matches"]),
        "max_goals": int(spec["max_goals"]),
        "half_life": float(spec["half_life"]),
    }


logger.info(
    "PB convergence patch activé : max_attempts=%d, half_life=%.0f→%.0f, "
    "bayesian=(n=%d, burn=%d, chains=%d, thin=%d, cores=%d), "
    "filtre max_goals=%d, floor=%d, fallback=%s",
    PB_MAX_ATTEMPTS,
    PB_HALF_LIFE_DEFAULT,
    PB_HALF_LIFE_SOFT,
    PB_BAYESIAN_N_SAMPLES,
    PB_BAYESIAN_BURN,
    PB_BAYESIAN_N_CHAINS,
    PB_BAYESIAN_THIN,
    PB_BAYESIAN_N_CORES,
    PB_MAX_GOALS_FILTER,
    PB_MIN_MATCHES_HARD_FLOOR,
    [m for m, _ in PB_FALLBACK_CHAIN],
)
