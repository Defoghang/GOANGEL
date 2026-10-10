import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

_TOKEN_SEPARATOR_RE = re.compile(r"[,;\s]+")


def _split_token_string(raw: str) -> List[str]:
    if not raw:
        return []
    parts = _TOKEN_SEPARATOR_RE.split(str(raw))
    out: List[str] = []
    for p in parts:
        v = p.strip().strip('"').strip("'").strip()
        if v:
            out.append(v)
    return out


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
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


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y", "o")


def load_environment() -> None:
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    else:
        load_dotenv(override=False)


load_environment()


def get_bzzoiro_tokens() -> List[str]:
    raw_pool = os.getenv("BZZOIRO_TOKENS", "").strip()
    raw_single = os.getenv("BZZOIRO_TOKEN", "").strip()
    raw_key = os.getenv("BZZOIRO_API_KEY", "").strip()

    candidates: List[str] = []
    candidates.extend(_split_token_string(raw_pool))
    candidates.extend(_split_token_string(raw_single))
    candidates.extend(_split_token_string(raw_key))

    seen: set = set()
    unique: List[str] = []
    for t in candidates:
        if len(t) < 10 or len(t) > 512:
            continue
        if any(ch.isspace() for ch in t):
            continue
        if t in seen:
            continue
        seen.add(t)
        unique.append(t)
    return unique


def get_workers_per_key() -> int:
    return _env_int("BZZOIRO_WORKERS_PER_KEY", 2, 1, 32)


def get_max_workers() -> int:
    return _env_int("BZZOIRO_MAX_WORKERS", 32, 1, 256)


def compute_total_workers(n_tokens: int) -> int:
    if n_tokens <= 0:
        return 1
    per_key = get_workers_per_key()
    cap = get_max_workers()
    return max(1, min(n_tokens * per_key, cap))


def get_rate_limit() -> Tuple[int, int]:
    limit = _env_int("BZZOIRO_RATE_LIMIT_PER_MINUTE", 10, 1, 600)
    margin = _env_int("BZZOIRO_RATE_LIMIT_SAFETY_MARGIN", 1, 0, 100)
    return limit, margin


def get_historical_seasons_lookback() -> int:
    return _env_int("BZZOIRO_HISTORICAL_SEASONS_LOOKBACK", 3, 1, 30)


def get_enrich_xg() -> bool:
    return _env_bool("BZZOIRO_ENRICH_XG", True)


def get_strict_tokens() -> bool:
    return _env_bool("GOANGEL_STRICT_TOKENS", True)


def get_cache_dir() -> Path:
    raw = os.getenv("GOANGEL_CACHE_DIR", "").strip()
    if not raw:
        return (BASE_DIR / "cache").resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


def get_log_level() -> str:
    return os.getenv("GOANGEL_LOG_LEVEL", "INFO").strip().upper() or "INFO"


def get_log_file() -> str:
    raw = os.getenv("GOANGEL_LOG_FILE", "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return str(p)


def get_random_seed() -> int:
    return _env_int("GOANGEL_RANDOM_SEED", 42, 0, 2**31 - 1)


def get_timezone() -> str:
    raw = os.getenv("GOANGEL_TIMEZONE", "UTC").strip()
    return raw if raw else "UTC"


def get_calibration_method() -> str:
    raw = os.getenv("GOANGEL_CALIBRATION_METHOD", "isotonic").strip().lower()
    if raw not in ("isotonic", "sigmoid"):
        return "isotonic"
    return raw


def get_temperature_default() -> float:
    return _env_float("GOANGEL_TEMPERATURE_DEFAULT", 1.15, 0.5, 5.0)


def get_ensemble_weight_range() -> Tuple[float, float]:
    lo = _env_float("GOANGEL_ENSEMBLE_MIN_WEIGHT", 0.0, 0.0, 1.0)
    hi = _env_float("GOANGEL_ENSEMBLE_MAX_WEIGHT", 0.60, 0.0, 1.0)
    if hi < lo:
        hi = lo
    return lo, hi


def get_min_matches_for_training() -> int:
    return _env_int("GOANGEL_MIN_MATCHES_FOR_TRAINING", 500, 50, 1_000_000)


def get_split_ratios() -> Tuple[float, float, float, float]:
    tr = _env_float("GOANGEL_TRAIN_RATIO", 0.65, 0.01, 0.99)
    va = _env_float("GOANGEL_VAL_RATIO", 0.10, 0.00, 0.99)
    ca = _env_float("GOANGEL_CALIB_RATIO", 0.10, 0.00, 0.99)
    te = _env_float("GOANGEL_TEST_RATIO", 0.15, 0.01, 0.99)
    total = tr + va + ca + te
    if total <= 0:
        return 0.65, 0.10, 0.10, 0.15
    return tr / total, va / total, ca / total, te / total


def get_walk_forward_folds() -> int:
    return _env_int("GOANGEL_WALK_FORWARD_FOLDS", 5, 2, 20)


def diagnostics() -> Dict[str, object]:
    tokens = get_bzzoiro_tokens()
    n = len(tokens)
    per_key = get_workers_per_key()
    cap = get_max_workers()
    total = compute_total_workers(n)
    rate, margin = get_rate_limit()
    return {
        "tokens": n,
        "workers_per_key": per_key,
        "max_workers": cap,
        "total_workers": total,
        "rate_limit_per_minute": rate,
        "rate_limit_safety_margin": margin,
        "seasons_lookback": get_historical_seasons_lookback(),
        "enrich_xg": get_enrich_xg(),
        "strict_tokens": get_strict_tokens(),
        "cache_dir": str(get_cache_dir()),
        "log_level": get_log_level(),
        "timezone": get_timezone(),
        "calibration": get_calibration_method(),
        "temperature_default": get_temperature_default(),
        "min_matches_for_training": get_min_matches_for_training(),
        "walk_forward_folds": get_walk_forward_folds(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(diagnostics(), indent=2, ensure_ascii=False))
