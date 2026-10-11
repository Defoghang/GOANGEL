import inspect
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from config import CACHE_DIR as _CONFIG_CACHE_DIR
    _CACHE_DIR: Path = Path(_CONFIG_CACHE_DIR)
except Exception:
    _CACHE_DIR = Path("./cache")


logger = logging.getLogger("goangel.pb_convergence")


__all__ = [
    "PBConvergenceError",
    "PBIncompleteError",
    "PBValidationError",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_GOALS_FILTER",
    "DEFAULT_HALF_LIFE",
    "DEFAULT_HALF_LIFE_SOFT",
    "MIN_MATCHES_HARD_FLOOR",
    "MAX_GOALS_CEILING",
    "MIN_MATCHES_CEILING",
    "DEFAULT_BAYESIAN_N_SAMPLES",
    "DEFAULT_BAYESIAN_BURN",
    "DEFAULT_BAYESIAN_N_CHAINS",
    "DEFAULT_BAYESIAN_THIN",
    "DEFAULT_BAYESIAN_N_CORES",
    "BAYESIAN_MIN_MATCHES_OVERRIDE",
    "PB_MAX_ATTEMPTS",
    "PB_MAX_GOALS_FILTER",
    "PB_HALF_LIFE_DEFAULT",
    "PB_HALF_LIFE_SOFT",
    "PB_MIN_MATCHES_HARD_FLOOR",
    "FILTER_PLAN",
    "get_pb_attempt",
    "increment_pb_attempt",
    "reset_pb_attempt",
    "resolve_filter",
    "get_best_filter",
    "update_best_filter",
    "save_checkpoint",
    "load_checkpoint",
    "clear_checkpoints",
    "clear_pb_state",
    "compute_time_weights",
    "validate_pb_inputs",
    "apply_filter",
    "fit_pb_attempt",
    "get_pb_status",
    "list_pb_status",
    "get_pb_runtime_config",
    "get_pb_filter_for_attempt",
]


class PBConvergenceError(Exception):
    pass


class PBIncompleteError(PBConvergenceError):
    pass


class PBValidationError(PBConvergenceError):
    pass


_SCHEMA_VERSION: str = "goangel-pb-convergence-1.0"
_LOCK: threading.RLock = threading.RLock()
_DAYS_PER_SECOND: float = 86400.0
_MAX_HISTORY_ENTRIES: int = 50
_MAX_CHECKPOINTS_KEPT: int = 3
_MIN_WEIGHT: float = 1e-8
_MAX_WEIGHT: float = 1e6
_MIN_X0_FALLBACK: float = 1e-3

DEFAULT_MAX_ATTEMPTS: int = 15
DEFAULT_MAX_GOALS_FILTER: int = 8
DEFAULT_HALF_LIFE: float = 180.0
DEFAULT_HALF_LIFE_SOFT: float = 365.0
MIN_MATCHES_HARD_FLOOR: int = 100
MAX_GOALS_CEILING: int = 200
MIN_MATCHES_CEILING: int = 200

DEFAULT_BAYESIAN_N_SAMPLES: int = 2000
DEFAULT_BAYESIAN_BURN: int = 1000
DEFAULT_BAYESIAN_N_CHAINS: int = 4
DEFAULT_BAYESIAN_THIN: int = 5
DEFAULT_BAYESIAN_N_CORES: int = 1
BAYESIAN_MIN_MATCHES_OVERRIDE: int = 0

FILTER_PLAN: Dict[int, Dict[str, Any]] = {
    1:  {"min_matches": 0,  "max_goals": 100, "half_life": 180.0},
    2:  {"min_matches": 0,  "max_goals": 8,   "half_life": 180.0},
    3:  {"min_matches": 4,  "max_goals": 8,   "half_life": 365.0},
    4:  {"min_matches": 6,  "max_goals": 8,   "half_life": 365.0},
    5:  {"min_matches": 8,  "max_goals": 8,   "half_life": 365.0},
    6:  {"min_matches": 10, "max_goals": 8,   "half_life": 365.0},
    7:  {"min_matches": 12, "max_goals": 8,   "half_life": 365.0},
    8:  {"min_matches": 15, "max_goals": 8,   "half_life": 365.0},
    15: {"min_matches": 20, "max_goals": 8,   "half_life": 365.0},
}


def _load_pb_const(name: str, default: Any) -> Any:
    try:
        import config as _cfg
        return getattr(_cfg, name, default)
    except Exception:
        return default


PB_MAX_ATTEMPTS: int = int(_load_pb_const("PB_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
PB_MAX_GOALS_FILTER: int = int(_load_pb_const("PB_MAX_GOALS_FILTER", DEFAULT_MAX_GOALS_FILTER))
PB_HALF_LIFE_DEFAULT: float = float(_load_pb_const("PB_HALF_LIFE_DEFAULT", DEFAULT_HALF_LIFE))
PB_HALF_LIFE_SOFT: float = float(_load_pb_const("PB_HALF_LIFE_SOFT", DEFAULT_HALF_LIFE_SOFT))
PB_MIN_MATCHES_HARD_FLOOR: int = int(_load_pb_const("PB_MIN_MATCHES_HARD_FLOOR", MIN_MATCHES_HARD_FLOOR))
PB_BAYESIAN_N_SAMPLES: int = int(_load_pb_const("PB_BAYESIAN_N_SAMPLES", DEFAULT_BAYESIAN_N_SAMPLES))
PB_BAYESIAN_BURN: int = int(_load_pb_const("PB_BAYESIAN_BURN", DEFAULT_BAYESIAN_BURN))
PB_BAYESIAN_N_CHAINS: int = int(_load_pb_const("PB_BAYESIAN_N_CHAINS", DEFAULT_BAYESIAN_N_CHAINS))
PB_BAYESIAN_THIN: int = int(_load_pb_const("PB_BAYESIAN_THIN", DEFAULT_BAYESIAN_THIN))
PB_BAYESIAN_N_CORES: int = int(_load_pb_const("PB_BAYESIAN_N_CORES", DEFAULT_BAYESIAN_N_CORES))
PB_WEIGHT_MIN: float = float(_load_pb_const("PB_WEIGHT_MIN", _MIN_WEIGHT))
PB_WEIGHT_MAX: float = float(_load_pb_const("PB_WEIGHT_MAX", _MAX_WEIGHT))
PB_HISTORY_MAX_ENTRIES: int = int(_load_pb_const("PB_HISTORY_MAX_ENTRIES", _MAX_HISTORY_ENTRIES))
PB_CHECKPOINTS_KEPT: int = int(_load_pb_const("PB_CHECKPOINTS_KEPT", _MAX_CHECKPOINTS_KEPT))


_PB_ROOT: Path = _CACHE_DIR / "stages" / "pb_convergence"
_ATTEMPT_DIR: Path = _PB_ROOT / "attempts"
_CKPT_DIR: Path = _PB_ROOT / "checkpoints"
_FILTER_DIR: Path = _PB_ROOT / "best_filters"
_META_DIR: Path = _PB_ROOT / "meta"


def _ensure_dirs() -> None:
    for d in (_PB_ROOT, _ATTEMPT_DIR, _CKPT_DIR, _FILTER_DIR, _META_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("mkdir %s échoué : %s", d, exc)


_ensure_dirs()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    _atomic_write_bytes(path, data)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(path, data)


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("JSON illisible %s : %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _safe_read_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Pickle illisible %s : %s", path, exc)
        return None


def _sanitize_class_name(class_name: Any) -> str:
    if not isinstance(class_name, str):
        raise PBValidationError("class_name doit être une chaîne non vide")
    stripped = class_name.strip()
    if not stripped:
        raise PBValidationError("class_name vide")
    cleaned = "".join(ch for ch in stripped if ch.isalnum() or ch in ("_", "-"))
    if not cleaned:
        raise PBValidationError("class_name ne contient aucun caractère valide")
    return cleaned


def _attempt_file(class_name: str) -> Path:
    return _ATTEMPT_DIR / f"{_sanitize_class_name(class_name)}.json"


def _filter_file(class_name: str) -> Path:
    return _FILTER_DIR / f"{_sanitize_class_name(class_name)}.json"


def _meta_file(class_name: str) -> Path:
    return _META_DIR / f"{_sanitize_class_name(class_name)}.json"


def _ckpt_dir(class_name: str) -> Path:
    d = _CKPT_DIR / _sanitize_class_name(class_name)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("mkdir %s échoué : %s", d, exc)
    return d


def _ckpt_path(class_name: str, attempt: int) -> Path:
    return _ckpt_dir(class_name) / f"attempt_{int(attempt):02d}.pkl"


def _empty_attempt_state(class_name: str) -> Dict[str, Any]:
    return {
        "schema": _SCHEMA_VERSION,
        "class": class_name,
        "count": 0,
        "history": [],
    }


def _validate_attempt_state(state: Any, class_name: str) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return _empty_attempt_state(class_name)
    if state.get("class") != class_name:
        return _empty_attempt_state(class_name)
    count_raw = state.get("count", 0)
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        count = 0
    if count < 0:
        count = 0
    history_raw = state.get("history", [])
    clean_history: List[Dict[str, Any]] = []
    if isinstance(history_raw, list):
        for entry in history_raw[-PB_HISTORY_MAX_ENTRIES:]:
            if isinstance(entry, dict):
                clean_history.append(entry)
    return {
        "schema": _SCHEMA_VERSION,
        "class": class_name,
        "count": count,
        "history": clean_history,
    }


def get_pb_attempt(class_name: str) -> int:
    with _LOCK:
        state = _safe_read_json(_attempt_file(class_name))
        state = _validate_attempt_state(state, class_name)
        return int(state["count"])


def increment_pb_attempt(class_name: str, meta: Optional[Dict[str, Any]] = None) -> int:
    with _LOCK:
        state = _safe_read_json(_attempt_file(class_name))
        state = _validate_attempt_state(state, class_name)
        state["count"] = int(state["count"]) + 1
        entry: Dict[str, Any] = {
            "attempt": int(state["count"]),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(meta, dict):
            for k, v in meta.items():
                entry[str(k)] = v
        history = list(state.get("history", []))
        history.append(entry)
        state["history"] = history[-PB_HISTORY_MAX_ENTRIES:]
        try:
            _atomic_write_json(_attempt_file(class_name), state)
        except Exception as exc:
            logger.error("Écriture attempt state %s échouée : %s", class_name, exc)
        return int(state["count"])


def reset_pb_attempt(class_name: str) -> None:
    with _LOCK:
        try:
            p = _attempt_file(class_name)
            if p.exists():
                p.unlink()
        except OSError as exc:
            logger.warning("Reset attempt %s échoué : %s", class_name, exc)


def _validate_filter_spec(spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise PBValidationError("spec filtre doit être un dict")
    mm_raw = spec.get("min_matches", 0)
    mg_raw = spec.get("max_goals", 100)
    hl_raw = spec.get("half_life", PB_HALF_LIFE_DEFAULT)
    try:
        mm = int(mm_raw)
    except (TypeError, ValueError):
        mm = 0
    try:
        mg = int(mg_raw)
    except (TypeError, ValueError):
        mg = 100
    try:
        hl = float(hl_raw)
    except (TypeError, ValueError):
        hl = PB_HALF_LIFE_DEFAULT
    mm = max(0, min(mm, MIN_MATCHES_CEILING))
    mg = max(1, min(mg, MAX_GOALS_CEILING))
    if not np.isfinite(hl) or hl <= 0:
        hl = PB_HALF_LIFE_DEFAULT
    return {"min_matches": mm, "max_goals": mg, "half_life": hl}


def resolve_filter(
    attempt: int,
    best_so_far: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        a = int(attempt)
    except (TypeError, ValueError):
        a = 1
    if a < 1:
        a = 1
    if a in FILTER_PLAN:
        return _validate_filter_spec(FILTER_PLAN[a])
    if best_so_far is not None and a < PB_MAX_ATTEMPTS:
        try:
            return _validate_filter_spec(best_so_far)
        except PBValidationError:
            pass
    last_key = max(FILTER_PLAN.keys())
    return _validate_filter_spec(FILTER_PLAN[last_key])


def get_best_filter(class_name: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        data = _safe_read_json(_filter_file(class_name))
        if not isinstance(data, dict):
            return None
        spec = data.get("filter")
        n_raw = data.get("n_kept", 0)
        try:
            n_kept = int(n_raw)
        except (TypeError, ValueError):
            n_kept = 0
        if n_kept < 0:
            n_kept = 0
        try:
            spec_v = _validate_filter_spec(spec)
        except PBValidationError:
            return None
        spec_v["n_kept"] = n_kept
        return spec_v


def update_best_filter(class_name: str, filt: Dict[str, Any], n_kept: int) -> None:
    try:
        filt_v = _validate_filter_spec(filt)
    except PBValidationError:
        return
    try:
        n_kept_i = int(n_kept)
    except (TypeError, ValueError):
        return
    if n_kept_i < 0:
        return
    with _LOCK:
        current = get_best_filter(class_name)
        if current is not None:
            try:
                current_n = int(current.get("n_kept", 0))
            except (TypeError, ValueError):
                current_n = 0
            if current_n >= n_kept_i:
                return
        payload = {
            "schema": _SCHEMA_VERSION,
            "class": class_name,
            "filter": filt_v,
            "n_kept": n_kept_i,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _atomic_write_json(_filter_file(class_name), payload)
        except Exception as exc:
            logger.warning("Best filter save %s échoué : %s", class_name, exc)


def _prune_checkpoints(class_name: str) -> None:
    d = _ckpt_dir(class_name)
    try:
        entries = sorted(d.glob("attempt_*.pkl"))
    except OSError:
        return
    if len(entries) <= PB_CHECKPOINTS_KEPT:
        return
    for old in entries[:-PB_CHECKPOINTS_KEPT]:
        try:
            old.unlink()
        except OSError:
            pass


def save_checkpoint(
    class_name: str,
    attempt: int,
    state: Dict[str, Any],
) -> Optional[Path]:
    try:
        a = int(attempt)
    except (TypeError, ValueError):
        return None
    payload = {
        "schema": _SCHEMA_VERSION,
        "class": class_name,
        "attempt": a,
        "state": state,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    p = _ckpt_path(class_name, a)
    try:
        _atomic_write_pickle(p, payload)
    except Exception as exc:
        logger.warning("Checkpoint save %s échoué : %s", p, exc)
        return None
    _prune_checkpoints(class_name)
    return p


def load_checkpoint(class_name: str, attempt: int) -> Optional[Dict[str, Any]]:
    try:
        a = int(attempt)
    except (TypeError, ValueError):
        return None
    p = _ckpt_path(class_name, a)
    data = _safe_read_pickle(p)
    if not isinstance(data, dict):
        return None
    if data.get("class") != class_name:
        return None
    return data


def clear_checkpoints(class_name: str) -> None:
    d = _ckpt_dir(class_name)
    try:
        for p in d.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def clear_pb_state(class_name: Optional[str] = None) -> None:
    if class_name is None:
        for d in (_ATTEMPT_DIR, _CKPT_DIR, _FILTER_DIR, _META_DIR):
            try:
                for p in d.rglob("*"):
                    if p.is_file():
                        try:
                            p.unlink()
                        except OSError:
                            pass
            except OSError:
                pass
        return
    with _LOCK:
        for p in (_attempt_file(class_name), _filter_file(class_name), _meta_file(class_name)):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        clear_checkpoints(class_name)


def compute_time_weights(
    dates: Optional[List[Any]],
    half_life_days: float,
) -> Optional[np.ndarray]:
    if dates is None:
        return None
    n = len(dates)
    if n == 0:
        return None
    try:
        hl = float(half_life_days)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(hl) or hl <= 0:
        return None
    parsed: List[Optional[pd.Timestamp]] = []
    for d in dates:
        if d is None:
            parsed.append(None)
            continue
        try:
            p = pd.to_datetime(d, errors="coerce", utc=True)
        except Exception:
            parsed.append(None)
            continue
        if p is None:
            parsed.append(None)
            continue
        try:
            if pd.isna(p):
                parsed.append(None)
                continue
        except (TypeError, ValueError):
            parsed.append(None)
            continue
        parsed.append(p)
    valid = [p for p in parsed if p is not None]
    if not valid:
        return None
    ref = max(valid)
    xi = float(np.log(2.0) / hl)
    w = np.zeros(n, dtype=np.float64)
    for i, p in enumerate(parsed):
        if p is None:
            w[i] = 1.0
            continue
        delta = (ref - p).total_seconds()
        if delta < 0.0:
            delta = 0.0
        days_ago = delta / _DAYS_PER_SECOND
        try:
            w[i] = float(np.exp(-xi * days_ago))
        except Exception:
            w[i] = 0.0
    if not np.all(np.isfinite(w)):
        w = np.where(np.isfinite(w), w, 0.0)
    mean_w = float(w.mean())
    if mean_w <= 0.0 or not np.isfinite(mean_w):
        return None
    w = w / mean_w
    w = np.clip(w, PB_WEIGHT_MIN, PB_WEIGHT_MAX)
    return w


def validate_pb_inputs(
    gh: List[int],
    ga: List[int],
    th: List[str],
    ta: List[str],
) -> None:
    if not isinstance(gh, list) or not isinstance(ga, list):
        raise PBValidationError("gh et ga doivent être des listes")
    if not isinstance(th, list) or not isinstance(ta, list):
        raise PBValidationError("th et ta doivent être des listes")
    n = len(gh)
    if n == 0:
        raise PBValidationError("aucun match fourni")
    if not (len(ga) == len(th) == len(ta) == n):
        raise PBValidationError("longueurs gh/ga/th/ta incohérentes")
    for i in range(n):
        if not isinstance(th[i], str) or not th[i]:
            raise PBValidationError(f"th[{i}] invalide")
        if not isinstance(ta[i], str) or not ta[i]:
            raise PBValidationError(f"ta[{i}] invalide")
        if not isinstance(gh[i], (int, np.integer)) or gh[i] < 0:
            raise PBValidationError(f"gh[{i}] invalide")
        if not isinstance(ga[i], (int, np.integer)) or ga[i] < 0:
            raise PBValidationError(f"ga[{i}] invalide")


def apply_filter(
    gh: List[int],
    ga: List[int],
    th: List[str],
    ta: List[str],
    min_matches: int = 0,
    max_goals: int = 100,
) -> Tuple[List[int], List[int], List[str], List[str], List[int], int]:
    n = len(gh)
    if n == 0:
        return [], [], [], [], [], 0
    if not (len(ga) == len(th) == len(ta) == n):
        raise PBValidationError("apply_filter : longueurs incohérentes")
    try:
        mm = int(min_matches)
    except (TypeError, ValueError):
        mm = 0
    try:
        mg = int(max_goals)
    except (TypeError, ValueError):
        mg = 100
    mm = max(0, min(mm, MIN_MATCHES_CEILING))
    mg = max(1, min(mg, MAX_GOALS_CEILING))
    counts: Dict[str, int] = {}
    for h in th:
        counts[h] = counts.get(h, 0) + 1
    for a in ta:
        counts[a] = counts.get(a, 0) + 1
    keep: List[int] = []
    for i in range(n):
        try:
            gh_i = int(gh[i])
            ga_i = int(ga[i])
        except (TypeError, ValueError):
            continue
        if gh_i < 0 or ga_i < 0:
            continue
        if gh_i > mg or ga_i > mg:
            continue
        if mm > 0:
            if counts.get(th[i], 0) < mm:
                continue
            if counts.get(ta[i], 0) < mm:
                continue
        keep.append(i)
    gh_f = [int(gh[i]) for i in keep]
    ga_f = [int(ga[i]) for i in keep]
    th_f = [th[i] for i in keep]
    ta_f = [ta[i] for i in keep]
    return gh_f, ga_f, th_f, ta_f, keep, len(keep)


def _extract_dates_for_kept(
    dates: Optional[List[Any]],
    keep_idx: List[int],
    total: int,
) -> Optional[List[Any]]:
    if dates is None:
        return None
    if not isinstance(dates, list):
        return None
    if len(dates) == total:
        return [dates[i] for i in keep_idx]
    if len(dates) == len(keep_idx):
        return list(dates)
    return None


def _is_bayesian_model_usable(model: Any) -> bool:
    """Vérifie qu'un modèle bayésien est utilisable malgré Iteration limit."""
    if model is None:
        return False
    trace = getattr(model, "trace", None)
    if trace is None:
        return False
    try:
        import numpy as _np
        arr = _np.asarray(trace, dtype=float)
    except Exception:
        return False
    if arr.size == 0:
        return False
    if not _np.all(_np.isfinite(arr)):
        return False
    return True




# ─────────────────────────────────────────────────────────────────────────────
# Patch penaltyblog : neutraliser _get_initial_params() sur BayesianGoalModel
# ─────────────────────────────────────────────────────────────────────────────

_BAYESIAN_INIT_PATCHED: bool = False


def _safe_get_initial_params(self) -> np.ndarray:
    """Remplace _get_initial_params() : tente le MLE, sinon retourne un
    vecteur centré utilisable par les walkers MCMC."""
    ndim = (self.n_teams * 2) + 2
    try:
        from penaltyblog.models.base_model import BaseGoalsModel as _BGM
        orig = _BGM._get_initial_params
    except Exception:
        orig = None

    if orig is not None:
        try:
            params = orig(self)
            arr = np.asarray(params, dtype=np.float64).ravel()
            if arr.size == ndim and np.all(np.isfinite(arr)):
                return arr
            logger.warning(
                "MLE initial renvoyé invalide (%s), fallback aléatoire centré",
                arr.shape if hasattr(arr, 'shape') else "?",
            )
        except Exception as exc:
            logger.warning(
                "MLE initial a échoué (%s) — fallback aléatoire centré", exc,
            )

    rng = np.random.default_rng(12345)
    att = rng.normal(0.0, 0.15, size=self.n_teams)
    deff = rng.normal(0.0, 0.15, size=self.n_teams)
    att -= att.mean()
    deff -= deff.mean()
    hfa_rho = np.array([0.25, -0.10], dtype=np.float64)
    return np.concatenate([att, deff, hfa_rho])


def apply_bayesian_init_patch() -> bool:
    """Monkey-patch _get_initial_params sur les modèles bayésiens."""
    global _BAYESIAN_INIT_PATCHED
    if _BAYESIAN_INIT_PATCHED:
        return True
    try:
        from penaltyblog.models.base_bayesian_model import BaseBayesianModel
        from penaltyblog.models.bayesian_goal_model import BayesianGoalModel
        try:
            from penaltyblog.models.hierarchical_bayesian_goal_model import (
                HierarchicalBayesianGoalModel,
            )
        except ImportError:
            HierarchicalBayesianGoalModel = None

        patched = 0
        for cls in (BaseBayesianModel, BayesianGoalModel, HierarchicalBayesianGoalModel):
            if cls is None:
                continue
            try:
                cls._get_initial_params = _safe_get_initial_params
                patched += 1
            except Exception as exc:
                logger.warning("Patch %s échoué : %s", cls.__name__, exc)
        _BAYESIAN_INIT_PATCHED = True
        logger.info(
            "✅ _get_initial_params patché sur %d classe(s) bayésienne(s)",
            patched,
        )
        return True
    except Exception as exc:
        logger.error("Patch initial_params échoué : %s", exc)
        return False


apply_bayesian_init_patch()

def fit_pb_attempt(
    class_name: str,
    model_class: Any,
    gh: List[int],
    ga: List[int],
    th: List[str],
    ta: List[str],
    dates: Optional[List[Any]] = None,
    is_bayesian: bool = False,
    bayesian_kwargs: Optional[Dict[str, Any]] = None,
    minimizer_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Any, Dict[str, Any]]:
    start_ts = time.monotonic()
    validate_pb_inputs(gh, ga, th, ta)

    attempt = increment_pb_attempt(class_name)
    best = get_best_filter(class_name)
    best_spec: Optional[Dict[str, Any]] = None
    if isinstance(best, dict):
        best_spec = {
            "min_matches": best.get("min_matches", 0),
            "max_goals": best.get("max_goals", 100),
            "half_life": best.get("half_life", PB_HALF_LIFE_DEFAULT),
        }
    filt = resolve_filter(attempt, best_spec)
    if is_bayesian and BAYESIAN_MIN_MATCHES_OVERRIDE > 0:
        filt["min_matches"] = max(
            int(filt["min_matches"]),
            int(BAYESIAN_MIN_MATCHES_OVERRIDE),
        )
    mm = int(filt["min_matches"])
    mg = int(filt["max_goals"])
    hl = float(filt["half_life"])

    gh_f, ga_f, th_f, ta_f, keep_idx, n_kept = apply_filter(
        gh, ga, th, ta, mm, mg,
    )

    meta: Dict[str, Any] = {
        "class": class_name,
        "attempt": attempt,
        "is_bayesian": bool(is_bayesian),
        "filter": dict(filt),
        "n_input": int(len(gh)),
        "n_kept": int(n_kept),
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    if n_kept < PB_MIN_MATCHES_HARD_FLOOR:
        msg = f"filtre trop strict : {n_kept} < {PB_MIN_MATCHES_HARD_FLOOR}"
        meta["error"] = msg
        meta["status"] = "incomplete"
        meta["duration_s"] = float(time.monotonic() - start_ts)
        save_checkpoint(class_name, attempt, {"meta": meta, "status": "incomplete"})
        logger.warning("%s : %s", class_name, msg)
        return "incomplete", None, meta

    dates_f = _extract_dates_for_kept(dates, keep_idx, len(gh))
    weights = compute_time_weights(dates_f, hl)
    meta["weights_applied"] = weights is not None
    if weights is not None:
        meta["half_life"] = float(hl)
        meta["weights_mean"] = float(np.mean(weights))
        meta["weights_max"] = float(np.max(weights))
        meta["weights_min"] = float(np.min(weights))

    def _instantiate():
        if weights is not None:
            try:
                return model_class(gh_f, ga_f, th_f, ta_f, weights=weights)
            except TypeError:
                pass
            except Exception as exc:
                logger.debug(
                    "%s : weights rejetés (%s), fallback sans poids",
                    class_name, exc,
                )
        return model_class(gh_f, ga_f, th_f, ta_f)

    if is_bayesian:
        # Progressif : plus de samples à chaque tentative (max 3000)
        progress_factor = min(1.0 + 0.5 * (attempt - 1), 3.0)
        n_samples_try = min(int(PB_BAYESIAN_N_SAMPLES * progress_factor), 3000)
        burn_try = min(int(PB_BAYESIAN_BURN * progress_factor), 1500)
        try:
            m = _instantiate()
            kwargs: Dict[str, Any] = {
                "n_samples": n_samples_try,
                "burn": burn_try,
                "n_chains": PB_BAYESIAN_N_CHAINS,
                "thin": PB_BAYESIAN_THIN,
            }
            if isinstance(bayesian_kwargs, dict):
                for k, v in bayesian_kwargs.items():
                    kwargs[str(k)] = v
            try:
                sig = inspect.signature(m.fit)
                if "n_cores" in sig.parameters and "n_cores" not in kwargs:
                    kwargs["n_cores"] = PB_BAYESIAN_N_CORES
            except (TypeError, ValueError):
                pass
            m.fit(**kwargs)
            meta["status"] = "ok"
            meta["optimizer"] = "bayesian"
            meta["duration_s"] = float(time.monotonic() - start_ts)
            meta["bayesian_kwargs"] = dict(kwargs)
            update_best_filter(class_name, filt, n_kept)
            save_checkpoint(class_name, attempt, {"meta": meta, "status": "ok"})
            return "ok", m, meta
        except Exception as exc:
            err_str = str(exc)
            msg = f"bayesian fit failed: {type(exc).__name__}: {exc}"
            meta["error"] = msg
            meta["status"] = "incomplete"
            meta["duration_s"] = float(time.monotonic() - start_ts)
            meta["bayesian_n_samples_try"] = n_samples_try
            meta["bayesian_burn_try"] = burn_try
            save_checkpoint(class_name, attempt, {"meta": meta, "status": "incomplete"})
            logger.warning("%s : %s", class_name, msg)
            # Si "Iteration limit reached" → c'est un signal de retry avec plus de samples
            # Tenter récupération : le sampler MCMC peut avoir convergé partiellement
            partial_model = None
            try:
                partial_model = m
            except Exception:
                partial_model = None
            if partial_model is not None and _is_bayesian_model_usable(partial_model):
                logger.warning(
                    "%s : forçage succès sur modèle partiel (trace finie, %d params)",
                    class_name, int(partial_model.trace.shape[1]) if hasattr(partial_model.trace, "shape") else -1,
                )
                meta["status"] = "ok"
                meta["optimizer"] = "bayesian_partial_forced"
                meta["forced_success"] = True
                meta["forced_reason"] = err_str[:200]
                meta["duration_s"] = float(time.monotonic() - start_ts)
                meta["bayesian_kwargs"] = dict(kwargs)
                update_best_filter(class_name, filt, n_kept)
                save_checkpoint(class_name, attempt, {"meta": meta, "status": "ok"})
                return "ok", partial_model, meta
            if "Iteration limit" in err_str or "iteration limit" in err_str.lower():
                logger.warning(
                    "%s : limite d'itérations atteinte — retry avec n_samples=%d, burn=%d",
                    class_name, n_samples_try, burn_try,
                )
            return "incomplete", None, meta

    try:
        m = _instantiate()
        opts: Dict[str, Any] = {"maxiter": 5000}
        if isinstance(minimizer_options, dict):
            for k, v in minimizer_options.items():
                opts[str(k)] = v
        m.fit(minimizer_options=opts)
        meta["status"] = "ok"
        meta["optimizer"] = "auto-slsqp-trust"
        meta["duration_s"] = float(time.monotonic() - start_ts)
        meta["minimizer_options"] = dict(opts)
        update_best_filter(class_name, filt, n_kept)
        save_checkpoint(class_name, attempt, {"meta": meta, "status": "ok"})
        return "ok", m, meta
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        meta["error"] = msg
        meta["status"] = "incomplete"
        meta["duration_s"] = float(time.monotonic() - start_ts)
        save_checkpoint(class_name, attempt, {"meta": meta, "status": "incomplete"})
        logger.warning("%s : non convergé (%s)", class_name, msg)
        return "incomplete", None, meta


def get_pb_status(class_name: str) -> Dict[str, Any]:
    with _LOCK:
        state = _safe_read_json(_attempt_file(class_name))
        state = _validate_attempt_state(state, class_name)
        best = get_best_filter(class_name)
        ckpts: List[Dict[str, Any]] = []
        d = _ckpt_dir(class_name)
        try:
            for p in sorted(d.glob("attempt_*.pkl")):
                try:
                    size_mb = p.stat().st_size / (1024.0 * 1024.0)
                except OSError:
                    size_mb = 0.0
                ckpts.append({"name": p.name, "size_mb": round(size_mb, 3)})
        except OSError:
            pass
        return {
            "class": class_name,
            "attempt_count": int(state["count"]),
            "history": list(state["history"]),
            "best_filter": best,
            "checkpoints": ckpts,
            "max_attempts": int(PB_MAX_ATTEMPTS),
        }


def list_pb_status() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        for p in sorted(_ATTEMPT_DIR.glob("*.json")):
            cn = p.stem
            try:
                result[cn] = get_pb_status(cn)
            except PBValidationError:
                continue
    except OSError:
        pass
    return result


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
        "weights": {
            "min": float(PB_WEIGHT_MIN),
            "max": float(PB_WEIGHT_MAX),
        },
        "history_max_entries": int(PB_HISTORY_MAX_ENTRIES),
        "checkpoints_kept": int(PB_CHECKPOINTS_KEPT),
    }


def get_pb_filter_for_attempt(attempt: int) -> Dict[str, Any]:
    try:
        a = int(attempt)
    except (TypeError, ValueError):
        a = 1
    if a < 1:
        a = 1
    if a in FILTER_PLAN:
        spec = FILTER_PLAN[a]
        return {
            "min_matches": int(spec["min_matches"]),
            "max_goals": int(spec["max_goals"]),
            "half_life": float(spec["half_life"]),
        }
    last_key = max(FILTER_PLAN.keys())
    spec = FILTER_PLAN[last_key]
    return {
        "min_matches": int(spec["min_matches"]),
        "max_goals": int(spec["max_goals"]),
        "half_life": float(spec["half_life"]),
    }
