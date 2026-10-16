import hashlib
import json
import logging
import lzma
import os
import pickle
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("goangel.checkpoint_all")

_LOCK = threading.RLock()
_SCHEMA = "goangel-checkpoint-2.0"
_COMPRESS_THRESHOLD = 5 * 1024 * 1024
_ROOT = Path("./cache/stages/checkpoints")
_LEGACY_ROOTS = [
    Path("./cache/stages/checkpoints_universal"),
    Path("./cache/stages/pb_convergence/checkpoints"),
    Path("./cache/stages/pb_convergence/wf_checkpoints"),
]

ML_MODELS: Tuple[str, ...] = ("XGB", "LGBM", "CatBoost", "RF", "DL")
PB_MODELS: Tuple[str, ...] = (
    "PoissonGoalsModel",
    "DixonColesGoalModel",
    "BivariatePoissonGoalModel",
    "NegativeBinomialGoalModel",
    "ZeroInflatedPoissonGoalsModel",
    "WeibullCopulaGoalsModel",
    "BayesianGoalModel",
    "HierarchicalBayesianGoalModel",
)
BAYESIAN_MODELS: Tuple[str, ...] = ("BayesianGoalModel", "HierarchicalBayesianGoalModel")
FINALIZE_KEYS: Tuple[str, ...] = (
    "finalize_prep",
    "finalize_xgb",
    "finalize_lgbm",
    "finalize_catboost",
    "finalize_rf",
    "finalize_dl",
    "finalize_pb_poisson",
    "finalize_pb_dixon_coles",
    "finalize_pb_bivariate",
    "finalize_pb_negbinomial",
    "finalize_pb_zeroinflated",
    "finalize_pb_weibull",
    "finalize_pb_bayesian",
    "finalize_pb_hierarchical",
    "finalize_agg",
)

STEP_DATA_LOADED = "data_loaded"
STEP_SPLIT_DONE = "split_done"
STEP_TRAIN_STARTED = "train_started"
STEP_TRAINED = "trained"
STEP_CALIBRATED = "calibrated"
STEP_WF_DONE = "wf_done"
STEP_FINALIZED = "finalized"
STEP_ATTEMPT = "attempt"
STEP_BEST_FILTER = "best_filter"
STEP_MCMC_STATE = "mcmc_state"
STEP_CONTEXT = "context"
STEP_PARTIAL = "partial"
STEP_AGG_DONE = "agg_done"


def _safe_key(key: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(key))


def _ensure_root() -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def _key_dir(key: str) -> Path:
    d = _ensure_root() / _safe_key(key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_bytes(path: Path, data: bytes) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
    return len(data)


def _atomic_write_pickle(path: Path, payload: Any) -> int:
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if len(raw) >= _COMPRESS_THRESHOLD:
        data = lzma.compress(raw, preset=6)
    else:
        data = raw
    return _atomic_write_bytes(path, data)


def _atomic_read_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            magic = f.read(6)
        if magic[:6] == b"\xfd7zXZ\x00":
            with open(path, "rb") as f:
                return pickle.loads(lzma.decompress(f.read()))
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Checkpoint illisible %s : %s", path, exc)
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> int:
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    return _atomic_write_bytes(path, data)


def _atomic_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("JSON illisible %s : %s", path, exc)
        return None


def _step_path(entity_key: str, step_name: str) -> Path:
    return _key_dir(entity_key) / f"{_safe_key(step_name)}.pkl"


def _manifest_path(entity_key: str) -> Path:
    return _key_dir(entity_key) / "_manifest.json"


def _read_manifest(entity_key: str) -> Dict[str, Any]:
    data = _atomic_read_json(_manifest_path(entity_key))
    if not isinstance(data, dict):
        return {"schema": _SCHEMA, "entity_key": entity_key, "steps": {}}
    data.setdefault("steps", {})
    return data


def _write_manifest(entity_key: str, manifest: Dict[str, Any]) -> None:
    manifest["schema"] = _SCHEMA
    manifest["entity_key"] = entity_key
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(_manifest_path(entity_key), manifest)


def save_step(
    entity_key: str,
    step_name: str,
    payload: Any,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Optional[Path]:
    with _LOCK:
        try:
            path = _step_path(entity_key, step_name)
            if path.exists() and not overwrite:
                return path
            wrapper = {
                "schema": _SCHEMA,
                "entity_key": entity_key,
                "step_name": step_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "metadata": metadata or {},
            }
            size = _atomic_write_pickle(path, wrapper)
            manifest = _read_manifest(entity_key)
            manifest["steps"][step_name] = {
                "file": path.name,
                "size": size,
                "ts": wrapper["timestamp"],
                "metadata": metadata or {},
            }
            _write_manifest(entity_key, manifest)
            logger.info(
                "checkpoint_save key=%s step=%s size_kb=%.2f",
                entity_key, step_name, size / 1024.0,
            )
            return path
        except Exception as exc:
            logger.error(
                "checkpoint_save_failed key=%s step=%s err=%s",
                entity_key, step_name, exc,
            )
            return None


def load_step(entity_key: str, step_name: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        wrapper = _atomic_read_pickle(_step_path(entity_key, step_name))
        if not isinstance(wrapper, dict):
            return None
        if wrapper.get("schema") != _SCHEMA:
            return None
        return wrapper


def load_payload(entity_key: str, step_name: str) -> Optional[Any]:
    wrapper = load_step(entity_key, step_name)
    if wrapper is None:
        return None
    return wrapper.get("payload")


def has_step(entity_key: str, step_name: str) -> bool:
    with _LOCK:
        return _step_path(entity_key, step_name).exists()


def delete_step(entity_key: str, step_name: str) -> bool:
    with _LOCK:
        path = _step_path(entity_key, step_name)
        existed = path.exists()
        if existed:
            try:
                path.unlink()
                manifest = _read_manifest(entity_key)
                manifest.get("steps", {}).pop(step_name, None)
                _write_manifest(entity_key, manifest)
            except OSError:
                pass
        return existed


def list_steps(entity_key: str) -> List[str]:
    with _LOCK:
        manifest = _read_manifest(entity_key)
        return sorted(manifest.get("steps", {}).keys())


def list_all_entities() -> List[str]:
    with _LOCK:
        if not _ROOT.exists():
            return []
        return sorted([d.name for d in _ROOT.iterdir() if d.is_dir()])


def get_entity_manifest(entity_key: str) -> Dict[str, Any]:
    with _LOCK:
        return _read_manifest(entity_key)


def clear_entity(entity_key: str) -> int:
    with _LOCK:
        d = _key_dir(entity_key)
        n = 0
        for p in d.glob("*"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n


def clear_all() -> int:
    with _LOCK:
        if not _ROOT.exists():
            return 0
        n = 0
        for d in list(_ROOT.iterdir()):
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                    n += 1
                except OSError:
                    pass
        return n


def save_fold(entity_key: str, fold_idx: int, payload: Any) -> Optional[Path]:
    return save_step(entity_key, f"fold_{int(fold_idx):03d}", payload)


def load_fold(entity_key: str, fold_idx: int) -> Optional[Dict[str, Any]]:
    return load_step(entity_key, f"fold_{int(fold_idx):03d}")


def list_folds(entity_key: str) -> Dict[int, Dict[str, Any]]:
    with _LOCK:
        d = _key_dir(entity_key)
        out: Dict[int, Dict[str, Any]] = {}
        for p in sorted(d.glob("fold_*.pkl")):
            try:
                idx = int(p.stem.split("_")[1])
                wrapper = _atomic_read_pickle(p)
                if isinstance(wrapper, dict):
                    out[idx] = wrapper.get("payload", {})
            except Exception:
                continue
        return out


def clear_folds(entity_key: str) -> int:
    with _LOCK:
        d = _key_dir(entity_key)
        n = 0
        for p in d.glob("fold_*.pkl"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n


def ml_key(name: str) -> str:
    return f"ml_{name}"


def pb_key(name: str) -> str:
    return f"pb_{name}"


def finalize_key(name: str) -> str:
    return name if name.startswith("finalize") else f"finalize_{name}"


def is_ml_done(name: str) -> bool:
    return has_step(ml_key(name), STEP_TRAINED)


def is_pb_done(name: str) -> bool:
    return has_step(pb_key(name), STEP_TRAINED)


def is_finalize_pb_done(name: str) -> bool:
    return has_step(pb_key(name), STEP_FINALIZED)


def get_ml_state(name: str) -> Dict[str, Any]:
    return get_entity_manifest(ml_key(name))


def get_pb_state(name: str) -> Dict[str, Any]:
    return get_entity_manifest(pb_key(name))


def get_all_model_states() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ml": {},
        "pb": {},
        "finalize": {},
        "complete": {},
    }
    for m in ML_MODELS:
        steps = list_steps(ml_key(m))
        out["ml"][m] = {
            "trained": STEP_TRAINED in steps,
            "calibrated": STEP_CALIBRATED in steps,
            "wf_done": STEP_WF_DONE in steps,
            "steps": steps,
        }
    for m in PB_MODELS:
        steps = list_steps(pb_key(m))
        out["pb"][m] = {
            "trained": STEP_TRAINED in steps,
            "finalized": STEP_FINALIZED in steps,
            "folds": sorted(list_folds(pb_key(m)).keys()),
            "steps": steps,
        }
    for k in FINALIZE_KEYS:
        steps = list_steps(finalize_key(k))
        out["finalize"][k] = {
            "done": len(steps) > 0,
            "steps": steps,
        }
    ml_done = sum(1 for m in ML_MODELS if out["ml"][m]["trained"])
    pb_done = sum(1 for m in PB_MODELS if out["pb"][m]["trained"])
    finalize_done = sum(1 for k in FINALIZE_KEYS if out["finalize"][k]["done"])
    out["complete"] = {
        "ml_done": ml_done,
        "ml_total": len(ML_MODELS),
        "pb_done": pb_done,
        "pb_total": len(PB_MODELS),
        "finalize_done": finalize_done,
        "finalize_total": len(FINALIZE_KEYS),
        "all_done": (
            ml_done == len(ML_MODELS)
            and pb_done == len(PB_MODELS)
            and finalize_done == len(FINALIZE_KEYS)
        ),
    }
    return out


def get_state_digest(entity_key: str) -> str:
    with _LOCK:
        d = _key_dir(entity_key)
        h = hashlib.sha256()
        for p in sorted(d.glob("*.pkl")):
            h.update(p.name.encode("utf-8"))
            try:
                st = p.stat()
                h.update(str(st.st_size).encode("utf-8"))
                h.update(str(st.st_mtime_ns).encode("utf-8"))
            except OSError:
                pass
        return h.hexdigest()[:16]


def total_size_mb() -> float:
    with _LOCK:
        if not _ROOT.exists():
            return 0.0
        total = 0
        for p in _ROOT.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total / (1024.0 * 1024.0)


def get_full_report() -> Dict[str, Any]:
    with _LOCK:
        states = get_all_model_states()
        states["total_size_mb"] = round(total_size_mb(), 2)
        states["entities"] = list_all_entities()
        states["root"] = str(_ROOT)
        return states


def migrate_legacy_layout() -> Dict[str, int]:
    stats = {"migrated": 0, "skipped": 0}
    with _LOCK:
        for legacy_root in _LEGACY_ROOTS:
            if not legacy_root.exists():
                continue
            for sub in list(legacy_root.iterdir()):
                if not sub.is_dir():
                    continue
                key = _safe_key(sub.name)
                target = _key_dir(key)
                for p in sub.glob("*.pkl"):
                    dest = target / p.name
                    if dest.exists():
                        stats["skipped"] += 1
                        continue
                    try:
                        shutil.copy2(p, dest)
                        stats["migrated"] += 1
                    except OSError:
                        stats["skipped"] += 1
    if stats["migrated"] > 0:
        logger.info(
            "checkpoint_migration migrated=%d skipped=%d",
            stats["migrated"], stats["skipped"],
        )
    return stats


def cleanup_orphan_entities(keep_keys: Iterable[str]) -> int:
    keep_set = set(_safe_key(k) for k in keep_keys)
    removed = 0
    with _LOCK:
        if not _ROOT.exists():
            return 0
        for d in list(_ROOT.iterdir()):
            if not d.is_dir():
                continue
            if d.name in keep_set:
                continue
            try:
                shutil.rmtree(d)
                removed += 1
            except OSError:
                pass
    return removed


def save_ml_data_loaded(model_name: str, n_samples: int, n_features: int) -> Optional[Path]:
    return save_step(
        ml_key(model_name),
        STEP_DATA_LOADED,
        {"n_samples": int(n_samples), "n_features": int(n_features)},
    )


def save_ml_split(
    model_name: str,
    cut_train: int,
    cut_val: int,
    cut_calib: int,
    n_total: int,
) -> Optional[Path]:
    return save_step(
        ml_key(model_name),
        STEP_SPLIT_DONE,
        {
            "cut_train": int(cut_train),
            "cut_val": int(cut_val),
            "cut_calib": int(cut_calib),
            "n_total": int(n_total),
        },
    )


def save_ml_trained(
    model_name: str,
    model_path: str,
    calib_ll: Optional[float] = None,
    accuracy: Optional[float] = None,
) -> Optional[Path]:
    return save_step(
        ml_key(model_name),
        STEP_TRAINED,
        {
            "model_path": str(model_path),
            "calib_ll": calib_ll,
            "accuracy": accuracy,
        },
    )


def save_ml_calibrated(
    model_name: str,
    calib_method: str,
    calib_ll: Optional[float] = None,
) -> Optional[Path]:
    return save_step(
        ml_key(model_name),
        STEP_CALIBRATED,
        {
            "calib_method": calib_method,
            "calib_ll": calib_ll,
        },
    )


def save_pb_attempt(
    class_name: str,
    attempt: int,
    filter_spec: Dict[str, Any],
    n_kept: int,
    status: str,
    duration_s: float,
) -> Optional[Path]:
    key = pb_key(class_name)
    return save_step(
        key,
        f"attempt_{int(attempt):03d}",
        {
            "attempt": int(attempt),
            "filter": filter_spec,
            "n_kept": int(n_kept),
            "status": status,
            "duration_s": float(duration_s),
        },
    )


def save_pb_best_filter(
    class_name: str,
    filter_spec: Dict[str, Any],
    n_kept: int,
) -> Optional[Path]:
    return save_step(
        pb_key(class_name),
        STEP_BEST_FILTER,
        {"filter": filter_spec, "n_kept": int(n_kept)},
        overwrite=True,
    )


def load_pb_best_filter(class_name: str) -> Optional[Dict[str, Any]]:
    return load_payload(pb_key(class_name), STEP_BEST_FILTER)


def save_pb_trained(
    class_name: str,
    model_path: str,
    attempt: int,
) -> Optional[Path]:
    return save_step(
        pb_key(class_name),
        STEP_TRAINED,
        {"model_path": str(model_path), "attempt": int(attempt)},
    )


def save_pb_mcmc_state(
    class_name: str,
    n_samples: int,
    burn: int,
    n_chains: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    payload = {
        "n_samples": int(n_samples),
        "burn": int(burn),
        "n_chains": int(n_chains),
    }
    if extra:
        payload.update(extra)
    return save_step(pb_key(class_name), STEP_MCMC_STATE, payload)


def save_pb_wf_fold(
    class_name: str,
    fold_idx: int,
    log_loss: float,
    n_val: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    payload = {
        "log_loss": float(log_loss),
        "n_val": int(n_val),
    }
    if extra:
        payload.update(extra)
    return save_fold(pb_key(class_name), fold_idx, payload)


def load_pb_wf_folds(class_name: str) -> Dict[int, Dict[str, Any]]:
    return list_folds(pb_key(class_name))


def save_pb_finalized(
    class_name: str,
    calib_ll: float,
    wf_ll: float,
    calib_status: str,
    n_kept: int,
) -> Optional[Path]:
    return save_step(
        pb_key(class_name),
        STEP_FINALIZED,
        {
            "calib_ll": float(calib_ll),
            "wf_ll": float(wf_ll),
            "calib_status": calib_status,
            "n_kept": int(n_kept),
        },
    )


def save_finalize_context(context: Dict[str, Any]) -> Optional[Path]:
    return save_step(finalize_key("finalize_prep"), STEP_CONTEXT, context)


def load_finalize_context() -> Optional[Dict[str, Any]]:
    return load_payload(finalize_key("finalize_prep"), STEP_CONTEXT)


def save_finalize_partial(key: str, payload: Dict[str, Any]) -> Optional[Path]:
    return save_step(finalize_key(key), STEP_PARTIAL, payload)


def load_finalize_partial(key: str) -> Optional[Dict[str, Any]]:
    return load_payload(finalize_key(key), STEP_PARTIAL)


def has_finalize_partial(key: str) -> bool:
    return has_step(finalize_key(key), STEP_PARTIAL)


def list_finalize_partials() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k in FINALIZE_KEYS:
        if has_finalize_partial(k):
            payload = load_finalize_partial(k)
            if payload is not None:
                out[k] = payload
    return out


def save_finalize_agg(payload: Dict[str, Any]) -> Optional[Path]:
    return save_step(finalize_key("finalize_agg"), STEP_AGG_DONE, payload)


def load_finalize_agg() -> Optional[Dict[str, Any]]:
    return load_payload(finalize_key("finalize_agg"), STEP_AGG_DONE)


def reset_ml(model_name: Optional[str] = None) -> int:
    if model_name is None:
        n = 0
        for m in ML_MODELS:
            n += clear_entity(ml_key(m))
        return n
    return clear_entity(ml_key(model_name))


def reset_pb(class_name: Optional[str] = None) -> int:
    if class_name is None:
        n = 0
        for m in PB_MODELS:
            n += clear_entity(pb_key(m))
        return n
    return clear_entity(pb_key(class_name))


def reset_finalize(key: Optional[str] = None) -> int:
    if key is None:
        n = 0
        for k in FINALIZE_KEYS:
            n += clear_entity(finalize_key(k))
        return n
    return clear_entity(finalize_key(key))


def reset_all_models() -> Dict[str, int]:
    return {
        "ml": reset_ml(),
        "pb": reset_pb(),
        "finalize": reset_finalize(),
    }


def bootstrap() -> Dict[str, Any]:
    _ensure_root()
    stats = migrate_legacy_layout()
    report = get_full_report()
    report["migration"] = stats
    return report


bootstrap()
