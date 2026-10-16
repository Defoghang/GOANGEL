"""GOANGEL — Checkpoint universel pour tous les modèles (ML + PB + Finalize)."""
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("goangel.training_checkpoint")

_LOCK = threading.RLock()
_ROOT = Path("./cache/stages/checkpoints_universal")
_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(name))


def _key_dir(model_key: str) -> Path:
    d = _ROOT / _safe_name(model_key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _atomic_read(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Checkpoint illisible %s : %s", path, exc)
        return None


def save_step(model_key: str, step_name: str, payload: Dict[str, Any]) -> Path:
    """Sauvegarde une étape. Retourne le chemin."""
    with _LOCK:
        d = _key_dir(model_key)
        p = d / f"{_safe_name(step_name)}.pkl"
        data = dict(payload)
        data["__ts__"] = time.time()
        data["__model_key__"] = model_key
        data["__step__"] = step_name
        _atomic_write(p, data)
        logger.info("💾 Checkpoint [%s] %s → %s", model_key, step_name, p.name)
        return p


def load_step(model_key: str, step_name: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        d = _key_dir(model_key)
        p = d / f"{_safe_name(step_name)}.pkl"
        return _atomic_read(p)


def has_step(model_key: str, step_name: str) -> bool:
    with _LOCK:
        d = _key_dir(model_key)
        return (d / f"{_safe_name(step_name)}.pkl").exists()


def save_fold(model_key: str, fold_idx: int, payload: Dict[str, Any]) -> Path:
    return save_step(model_key, f"fold_{int(fold_idx):03d}", payload)


def load_fold(model_key: str, fold_idx: int) -> Optional[Dict[str, Any]]:
    return load_step(model_key, f"fold_{int(fold_idx):03d}")


def list_folds(model_key: str) -> Dict[int, Dict[str, Any]]:
    with _LOCK:
        d = _key_dir(model_key)
        out: Dict[int, Dict[str, Any]] = {}
        for fp in sorted(d.glob("fold_*.pkl")):
            try:
                idx = int(fp.stem.split("_")[1])
                payload = _atomic_read(fp)
                if payload is not None:
                    out[idx] = payload
            except Exception:
                continue
        return out


def list_steps(model_key: str) -> List[str]:
    with _LOCK:
        d = _key_dir(model_key)
        return [p.stem for p in sorted(d.glob("*.pkl")) if not p.stem.startswith("fold_")]


def clear_model(model_key: str) -> int:
    with _LOCK:
        d = _key_dir(model_key)
        n = 0
        for p in d.glob("*.pkl"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n


def get_state(model_key: str) -> Dict[str, Any]:
    with _LOCK:
        return {
            "model_key": model_key,
            "steps": list_steps(model_key),
            "folds": sorted(list_folds(model_key).keys()),
        }
