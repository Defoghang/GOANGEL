"""GOANGEL — finalize éclaté : contexte partagé + résultats partiels."""
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("goangel.finalize_split")

FINALIZE_DIR: Path = Path("./cache/stages/finalize")
FINALIZE_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE: Path = FINALIZE_DIR / "context.pkl"


def _atomic_write(path: Path, obj: Any) -> None:
    import lzma
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    data = lzma.compress(raw, preset=3) if len(raw) > 5 * 1024 * 1024 else raw
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _atomic_read(path: Path) -> Any:
    import lzma
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic[:6] == b"\xfd7zXZ\x00":
        with open(path, "rb") as f:
            return pickle.loads(lzma.decompress(f.read()))
    with open(path, "rb") as f:
        return pickle.load(f)


def save_context(ctx: Dict[str, Any]) -> Path:
    _atomic_write(CONTEXT_FILE, ctx)
    return CONTEXT_FILE


def load_context() -> Optional[Dict[str, Any]]:
    if not CONTEXT_FILE.exists():
        return None
    try:
        return _atomic_read(CONTEXT_FILE)
    except Exception as exc:
        logger.error("Lecture context échouée : %s", exc)
        return None


def partial_path(model_key: str) -> Path:
    return FINALIZE_DIR / f"{model_key}.pkl"


def save_partial(model_key: str, payload: Dict[str, Any]) -> Path:
    p = partial_path(model_key)
    _atomic_write(p, payload)
    return p


def load_partial(model_key: str) -> Optional[Dict[str, Any]]:
    p = partial_path(model_key)
    if not p.exists():
        return None
    try:
        return _atomic_read(p)
    except Exception as exc:
        logger.error("Lecture partial %s échouée : %s", model_key, exc)
        return None


def list_partials() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(FINALIZE_DIR.glob("*.pkl")):
        if p.name == "context.pkl":
            continue
        key = p.stem
        try:
            out[key] = _atomic_read(p)
        except Exception:
            continue
    return out
