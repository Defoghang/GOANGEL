import functools
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from scipy.optimize import minimize as _ORIGINAL_MINIMIZE


logger = logging.getLogger("goangel.penaltyblog_patch")


_PATCH_LOCK: threading.RLock = threading.RLock()
_PATCH_APPLIED: bool = False
_ORIGINAL_REF: Callable[..., Any] = _ORIGINAL_MINIMIZE
_PATCH_TIMESTAMP: Optional[float] = None


_NO_CONSTRAINT_METHODS = frozenset({"Powell", "Nelder-Mead", "CG", "BFGS", "L-BFGS-B", "TNC"})
_NO_BOUNDS_METHODS = frozenset({"Nelder-Mead", "CG", "BFGS"})


_DEFAULT_FALLBACK_CHAIN: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    (
        "trust-constr",
        {
            "maxiter": 5000,
            "gtol": 1e-6,
            "xtol": 1e-8,
            "barrier_tol": 1e-8,
            "initial_tr_radius": 1.0,
            "verbose": 0,
        },
    ),
    (
        "Powell",
        {
            "maxiter": 5000,
            "xtol": 1e-6,
            "ftol": 1e-6,
        },
    ),
    (
        "L-BFGS-B",
        {
            "maxiter": 5000,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 50,
        },
    ),
)


_MAX_FALLBACK_ATTEMPTS: int = len(_DEFAULT_FALLBACK_CHAIN)
_MIN_X0_SIZE: int = 1
_MAX_X0_SIZE: int = 500000
_FUN_FINITE_TOLERANCE: float = 1e-300


def _resolve_method(
    method: Optional[str],
    bounds: Any,
    constraints: Any,
) -> str:
    if method is not None:
        return str(method)
    has_constraints = bool(constraints) if constraints is not None else False
    has_bounds = bounds is not None and not (
        isinstance(bounds, (list, tuple)) and len(bounds) == 0
    )
    if has_constraints:
        if has_bounds:
            return "SLSQP"
        return "COBYLA"
    if has_bounds:
        return "L-BFGS-B"
    return "BFGS"


def _sanitize_options(options: Any) -> Dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        try:
            return dict(options)
        except Exception:
            return {}
    return dict(options)


def _sanitize_x0(x0: Any) -> Optional[np.ndarray]:
    if x0 is None:
        return None
    try:
        arr = np.asarray(x0, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 1:
        try:
            arr = arr.ravel()
        except Exception:
            return None
    if arr.size < _MIN_X0_SIZE or arr.size > _MAX_X0_SIZE:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _result_is_usable(result: Any) -> bool:
    if result is None:
        return False
    x = getattr(result, "x", None)
    if x is None:
        return False
    try:
        xarr = np.asarray(x, dtype=np.float64)
    except Exception:
        return False
    if xarr.size == 0:
        return False
    if not np.all(np.isfinite(xarr)):
        return False
    fun = getattr(result, "fun", None)
    if fun is not None:
        try:
            f = float(fun)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(f):
            return False
    return True


def _extract_fun(result: Any) -> Optional[float]:
    if result is None:
        return None
    fun = getattr(result, "fun", None)
    if fun is None:
        return None
    try:
        f = float(fun)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _extract_success(result: Any) -> bool:
    if result is None:
        return False
    s = getattr(result, "success", None)
    if s is None:
        return False
    try:
        return bool(s)
    except Exception:
        return False


def _extract_message(result: Any) -> str:
    if result is None:
        return "no result"
    msg = getattr(result, "message", None)
    if msg is None:
        return "unknown"
    try:
        return str(msg)
    except Exception:
        return "unknown"


def _result_is_better(new_result: Any, old_result: Any) -> bool:
    if not _result_is_usable(new_result):
        return False
    new_fun = _extract_fun(new_result)
    if new_fun is None:
        return False
    old_fun = _extract_fun(old_result)
    if old_fun is None:
        return True
    return float(new_fun) < float(old_fun)


def _safe_call(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.debug("Appel optimiseur échoué : %s", exc)
        return None


def _adjust_for_method(
    method: str,
    bounds: Any,
    constraints: Any,
    jac: Any,
) -> Tuple[Any, Any, Any]:
    adj_bounds = bounds
    adj_constraints = constraints
    adj_jac = jac

    if method in _NO_CONSTRAINT_METHODS:
        if adj_constraints:
            logger.debug(
                "Patch : %s ne supporte pas les contraintes → retirées",
                method,
            )
        adj_constraints = ()

    if method in _NO_BOUNDS_METHODS:
        if adj_bounds is not None:
            logger.debug(
                "Patch : %s ne supporte pas les bounds → retirés",
                method,
            )
        adj_bounds = None

    if method == "trust-constr":
        if adj_jac is not None and not isinstance(adj_jac, str):
            adj_jac = None
    elif method == "Powell":
        adj_jac = None
    elif method == "Nelder-Mead":
        adj_jac = None

    return adj_bounds, adj_constraints, adj_jac


def _build_fallback_options(
    base_options: Dict[str, Any],
    fallback_options: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for k, v in fallback_options.items():
        merged[str(k)] = v
    for k, v in base_options.items():
        key = str(k)
        if key in merged:
            continue
        merged[key] = v
    return merged


def _patched_minimize(
    fun: Callable[..., Any],
    x0: Any,
    args: Tuple[Any, ...] = (),
    method: Optional[str] = None,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Any = None,
    constraints: Any = (),
    tol: Any = None,
    callback: Any = None,
    options: Any = None,
    **extra_kwargs: Any,
) -> Any:
    if fun is None or not callable(fun):
        raise TypeError("minimize: 'fun' doit être callable")

    x0_arr = _sanitize_x0(x0)
    if x0_arr is None:
        raise ValueError("minimize: 'x0' invalide")

    resolved_method = _resolve_method(method, bounds, constraints)
    base_options = _sanitize_options(options)

    adj_bounds, adj_constraints, adj_jac = _adjust_for_method(
        resolved_method, bounds, constraints, jac,
    )

    primary_call_kwargs: Dict[str, Any] = {
        "fun": fun,
        "x0": x0_arr,
        "args": args,
        "method": resolved_method,
        "jac": adj_jac,
        "hess": hess,
        "hessp": hessp,
        "bounds": adj_bounds,
        "constraints": adj_constraints,
        "tol": tol,
        "callback": callback,
        "options": base_options,
    }
    for k, v in extra_kwargs.items():
        if k not in primary_call_kwargs:
            primary_call_kwargs[k] = v

    primary_result = _safe_call(_ORIGINAL_REF, **primary_call_kwargs)

    if _extract_success(primary_result) and _result_is_usable(primary_result):
        return primary_result

    needs_fallback = resolved_method in ("SLSQP", "COBYLA", "trust-constr")
    if not needs_fallback:
        if primary_result is not None:
            return primary_result
        return _safe_call(_ORIGINAL_REF, **primary_call_kwargs)

    if primary_result is not None:
        logger.warning(
            "Optimiseur %s échoué (%s, fun=%s) → fallback automatique",
            resolved_method,
            _extract_message(primary_result),
            _extract_fun(primary_result),
        )

    best_result = primary_result

    for fallback_method, fallback_options in _DEFAULT_FALLBACK_CHAIN:
        if fallback_method == resolved_method:
            continue

        fb_bounds, fb_constraints, fb_jac = _adjust_for_method(
            fallback_method, bounds, constraints, jac,
        )

        if fallback_method == "trust-constr":
            fb_options = _build_fallback_options(
                {"maxiter": base_options.get("maxiter", 5000)},
                fallback_options,
            )
            if fb_jac is None:
                fb_jac_to_use: Any = "2-point"
            else:
                fb_jac_to_use = fb_jac
        elif fallback_method == "Powell":
            fb_options = _build_fallback_options(
                {"maxiter": base_options.get("maxiter", 5000)},
                fallback_options,
            )
            fb_jac_to_use = None
        elif fallback_method == "L-BFGS-B":
            fb_options = _build_fallback_options(
                {"maxiter": base_options.get("maxiter", 5000)},
                fallback_options,
            )
            if fb_jac is None:
                fb_jac_to_use = "2-point"
            else:
                fb_jac_to_use = fb_jac
        else:
            fb_options = _build_fallback_options(
                {"maxiter": base_options.get("maxiter", 5000)},
                fallback_options,
            )
            fb_jac_to_use = fb_jac

        if callable(fb_jac_to_use):
            try:
                _probe = fb_jac_to_use(np.asarray(x0_arr, dtype=np.float64))
                _ = np.asarray(_probe)
            except Exception:
                fb_jac_to_use = "2-point" if fallback_method != "Powell" else None

        call_kwargs: Dict[str, Any] = {
            "fun": fun,
            "x0": x0_arr,
            "args": args,
            "method": fallback_method,
            "jac": fb_jac_to_use,
            "hess": None,
            "hessp": None,
            "bounds": fb_bounds,
            "constraints": fb_constraints,
            "tol": tol,
            "callback": callback,
            "options": fb_options,
        }

        try:
            fb_result = _ORIGINAL_REF(**call_kwargs)
        except Exception as exc:
            logger.debug(
                "Fallback %s a levé : %s", fallback_method, exc,
            )
            continue

        if _extract_success(fb_result) and _result_is_usable(fb_result):
            logger.info(
                "Fallback %s : SUCCESS (fun=%.6f, nit=%s)",
                fallback_method,
                _extract_fun(fb_result) or float("nan"),
                getattr(fb_result, "nit", "?"),
            )
            return fb_result

        if _result_is_better(fb_result, best_result):
            best_result = fb_result
            logger.info(
                "Fallback %s : amélioration (fun=%.6f) conservée",
                fallback_method,
                _extract_fun(fb_result) or float("nan"),
            )

    if best_result is not None:
        return best_result

    return _safe_call(_ORIGINAL_REF, **primary_call_kwargs)


def apply_patch() -> bool:
    global _PATCH_APPLIED, _PATCH_TIMESTAMP
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return True
        try:
            import scipy.optimize as _scipy_opt
            current = getattr(_scipy_opt, "minimize", None)
            if current is None:
                logger.error("scipy.optimize.minimize introuvable")
                return False
            if getattr(current, "__name__", "") == "_patched_minimize":
                _PATCH_APPLIED = True
                _PATCH_TIMESTAMP = time.time()
                return True
            _scipy_opt.minimize = _patched_minimize
            try:
                import scipy.optimize._minimize as _min_mod
                _min_mod.minimize = _patched_minimize
            except Exception:
                pass
            _PATCH_APPLIED = True
            _PATCH_TIMESTAMP = time.time()
            logger.info(
                "✅ penaltyblog_method_patch appliqué "
                "(fallback chain: trust-constr → Powell → L-BFGS-B)"
            )
            return True
        except Exception as exc:
            logger.error("Application patch échouée : %s", exc)
            return False


def is_patch_applied() -> bool:
    with _PATCH_LOCK:
        return bool(_PATCH_APPLIED)


def get_patch_info() -> Dict[str, Any]:
    with _PATCH_LOCK:
        return {
            "applied": bool(_PATCH_APPLIED),
            "timestamp": _PATCH_TIMESTAMP,
            "fallback_chain": [m for m, _ in _DEFAULT_FALLBACK_CHAIN],
            "max_fallback_attempts": int(_MAX_FALLBACK_ATTEMPTS),
        }


apply_patch()
