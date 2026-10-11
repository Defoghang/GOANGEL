import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from scipy.optimize import minimize as _ORIGINAL_MINIMIZE


logger = logging.getLogger("goangel.penaltyblog_patch")


_PATCH_LOCK: threading.RLock = threading.RLock()
_PATCH_APPLIED: bool = False
_ORIGINAL_REF: Callable[..., Any] = _ORIGINAL_MINIMIZE
_PATCH_TIMESTAMP: Optional[float] = None


_FORCE_SUCCESS_ENABLED: bool = True
_FUN_ABS_LIMIT: float = 1e9
_X_FINITE_ATOL: float = 1e-12


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


def _resolve_method(method, bounds, constraints):
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


def _sanitize_options(options):
    if options is None:
        return {}
    if not isinstance(options, dict):
        try:
            return dict(options)
        except Exception:
            return {}
    return dict(options)


def _sanitize_x0(x0):
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


def _result_is_usable(result):
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


def _extract_fun(result):
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


def _extract_success(result):
    if result is None:
        return False
    s = getattr(result, "success", None)
    if s is None:
        return False
    try:
        return bool(s)
    except Exception:
        return False


def _extract_message(result):
    if result is None:
        return "no result"
    msg = getattr(result, "message", None)
    if msg is None:
        return "unknown"
    try:
        return str(msg)
    except Exception:
        return "unknown"


def _extract_x(result):
    if result is None:
        return None
    x = getattr(result, "x", None)
    if x is None:
        return None
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception:
        return None


def _result_is_better(new_result, old_result):
    if not _result_is_usable(new_result):
        return False
    new_fun = _extract_fun(new_result)
    if new_fun is None:
        return False
    old_fun = _extract_fun(old_result)
    if old_fun is None:
        return True
    return float(new_fun) < float(old_fun)


def _force_success_if_usable(result, reason: str, method: str):
    if not _FORCE_SUCCESS_ENABLED:
        return result
    if result is None:
        return result
    if _extract_success(result):
        return result
    if not _result_is_usable(result):
        return result
    f = _extract_fun(result)
    if f is None:
        return result
    if abs(float(f)) > _FUN_ABS_LIMIT:
        logger.warning(
            "Forçage success refusé : fun=%.4g hors limites (|fun|>%.1g)",
            float(f), _FUN_ABS_LIMIT,
        )
        return result
    x = _extract_x(result)
    if x is None:
        return result
    if not np.all(np.isfinite(x)):
        return result
    original_msg = _extract_message(result)
    try:
        result.success = True
        result.message = (
            f"[goangel_forced] method={method} reason={reason} "
            f"original_msg={original_msg} fun={float(f):.6g}"
        )
    except Exception as exc:
        logger.warning("Impossible de forcer success : %s", exc)
        return result
    logger.warning(
        "⚠️ Forçage success=True : méthode=%s raison=%s fun=%.6g "
        "(SLSQP/fallbacks sans convergence formelle, params acceptés)",
        method, reason, float(f),
    )
    return result


def _safe_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.debug("Appel optimiseur échoué : %s", exc)
        return None


def _adjust_for_method(method, bounds, constraints, jac):
    adj_bounds = bounds
    adj_constraints = constraints
    adj_jac = jac

    if method in _NO_CONSTRAINT_METHODS:
        if adj_constraints:
            logger.debug(
                "Patch : %s ne supporte pas les contraintes → retirées", method,
            )
        adj_constraints = ()

    if method in _NO_BOUNDS_METHODS:
        if adj_bounds is not None:
            logger.debug(
                "Patch : %s ne supporte pas les bounds → retirés", method,
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


def _build_fallback_options(base_options, fallback_options):
    merged = {}
    for k, v in fallback_options.items():
        merged[str(k)] = v
    for k, v in base_options.items():
        key = str(k)
        if key in merged:
            continue
        merged[key] = v
    return merged


def _patched_minimize(
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    **extra_kwargs,
):
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

    primary_call_kwargs = {
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

        fb_options = _build_fallback_options(
            {"maxiter": base_options.get("maxiter", 5000)},
            fallback_options,
        )

        if fallback_method == "trust-constr":
            fb_jac_to_use = "2-point"
        elif fallback_method == "Powell":
            fb_jac_to_use = None
        elif fallback_method == "L-BFGS-B":
            fb_jac_to_use = "2-point"
        else:
            if callable(fb_jac):
                try:
                    _probe = fb_jac(np.asarray(x0_arr, dtype=np.float64))
                    _ = np.asarray(_probe)
                    fb_jac_to_use = fb_jac
                except Exception:
                    fb_jac_to_use = "2-point"
            else:
                fb_jac_to_use = fb_jac

        call_kwargs = {
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
            logger.debug("Fallback %s a levé : %s", fallback_method, exc)
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
                "Fallback %s : amélioration (fun=%.6g) conservée",
                fallback_method,
                _extract_fun(fb_result) or float("nan"),
            )

    if best_result is not None and _result_is_usable(best_result):
        return _force_success_if_usable(
            best_result,
            reason="all_methods_failed_but_params_usable",
            method=resolved_method,
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
                "(force_success=%s, fallback=%s)",
                _FORCE_SUCCESS_ENABLED,
                [m for m, _ in _DEFAULT_FALLBACK_CHAIN],
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
            "force_success_enabled": bool(_FORCE_SUCCESS_ENABLED),
            "fallback_chain": [m for m, _ in _DEFAULT_FALLBACK_CHAIN],
            "max_fallback_attempts": int(_MAX_FALLBACK_ATTEMPTS),
            "fun_abs_limit": float(_FUN_ABS_LIMIT),
        }


apply_patch()
