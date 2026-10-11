#!/usr/bin/env python3
"""Audit pipeline GOANGEL — fuites & sauts de matchs.
Fonctionne sur le runner GitHub (cache-auto cloné en local).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = [0]
FAIL = [0]
WARN = [0]


def ok(msg):
    PASS[0] += 1
    print(f"{GREEN}✅ {msg}{RESET}", flush=True)


def fail(msg):
    FAIL[0] += 1
    print(f"{RED}❌ {msg}{RESET}", flush=True)


def warn(msg):
    WARN[0] += 1
    print(f"{YELLOW}⚠️  {msg}{RESET}", flush=True)


def head(msg):
    print(f"\n{BOLD}{CYAN}═══ {msg} ═══{RESET}", flush=True)


def _read_pickle_any(path):
    import lzma, gzip
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic[:6] == b"\xfd7zXZ\x00":
        with open(path, "rb") as f:
            return pickle.loads(lzma.decompress(f.read()))
    if magic[:2] == b"\x1f\x8b":
        with open(path, "rb") as f:
            return pickle.loads(gzip.decompress(f.read()))
    with open(path, "rb") as f:
        return pickle.load(f)


def run_audit(cache_dir="cache"):
    cache = Path(cache_dir)
    features_file = cache / "stages" / "features.pkl"
    historical_file = cache / "historical.pkl"
    metadata_file = cache / "stages" / "metadata.pkl"

    head("1. Présence artefacts")
    _CRITICAL = {"features.pkl", "historical.pkl"}
    for f in (features_file, historical_file, metadata_file):
        if not f.exists():
            fail(f"Introuvable : {f}")
            if f.name in _CRITICAL:
                return 1
            continue
        size = f.stat().st_size
        if size < 1024:
            if f.name in _CRITICAL:
                fail(f"{f.name} = {size} bytes (pointeur LFS non résolu)")
                return 1
            else:
                warn(f"{f.name} = {size} bytes (pointeur LFS, non critique)")
                continue
        ok(f"{f.name} : {size/1024/1024:.1f} MB")

    try:
        df_past = _read_pickle_any(historical_file)
        ok(f"historical.pkl : {len(df_past)} matchs")
    except Exception as exc:
        fail(f"Lecture historical : {exc}")
        return 1

    try:
        payload = _read_pickle_any(features_file)
        X = payload["X"]
        y = payload["y"]
        gh_all = payload["gh_all"]
        ga_all = payload["ga_all"]
        ok(f"features.pkl : X{X.shape} y{y.shape}")
    except Exception as exc:
        fail(f"Lecture features : {exc}")
        return 1

    head("2. Chronologie")
    if "date_parsed" not in df_past.columns:
        fail("date_parsed absent")
    else:
        dates = pd.to_datetime(df_past["date_parsed"], errors="coerce", utc=True)
        if dates.isna().any():
            fail(f"{dates.isna().sum()} dates invalides")
        else:
            ok("Toutes les dates valides")
        if dates.is_monotonic_increasing:
            ok("Dates triées chronologiquement")
        else:
            warn("Dates non triées (vérifier sort_values)")
        span = (dates.max() - dates.min()).days
        ok(f"Plage : {span} jours ({span/365:.1f} ans)")
        print(f"   Du {dates.min().date()} au {dates.max().date()}")

    head("3. Comptage matchs")
    n_hist = len(df_past)
    n_feat = X.shape[0]
    print(f"   historical : {n_hist}")
    print(f"   features   : {n_feat}")
    ratio = n_feat / max(n_hist, 1)
    if ratio >= 0.98:
        ok(f"Ratio features/historical = {ratio*100:.1f}% → pas de warmup")
    elif ratio >= 0.90:
        warn(f"Ratio = {ratio*100:.1f}% → warmup modéré")
    else:
        fail(f"Ratio = {ratio*100:.1f}% → {n_hist - n_feat} matchs skippés (fuite potentielle)")

    head("4. Split chronologique")
    try:
        import config as c
        tr, va, ca, te = c.TRAIN_RATIO, c.VAL_RATIO, c.CALIB_RATIO, c.TEST_RATIO
    except Exception:
        tr, va, ca, te = 0.65, 0.10, 0.10, 0.15
    total = tr + va + ca + te
    if abs(total - 1.0) > 1e-6:
        fail(f"Ratios somment à {total} ≠ 1.0")
    else:
        ok(f"Ratios OK : {tr}/{va}/{ca}/{te}")

    n = n_feat
    cut_tr = int(n * tr)
    cut_va = int(n * (tr + va))
    cut_ca = int(n * (tr + va + ca))
    print(f"   TRAIN : [0:{cut_tr}]      = {cut_tr}")
    print(f"   VAL   : [{cut_tr}:{cut_va}]  = {cut_va-cut_tr}")
    print(f"   CALIB : [{cut_va}:{cut_ca}]  = {cut_ca-cut_va}")
    print(f"   TEST  : [{cut_ca}:{n}]      = {n-cut_ca}")

    head("5. Distribution labels")
    u, cnt = np.unique(y, return_counts=True)
    names = {0: "HOME", 1: "DRAW", 2: "AWAY"}
    for ui, ci in zip(u.tolist(), cnt.tolist()):
        print(f"   {names.get(ui, ui):5s} : {ci:>6} ({ci/n*100:.1f}%)")

    head("6. Filtre attempt 1 PB")
    try:
        import pb_convergence as pbc
        p1 = pbc.FILTER_PLAN.get(1, {})
        mm = p1.get("min_matches", -1)
        mg = p1.get("max_goals", -1)
        if mm == 0 and mg >= 100:
            ok(f"Attempt 1 : min_matches={mm}, max_goals={mg} → tous les matchs")
        else:
            warn(f"Attempt 1 : min_matches={mm}, max_goals={mg} → filtré")
    except Exception as exc:
        warn(f"Import pb_convergence : {exc}")

    head("7. Détection fuite (LogReg sur échantillon)")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        smp = min(5000, X.shape[0])
        Xs = np.where(np.isfinite(X[:smp]), X[:smp], 0.0)
        Xs = StandardScaler().fit_transform(Xs)
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(Xs, y[:smp])
        acc = clf.score(Xs, y[:smp])
        print(f"   LogReg train accuracy : {acc*100:.1f}%")
        if acc > 0.95:
            fail(f"Accuracy {acc*100:.1f}% > 95% → FUITE")
        elif acc > 0.75:
            warn(f"Accuracy {acc*100:.1f}% → vérifier")
        else:
            ok(f"Accuracy {acc*100:.1f}% → pas de fuite triviale")
    except Exception as exc:
        warn(f"Test fuite : {exc}")

    head("8. Qualité features")
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    n_tot = X.size
    print(f"   Total      : {n_tot:,}")
    print(f"   NaN        : {n_nan:,} ({n_nan/n_tot*100:.2f}%)")
    print(f"   Inf        : {n_inf:,} ({n_inf/n_tot*100:.2f}%)")
    col_nan = np.isnan(X).mean(axis=0)
    empty = int((col_nan > 0.99).sum())
    if empty > 0:
        warn(f"{empty} features 99%+ NaN")
    else:
        ok("Aucune feature vide")
    col_std = np.nanstd(X, axis=0)
    const = int((col_std < 1e-10).sum())
    if const > 10:
        warn(f"{const} features constantes")
    elif const > 0:
        print(f"   {const} features constantes (OK)")
    else:
        ok("Aucune feature constante")

    head("RÉSUMÉ")
    print()
    print(f"{GREEN}✅ PASS : {PASS[0]}{RESET}")
    print(f"{YELLOW}⚠️  WARN : {WARN[0]}{RESET}")
    print(f"{RED}❌ FAIL : {FAIL[0]}{RESET}")
    print()
    if FAIL[0] == 0:
        print(f"{GREEN}{BOLD}🎉 AUDIT RÉUSSI{RESET}")
        return 0
    print(f"{RED}{BOLD}❌ {FAIL[0]} PROBLÈME(S){RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(run_audit("cache"))
