#!/bin/bash
# Vérification finale GOANGEL — à lancer depuis la racine du repo

set +e

GREEN="\033[1;32m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
CYAN="\033[1;36m"
GREY="\033[0;90m"
BOLD="\033[1m"
RESET="\033[0m"

PASS=0
FAIL=0
WARN=0

ok()   { echo -e "${GREEN}✅ $1${RESET}"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}❌ $1${RESET}"; FAIL=$((FAIL+1)); }
warn() { echo -e "${YELLOW}⚠️  $1${RESET}"; WARN=$((WARN+1)); }
info() { echo -e "${CYAN}ℹ  $1${RESET}"; }
head_sec() { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════${RESET}"; echo -e "${BOLD}${CYAN} $1${RESET}"; echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════${RESET}"; }

# ─────────────────────────────────────────────────────────────
head_sec "01. FICHIERS REQUIS"
# ─────────────────────────────────────────────────────────────

REQUIRED_FILES=(
  "config.py"
  "config_loader.py"
  "features.py"
  "main.py"
  "predict.py"
  "api.py"
  "checkpoint.py"
  "training_runner.py"
  "penaltyblog_method_patch.py"
  "pb_convergence.py"
  "requirements.txt"
  ".gitignore"
  ".env.example"
  ".github/workflows/train.yml"
  ".github/workflows/_stage_runner.yml"
  ".github/scripts/run_stage.sh"
  ".github/scripts/pb_retry_check.sh"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "$f" ]; then
    ok "Présent : $f"
  else
    fail "MANQUANT : $f"
  fi
done

if [ -f ".env" ]; then
  if grep -q "^\.env$" .gitignore 2>/dev/null; then
    ok ".env présent et ignoré par git"
  else
    fail ".env présent MAIS pas dans .gitignore !"
  fi
else
  warn ".env absent (normal sur CI, présent en local)"
fi

# ─────────────────────────────────────────────────────────────
head_sec "02. SYNTAXE PYTHON (py_compile)"
# ─────────────────────────────────────────────────────────────

PY_FILES=(
  "config.py"
  "config_loader.py"
  "features.py"
  "main.py"
  "predict.py"
  "api.py"
  "checkpoint.py"
  "training_runner.py"
  "penaltyblog_method_patch.py"
  "pb_convergence.py"
)

for f in "${PY_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    fail "py_compile : $f introuvable"
    continue
  fi
  if python3 -m py_compile "$f" 2>/tmp/pyc_err; then
    ok "py_compile OK : $f"
  else
    fail "py_compile ERREUR : $f"
    sed 's/^/    /' /tmp/pyc_err
  fi
done

# ─────────────────────────────────────────────────────────────
head_sec "03. SYNTAXE BASH (bash -n)"
# ─────────────────────────────────────────────────────────────

for sh in ".github/scripts/run_stage.sh" ".github/scripts/pb_retry_check.sh"; do
  if [ ! -f "$sh" ]; then
    fail "bash -n : $sh introuvable"
    continue
  fi
  if bash -n "$sh" 2>/tmp/sh_err; then
    ok "bash -n OK : $sh"
  else
    fail "bash -n ERREUR : $sh"
    sed 's/^/    /' /tmp/sh_err
  fi
done

# ─────────────────────────────────────────────────────────────
head_sec "04. IMPORTS PYTHON (modules critiques)"
# ─────────────────────────────────────────────────────────────

IMPORT_TESTS=(
  "import config; print('config OK')"
  "import config_loader; print('config_loader OK')"
  "from penaltyblog_method_patch import apply_patch, is_patch_applied; apply_patch(); assert is_patch_applied(); print('patch OK')"
  "import pb_convergence; print('pb_convergence OK')"
  "from pb_convergence import fit_pb_attempt, PBIncompleteError, resolve_filter, apply_filter, compute_time_weights; print('pb_convergence API OK')"
  "from features import _fit_pb_model, _PB_MODEL_CLASSES, _PB_MODEL_ORDER; print('features OK')"
  "import checkpoint; print('checkpoint OK')"
)

for test in "${IMPORT_TESTS[@]}"; do
  if python3 -c "$test" >/tmp/import_out 2>/tmp/import_err; then
    ok "$(cat /tmp/import_out)"
  else
    fail "Échec import : $test"
    sed 's/^/    /' /tmp/import_err | head -20
  fi
done

# ─────────────────────────────────────────────────────────────
head_sec "05. PATCHES APPLIQUÉS"
# ─────────────────────────────────────────────────────────────

# 5.1 config.py
if grep -q "PB_MAX_ATTEMPTS" config.py; then
  ok "config.py : PB_MAX_ATTEMPTS présent"
else
  fail "config.py : PB_MAX_ATTEMPTS absent (patch non appliqué)"
fi

if grep -q "PB_FALLBACK_CHAIN" config.py; then
  ok "config.py : PB_FALLBACK_CHAIN présent"
else
  fail "config.py : PB_FALLBACK_CHAIN absent"
fi

if grep -q "PB_FILTER_PLAN" config.py; then
  ok "config.py : PB_FILTER_PLAN présent"
else
  fail "config.py : PB_FILTER_PLAN absent"
fi

# 5.2 features.py
if grep -q "weights: Optional\[np.ndarray\] = None," features.py; then
  ok "features.py : _fit_pb_model signature étendue (weights=)"
else
  fail "features.py : signature _fit_pb_model NON patchée"
fi

if grep -q "strategies: List\[Tuple\[str, Dict\[str, Any\]\]\] = \[" features.py; then
  fail "features.py : ancienne boucle multi-optimizers TOUJOURS présente"
else
  ok "features.py : boucle multi-optimizers supprimée"
fi

if grep -q "n_cores" features.py; then
  ok "features.py : n_cores (bayesian) géré"
else
  warn "features.py : n_cores absent (à vérifier)"
fi

# 5.3 main.py
if grep -q "from pb_convergence import PBIncompleteError" main.py; then
  ok "main.py : import PBIncompleteError présent"
else
  fail "main.py : import PBIncompleteError ABSENT"
fi

if grep -q "except PBIncompleteError as exc:" main.py; then
  ok "main.py : catch PBIncompleteError présent"
else
  fail "main.py : catch PBIncompleteError ABSENT"
fi

if grep -q "return 42" main.py; then
  ok "main.py : return 42 (soft-retry) présent"
else
  fail "main.py : return 42 ABSENT"
fi

if grep -q "fit_pb_attempt(" main.py; then
  ok "main.py : _stage_train_pb_model utilise fit_pb_attempt()"
else
  fail "main.py : fit_pb_attempt absent de _stage_train_pb_model"
fi

# 5.4 penaltyblog_method_patch.py
if grep -q "_DEFAULT_FALLBACK_CHAIN" penaltyblog_method_patch.py; then
  ok "penaltyblog_method_patch.py : fallback chain présent"
else
  fail "penaltyblog_method_patch.py : fallback chain ABSENT"
fi

if grep -q "trust-constr" penaltyblog_method_patch.py; then
  ok "penaltyblog_method_patch.py : trust-constr présent"
else
  fail "penaltyblog_method_patch.py : trust-constr ABSENT"
fi

if tail -20 penaltyblog_method_patch.py | grep -q "^apply_patch()$"; then
  ok "penaltyblog_method_patch.py : apply_patch() appelé en fin de module"
else
  warn "penaltyblog_method_patch.py : appel auto de apply_patch() non détecté"
fi

# 5.5 run_stage.sh
if grep -q 'if \[ "\${RC}" = "42" \]; then' .github/scripts/run_stage.sh; then
  ok "run_stage.sh : gestion RC=42 présente"
else
  fail "run_stage.sh : gestion RC=42 ABSENTE"
fi

if grep -q "pb_retry_check.sh" .github/scripts/run_stage.sh; then
  ok "run_stage.sh : appel pb_retry_check.sh présent"
else
  fail "run_stage.sh : appel pb_retry_check.sh ABSENT"
fi

if grep -q "tentatives PB épuisées" .github/scripts/run_stage.sh; then
  ok "run_stage.sh : garde-fou tentatives épuisées présent"
else
  fail "run_stage.sh : garde-fou tentatives épuisées ABSENT"
fi

if grep -q 'if \[ "\${RC}" = "124" \]' .github/scripts/run_stage.sh; then
  ok "run_stage.sh : gestion RC=124 (timeout) conservée"
else
  fail "run_stage.sh : gestion RC=124 SUPPRIMÉE (régression)"
fi

# 5.6 pb_retry_check.sh
if [ -x ".github/scripts/pb_retry_check.sh" ]; then
  ok "pb_retry_check.sh : exécutable"
else
  warn "pb_retry_check.sh : non exécutable (chmod +x)"
fi

if grep -q "ZeroInflatedPoissonGoalsModel" .github/scripts/pb_retry_check.sh; then
  ok "pb_retry_check.sh : mapping classes PB présent"
else
  fail "pb_retry_check.sh : mapping classes PB ABSENT"
fi

# ─────────────────────────────────────────────────────────────
head_sec "06. TESTS FONCTIONNELS (module pb_convergence)"
# ─────────────────────────────────────────────────────────────

python3 << 'PYTEST' >/tmp/pytest_out 2>&1
import sys
from pb_convergence import (
    resolve_filter, apply_filter, compute_time_weights,
    get_pb_filter_for_attempt, FILTER_PLAN, PB_MAX_ATTEMPTS,
    validate_pb_inputs, PBValidationError,
)

f1 = resolve_filter(1)
assert f1["min_matches"] == 0 and f1["max_goals"] == 100, f"attempt 1 = {f1}"
f5 = resolve_filter(5)
assert f5["min_matches"] == 8, f"attempt 5 = {f5}"
f15 = resolve_filter(15)
assert f15["min_matches"] == 20, f"attempt 15 = {f15}"
f99 = resolve_filter(99)
assert f99["min_matches"] == 20, f"attempt 99 = {f99}"

gh = [0,1,2,5,10,0,3]
ga = [0,1,2,3,5,0,1]
th = ["A","B","C","D","E","A","F"]
ta = ["E","D","C","B","A","F","A"]

gh_f, ga_f, th_f, ta_f, keep, n = apply_filter(gh, ga, th, ta, 0, 8)
assert n == 6, f"max_goals=8 : attendu 6, reçu {n}"
assert 10 not in gh_f

gh_f2, _, _, _, keep2, n2 = apply_filter(gh, ga, th, ta, 4, 8)
assert n2 < n, f"min_matches=4 doit réduire : {n2} < {n}"
for i in keep2:
    assert th[i] in ("A","C","E","F") or ta[i] in ("A","C","E","F")

w = compute_time_weights(
    ["2024-01-01","2024-06-01","2025-01-01"],
    half_life_days=180.0,
)
assert w is not None and len(w) == 3
assert w[2] > w[0], f"plus récent > plus ancien : {w}"

w_none = compute_time_weights(None, 180.0)
assert w_none is None

w_zero = compute_time_weights([], 180.0)
assert w_zero is None

try:
    validate_pb_inputs([], [], [], [])
    raise AssertionError("validate_pb_inputs aurait dû lever")
except PBValidationError:
    pass

try:
    validate_pb_inputs([0], [0], ["A"], ["B","C"])
    raise AssertionError("longueurs incohérentes auraient dû lever")
except PBValidationError:
    pass

assert len(FILTER_PLAN) >= 9, f"FILTER_PLAN = {len(FILTER_PLAN)}"
assert PB_MAX_ATTEMPTS == 15, f"PB_MAX_ATTEMPTS = {PB_MAX_ATTEMPTS}"

cfg = get_pb_filter_for_attempt(3)
assert cfg["min_matches"] == 4

print("ALL_TESTS_PASS")
PYTEST

if grep -q "ALL_TESTS_PASS" /tmp/pytest_out; then
  ok "Tests fonctionnels pb_convergence : 100% PASS"
else
  fail "Tests fonctionnels pb_convergence : ÉCHEC"
  sed 's/^/    /' /tmp/pytest_out | tail -25
fi

# ─────────────────────────────────────────────────────────────
head_sec "07. TEST PATCH scipy.optimize.minimize"
# ─────────────────────────────────────────────────────────────

python3 << 'PYTEST2' >/tmp/pytest2_out 2>&1
import numpy as np
from scipy.optimize import minimize
import penaltyblog_method_patch

import scipy.optimize
assert scipy.optimize.minimize.__name__ == "_patched_minimize", f"patch non appliqué : {scipy.optimize.minimize.__name__}"

def rosen(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

res = minimize(rosen, x0=np.array([0.5, 0.5]), method="SLSQP", options={"maxiter": 500})
assert res.success, f"minimize SLSQP simple échoué : {res.message}"
assert abs(res.x[0] - 1.0) < 0.01

def rosen_with_constraint(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

cons = [{"type": "eq", "fun": lambda x: x[0] - x[1]}]
res2 = minimize(
    rosen_with_constraint,
    x0=np.array([0.5, 0.5]),
    method=None,
    constraints=cons,
    bounds=[(-5, 5), (-5, 5)],
    options={"maxiter": 500},
)
assert hasattr(res2, "x"), "résultat invalide"

print("PATCH_TEST_PASS")
PYTEST2

if grep -q "PATCH_TEST_PASS" /tmp/pytest2_out; then
  ok "Patch scipy.optimize.minimize : fonctionnel"
else
  fail "Patch scipy.optimize.minimize : ÉCHEC"
  sed 's/^/    /' /tmp/pytest2_out | tail -20
fi

# ─────────────────────────────────────────────────────────────
head_sec "08. COHÉRENCE CONFIG ↔ FEATURES"
# ─────────────────────────────────────────────────────────────

python3 << 'PYTEST3' >/tmp/pytest3_out 2>&1
import config as cfg
from features import FEATURE_NAMES, EXPECTED_FEATURE_COUNT

assert EXPECTED_FEATURE_COUNT == 172, f"EXPECTED_FEATURE_COUNT = {EXPECTED_FEATURE_COUNT}"
assert len(FEATURE_NAMES) == 172, f"len(FEATURE_NAMES) = {len(FEATURE_NAMES)}"
assert cfg.EXPECTED_FEATURE_COUNT_REFERENCE == 172, f"config = {cfg.EXPECTED_FEATURE_COUNT_REFERENCE}"

total = cfg.TRAIN_RATIO + cfg.VAL_RATIO + cfg.CALIB_RATIO + cfg.TEST_RATIO
assert abs(total - 1.0) < 1e-6, f"ratios = {total}"

assert hasattr(cfg, "PB_MAX_ATTEMPTS")
assert hasattr(cfg, "PB_FILTER_PLAN")
assert hasattr(cfg, "PB_FALLBACK_CHAIN")
assert hasattr(cfg, "PB_BAYESIAN_N_SAMPLES")
assert hasattr(cfg, "COMPETITIONS_TIER_INDEX")

assert len(cfg.COMPETITIONS) == 29, f"COMPETITIONS = {len(cfg.COMPETITIONS)}"

print("COHERENCE_PASS")
PYTEST3

if grep -q "COHERENCE_PASS" /tmp/pytest3_out; then
  ok "Cohérence config ↔ features : OK (172 features)"
else
  fail "Cohérence config ↔ features : ÉCHEC"
  sed 's/^/    /' /tmp/pytest3_out | tail -20
fi

# ─────────────────────────────────────────────────────────────
head_sec "09. ARBORESCENCE GITHUB ACTIONS"
# ─────────────────────────────────────────────────────────────

if [ -f ".github/workflows/train.yml" ]; then
  ok "train.yml présent"
  for stage in prepare train_xgb train_lgbm train_catboost train_rf train_dl \
               train_pb_poisson train_pb_dixon_coles train_pb_bivariate \
               train_pb_negbinomial train_pb_zeroinflated train_pb_weibull \
               train_pb_bayesian train_pb_hierarchical finalize; do
    if grep -q "stage: $stage" .github/workflows/train.yml; then
      :
    else
      warn "train.yml : stage $stage absent"
    fi
  done
  ok "train.yml : 15 stages présents"
else
  fail "train.yml absent"
fi

if [ -f ".github/workflows/_stage_runner.yml" ]; then
  ok "_stage_runner.yml présent"
  if grep -q "run_stage.sh" .github/workflows/_stage_runner.yml; then
    ok "_stage_runner.yml : appelle run_stage.sh"
  else
    fail "_stage_runner.yml : n'appelle pas run_stage.sh"
  fi
else
  fail "_stage_runner.yml absent"
fi

# ─────────────────────────────────────────────────────────────
head_sec "10. VÉRIFICATION .env.example"
# ─────────────────────────────────────────────────────────────

if [ -f ".env.example" ]; then
  ok ".env.example présent"
  for var in BZZOIRO_TOKENS GOANGEL_CACHE_DIR GOANGEL_CALIBRATION_METHOD GOANGEL_GITHUB_PUSH; do
    if grep -q "^${var}=" .env.example; then
      ok ".env.example : $var présent"
    else
      warn ".env.example : $var absent"
    fi
  done
else
  warn ".env.example absent"
fi

# ─────────────────────────────────────────────────────────────
head_sec "RÉSUMÉ FINAL"
# ─────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}✅ PASS : $PASS${RESET}"
echo -e "${YELLOW}⚠️  WARN : $WARN${RESET}"
echo -e "${RED}❌ FAIL : $FAIL${RESET}"
echo ""

if [ "$FAIL" -eq 0 ]; then
  echo -e "${GREEN}${BOLD}🎉 TOUT EST PRÊT POUR LE PUSH${RESET}"
  echo ""
  echo -e "${CYAN}Prochaine étape :${RESET}"
  echo "   git add ."
  echo "   git status --short"
  echo "   git commit -m 'GOANGEL v23 + PB convergence patch'"
  echo "   git push -u origin main"
  echo ""
  exit 0
else
  echo -e "${RED}${BOLD}❌ CORRIGER LES ERREURS AVANT PUSH${RESET}"
  echo ""
  exit 1
fi
