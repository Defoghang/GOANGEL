#!/bin/bash
set +e

if [ -z "${STAGE}" ]; then
  echo "STAGE env var requis" >&2
  exit 2
fi

export GH_TOKEN="${GITHUB_TOKEN}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "═══════════════════════════════════════════════════════════"
echo "  GOANGEL STAGE : ${STAGE}"
echo "  Repo          : ${GITHUB_REPOSITORY}"
echo "  Run ID        : ${GITHUB_RUN_ID}"
echo "  Attempt       : ${GITHUB_RUN_ATTEMPT}"
echo "═══════════════════════════════════════════════════════════"

# Git LFS (préinstallé sur ubuntu-latest)
if ! command -v git-lfs &>/dev/null; then
  echo "Installation git-lfs..."
  sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
fi
git lfs install --skip-repo

pip install --upgrade pip -q
pip install -q requests python-dotenv numpy pandas scipy scikit-learn
pip install -q xgboost lightgbm catboost penaltyblog
pip install -q torch --index-url https://download.pytorch.org/whl/cpu

cat > .env <<ENVEOF
BZZOIRO_TOKENS="${TOKENS}"
BZZOIRO_WORKERS_PER_KEY=${WORKERS_PER_KEY}
BZZOIRO_MAX_WORKERS=${MAX_WORKERS}
BZZOIRO_RATE_LIMIT_PER_MINUTE=${RATE_LIMIT}
BZZOIRO_RATE_LIMIT_SAFETY_MARGIN=1
BZZOIRO_HISTORICAL_SEASONS_LOOKBACK=${SEASONS}
BZZOIRO_ENRICH_XG=${ENRICH}
API_MAX_RETRIES=6
API_RETRY_BACKOFF_SECONDS=0.4
API_TIMEOUT_SECONDS=25
GOANGEL_CACHE_DIR=./cache
GOANGEL_LOG_LEVEL=INFO
GOANGEL_GITHUB_PUSH=true
GOANGEL_GITHUB_REPO=${GITHUB_REPOSITORY}
GOANGEL_GITHUB_USER=${GITHUB_REPOSITORY_OWNER}
GOANGEL_GITHUB_BRANCH=cache-auto
GH_PAT=${GITHUB_TOKEN}
GOANGEL_MAX_ATTEMPTS_PER_MODEL=${MAX_ATTEMPTS}
GOANGEL_PB_BAYESIAN_N_SAMPLES=750
GOANGEL_PB_BAYESIAN_BURN=400
GOANGEL_PB_BAYESIAN_N_CHAINS=1
GOANGEL_PB_BAYESIAN_THIN=2
GOANGEL_PB_BAYESIAN_N_CORES=1
ENVEOF

mkdir -p cache
if gh api "repos/${GITHUB_REPOSITORY}/branches/cache-auto" >/dev/null 2>&1; then
  echo "↻ Restauration cache-auto..."
  rm -rf /tmp/ckpt_restore
  git clone --depth 1 --branch cache-auto \
    "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
    /tmp/ckpt_restore 2>/dev/null

  # ← Résoudre les pointeurs LFS en vrais blobs
  if [ -d /tmp/ckpt_restore ]; then
    cd /tmp/ckpt_restore
    if ! git lfs pull 2>&1 | tail -3; then
      echo "⚠️  git lfs pull échoué (continuera avec les fichiers disponibles)"
    fi
    cd "${GITHUB_WORKSPACE}"
  fi

  if [ -d /tmp/ckpt_restore/cache ]; then
    cp -r /tmp/ckpt_restore/cache/. cache/ 2>/dev/null || true
  fi
fi

# ─── Audit pipeline (fuites & sauts de matchs) ───
if [ -f "audit_pipeline.py" ] && [ -f "cache/stages/features.pkl" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "  AUDIT PIPELINE (fuites & sauts)"
  echo "═══════════════════════════════════════════════════════════"
  python3 -u audit_pipeline.py 2>&1 | tee -a audit.log || true
  echo ""
fi

if [ -f "cache/stages/done/${STAGE}.done" ]; then
  echo "✅ Stage ${STAGE} déjà terminé (marqueur .done présent) — skip"
  if [ "${IS_RETRY}" = "true" ]; then
    NEXT=$(python3 main.py --stage next 2>/dev/null | grep '^NEXT:' | cut -d: -f2 | tr -d '[:space:]')
    echo "🔗 Stage suivant détecté (skip chain) : ${NEXT}"
    if [ -n "${NEXT}" ] && [ "${NEXT}" != "DONE" ]; then
      sleep 12
      gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \
        -f seasons_lookback=${SEASONS} \
        -f enrich_xg=${ENRICH} \
        -f workers_per_key=${WORKERS_PER_KEY} \
        -f max_workers=${MAX_WORKERS} \
        -f rate_limit_per_minute=${RATE_LIMIT} \
        -f max_attempts_per_model=${MAX_ATTEMPTS} \
        -f force_stage=${NEXT} \
        -f is_retry=true
      echo "🚀 Nouveau run déclenché pour ${NEXT}"
    else
      echo "🏁 Tous les stages sont done — DONE"
    fi
  fi
  exit 0
fi

mkdir -p cache/stages/attempts
ATTEMPT_FILE="cache/stages/attempts/${STAGE}.json"
if [ -f "${ATTEMPT_FILE}" ]; then
  ATTEMPT_COUNT=$(python3 -c "import json; print(json.load(open('${ATTEMPT_FILE}')).get('count',0))" 2>/dev/null || echo 0)
else
  ATTEMPT_COUNT=0
fi

if [ "${ATTEMPT_COUNT}" -ge "${MAX_ATTEMPTS}" ]; then
  echo "❌ Stage ${STAGE} : ${MAX_ATTEMPTS} tentatives épuisées — ARRÊT DÉFINITIF"
  exit 3
fi

NEW_COUNT=$((ATTEMPT_COUNT + 1))
echo "{\"count\": ${NEW_COUNT}, \"stage\": \"${STAGE}\", \"last_attempt\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"max\": ${MAX_ATTEMPTS}}" > "${ATTEMPT_FILE}"
echo "▶️  Tentative ${NEW_COUNT}/${MAX_ATTEMPTS} pour ${STAGE}"

START_TS=$(date +%s)
timeout --signal=TERM --kill-after=30s 19800 \
  python3 -u main.py --stage "${STAGE}" 2>&1 | tee training.log
RC=${PIPESTATUS[0]}
DURATION=$(($(date +%s) - START_TS))
echo "🏁 Stage ${STAGE} terminé RC=${RC} en ${DURATION}s"

echo "↻ Push cache-auto..."
cd /tmp && rm -rf gopush
git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" gopush 2>/dev/null
cd gopush
git config user.email "actions@users.noreply.github.com"
git config user.name "GOANGEL Auto"
git checkout -B cache-auto

# ── Git LFS : tracker les gros fichiers AVANT git add ──
git lfs install --local --skip-repo
git lfs track "*.pkl"
git lfs track "*.pkl.xz"
git lfs track "*.h5"
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes

# Configurer LFS pour ce repo (upload direct via HTTPS)
git config lfs.https://github.com/${GITHUB_REPOSITORY}.git/info/lfs.access basic
git config lfs.concurrenttransfers 4

mkdir -p cache
if [ -d "${GITHUB_WORKSPACE}/cache" ]; then
  cp -r "${GITHUB_WORKSPACE}/cache/." cache/ 2>/dev/null || true
fi
git add -A
git commit -m "checkpoint ${STAGE} run${GITHUB_RUN_ID} att${NEW_COUNT} rc${RC} $(date +%s)" || echo "no changes"

PUSH_OK=0
for push_attempt in 1 2 3; do
  echo "  Push tentative ${push_attempt}/3..."
  if git push origin cache-auto --force 2>&1; then
    PUSH_OK=1
    break
  fi
  sleep 5
done
cd "${GITHUB_WORKSPACE}"

if [ "${PUSH_OK}" != "1" ]; then
  echo "❌ PUSH cache-auto ÉCHOUÉ après 3 tentatives"
  echo "::error::Impossible de persister l'état sur GitHub — ARRÊT DE LA CHAÎNE"
  exit 5
fi
echo "✅ cache-auto pushé (stage=${STAGE})"

if [ "${RC}" = "0" ]; then
  if [ "${IS_RETRY}" = "true" ]; then
    NEXT=$(python3 main.py --stage next 2>/dev/null | grep '^NEXT:' | cut -d: -f2 | tr -d '[:space:]')
    echo "🔗 Stage suivant détecté : ${NEXT}"

    if [ -z "${NEXT}" ] || [ "${NEXT}" = "DONE" ]; then
      echo ""
      echo "═══════════════════════════════════════════════════════════"
      echo "  🎉 PIPELINE COMPLET — TOUS LES STAGES SONT TERMINÉS"
      echo "═══════════════════════════════════════════════════════════"
      exit 0
    fi

    sleep 20

    TRIGGERED=0
    for trigger_attempt in 1 2 3; do
      echo "  Trigger ${trigger_attempt}/3 pour ${NEXT}..."
      TRIGGER_OUTPUT=$(gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \
        -f seasons_lookback=${SEASONS} \
        -f enrich_xg=${ENRICH} \
        -f workers_per_key=${WORKERS_PER_KEY} \
        -f max_workers=${MAX_WORKERS} \
        -f rate_limit_per_minute=${RATE_LIMIT} \
        -f max_attempts_per_model=${MAX_ATTEMPTS} \
        -f force_stage=${NEXT} \
        -f is_retry=true 2>&1)
      TRIGGER_RC=$?
      echo "  gh workflow run output : ${TRIGGER_OUTPUT}"
      if [ "${TRIGGER_RC}" = "0" ]; then
        echo "  ✅ Trigger réussi (RC=0)"
        TRIGGERED=1
        break
      fi
      echo "  ⚠️  Trigger RC=${TRIGGER_RC} — retry dans 10s..."
      echo "  ⚠️  Trigger échoué, retry dans 10s..."
      sleep 10
    done

    if [ "${TRIGGERED}" = "1" ]; then
      echo "🚀 Nouveau run déclenché pour ${NEXT}"
      sleep 5
      echo "─── Runs récents ───"
      gh run list --repo "${GITHUB_REPOSITORY}" --workflow=train.yml --limit 3 || true
    else
      echo ""
      echo "═══════════════════════════════════════════════════════════"
      echo "  ❌ ÉCHEC DÉCLENCHEMENT WORKFLOW pour ${NEXT}"
      echo "  Après 3 tentatives — ARRÊT DE LA CHAÎNE"
      echo "═══════════════════════════════════════════════════════════"
      echo "::error::Impossible de déclencher le stage suivant ${NEXT}"
      exit 4
    fi
  fi

  exit 0
fi

if [ "${RC}" = "42" ]; then
  echo "⏸️  PB INCOMPLET sur ${STAGE} (RC=42) — soft-retry nouvelle tentative"
  if [ -f ".github/scripts/pb_retry_check.sh" ]; then
    bash .github/scripts/pb_retry_check.sh "${STAGE}" "${MAX_ATTEMPTS}"
    RC_PB=$?
    if [ "${RC_PB}" = "1" ]; then
      echo "❌ Stage ${STAGE} : ${MAX_ATTEMPTS} tentatives PB épuisées — ARRÊT DÉFINITIF"
      exit 3
    fi
  fi
  sleep 12
  gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \
    -f seasons_lookback=${SEASONS} \
    -f enrich_xg=${ENRICH} \
    -f workers_per_key=${WORKERS_PER_KEY} \
    -f max_workers=${MAX_WORKERS} \
    -f rate_limit_per_minute=${RATE_LIMIT} \
    -f max_attempts_per_model=${MAX_ATTEMPTS} \
    -f force_stage=${STAGE} \
    -f is_retry=true
  exit 1
fi

if [ "${RC}" = "124" ] || [ "${RC}" = "143" ]; then
  echo "⏰ TIMEOUT volontaire sur ${STAGE} (RC=${RC}) — relance dans un nouveau run"
  sleep 15
  gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \
    -f seasons_lookback=${SEASONS} \
    -f enrich_xg=${ENRICH} \
    -f workers_per_key=${WORKERS_PER_KEY} \
    -f max_workers=${MAX_WORKERS} \
    -f rate_limit_per_minute=${RATE_LIMIT} \
    -f max_attempts_per_model=${MAX_ATTEMPTS} \
    -f force_stage=${STAGE} \
    -f is_retry=true
  exit 1
fi

echo "❌ HARD FAILURE sur ${STAGE} (RC=${RC}) — ARRÊT DE LA CHAÎNE"
exit 1
