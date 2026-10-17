#!/bin/bash
set +e

if [ -z "${STAGE}" ]; then
  echo "STAGE env var requis" >&2
  exit 2
fi

export GH_TOKEN="${GITHUB_TOKEN}"

echo "═══════════════════════════════════════════════════════════"
echo "  GOANGEL STAGE : ${STAGE}"
echo "  Repo          : ${GITHUB_REPOSITORY}"
echo "  Run ID        : ${GITHUB_RUN_ID}"
echo "  Attempt       : ${GITHUB_RUN_ATTEMPT}"
echo "═══════════════════════════════════════════════════════════"

lfs_download_by_api() {
  local FILE_PATH="$1"
  local DEST_PATH="$2"
  local POINTER_URL="https://raw.githubusercontent.com/${GITHUB_REPOSITORY}/cache-auto/${FILE_PATH}"

  local POINTER_CONTENT
  POINTER_CONTENT=$(curl -sS -H "Authorization: token ${GITHUB_TOKEN}" "${POINTER_URL}" 2>/dev/null)

  local OID SIZE
  OID=$(echo "${POINTER_CONTENT}" | grep -oE 'sha256:[a-f0-9]+' | cut -d: -f2 | head -1)
  SIZE=$(echo "${POINTER_CONTENT}" | grep -oE 'size [0-9]+' | awk '{print $2}' | head -1)

  if [ -z "${OID}" ] || [ -z "${SIZE}" ]; then
    return 1
  fi

  local BATCH_RESPONSE
  BATCH_RESPONSE=$(curl -sS -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.git-lfs+json" \
    -H "Content-Type: application/vnd.git-lfs+json" \
    -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"${OID}\",\"size\":${SIZE}}]}" \
    "https://github.com/${GITHUB_REPOSITORY}.git/info/lfs/objects/batch" 2>/dev/null)

  local DOWNLOAD_URL
  DOWNLOAD_URL=$(echo "${BATCH_RESPONSE}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for o in d.get('objects', []):
        a = o.get('actions', {})
        if 'download' in a:
            print(a['download']['href'])
            break
except: pass
" 2>/dev/null)

  if [ -z "${DOWNLOAD_URL}" ]; then
    return 1
  fi

  mkdir -p "$(dirname "${DEST_PATH}")"
  if curl -sS -L -o "${DEST_PATH}" "${DOWNLOAD_URL}" 2>/dev/null; then
    local DL_SIZE
    DL_SIZE=$(stat -c%s "${DEST_PATH}" 2>/dev/null || echo 0)
    [ "${DL_SIZE}" = "${SIZE}" ] && return 0
  fi
  return 1
}

if ! command -v git-lfs &>/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
fi
git lfs install --skip-repo >/dev/null 2>&1 || true

if ! command -v rclone &>/dev/null; then
  curl -sS https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || true
fi

echo "🔍 Diagnostic rclone :"
echo "  rclone binary : $(command -v rclone || echo 'ABSENT')"
echo "  RCLONE_CONFIG : ${#RCLONE_CONFIG} caractères"

if [ -z "${RCLONE_CONFIG}" ]; then
  echo "  ❌ RCLONE_CONFIG NON DÉFINI"
else
  echo "${RCLONE_CONFIG}" | base64 -d > /tmp/rclone.conf 2>/dev/null
  B64_RC=$?
  if [ "$B64_RC" = "0" ] && [ -s /tmp/rclone.conf ]; then
    chmod 600 /tmp/rclone.conf
    echo "  ✅ rclone.conf créé : $(stat -c%s /tmp/rclone.conf) bytes"
  else
    echo "  ⚠️  Décodage base64 échoué (RC=$B64_RC), fallback brut"
    echo "${RCLONE_CONFIG}" > /tmp/rclone.conf
    chmod 600 /tmp/rclone.conf
  fi
fi

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
GOANGEL_PB_BAYESIAN_N_SAMPLES=1200
GOANGEL_PB_BAYESIAN_BURN=600
GOANGEL_PB_BAYESIAN_N_CHAINS=1
GOANGEL_PB_BAYESIAN_THIN=1
GOANGEL_PB_BAYESIAN_N_CORES=1
ENVEOF

mkdir -p cache

echo "☁️  Restauration cache depuis Google Drive + GitHub..."

mkdir -p cache/stages/done cache/stages/attempts cache/stages/models

# ─── 1. Drive en priorité (gros fichiers) ───
if [ -f /tmp/rclone.conf ]; then
  mkdir -p cache/stages/done cache/stages/attempts cache/stages/models cache/stages/finalize
  echo "  📥 Drive : cache/..."
  rclone --config /tmp/rclone.conf copy gdrive:GOANGEL/cache/ cache/ \
    --transfers 8 --checkers 16 \
    --exclude "*.tmp" --exclude "*.log" \
    --ignore-existing --ignore-errors \
    --stats-one-line --stats 10s 2>&1 | tail -3 || true

  echo "  📥 Drive : cache/stages/models/ (modèles critiques)"
  rclone --config /tmp/rclone.conf copy gdrive:GOANGEL/cache/stages/models/ cache/stages/models/ \
    --transfers 8 --checkers 16 --ignore-existing 2>&1 | tail -2 || true

  mkdir -p cache/stages/finalize cache/stages/models
  echo "  📥 Drive : cache/stages/finalize/"
  rclone --config /tmp/rclone.conf copy gdrive:GOANGEL/cache/stages/finalize/ cache/stages/finalize/ \
    --transfers 8 --checkers 16 --ignore-existing --ignore-errors 2>&1 | tail -2 || true

  echo "  🔍 Vérification :"
  for f in cache/historical.pkl cache/stages/features.pkl; do
    if [ -f "$f" ]; then
      SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
      HB=$(head -c 4 "$f" 2>/dev/null || echo "")
      if [ "$HB" = "vers" ] || [ "$SZ" -lt 1000 ]; then
        echo "     ❌ $f : pointeur LFS ($SZ bytes)"
      else
        echo "     ✅ $f : $SZ bytes"
      fi
    fi
  done
fi

# ─── 2. GitHub : marqueurs .done + compteurs .json + metadata ───
if gh api "repos/${GITHUB_REPOSITORY}/branches/cache-auto" >/dev/null 2>&1; then
  rm -rf /tmp/ckpt_restore
  git clone --depth 1 --branch cache-auto \
    "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
    /tmp/ckpt_restore 2>&1 | tail -2

  if [ -d /tmp/ckpt_restore/cache ]; then
    echo "  📥 GitHub : marqueurs (.done, .json) + metadata"
    mkdir -p cache/stages/done cache/stages/attempts
    cp -f /tmp/ckpt_restore/cache/stages/done/*.done cache/stages/done/ 2>/dev/null || true
    cp -f /tmp/ckpt_restore/cache/stages/attempts/*.json cache/stages/attempts/ 2>/dev/null || true
    cp -f /tmp/ckpt_restore/cache/stages/metadata.pkl cache/stages/metadata.pkl 2>/dev/null || true
  fi
fi


if [ -f "cache/stages/done/${STAGE}.done" ]; then
  echo "✅ Stage ${STAGE} déjà terminé (marqueur .done présent) — skip"
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
python3 -u main.py --stage "${STAGE}" 2>&1 | tee training.log
RC=${PIPESTATUS[0]}
DURATION=$(($(date +%s) - START_TS))
echo "🏁 Stage ${STAGE} terminé RC=${RC} en ${DURATION}s"

echo "☁️  Push cache → Google Drive + GitHub (marqueurs)..."

# ─── 1. Push Drive (tout le cache) ───
if [ -f /tmp/rclone.conf ]; then
  # Garde-fou : ne pusher QUE si cache local est substantiel (> 10 MB)
  CACHE_MB=$(du -sm cache 2>/dev/null | awk '{print $1}' || echo 0)
  echo "  📊 Cache local : ${CACHE_MB} MB"

  if [ "$CACHE_MB" -lt 10 ]; then
    echo "  ⚠️  Cache trop petit (${CACHE_MB} MB < 10 MB) — skip Drive push (évite écrasement)"
  else
    echo "  📤 Drive : copy cache/ (non destructif)..."
    rclone --config /tmp/rclone.conf copy cache/ gdrive:GOANGEL/cache/ \
      --transfers 8 --checkers 16 \
      --exclude "*.tmp" --exclude "*.log" \
      --exclude "checkpoints/**" \
      --stats-one-line --stats 10s 2>&1 | tail -3 || true
    echo "  ✅ Drive pushé"

    # Push spécifique des modèles (sans --ignore-existing pour mise à jour)
    if [ -d cache/stages/models ] && [ "$(ls -A cache/stages/models 2>/dev/null)" ]; then
      echo "  📤 Drive : cache/stages/models/ (mise à jour)"
      rclone --config /tmp/rclone.conf copy cache/stages/models/ gdrive:GOANGEL/cache/stages/models/ \
        --transfers 8 --checkers 16 2>&1 | tail -2 || true
    fi
  fi
fi

# ─── 2. Push GitHub cache-auto (marqueurs uniquement) ───
echo "  📤 GitHub cache-auto (marqueurs)..."

# Créer .gitignore pour cache-auto (exclure tous les .pkl sauf metadata)
mkdir -p /tmp/goangel-gitignore
cat > /tmp/goangel-gitignore/.gitignore <<'GIEOF'
# Cache local (gros fichiers → Drive)
cache/**/*.pkl
cache/*.pkl
!cache/stages/metadata.pkl

# Logs & temp
*.tmp
*.log
checkpoints/
checkpoints/**
cache/checkpoints/
GIEOF

cd /tmp && rm -rf gopush
git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" gopush 2>/dev/null
cd gopush
git config user.email "actions@users.noreply.github.com"
git config user.name "GOANGEL Auto"
git checkout -B cache-auto

# Copier .gitignore
cp /tmp/goangel-gitignore/.gitignore .gitignore

# Nettoyer les .pkl anciens du repo (migration LFS → Drive)
git rm -r --cached cache/ 2>/dev/null || true

# Recréer la structure avec SEULEMENT les petits fichiers
mkdir -p cache/stages/done cache/stages/attempts
cp -f "${GITHUB_WORKSPACE}/cache/stages/done/"*.done cache/stages/done/ 2>/dev/null || true
cp -f "${GITHUB_WORKSPACE}/cache/stages/attempts/"*.json cache/stages/attempts/ 2>/dev/null || true
cp -f "${GITHUB_WORKSPACE}/cache/stages/metadata.pkl" cache/stages/metadata.pkl 2>/dev/null || true

# Supprimer LFS tracking (.gitattributes)
rm -f .gitattributes

git add -A
git commit -m "checkpoint ${STAGE} run${GITHUB_RUN_ID} att${NEW_COUNT} rc${RC} $(date +%s)" || echo "no changes"

PUSH_OK=0
for push_attempt in 1 2 3; do
  echo "  Push GitHub tentative ${push_attempt}/3..."
  if git push origin cache-auto --force 2>&1 | tail -3; then
    PUSH_OK=1
    break
  fi
  sleep 5
done
cd "${GITHUB_WORKSPACE}"

if [ "${PUSH_OK}" != "1" ]; then
  echo "  ⚠️  GitHub push échoué (non critique, Drive reste source de vérité)"
fi

echo "✅ Cache sauvegardé (Drive + GitHub)"

if [ -f "gdrive_sync.py" ]; then
  echo "☁️  Backup cache vers Google Drive..."
  python3 gdrive_sync.py backup 2>&1 | tail -10 || true
fi

if [ "${RC}" = "0" ]; then
  echo "✅ Stage ${STAGE} terminé avec succès (RC=0)"
  exit 0
fi

if [ "${RC}" = "42" ]; then
  echo "⏸️  PB INCOMPLET sur ${STAGE} (RC=42) — soft-retry"
  exit 1
fi

if [ "${RC}" = "124" ] || [ "${RC}" = "143" ]; then
  echo "⏰ TIMEOUT sur ${STAGE} (RC=${RC})"
  exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  ❌ HARD FAILURE sur ${STAGE} (RC=${RC})"
echo "  🛑 ARRÊT TOTAL DE LA CHAÎNE"
echo "═══════════════════════════════════════════════════════════"
exit 1
