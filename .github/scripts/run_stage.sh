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

echo "↻ Restauration cache-auto..."
if gh api "repos/${GITHUB_REPOSITORY}/branches/cache-auto" >/dev/null 2>&1; then
  rm -rf /tmp/ckpt_restore
  export GIT_LFS_SKIP_SMUDGE=1

  git lfs install --local 2>&1 | tail -1

  git clone --depth 1 --branch cache-auto \
    "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
    /tmp/ckpt_restore 2>&1 | tail -3

  if [ -d /tmp/ckpt_restore ]; then
    cd /tmp/ckpt_restore

    git remote set-url origin \
      "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

    git lfs install --local >/dev/null 2>&1 || true

    echo "  📥 git lfs fetch --all..."
    git lfs fetch --all 2>&1 | tail -2 || true

    echo "  📥 git lfs checkout..."
    git lfs checkout 2>&1 | tail -2 || true

    echo "  📥 git lfs pull origin cache-auto..."
    git lfs pull origin cache-auto 2>&1 | tail -2 || true

    cd "${GITHUB_WORKSPACE}"

    cp -r /tmp/ckpt_restore/cache/. cache/ 2>/dev/null || true

    echo "  🔍 Vérification post-copie :"
    LFS_POINTER=0
    for f in cache/historical.pkl cache/stages/features.pkl; do
      if [ -f "$f" ]; then
        HB=$(head -c 4 "$f" 2>/dev/null || echo "")
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$HB" = "vers" ] || [ "$SZ" -lt 1000 ]; then
          echo "     ❌ POINTEUR LFS : $f ($SZ bytes)"
          LFS_POINTER=1
        else
          echo "     ✅ $f : $SZ bytes"
        fi
      fi
    done

    if [ "$LFS_POINTER" = "1" ]; then
      echo "  ⚠️  Fallback : download LFS par API directe..."
      for f in cache/historical.pkl cache/stages/features.pkl; do
        echo "     Téléchargement : $f"
        lfs_download_by_api "$f" "$f" || true
      done

      echo "  🔍 Vérification après download direct :"
      for f in cache/historical.pkl cache/stages/features.pkl; do
        if [ -f "$f" ]; then
          SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
          HB=$(head -c 4 "$f" 2>/dev/null || echo "")
          if [ "$HB" = "vers" ] || [ "$SZ" -lt 1000 ]; then
            echo "     ❌ $f : encore pointeur ($SZ bytes)"
          else
            echo "     ✅ $f : $SZ bytes (OK)"
          fi
        fi
      done
    fi
  fi
fi

if [ -f "gdrive_sync.py" ]; then
  echo "☁️  Restauration cache depuis Google Drive..."
  python3 gdrive_sync.py restore 2>&1 | tail -10 || true
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

echo "↻ Push cache-auto..."
cd /tmp && rm -rf gopush
git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" gopush 2>/dev/null
cd gopush
git config user.email "actions@users.noreply.github.com"
git config user.name "GOANGEL Auto"
git checkout -B cache-auto

rm -rf cache/checkpoints 2>/dev/null || true

git lfs install --local >/dev/null 2>&1 || true
git lfs track "*.pkl" >/dev/null 2>&1 || true
git lfs track "*.pkl.xz" >/dev/null 2>&1 || true
git lfs track "*.h5" >/dev/null 2>&1 || true
git lfs track "*.pt" >/dev/null 2>&1 || true
git lfs track "*.pth" >/dev/null 2>&1 || true
git add .gitattributes 2>/dev/null || true

mkdir -p cache
if [ -d "${GITHUB_WORKSPACE}/cache" ]; then
  cp -r "${GITHUB_WORKSPACE}/cache/." cache/ 2>/dev/null || true
fi

git add -A
git add -f cache/ 2>/dev/null || true
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
  exit 5
fi
echo "✅ cache-auto pushé (stage=${STAGE})"

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
