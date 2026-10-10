import os
import shutil
import sys
from pathlib import Path

CANDIDATES = (
    Path(".github") / "scripts" / "run_stage.sh",
    Path(".github/scripts/run_stage.sh"),
    Path("run_stage.sh"),
)

BACKUP_SUFFIX = ".bak_pre_pb_patch"

OLD_RC_BLOCK = '''if [ "${RC}" = "124" ] || [ "${RC}" = "143" ]; then
  echo "⏰ TIMEOUT volontaire sur ${STAGE} (RC=${RC}) — relance dans un nouveau run"
  sleep 15
  gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \\
    -f seasons_lookback=${SEASONS} \\
    -f enrich_xg=${ENRICH} \\
    -f workers_per_key=${WORKERS_PER_KEY} \\
    -f max_workers=${MAX_WORKERS} \\
    -f rate_limit_per_minute=${RATE_LIMIT} \\
    -f max_attempts_per_model=${MAX_ATTEMPTS} \\
    -f force_stage=${STAGE} \\
    -f is_retry=true
  exit 1
fi
'''

NEW_RC_BLOCK = '''if [ "${RC}" = "42" ]; then
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
  gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \\
    -f seasons_lookback=${SEASONS} \\
    -f enrich_xg=${ENRICH} \\
    -f workers_per_key=${WORKERS_PER_KEY} \\
    -f max_workers=${MAX_WORKERS} \\
    -f rate_limit_per_minute=${RATE_LIMIT} \\
    -f max_attempts_per_model=${MAX_ATTEMPTS} \\
    -f force_stage=${STAGE} \\
    -f is_retry=true
  exit 1
fi

if [ "${RC}" = "124" ] || [ "${RC}" = "143" ]; then
  echo "⏰ TIMEOUT volontaire sur ${STAGE} (RC=${RC}) — relance dans un nouveau run"
  sleep 15
  gh workflow run train.yml --repo "${GITHUB_REPOSITORY}" \\
    -f seasons_lookback=${SEASONS} \\
    -f enrich_xg=${ENRICH} \\
    -f workers_per_key=${WORKERS_PER_KEY} \\
    -f max_workers=${MAX_WORKERS} \\
    -f rate_limit_per_minute=${RATE_LIMIT} \\
    -f max_attempts_per_model=${MAX_ATTEMPTS} \\
    -f force_stage=${STAGE} \\
    -f is_retry=true
  exit 1
fi
'''

NEW_PB_RETRY_HELPER = '''#!/bin/bash
set +e

STAGE="$1"
MAX_ATTEMPTS="${2:-15}"

if [ -z "${STAGE}" ]; then
  echo "usage: pb_retry_check.sh <stage> [max_attempts]" >&2
  exit 0
fi

if [ -z "${MAX_ATTEMPTS}" ]; then
  MAX_ATTEMPTS=15
fi

PB_STATE_FILE="cache/stages/pb_convergence/attempts/${STAGE}.json"

if [ ! -f "${PB_STATE_FILE}" ]; then
  PB_STAGE_NAME=$(echo "${STAGE}" | sed 's/^train_pb_//')
  case "${PB_STAGE_NAME}" in
    poisson)           PB_CLASS="PoissonGoalsModel" ;;
    dixon_coles)       PB_CLASS="DixonColesGoalModel" ;;
    bivariate)         PB_CLASS="BivariatePoissonGoalModel" ;;
    negbinomial)       PB_CLASS="NegativeBinomialGoalModel" ;;
    zeroinflated)      PB_CLASS="ZeroInflatedPoissonGoalsModel" ;;
    weibull)           PB_CLASS="WeibullCopulaGoalsModel" ;;
    bayesian)          PB_CLASS="BayesianGoalModel" ;;
    hierarchical)      PB_CLASS="HierarchicalBayesianGoalModel" ;;
    *)                 PB_CLASS="" ;;
  esac
  if [ -n "${PB_CLASS}" ]; then
    PB_STATE_FILE="cache/stages/pb_convergence/attempts/${PB_CLASS}.json"
  fi
fi

if [ ! -f "${PB_STATE_FILE}" ]; then
  exit 0
fi

PB_ATTEMPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('count',0))" "${PB_STATE_FILE}" 2>/dev/null || echo 0)

if [ "${PB_ATTEMPT}" -ge "${MAX_ATTEMPTS}" ]; then
  exit 1
fi

exit 0
'''

MARKER_NEW = 'if [ "${RC}" = "42" ]; then'
MARKER_HELPER_CALL = 'pb_retry_check.sh'


def _find_script() -> Path:
    for c in CANDIDATES:
        if c.exists():
            return c
    raise SystemExit("❌ run_stage.sh introuvable (ni .github/scripts/run_stage.sh ni ./run_stage.sh)")


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _backup(path: Path) -> None:
    try:
        b = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        shutil.copy2(path, b)
    except Exception as exc:
        print(f"⚠️  Backup non créé : {exc}")


def _patch_run_stage(src: str) -> str:
    if MARKER_NEW in src and MARKER_HELPER_CALL in src:
        return src

    if OLD_RC_BLOCK in src:
        return src.replace(OLD_RC_BLOCK, NEW_RC_BLOCK, 1)

    old_line = 'if [ "${RC}" = "124" ] || [ "${RC}" = "143" ]; then'
    if old_line in src:
        src = src.replace(
            old_line,
            'if [ "${RC}" = "124" ] || [ "${RC}" = "143" ] || [ "${RC}" = "42" ]; then',
            1,
        )
        old_echo = 'echo "⏰ TIMEOUT volontaire sur ${STAGE} (RC=${RC}) — relance dans un nouveau run"'
        new_echo = (
            'if [ "${RC}" = "42" ]; then\n'
            '    echo "⏸️  PB INCOMPLET sur ${STAGE} (RC=42) — soft-retry nouvelle tentative"\n'
            '    if [ -f ".github/scripts/pb_retry_check.sh" ]; then\n'
            '      bash .github/scripts/pb_retry_check.sh "${STAGE}" "${MAX_ATTEMPTS}"\n'
            '      RC_PB=$?\n'
            '      if [ "${RC_PB}" = "1" ]; then\n'
            '        echo "❌ Stage ${STAGE} : ${MAX_ATTEMPTS} tentatives PB épuisées — ARRÊT DÉFINITIF"\n'
            '        exit 3\n'
            '      fi\n'
            '    fi\n'
            '  else\n'
            '    echo "⏰ TIMEOUT volontaire sur ${STAGE} (RC=${RC}) — relance dans un nouveau run"\n'
            '  fi'
        )
        if old_echo in src:
            src = src.replace(old_echo, new_echo, 1)
        return src

    raise SystemExit(
        "❌ run_stage.sh : bloc RC=124/143 introuvable — patch impossible sans cible"
    )


def main() -> int:
    script = _find_script()
    src = script.read_text(encoding="utf-8")

    if MARKER_NEW in src and MARKER_HELPER_CALL in src:
        print(f"⏭  {script} : déjà patché — rien à faire")
        return 0

    _backup(script)
    new_src = _patch_run_stage(src)
    _atomic_write(script, new_src)

    try:
        os.chmod(script, 0o755)
    except OSError:
        pass

    check = script.read_text(encoding="utf-8")
    if MARKER_NEW not in check:
        print("❌ Patch RC=42 absent après écriture")
        return 3
    if MARKER_HELPER_CALL not in check:
        print("❌ Appel pb_retry_check.sh absent après écriture")
        return 4
    if 'if [ "${RC}" = "124" ] || [ "${RC}" = "143" ] || [ "${RC}" = "42" ]; then' in check:
        pass
    elif 'if [ "${RC}" = "42" ]; then' in check and 'if [ "${RC}" = "124" ]' in check:
        pass
    else:
        print("⚠️  Vérification structurelle imprécise — lecture manuelle conseillée")

    helper_dir = script.parent
    helper_path = helper_dir / "pb_retry_check.sh"
    _atomic_write(helper_path, NEW_PB_RETRY_HELPER)
    try:
        os.chmod(helper_path, 0o755)
    except OSError:
        pass

    if not helper_path.exists():
        print("❌ pb_retry_check.sh non créé")
        return 5

    print(f"✅ {script} patché :")
    print("   - RC=42 → soft-retry (relance workflow nouvelle tentative)")
    print("   - garde-fou : arrêt définitif si tentatives PB épuisées")
    print(f"✅ {helper_path} créé (compteur tentatives PB par stage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
