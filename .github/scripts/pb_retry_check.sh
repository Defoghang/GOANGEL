#!/bin/bash
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
