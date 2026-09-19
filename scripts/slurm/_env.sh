# Shared environment for Slurm jobs and submission helpers. Source this file.
# The caller locates it before sourcing because Slurm runs a spool copy of jobs.
GAMINGMASTER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd) || return 1
export GAMINGMASTER_ROOT
export PYTHONPATH="${GAMINGMASTER_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$GAMINGMASTER_ROOT" || return 1
