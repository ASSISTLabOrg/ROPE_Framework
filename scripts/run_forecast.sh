#!/usr/bin/env bash
# run_forecast.sh — Submit an upper-atmosphere density forecast job
#
# Usage (local):
#   ./scripts/run_forecast.sh
#   ./scripts/run_forecast.sh --start-time 2025-01-15T00:00:00Z --forecast-length 720
#
# Usage (SLURM):
#   Set SCHEDULER=slurm below, then:
#   sbatch scripts/run_forecast.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit these to match your installation
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FORECAST_BIN="${REPO_ROOT}/build/examples/forecast"
CONFIG_FILE="${REPO_ROOT}/config/forecast_job.conf"
OUTPUT_DIR="${REPO_ROOT}/output"

# ---------------------------------------------------------------------------
# Forecast parameters
# These values are used when no --config flag is given, or they override the
# config file when passed on the command line.
# ---------------------------------------------------------------------------
START_TIME="2025-01-15T12:00:00Z"   # UTC, ISO 8601
FORECAST_LENGTH=360                  # minutes

PREDICTOR_MODEL=""   # leave empty to use value from CONFIG_FILE
DECODER_MODEL=""     # leave empty to use value from CONFIG_FILE
ENCODER_MODEL=""     # optional; leave empty if not needed

# ---------------------------------------------------------------------------
# Execution settings
# ---------------------------------------------------------------------------
NUM_THREADS=4
USE_GPU=false
LOG_FILE=""          # empty = stdout only; set a path to tee to file

# ---------------------------------------------------------------------------
# SLURM settings (only used when SCHEDULER=slurm)
# ---------------------------------------------------------------------------
SCHEDULER="local"    # "local" or "slurm"
SLURM_PARTITION="compute"
SLURM_NODES=1
SLURM_NTASKS=1
SLURM_CPUS_PER_TASK=4
SLURM_MEM="8G"
SLURM_TIME="02:00:00"
SLURM_JOB_NAME="upatmocast"
SLURM_OUTPUT="${REPO_ROOT}/logs/slurm_%j.out"

# ---------------------------------------------------------------------------
# Parse command-line overrides
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-time)       START_TIME="$2";       shift 2 ;;
        --forecast-length)  FORECAST_LENGTH="$2";  shift 2 ;;
        --predictor)        PREDICTOR_MODEL="$2";  shift 2 ;;
        --decoder)          DECODER_MODEL="$2";    shift 2 ;;
        --encoder)          ENCODER_MODEL="$2";    shift 2 ;;
        --config)           CONFIG_FILE="$2";      shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";       shift 2 ;;
        --threads)          NUM_THREADS="$2";      shift 2 ;;
        --gpu)              USE_GPU=true;           shift   ;;
        --log)              LOG_FILE="$2";         shift 2 ;;
        --slurm)            SCHEDULER="slurm";     shift   ;;
        --help)
            grep '^#' "$0" | head -10
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate binary
# ---------------------------------------------------------------------------
if [[ ! -x "${FORECAST_BIN}" ]]; then
    echo "ERROR: forecast binary not found at ${FORECAST_BIN}" >&2
    echo "Build the project first:" >&2
    echo "  cd ${REPO_ROOT} && ./build.sh" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
ARGS=()

# Config file (provides defaults; CLI args below will override)
if [[ -f "${CONFIG_FILE}" ]]; then
    ARGS+=(--config "${CONFIG_FILE}")
fi

# Mandatory forecast parameters (override config file)
ARGS+=(--start-time       "${START_TIME}")
ARGS+=(--forecast-length  "${FORECAST_LENGTH}")

# Model overrides (only if explicitly set above or via CLI)
[[ -n "${PREDICTOR_MODEL}" ]] && ARGS+=(--predictor "${PREDICTOR_MODEL}")
[[ -n "${DECODER_MODEL}"   ]] && ARGS+=(--decoder   "${DECODER_MODEL}")
[[ -n "${ENCODER_MODEL}"   ]] && ARGS+=(--encoder   "${ENCODER_MODEL}")

# Inference settings
ARGS+=(--threads "${NUM_THREADS}")
[[ "${USE_GPU}" == "true" ]] && ARGS+=(--gpu)

# Auto-named output inside OUTPUT_DIR
# Derive a filename: forecast_YYYYMMDD_HHMM_Nmin.bin
START_COMPACT=$(echo "${START_TIME}" | sed 's/[-T:]//g' | cut -c1-12)
OUTPUT_FILE="${OUTPUT_DIR}/forecast_${START_COMPACT}_${FORECAST_LENGTH}min.bin"
ARGS+=(--output "${OUTPUT_FILE}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
run_local() {
    echo "=== upatmocast forecast job ==="
    echo "  Binary:          ${FORECAST_BIN}"
    echo "  Config file:     ${CONFIG_FILE}"
    echo "  Start time (UTC):${START_TIME}"
    echo "  Forecast length: ${FORECAST_LENGTH} min"
    echo "  Output:          ${OUTPUT_FILE}"
    echo ""

    if [[ -n "${LOG_FILE}" ]]; then
        mkdir -p "$(dirname "${LOG_FILE}")"
        "${FORECAST_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    else
        "${FORECAST_BIN}" "${ARGS[@]}"
    fi
}

run_slurm() {
    mkdir -p "$(dirname "${SLURM_OUTPUT}")"

    # Rebuild ARGS as a single quoted string for the sbatch here-doc
    ARGS_STR="${ARGS[*]}"

    sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${SLURM_OUTPUT}

set -euo pipefail
echo "Job \${SLURM_JOB_ID} starting on \$(hostname) at \$(date -u)"
${FORECAST_BIN} ${ARGS_STR}
echo "Job \${SLURM_JOB_ID} finished at \$(date -u)"
EOF
    echo "SLURM job submitted.  Output: ${SLURM_OUTPUT}"
}

case "${SCHEDULER}" in
    local) run_local ;;
    slurm) run_slurm ;;
    *)
        echo "ERROR: Unknown scheduler '${SCHEDULER}'. Choose 'local' or 'slurm'." >&2
        exit 1
        ;;
esac
