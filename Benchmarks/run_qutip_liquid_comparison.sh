#!/usr/bin/env bash
# Orchestrates the QuTiP-vs-LiQuID comparison pipeline:
#   1. run the QuTiP benchmarks (pure decay + driven, plus the Pg pathology check)
#   2. make sure a LiQuID results file is present (placeholder until the
#      LiQuID-generation scripts are wired in)
#   3. plot the comparison figure from the measured JSONL data
#
# Usage:
#   ./run_qutip_liquid_comparison.sh                              # full run, default settings
#   ./run_qutip_liquid_comparison.sh --driven-only                # skip pure-decay sweep
#   ./run_qutip_liquid_comparison.sh --skip-benchmarks            # just re-plot existing results
#   ./run_qutip_liquid_comparison.sh --eps "0.10 0.05 0.02" --ntraj-max 100000 --seed 1
#   ./run_qutip_liquid_comparison.sh --liquid-dir /path/to/LiQuID/benchmarks/results
#       # convert LiQuID's real bench_adaptive_stopping.csv (Table 4) and
#       # bench2_main.csv (Table 5) instead of using the placeholder values
#   ./run_qutip_liquid_comparison.sh --liquid-adaptive-csv a.csv --liquid-variance-csv b.csv
#   ./run_qutip_liquid_comparison.sh --liquid-results path/to/already_converted_liquid.jsonl
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RESULTS_DIR="results"
QUTIP_OUT="$RESULTS_DIR/qutip_results.jsonl"
LIQUID_RESULTS="$RESULTS_DIR/liquid_results.jsonl"
LIQUID_PLACEHOLDER="liquid_results.example.jsonl"
OUT_PREFIX="$RESULTS_DIR/qutip_vs_liquid_comparison"

EPS_LIST="0.10 0.05 0.02 0.01"
DRIVEN_EPS_LIST="0.10 0.05"
NTRAJ_MAX=200000
PG_NTRAJ_MAX=5000
SEED=0
DRIVEN_ONLY=0
SKIP_BENCHMARKS=0
SKIP_PG_CHECK=0
LIQUID_DIR=""
LIQUID_ADAPTIVE_CSV=""
LIQUID_VARIANCE_CSV=""

usage() {
    grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eps) EPS_LIST="$2"; DRIVEN_EPS_LIST="$2"; shift 2 ;;
        --ntraj-max) NTRAJ_MAX="$2"; shift 2 ;;
        --pg-ntraj-max) PG_NTRAJ_MAX="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --liquid-results) LIQUID_RESULTS="$2"; shift 2 ;;
        --liquid-dir) LIQUID_DIR="$2"; shift 2 ;;
        --liquid-adaptive-csv) LIQUID_ADAPTIVE_CSV="$2"; shift 2 ;;
        --liquid-variance-csv) LIQUID_VARIANCE_CSV="$2"; shift 2 ;;
        --driven-only) DRIVEN_ONLY=1; shift ;;
        --skip-benchmarks) SKIP_BENCHMARKS=1; shift ;;
        --skip-pg-check) SKIP_PG_CHECK=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

#mkdir -p "$RESULTS_DIR"

# --- LiQuID data --------------------------------------------------------
# Priority:
#   1. Explicit CLI arguments given -> convert specified CSVs
#   2. Default local CSVs exist in results/ -> convert them automatically
#   3. $LIQUID_RESULTS already exists -> reuse it as-is
#   4. Otherwise -> fall back to the manuscript-value placeholder
if [[ -n "$LIQUID_DIR" || -n "$LIQUID_ADAPTIVE_CSV" || -n "$LIQUID_VARIANCE_CSV" ]]; then
    echo "==> Converting specified LiQuID benchmark CSVs -> $LIQUID_RESULTS"
    CONVERT_ARGS=(--out "$LIQUID_RESULTS")
    [[ -n "$LIQUID_DIR" ]] && CONVERT_ARGS+=(--liquid-results-dir "$LIQUID_DIR")
    [[ -n "$LIQUID_ADAPTIVE_CSV" ]] && CONVERT_ARGS+=(--adaptive-stopping-csv "$LIQUID_ADAPTIVE_CSV")
    [[ -n "$LIQUID_VARIANCE_CSV" ]] && CONVERT_ARGS+=(--variance-csv "$LIQUID_VARIANCE_CSV")
    python3 liquid_csv_to_jsonl.py "${CONVERT_ARGS[@]}"
elif [[ -f "results/bench_adaptive_stopping.csv" && -f "results/bench2_main.csv" ]]; then
    echo "==> Converting default LiQuID benchmark CSVs -> $LIQUID_RESULTS"
    python3 liquid_csv_to_jsonl.py \
        --adaptive-stopping-csv results/bench_adaptive_stopping.csv \
        --variance-csv results/bench2_main.csv \
        --out "$LIQUID_RESULTS"
elif [[ -f "$LIQUID_RESULTS" ]]; then
    echo "==> Reusing existing $LIQUID_RESULTS"
else
    echo "NOTE: $LIQUID_RESULTS not found and no local CSVs / --liquid-dir given."
    echo "      Using placeholder values from $LIQUID_PLACEHOLDER (manuscript Table 4/5)."
    echo "      Pass --liquid-dir /path/to/LiQuID/benchmarks/results to use real LiQuID output."
    cp "$LIQUID_PLACEHOLDER" "$LIQUID_RESULTS"
fi
# --- QuTiP benchmarks -----------------------------------------------------
if [[ "$SKIP_BENCHMARKS" -eq 0 ]]; then
    rm -f "$QUTIP_OUT"

    if [[ "$DRIVEN_ONLY" -eq 1 ]]; then
        echo "==> Running driven-qubit benchmark only..."
        python3 qutip_benchmark_driven.py \
            --eps $DRIVEN_EPS_LIST --ntraj-max "$NTRAJ_MAX" --seed "$SEED" \
            --out "$QUTIP_OUT"
    else
        echo "==> Running pure-decay + driven-qubit benchmark..."
        python3 qutip_benchmark.py \
            --eps $EPS_LIST --ntraj-max "$NTRAJ_MAX" --seed "$SEED" \
            --out "$QUTIP_OUT"
    fi

    if [[ "$SKIP_PG_CHECK" -eq 0 ]]; then
        echo "==> Running Pg pathology check..."
        python3 qutip_pg_check.py \
            --eps $DRIVEN_EPS_LIST --ntraj-max "$PG_NTRAJ_MAX" --seed "$SEED" \
            --out "$QUTIP_OUT"
    fi
else
    echo "==> Skipping benchmarks (--skip-benchmarks), reusing $QUTIP_OUT"
    if [[ ! -f "$QUTIP_OUT" ]]; then
        echo "ERROR: $QUTIP_OUT does not exist. Remove --skip-benchmarks to generate it." >&2
        exit 1
    fi
fi

# --- Plot -------------------------------------------------------------
#echo "==> Generating comparison figure..."
#python3 make_qutip_comparison.py \
#    --qutip-results "$QUTIP_OUT" \
#    --liquid-results "$LIQUID_RESULTS" \
#    --out-prefix "$OUT_PREFIX"

echo "==> Done."
echo "    Results:  $QUTIP_OUT"
#echo "    Figure:   $OUT_PREFIX.pdf / $OUT_PREFIX.png"
