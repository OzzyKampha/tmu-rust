#!/usr/bin/env bash
# compare_new.sh — Rust tmu-rs vs Python TMU for the three new model types.
#
# Runs the Rust examples (regression, convolutional, composite) and then the
# matching Python TMU implementations so you can compare accuracy side-by-side.
#
# Usage:
#   bash scripts/compare_new.sh                  # all three models
#   bash scripts/compare_new.sh --regressor      # TMRegressor only
#   bash scripts/compare_new.sh --conv           # ConvolutionalTM only
#   bash scripts/compare_new.sh --composite      # TMCompositeClassifier only
#   bash scripts/compare_new.sh --rust-only      # skip Python (no TMU needed)
#   bash scripts/compare_new.sh --python-only    # skip Rust build
#   bash scripts/compare_new.sh --native         # RUSTFLAGS="-C target-cpu=native"
#   bash scripts/compare_new.sh --parallel       # Rayon parallel training
#
# The Python side requires:   pip install tmu numpy
# TMU install:  pip install git+https://github.com/cair/tmu.git

set -euo pipefail
cd "$(dirname "$0")/.."

# ── flags ──────────────────────────────────────────────────────────────────────
DO_REGRESSOR=0; DO_CONV=0; DO_COMPOSITE=0; ALL=1
RUST_ONLY=0; PYTHON_ONLY=0; NATIVE=0; PARALLEL=0

for arg in "$@"; do
    case "$arg" in
        --regressor)   DO_REGRESSOR=1; ALL=0 ;;
        --conv)        DO_CONV=1;      ALL=0 ;;
        --composite)   DO_COMPOSITE=1; ALL=0 ;;
        --rust-only)   RUST_ONLY=1 ;;
        --python-only) PYTHON_ONLY=1 ;;
        --native)      NATIVE=1 ;;
        --parallel)    PARALLEL=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

if [[ $ALL -eq 1 ]]; then
    DO_REGRESSOR=1; DO_CONV=1; DO_COMPOSITE=1
fi

# ── build ──────────────────────────────────────────────────────────────────────
if [[ $PYTHON_ONLY -eq 0 ]]; then
    RUSTFLAGS_EXTRA=""
    FEATURES_FLAG=""
    [[ $NATIVE -eq 1 ]]   && RUSTFLAGS_EXTRA="-C target-cpu=native"
    [[ $PARALLEL -eq 1 ]] && FEATURES_FLAG="--features parallel"

    echo "═══════════════════════════════════════════════════════════"
    echo "  Building Rust (release)…"
    [[ -n "$RUSTFLAGS_EXTRA" ]] && echo "  RUSTFLAGS: $RUSTFLAGS_EXTRA"
    [[ -n "$FEATURES_FLAG"   ]] && echo "  Features : $FEATURES_FLAG"
    echo "═══════════════════════════════════════════════════════════"

    RUSTFLAGS="$RUSTFLAGS_EXTRA" cargo build --release $FEATURES_FLAG \
        --example regression \
        --example convolutional \
        --example composite \
        2>&1 | grep -E "^(error|warning\[)" || true
fi

# ── helper ─────────────────────────────────────────────────────────────────────
RUN_RUST() {
    local example="$1"
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "  Rust tmu-rs: $example"
    echo "───────────────────────────────────────────────────────────"
    RUSTFLAGS="${RUSTFLAGS_EXTRA:-}" ./target/release/examples/"$example"
}

RUN_PYTHON() {
    local flag="$1"
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "  Python TMU: $flag"
    echo "───────────────────────────────────────────────────────────"
    python3 scripts/compare_new_models.py "$flag"
}

# ── TMRegressor ────────────────────────────────────────────────────────────────
if [[ $DO_REGRESSOR -eq 1 ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  1 / 3  TMRegressor"
    echo "═══════════════════════════════════════════════════════════"

    [[ $PYTHON_ONLY -eq 0 ]] && RUN_RUST regression
    [[ $RUST_ONLY   -eq 0 ]] && RUN_PYTHON --regressor
fi

# ── ConvolutionalTM ───────────────────────────────────────────────────────────
if [[ $DO_CONV -eq 1 ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  2 / 3  ConvolutionalTM"
    echo "═══════════════════════════════════════════════════════════"

    [[ $PYTHON_ONLY -eq 0 ]] && RUN_RUST convolutional
    [[ $RUST_ONLY   -eq 0 ]] && RUN_PYTHON --conv
fi

# ── TMCompositeClassifier ─────────────────────────────────────────────────────
if [[ $DO_COMPOSITE -eq 1 ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  3 / 3  TMCompositeClassifier"
    echo "═══════════════════════════════════════════════════════════"

    [[ $PYTHON_ONLY -eq 0 ]] && RUN_RUST composite
    [[ $RUST_ONLY   -eq 0 ]] && RUN_PYTHON --composite
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done."
echo "═══════════════════════════════════════════════════════════"
