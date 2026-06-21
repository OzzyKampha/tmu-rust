#!/usr/bin/env bash
# compare.sh — Side-by-side Rust vs Python TMU throughput benchmark.
#
# Usage:
#   bash scripts/compare.sh                    # Rust sequential + Python large
#   bash scripts/compare.sh --parallel         # also run Rust with Rayon
#   bash scripts/compare.sh --native           # RUSTFLAGS="-C target-cpu=native"
#   bash scripts/compare.sh --small            # Python --small (NoisyXOR-scale accuracy check)
#   bash scripts/compare.sh --parallel --native
#
# Note: --small applies to the Python run only; the Rust bench_training example
# always uses the large (IMDb-scale) config. For a small Rust accuracy check,
# run: cargo run --release --example noisy_xor
#
# See BENCHMARKS.md for setup instructions and result interpretation.

set -euo pipefail
cd "$(dirname "$0")/.."

PARALLEL=0
NATIVE=0
SMALL=0

for arg in "$@"; do
    case "$arg" in
        --parallel) PARALLEL=1 ;;
        --native)   NATIVE=1 ;;
        --small)    SMALL=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Rust build flags ──────────────────────────────────────────────────────────
RUSTFLAGS_EXTRA=""
if [[ $NATIVE -eq 1 ]]; then
    RUSTFLAGS_EXTRA="-C target-cpu=native"
    echo "RUSTFLAGS: $RUSTFLAGS_EXTRA"
fi

# ── Python availability check ─────────────────────────────────────────────────
PYTHON_OK=0
if python3 -c "from tmu.models.classification.vanilla_classifier import TMClassifier" 2>/dev/null; then
    PYTHON_OK=1
else
    echo "WARNING: tmu Python package not found — skipping Python benchmark."
    echo "         Install with: pip install tmu"
fi

# ── Build Rust ────────────────────────────────────────────────────────────────
echo ""
echo "=== Building Rust (release) ==="
RUSTFLAGS="$RUSTFLAGS_EXTRA" cargo build --release --example bench_training
if [[ $PARALLEL -eq 1 ]]; then
    RUSTFLAGS="$RUSTFLAGS_EXTRA" cargo build --release --features parallel --example bench_training
fi

# Capture output to temp files for summary table parsing.
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

rust_seq_out="$TMPD/rust_seq.txt"
rust_par_out="$TMPD/rust_par.txt"
py_out="$TMPD/python.txt"

# ── Rust sequential ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  RUST  (sequential)"
echo "══════════════════════════════════════════════"
RUSTFLAGS="$RUSTFLAGS_EXTRA" \
    cargo run --release --example bench_training 2>/dev/null \
    | tee "$rust_seq_out"

# ── Rust parallel ─────────────────────────────────────────────────────────────
if [[ $PARALLEL -eq 1 ]]; then
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  RUST  (parallel, --features parallel)"
    echo "══════════════════════════════════════════════"
    RUSTFLAGS="$RUSTFLAGS_EXTRA" \
        cargo run --release --features parallel --example bench_training 2>/dev/null \
        | tee "$rust_par_out"
fi

# ── Python ────────────────────────────────────────────────────────────────────
if [[ $PYTHON_OK -eq 1 ]]; then
    PY_ARGS=""
    [[ $SMALL -eq 1 ]] && PY_ARGS="--small"
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  PYTHON TMU  (compare_tmu.py${PY_ARGS:+ $PY_ARGS})"
    echo "══════════════════════════════════════════════"
    python3 scripts/compare_tmu.py $PY_ARGS | tee "$py_out"
fi

# ── Parse summary metrics ─────────────────────────────────────────────────────
extract_median_ms() {
    grep -E "^\s+median" "$1" 2>/dev/null \
        | grep -oE 'median\s+[0-9]+\.[0-9]+' \
        | grep -oE '[0-9]+\.[0-9]+' \
        | tail -1   # tail: pick last block when --both runs two configs
}
extract_mcps() {
    grep "throughput" "$1" 2>/dev/null \
        | grep -oE '[0-9]+\.[0-9]+ Mclause-updates/s' \
        | grep -oE '[0-9]+\.[0-9]+' \
        | tail -1
}

RUST_SEQ_MS=$(extract_median_ms "$rust_seq_out")
RUST_SEQ_MCPS=$(extract_mcps "$rust_seq_out")
RUST_PAR_MS=""
RUST_PAR_MCPS=""
if [[ $PARALLEL -eq 1 && -f "$rust_par_out" ]]; then
    RUST_PAR_MS=$(extract_median_ms "$rust_par_out")
    RUST_PAR_MCPS=$(extract_mcps "$rust_par_out")
fi
PY_MS=""
PY_MCPS=""
if [[ $PYTHON_OK -eq 1 && -f "$py_out" ]]; then
    PY_MS=$(extract_median_ms "$py_out")
    PY_MCPS=$(extract_mcps "$py_out")
fi

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "  (config: 2 classes · 1000 features · 10000 clauses/class · IMDb-scale)"
echo "══════════════════════════════════════════════════════════════════════"
printf "  %-32s  %12s  %16s\n" "Runner" "Median ms" "Mclause-ups/s"
echo "  ────────────────────────────────────────────────────────────────────"

[[ -n "$RUST_SEQ_MS" ]] && \
    printf "  %-32s  %12s  %16s\n" \
        "Rust (sequential)" "${RUST_SEQ_MS} ms" "${RUST_SEQ_MCPS:-N/A}"

[[ -n "$RUST_PAR_MS" ]] && \
    printf "  %-32s  %12s  %16s\n" \
        "Rust (parallel, Rayon)" "${RUST_PAR_MS} ms" "${RUST_PAR_MCPS:-N/A}"

[[ -n "$PY_MS" ]] && \
    printf "  %-32s  %12s  %16s\n" \
        "Python TMU (C extension)" "${PY_MS} ms" "${PY_MCPS:-N/A}"

echo "══════════════════════════════════════════════════════════════════════"

# Speedup: compute inline via python3 to avoid bash floating-point limitations.
if [[ -n "$RUST_SEQ_MS" && -n "$PY_MS" ]]; then
    SPEEDUP=$(python3 -c "print(f'{float(\"$PY_MS\") / float(\"$RUST_SEQ_MS\"):.1f}')" 2>/dev/null || echo "N/A")
    echo "  Rust sequential speedup over Python: ${SPEEDUP}x"
fi
if [[ $PARALLEL -eq 1 && -n "$RUST_PAR_MS" && -n "$PY_MS" ]]; then
    SPEEDUP=$(python3 -c "print(f'{float(\"$PY_MS\") / float(\"$RUST_PAR_MS\"):.1f}')" 2>/dev/null || echo "N/A")
    echo "  Rust parallel  speedup over Python: ${SPEEDUP}x"
fi
echo ""
echo "See BENCHMARKS.md for methodology and interpretation notes."
