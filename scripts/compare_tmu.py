#!/usr/bin/env python3
"""
Python TMU benchmark — mirrors bench_training.rs output format.

Modes:
  python scripts/compare_tmu.py            # large scale (IMDb-scale, speed)
  python scripts/compare_tmu.py --small    # small scale (NoisyXOR-scale, accuracy)
  python scripts/compare_tmu.py --both     # both configs sequentially
"""
import sys
import argparse
import time

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is required.  Install with: pip install numpy")

try:
    from tmu.models.classification.vanilla_classifier import TMClassifier
except ImportError:
    sys.exit(
        "ERROR: tmu is not installed.\n"
        "Install with:  pip install tmu\n"
        "  or from source: pip install git+https://github.com/cair/tmu.git"
    )

# ── Configs ──────────────────────────────────────────────────────────────────
#
# n_clauses_per_cls = clauses per class, matching Rust's TsetlinMachine::with_config()
# third argument (clauses_per_class).
# TMClassifier takes number_of_clauses as the TOTAL across all classes, so we
# pass n_clauses_per_cls * n_classes when constructing the model.

LARGE = dict(
    label             = "large (IMDb-scale)",
    n_features        = 1_000,
    n_clauses_per_cls = 10_000,   # matches bench_training.rs n_clauses
    n_classes         = 2,
    threshold         = 8_000,
    s                 = 2.0,
    state_bits        = 8,
    n_train           = 2_000,
    n_warmup          = 2,
    n_bench           = 8,
    seed              = 42,
    print_accuracy    = False,
)

SMALL = dict(
    label             = "small (NoisyXOR-scale)",
    n_features        = 20,
    n_clauses_per_cls = 10,       # matches noisy_xor.rs clauses_per_class
    n_classes         = 2,
    threshold         = 10,
    s                 = 3.0,
    state_bits        = 8,
    n_train           = 2_000,
    n_warmup          = 0,
    n_bench           = 20,
    seed              = 42,
    print_accuracy    = True,
)


def make_xor_data(n_train: int, n_features: int, seed: int):
    """
    Synthetic XOR-labelled binary dataset: label = feature[0] XOR feature[1].

    Uses numpy PCG64 seeded at `seed`. Rust uses SplitMix64 — the data has the
    same statistical properties (i.i.d. Bernoulli(0.5) features, balanced XOR
    labels) but different bit sequences across languages.
    """
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(n_train, n_features), dtype=np.uint32)
    ys = (xs[:, 0] ^ xs[:, 1]).astype(np.uint32)
    return xs, ys


def run_bench(cfg: dict) -> dict:
    label             = cfg["label"]
    n_features        = cfg["n_features"]
    n_clauses_per_cls = cfg["n_clauses_per_cls"]
    n_classes         = cfg["n_classes"]
    threshold         = cfg["threshold"]
    s                 = cfg["s"]
    state_bits        = cfg["state_bits"]
    n_train           = cfg["n_train"]
    n_warmup          = cfg["n_warmup"]
    n_bench           = cfg["n_bench"]
    seed              = cfg["seed"]
    print_accuracy    = cfg["print_accuracy"]

    xs, ys = make_xor_data(n_train, n_features, seed)

    total_clauses = n_clauses_per_cls * n_classes
    # Matches bench_training.rs: n_train * 2 * n_clauses (where n_clauses = per-class)
    clause_updates_per_epoch = n_train * n_classes * n_clauses_per_cls

    tm = TMClassifier(
        number_of_clauses       = total_clauses,
        T                       = threshold,
        s                       = s,
        number_of_state_bits_ta = state_bits,
        weighted_clauses        = True,
        platform                = "CPU",
    )

    print(f"\nMode   : Python TMU  [{label}]")
    print(f"Config : {n_classes} classes · {n_features} features · "
          f"{n_clauses_per_cls} clauses/class · {n_train} training samples")
    print(f"Workload: {clause_updates_per_epoch // 1_000_000} M clause updates per epoch\n")

    header = f"{'epoch':>5}  {'ms':>9}  {'samples/s':>13}  {'Mclause-ups/s':>15}"
    if print_accuracy:
        header += f"  {'acc%':>6}"
    print(header)

    for _ in range(n_warmup):
        tm.fit(xs, ys)

    times_s = []
    for epoch in range(n_bench):
        t0 = time.perf_counter()
        tm.fit(xs, ys)
        elapsed_s = time.perf_counter() - t0
        times_s.append(elapsed_s)

        ms   = elapsed_s * 1_000.0
        sps  = n_train / elapsed_s
        mcps = clause_updates_per_epoch / elapsed_s / 1e6

        line = f"{epoch:>5}  {ms:>8.1f}  {sps:>13.0f}  {mcps:>15.1f}"
        if print_accuracy:
            preds = tm.predict(xs)
            acc   = 100.0 * float(np.mean(preds == ys))
            line += f"  {acc:>5.1f}%"
        print(line)

    # Use floor-division index to match Rust's `times[times.len() / 2]` after sort.
    times_s.sort()
    median_s = times_s[len(times_s) // 2]
    mean_s   = sum(times_s) / len(times_s)
    min_s    = times_s[0]
    max_s    = times_s[-1]

    print()
    print(f"── Summary ({n_bench} timed epochs) {'─' * 50}")
    print(f"  median {median_s*1000:7.1f} ms  |  mean {mean_s*1000:7.1f} ms  "
          f"|  min {min_s*1000:7.1f} ms  |  max {max_s*1000:7.1f} ms")
    print(f"  throughput  : {n_train / median_s:9.0f} samples/s       "
          f"{clause_updates_per_epoch / median_s / 1e6:7.1f} Mclause-updates/s")

    return {
        "label":    label,
        "median_s": median_s,
        "mean_s":   mean_s,
        "min_s":    min_s,
        "max_s":    max_s,
        "sps":      n_train / median_s,
        "mcps":     clause_updates_per_epoch / median_s / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Python TMU benchmark — mirrors bench_training.rs output format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--small", action="store_true",
                       help="Run small (NoisyXOR-scale) config only — accuracy check, fast")
    group.add_argument("--both",  action="store_true",
                       help="Run both small then large configs")
    parser.add_argument("--clauses", type=int, default=None,
                        help="Clauses per class for the large config (default: 10000). "
                             "Threshold scales proportionally.")
    args = parser.parse_args()

    if args.clauses is not None:
        LARGE["n_clauses_per_cls"] = args.clauses
        LARGE["threshold"] = max(1, 8_000 * args.clauses // 10_000)

    if args.small:
        run_bench(SMALL)
    elif args.both:
        run_bench(SMALL)
        run_bench(LARGE)
    else:
        run_bench(LARGE)


if __name__ == "__main__":
    main()
