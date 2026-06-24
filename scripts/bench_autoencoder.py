#!/usr/bin/env python3
"""
Python TMU autoencoder benchmark — mirrors bench_autoencoder.rs output format.

Modes:
  python scripts/bench_autoencoder.py            # large scale (throughput)
  python scripts/bench_autoencoder.py --small    # small scale (accuracy check)
  python scripts/bench_autoencoder.py --both     # both configs sequentially
"""
import sys
import argparse
import time

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is required.  Install with: pip install numpy")

try:
    from tmu.models.autoencoder.vanilla import TMAutoEncoder
except ImportError:
    sys.exit(
        "ERROR: tmu is not installed.\n"
        "Install with:  pip install tmu\n"
        "  or from source: pip install git+https://github.com/cair/tmu.git"
    )

# ── Configs ──────────────────────────────────────────────────────────────────
#
# clauses_per_output matches Rust's TMAutoEncoder::with_config() second argument.
# TMAutoEncoder takes number_of_clauses as the per-output count (unlike TMClassifier
# which takes the total across all classes).

LARGE = dict(
    label               = "large (throughput)",
    n_features          = 200,
    clauses_per_output  = 50,
    threshold           = 200,
    s                   = 2.0,
    state_bits          = 8,
    n_train             = 2_000,
    n_warmup            = 2,
    n_bench             = 8,
    seed                = 42,
    print_accuracy      = False,
)

SMALL = dict(
    label               = "small (accuracy check)",
    n_features          = 20,
    clauses_per_output  = 40,
    threshold           = 20,
    s                   = 3.9,
    state_bits          = 8,
    n_train             = 2_000,
    n_warmup            = 0,
    n_bench             = 20,
    seed                = 42,
    print_accuracy      = True,
)


def make_binary_data(n_train: int, n_features: int, seed: int):
    """Random i.i.d. Bernoulli(0.5) binary matrix, no labels needed."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(n_train, n_features), dtype=np.uint32)
    return xs


def run_bench(cfg: dict) -> dict:
    label              = cfg["label"]
    n_features         = cfg["n_features"]
    clauses_per_output = cfg["clauses_per_output"]
    threshold          = cfg["threshold"]
    s                  = cfg["s"]
    state_bits         = cfg["state_bits"]
    n_train            = cfg["n_train"]
    n_warmup           = cfg["n_warmup"]
    n_bench            = cfg["n_bench"]
    seed               = cfg["seed"]
    print_accuracy     = cfg["print_accuracy"]

    xs = make_binary_data(n_train, n_features, seed)

    # clause_updates_per_epoch = n_train × n_features × clauses_per_output
    # (all n_features outputs are updated per sample)
    clause_updates_per_epoch = n_train * n_features * clauses_per_output

    ae = TMAutoEncoder(
        number_of_clauses       = clauses_per_output,
        T                       = threshold,
        s                       = s,
        number_of_state_bits_ta = state_bits,
        weighted_clauses        = True,
        output_active           = np.arange(n_features),
        platform                = "CPU",
    )

    print(f"\nMode   : Python TMU  [{label}]")
    print(f"Config : {n_features} features · {clauses_per_output} clauses/output · "
          f"T={threshold} · s={s} · {n_train} training samples")
    print(f"Workload: {clause_updates_per_epoch // 1_000_000} M clause updates per epoch\n")

    header = f"{'epoch':>5}  {'ms':>9}  {'samples/s':>13}  {'Mclause-ups/s':>15}"
    if print_accuracy:
        header += f"  {'recon-acc':>9}"
    print(header)

    for _ in range(n_warmup):
        ae.fit(xs, number_of_examples=n_train)

    times_s = []
    for epoch in range(n_bench):
        t0 = time.perf_counter()
        ae.fit(xs, number_of_examples=n_train)
        elapsed_s = time.perf_counter() - t0
        times_s.append(elapsed_s)

        ms   = elapsed_s * 1_000.0
        sps  = n_train / elapsed_s
        mcps = clause_updates_per_epoch / elapsed_s / 1e6

        line = f"{epoch:>5}  {ms:>8.1f}  {sps:>13.0f}  {mcps:>15.1f}"
        if print_accuracy:
            recon = ae.predict(xs)
            acc = float(np.mean(recon == xs))
            line += f"  {acc:>9.4f}"
        print(line)

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
        description="Python TMU autoencoder benchmark — mirrors bench_autoencoder.rs output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--small", action="store_true",
                       help="Run small (accuracy check) config only")
    group.add_argument("--both",  action="store_true",
                       help="Run both small then large configs")
    args = parser.parse_args()

    if args.small:
        run_bench(SMALL)
    elif args.both:
        run_bench(SMALL)
        run_bench(LARGE)
    else:
        run_bench(LARGE)


if __name__ == "__main__":
    main()
