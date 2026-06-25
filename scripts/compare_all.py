#!/usr/bin/env python3
"""
compare_all.py — unified Python TMU comparison script for tmu-rs.

Covers all model types: TMClassifier, TMAutoEncoder, TMRegressor,
ConvolutionalTM (1-D and 2-D), and TMCompositeClassifier.

Usage:
  python scripts/compare_all.py                   # all models
  python scripts/compare_all.py --classifier      # TMClassifier (speed benchmark)
  python scripts/compare_all.py --autoencoder     # TMAutoEncoder (speed benchmark)
  python scripts/compare_all.py --regressor       # TMRegressor (accuracy)
  python scripts/compare_all.py --conv            # ConvolutionalTM 1-D (accuracy)
  python scripts/compare_all.py --conv2d          # ConvolutionalTM 2-D (accuracy)
  python scripts/compare_all.py --composite       # TMCompositeClassifier (accuracy)

  # classifier / autoencoder sub-modes:
  python scripts/compare_all.py --classifier --small     # NoisyXOR-scale
  python scripts/compare_all.py --classifier --both      # small then large
  python scripts/compare_all.py --autoencoder --small
  python scripts/compare_all.py --autoencoder --both

Prerequisites (for regressor, conv, conv2d, composite):
  python scripts/gen_shared_data.py    # writes data/cmp_*.bin
"""
import argparse
import pathlib
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy required — pip install numpy")

DATA = pathlib.Path("data")

# ── shared binary data helpers ────────────────────────────────────────────────

def _load(name, split, dtype, shape):
    path = DATA / f"cmp_{name}_{split}.bin"
    if not path.exists():
        sys.exit(
            f"ERROR: missing {path}\n"
            "Run:  python scripts/gen_shared_data.py"
        )
    arr = np.frombuffer(path.read_bytes(), dtype=np.dtype(dtype).newbyteorder("<"))
    return arr.reshape(shape)


def load_X(name, split, n, d):
    return _load(f"{name}_X", split, np.uint8, (n, d))

def load_y_f64(name, split, n):
    return _load(f"{name}_y", split, np.float64, (n,))

def load_y_u32(name, split, n):
    return _load(f"{name}_y", split, np.uint32, (n,))


# ── 1. TMClassifier (speed benchmark) ────────────────────────────────────────

CLASSIFIER_LARGE = dict(
    label             = "large (IMDb-scale)",
    n_features        = 1_000,
    n_clauses_per_cls = 10_000,
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

CLASSIFIER_SMALL = dict(
    label             = "small (NoisyXOR-scale)",
    n_features        = 20,
    n_clauses_per_cls = 10,
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


def _make_xor_data(n_train, n_features, seed):
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(n_train, n_features), dtype=np.uint32)
    ys = (xs[:, 0] ^ xs[:, 1]).astype(np.uint32)
    return xs, ys


def run_classifier(args):
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        sys.exit(
            "ERROR: tmu is not installed.\n"
            "Install with:  pip install tmu\n"
            "  or from source: pip install git+https://github.com/cair/tmu.git"
        )

    if args.small:
        configs = [CLASSIFIER_SMALL]
    elif args.both:
        configs = [CLASSIFIER_SMALL, CLASSIFIER_LARGE]
    else:
        configs = [CLASSIFIER_LARGE]

    for cfg in configs:
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

        xs, ys = _make_xor_data(n_train, n_features, seed)
        total_clauses = n_clauses_per_cls * n_classes
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


# ── 2. TMAutoEncoder (speed benchmark) ───────────────────────────────────────

AUTOENCODER_LARGE = dict(
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

AUTOENCODER_SMALL = dict(
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


def _make_structured_data(n_train, n_features, seed):
    half = n_features // 2
    rng = np.random.default_rng(seed)
    first = rng.integers(0, 2, size=(n_train, half), dtype=np.uint32)
    return np.concatenate([first, first], axis=1)


def run_autoencoder(args):
    try:
        from tmu.models.autoencoder.autoencoder import TMAutoEncoder
    except ImportError:
        sys.exit(
            "ERROR: tmu is not installed.\n"
            "Install with:  pip install tmu\n"
            "  or from source: pip install git+https://github.com/cair/tmu.git"
        )

    if args.small:
        configs = [AUTOENCODER_SMALL]
    elif args.both:
        configs = [AUTOENCODER_SMALL, AUTOENCODER_LARGE]
    else:
        configs = [AUTOENCODER_LARGE]

    for cfg in configs:
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

        xs = _make_structured_data(n_train, n_features, seed)
        total_clauses = n_features * clauses_per_output
        clause_updates_per_epoch = n_train * n_features * total_clauses

        ae = TMAutoEncoder(
            number_of_clauses       = total_clauses,
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


# ── 3. TMRegressor ────────────────────────────────────────────────────────────

def run_regressor(args):
    try:
        from tmu.models.regression.vanilla_regressor import TMRegressor
    except ImportError:
        print("[regressor] SKIP — tmu not installed or missing TMRegressor")
        return

    N_FEATURES = 20
    THRESHOLD  = 100
    N_CLAUSES  = 200
    S          = 3.0
    N_EPOCHS   = 60
    SCALE      = THRESHOLD / 5.0

    X_tr = load_X("regressor", "train", 5000, N_FEATURES).astype(np.uint32)
    y_tr = load_y_f64("regressor", "train", 5000)
    X_te = load_X("regressor", "test",  1000, N_FEATURES).astype(np.uint32)
    y_te = load_y_f64("regressor", "test",  1000)

    tm = TMRegressor(
        number_of_clauses=N_CLAUSES,
        T=THRESHOLD,
        s=S,
        number_of_state_bits_ta=8,
        weighted_clauses=True,
        platform="CPU",
    )

    print(f"\n{'─'*60}")
    print(f"  Python TMU  TMRegressor")
    print(f"  {N_FEATURES} features · {N_CLAUSES} clauses · T={THRESHOLD} · s={S}")
    print(f"  Target: count(features 0..4) × {SCALE:.0f}  →  y ∈ {{0,20,40,60,80,100}}")
    print(f"  (shared data from data/cmp_regressor_*.bin)")
    print(f"{'─'*60}")
    print(f"{'epoch':>5}  {'train MAE':>10}  {'test MAE':>10}")

    for epoch in range(1, N_EPOCHS + 1):
        tm.fit(X_tr, y_tr)
        if epoch % 5 == 0 or epoch == 1:
            tr_mae = float(np.mean(np.abs(tm.predict(X_tr) - y_tr)))
            te_mae = float(np.mean(np.abs(tm.predict(X_te) - y_te)))
            print(f"{epoch:>5}  {tr_mae:>10.3f}  {te_mae:>10.3f}")

    preds_te = tm.predict(X_te)
    final_mae  = float(np.mean(np.abs(preds_te - y_te)))
    final_rmse = float(np.sqrt(np.mean((preds_te - y_te) ** 2)))
    print(f"\nFinal test MAE:  {final_mae:.3f}")
    print(f"Final test RMSE: {final_rmse:.3f}")


# ── 4. ConvolutionalTM 1-D ────────────────────────────────────────────────────

def run_convolutional(args):
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        print("[convolutional] SKIP — tmu not installed or missing TMClassifier")
        return

    N_FEATURES  = 4
    KERNEL_SIZE = 2
    STRIDE      = 1
    N_PATCHES   = (N_FEATURES - KERNEL_SIZE) // STRIDE + 1   # 3
    N_CLAUSES   = 100
    THRESHOLD   = 50
    S           = 3.5
    N_EPOCHS    = 60

    X_tr = load_X("conv", "train", 5000, N_FEATURES).astype(np.uint32)
    y_tr = load_y_u32("conv", "train", 5000)
    X_te = load_X("conv", "test",  1000, N_FEATURES).astype(np.uint32)
    y_te = load_y_u32("conv", "test",  1000)

    X_tr_exp_list, y_tr_exp_list = [], []
    for p in range(N_PATCHES):
        X_tr_exp_list.append(X_tr[:, p * STRIDE:p * STRIDE + KERNEL_SIZE])
        y_tr_exp_list.append(y_tr)
    X_tr_exp = np.concatenate(X_tr_exp_list, axis=0)
    y_tr_exp = np.concatenate(y_tr_exp_list, axis=0)

    tm = TMClassifier(
        number_of_clauses=N_CLAUSES * 2,
        T=THRESHOLD,
        s=S,
        number_of_state_bits_ta=8,
        weighted_clauses=True,
        platform="CPU",
    )

    print(f"\n{'─'*60}")
    print(f"  Python TMU  ConvolutionalTM 1-D (patch-expanded training)")
    print(f"  {N_FEATURES} features · kernel={KERNEL_SIZE} · stride={STRIDE} · "
          f"{N_PATCHES} patches")
    print(f"  {N_CLAUSES} clauses/class · T={THRESHOLD} · s={S}")
    print(f"  Label: x[0] XOR x[1]  (signal always at patch 0; patches 1,2 = noise)")
    print(f"  (shared data from data/cmp_conv_*.bin)")
    print(f"{'─'*60}")
    print(f"{'epoch':>5}  {'train acc':>10}  {'test acc':>10}")

    def _vote(X, n):
        votes = np.zeros((n, 2), dtype=int)
        for p in range(N_PATCHES):
            patch = X[:, p * STRIDE:p * STRIDE + KERNEL_SIZE]
            preds = tm.predict(patch)
            votes[np.arange(n), preds] += 1
        return votes.argmax(axis=1)

    best_te = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        tm.fit(X_tr_exp, y_tr_exp)
        if epoch % 10 == 0 or epoch == 1:
            tr_acc = float(np.mean(_vote(X_tr, len(X_tr)) == y_tr))
            te_acc = float(np.mean(_vote(X_te, len(X_te)) == y_te))
            best_te = max(best_te, te_acc)
            print(f"{epoch:>5}  {tr_acc:>10.4f}  {te_acc:>10.4f}")

    final_acc = float(np.mean(_vote(X_te, len(X_te)) == y_te))
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Best  test accuracy: {best_te:.4f}  (across evaluated epochs)")


# ── 5. ConvolutionalTM 2-D ────────────────────────────────────────────────────

def run_conv2d(args):
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        print("[conv2d] SKIP — tmu not installed or missing TMClassifier")
        return

    INPUT_ROWS   = 2
    INPUT_COLS   = 4
    PATCH_ROWS   = 2
    PATCH_COLS   = 2
    STRIDE       = 1
    N_PATCH_ROWS = (INPUT_ROWS - PATCH_ROWS) // STRIDE + 1   # 1
    N_PATCH_COLS = (INPUT_COLS - PATCH_COLS) // STRIDE + 1   # 3
    N_PATCHES    = N_PATCH_ROWS * N_PATCH_COLS                # 3
    N_FEATURES   = INPUT_ROWS * INPUT_COLS                    # 8
    N_CLAUSES    = 100
    THRESHOLD    = 50
    S            = 3.5
    N_EPOCHS     = 60

    X_tr = load_X("conv2d", "train", 5000, N_FEATURES).astype(np.uint32)
    y_tr = load_y_u32("conv2d", "train", 5000)
    X_te = load_X("conv2d", "test",  1000, N_FEATURES).astype(np.uint32)
    y_te = load_y_u32("conv2d", "test",  1000)

    def _extract_patch(X, pr, pc):
        rows = []
        for r in range(PATCH_ROWS):
            start = (pr + r) * INPUT_COLS + pc
            rows.append(X[:, start:start + PATCH_COLS])
        return np.concatenate(rows, axis=1)

    X_tr_exp_list, y_tr_exp_list = [], []
    for pr in range(N_PATCH_ROWS):
        for pc in range(N_PATCH_COLS):
            X_tr_exp_list.append(_extract_patch(X_tr, pr, pc))
            y_tr_exp_list.append(y_tr)
    X_tr_exp = np.concatenate(X_tr_exp_list, axis=0)
    y_tr_exp = np.concatenate(y_tr_exp_list, axis=0)

    tm = TMClassifier(
        number_of_clauses=N_CLAUSES * 2,
        T=THRESHOLD,
        s=S,
        number_of_state_bits_ta=8,
        weighted_clauses=True,
        platform="CPU",
    )

    print(f"\n{'─'*60}")
    print(f"  Python TMU  ConvolutionalTM 2-D (patch-expanded training)")
    print(f"  {INPUT_ROWS}×{INPUT_COLS} image · patch={PATCH_ROWS}×{PATCH_COLS} · "
          f"stride={STRIDE} · {N_PATCHES} patches")
    print(f"  {N_CLAUSES} clauses/class · T={THRESHOLD} · s={S}")
    print(f"  Label: x[0,0] XOR x[1,0]  (vertical XOR, signal at patch (0,0))")
    print(f"  (shared data from data/cmp_conv2d_*.bin)")
    print(f"{'─'*60}")
    print(f"{'epoch':>5}  {'train acc':>10}  {'test acc':>10}")

    def _vote(X, n):
        votes = np.zeros((n, 2), dtype=int)
        for pr in range(N_PATCH_ROWS):
            for pc in range(N_PATCH_COLS):
                patch = _extract_patch(X, pr, pc)
                preds = tm.predict(patch)
                votes[np.arange(n), preds] += 1
        return votes.argmax(axis=1)

    best_te = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        tm.fit(X_tr_exp, y_tr_exp)
        if epoch % 10 == 0 or epoch == 1:
            tr_acc = float(np.mean(_vote(X_tr, len(X_tr)) == y_tr))
            te_acc = float(np.mean(_vote(X_te, len(X_te)) == y_te))
            best_te = max(best_te, te_acc)
            print(f"{epoch:>5}  {tr_acc:>10.4f}  {te_acc:>10.4f}")

    final_acc = float(np.mean(_vote(X_te, len(X_te)) == y_te))
    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print(f"Best  test accuracy: {best_te:.4f}  (across evaluated epochs)")


# ── 6. TMCompositeClassifier ──────────────────────────────────────────────────

def run_composite(args):
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        print("[composite] SKIP — tmu not installed or missing TMClassifier")
        return

    TMComposite = None
    try:
        from tmu.models.classification.composite_classifier import TMCompositeClassifier
        TMComposite = TMCompositeClassifier
    except ImportError:
        pass

    N_FEATURES    = 8
    N_CLASSES     = 4
    CLAUSES_EACH  = 20
    TOTAL_CLAUSES = 60
    THRESHOLD     = 20
    S             = 3.9
    N_EPOCHS      = 30

    X_tr = load_X("composite", "train", 5000, N_FEATURES).astype(np.uint32)
    y_tr = load_y_u32("composite", "train", 5000)
    X_te = load_X("composite", "test",  1000, N_FEATURES).astype(np.uint32)
    y_te = load_y_u32("composite", "test",  1000)

    print(f"\n{'─'*60}")
    print(f"  Python TMU  TMCompositeClassifier vs single")
    print(f"  {N_FEATURES} features · 4 classes · s={S} · 5% train noise")
    print(f"  (shared data from data/cmp_composite_*.bin)")

    if TMComposite is not None:
        composite = TMComposite()
        for _ in [10, 20, 30]:
            composite.add(TMClassifier(
                number_of_clauses=CLAUSES_EACH * N_CLASSES,
                T=THRESHOLD, s=S,
                number_of_state_bits_ta=8, weighted_clauses=True, platform="CPU",
            ))
        use_native = True
        print(f"  Composite: 3×{CLAUSES_EACH} clauses/class (native TMCompositeClassifier)")
    else:
        classifiers = [
            TMClassifier(
                number_of_clauses=CLAUSES_EACH * N_CLASSES,
                T=THRESHOLD, s=S,
                number_of_state_bits_ta=8, weighted_clauses=True, platform="CPU",
            )
            for _ in range(3)
        ]
        use_native = False
        print(f"  Composite: 3×{CLAUSES_EACH} clauses/class (manual ensemble)")

    single = TMClassifier(
        number_of_clauses=TOTAL_CLAUSES * N_CLASSES,
        T=THRESHOLD, s=S,
        number_of_state_bits_ta=8, weighted_clauses=True, platform="CPU",
    )

    print(f"  Single:    {TOTAL_CLAUSES} clauses/class")
    print(f"{'─'*60}")
    print(f"{'epoch':>5}  {'composite acc':>14}  {'single acc':>12}")

    def _composite_predict(X):
        if use_native:
            return composite.predict(X)
        votes = np.zeros((len(X), N_CLASSES), dtype=int)
        for clf in classifiers:
            preds = clf.predict(X)
            votes[np.arange(len(X)), preds] += 1
        return votes.argmax(axis=1)

    def _composite_fit(X, y):
        if use_native:
            composite.fit(X, y)
        else:
            for clf in classifiers:
                clf.fit(X, y)

    for epoch in range(1, N_EPOCHS + 1):
        _composite_fit(X_tr, y_tr)
        single.fit(X_tr, y_tr)
        if epoch % 5 == 0 or epoch == 1:
            ca = float(np.mean(_composite_predict(X_te) == y_te))
            sa = float(np.mean(single.predict(X_te) == y_te))
            print(f"{epoch:>5}  {ca:>14.4f}  {sa:>12.4f}")

    ca = float(np.mean(_composite_predict(X_te) == y_te))
    sa = float(np.mean(single.predict(X_te) == y_te))
    print(f"\nFinal composite accuracy: {ca:.4f}")
    print(f"Final single accuracy:    {sa:.4f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--classifier",  action="store_true",
                   help="TMClassifier speed benchmark (default: large scale)")
    p.add_argument("--autoencoder", action="store_true",
                   help="TMAutoEncoder speed benchmark (default: large scale)")
    p.add_argument("--regressor",   action="store_true",
                   help="TMRegressor accuracy comparison")
    p.add_argument("--conv",        action="store_true",
                   help="ConvolutionalTM 1-D accuracy comparison")
    p.add_argument("--conv2d",      action="store_true",
                   help="ConvolutionalTM 2-D accuracy comparison")
    p.add_argument("--composite",   action="store_true",
                   help="TMCompositeClassifier accuracy comparison")

    # sub-mode flags (only apply to classifier and autoencoder)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--small", action="store_true",
                      help="Use small (NoisyXOR/accuracy-check) config for classifier/autoencoder")
    mode.add_argument("--both",  action="store_true",
                      help="Run both small and large configs for classifier/autoencoder")

    args = p.parse_args()

    run_all = not (
        args.classifier or args.autoencoder or args.regressor
        or args.conv or args.conv2d or args.composite
    )

    if run_all or args.classifier:
        run_classifier(args)
    if run_all or args.autoencoder:
        run_autoencoder(args)
    if run_all or args.regressor:
        run_regressor(args)
    if run_all or args.conv:
        run_convolutional(args)
    if run_all or args.conv2d:
        run_conv2d(args)
    if run_all or args.composite:
        run_composite(args)


if __name__ == "__main__":
    main()
