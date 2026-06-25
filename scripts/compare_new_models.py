#!/usr/bin/env python3
"""
Accuracy comparison: Python TMU vs Rust tmu-rs for the three new model types.

  1. TMRegressor     — continuous output (count function, threshold=100)
  2. ConvolutionalTM — 1-D receptive-field clauses on position-invariant XOR
  3. TMCompositeClassifier — ensemble on 4-class XOR

Results are printed in a format that mirrors the Rust examples so that
you can run the Rust examples side-by-side and compare.

Usage:
  python scripts/compare_new_models.py            # all three models
  python scripts/compare_new_models.py --regressor
  python scripts/compare_new_models.py --conv
  python scripts/compare_new_models.py --composite
"""
import argparse
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy required — pip install numpy")

# ── shared data generators ────────────────────────────────────────────────────

def make_binary(n, n_features, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, n_features), dtype=np.uint32)


# ── 1. TMRegressor ────────────────────────────────────────────────────────────

def run_regressor():
    """
    Target: count of 1s in features 0..4, scaled to [0, 100].
    Matches examples/regression.rs exactly (same config, same split sizes).
    """
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
    SCALE      = THRESHOLD / 5.0   # 5 counting features → max count = 5

    def make(n, seed):
        xs = make_binary(n, N_FEATURES, seed)
        ys = (xs[:, :5].sum(axis=1) * SCALE).astype(float)
        return xs, ys

    X_tr, y_tr = make(5000, 1)
    X_te, y_te = make(1000, 2)

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


# ── 2. ConvolutionalTM ────────────────────────────────────────────────────────

def _extract_patches_1d(X, kernel_size, stride):
    """Extract 1-D patches from rows of X: shape (N, n_patches * kernel_size)."""
    n, n_features = X.shape
    n_patches = (n_features - kernel_size) // stride + 1
    patches = np.zeros((n, n_patches * kernel_size), dtype=np.uint32)
    for p in range(n_patches):
        start = p * stride
        patches[:, p * kernel_size:(p + 1) * kernel_size] = X[:, start:start + kernel_size]
    return patches, n_patches


def run_convolutional():
    """
    1-D XOR pattern that can appear at any of n_patches positions (position-invariant).
    The clause bank slides over each patch; weight tying lets the model find the pattern
    regardless of where it sits — the core use-case for a convolutional TM.

    Dataset:
      Each sample has N_FEATURES binary features.
      A random patch position p is chosen; label = XOR(features[p], features[p+1]).
      Other patches carry random noise.

    Python approach: pre-extract all patches (n_patches copies per sample), each
    of width kernel_size.  Train a flat TMClassifier on the concatenated patch
    features (shape N × kernel_size) — weight tying is achieved by replaying every
    patch to the same clause bank.

    Rust side: examples/convolutional.rs (fixed-position XOR, 3 patches).
    """
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        print("[convolutional] SKIP — tmu not installed or missing TMClassifier")
        return

    N_FEATURES  = 12
    KERNEL_SIZE = 2
    STRIDE      = 1
    N_PATCHES   = (N_FEATURES - KERNEL_SIZE) // STRIDE + 1   # 11
    N_CLAUSES   = 200    # per class
    THRESHOLD   = 100
    S           = 3.5
    N_EPOCHS    = 60

    def make(n, seed):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, size=(n, N_FEATURES), dtype=np.uint32)
        # Randomly choose which patch carries the label (position-invariant)
        pos = rng.integers(0, N_PATCHES, size=n)
        y = (X[np.arange(n), pos] ^ X[np.arange(n), pos + 1]).astype(np.uint32)
        return X, y

    X_tr, y_tr = make(5000, 1)
    X_te, y_te = make(1000, 2)

    # Expand training set: replicate each sample once per patch position.
    # This simulates weight tying — the TMClassifier sees every patch with
    # the same label, forcing it to learn a patch-position-agnostic rule.
    X_tr_exp_list = []
    y_tr_exp_list = []
    for p in range(N_PATCHES):
        X_tr_exp_list.append(X_tr[:, p * STRIDE:p * STRIDE + KERNEL_SIZE])
        y_tr_exp_list.append(y_tr)
    X_tr_exp = np.concatenate(X_tr_exp_list, axis=0)
    y_tr_exp = np.concatenate(y_tr_exp_list, axis=0)

    # Predict by majority vote over all patch positions.
    # For each test sample and each patch, predict; argmax of vote sum wins.
    # TMClassifier with 2 classes: predict returns 0/1; use predict_proba / raw scores.
    tm = TMClassifier(
        number_of_clauses=N_CLAUSES * 2,   # total (2 classes)
        T=THRESHOLD,
        s=S,
        number_of_state_bits_ta=8,
        weighted_clauses=True,
        platform="CPU",
    )

    print(f"\n{'─'*60}")
    print(f"  Python TMU  ConvolutionalTM (patch-expanded training)")
    print(f"  {N_FEATURES} features · kernel={KERNEL_SIZE} · stride={STRIDE} · "
          f"{N_PATCHES} patches")
    print(f"  {N_CLAUSES} clauses/class · T={THRESHOLD} · s={S}")
    print(f"  Dataset: position-invariant XOR (pattern at random patch)")
    print(f"{'─'*60}")
    print(f"{'epoch':>5}  {'train acc':>10}  {'test acc':>10}")

    for epoch in range(1, N_EPOCHS + 1):
        tm.fit(X_tr_exp, y_tr_exp)
        if epoch % 10 == 0 or epoch == 1:
            # Vote across patches for training acc
            tr_votes = np.zeros((len(X_tr), 2), dtype=int)
            for p in range(N_PATCHES):
                patch = X_tr[:, p * STRIDE:p * STRIDE + KERNEL_SIZE]
                preds = tm.predict(patch)
                tr_votes[np.arange(len(X_tr)), preds] += 1
            tr_acc = float(np.mean(tr_votes.argmax(axis=1) == y_tr))

            te_votes = np.zeros((len(X_te), 2), dtype=int)
            for p in range(N_PATCHES):
                patch = X_te[:, p * STRIDE:p * STRIDE + KERNEL_SIZE]
                preds = tm.predict(patch)
                te_votes[np.arange(len(X_te)), preds] += 1
            te_acc = float(np.mean(te_votes.argmax(axis=1) == y_te))

            print(f"{epoch:>5}  {tr_acc:>10.4f}  {te_acc:>10.4f}")

    # Final accuracy
    te_votes = np.zeros((len(X_te), 2), dtype=int)
    for p in range(N_PATCHES):
        patch = X_te[:, p * STRIDE:p * STRIDE + KERNEL_SIZE]
        preds = tm.predict(patch)
        te_votes[np.arange(len(X_te)), preds] += 1
    final_acc = float(np.mean(te_votes.argmax(axis=1) == y_te))
    print(f"\nFinal test accuracy: {final_acc:.4f}")


# ── 3. TMCompositeClassifier ──────────────────────────────────────────────────

def run_composite():
    """
    4-class XOR problem.  Mirrors examples/composite.rs exactly:
      y = 2*(x[0]^x[1]) + (x[2]^x[3])  ∈ {0, 1, 2, 3}
      5% label noise in training, 0% in test.

    Composite: 3 × 20 clauses/class.  Single: 60 clauses/class.
    """
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError:
        print("[composite] SKIP — tmu not installed or missing TMClassifier")
        return

    # Try to import TMCompositeClassifier (may not exist in all TMU versions)
    TMComposite = None
    try:
        from tmu.models.classification.composite_classifier import TMCompositeClassifier
        TMComposite = TMCompositeClassifier
    except ImportError:
        pass

    N_FEATURES   = 8
    N_CLASSES    = 4
    CLAUSES_EACH = 20
    TOTAL_CLAUSES = 60
    THRESHOLD    = 20
    S            = 3.9
    N_EPOCHS     = 30
    NOISE        = 0.05

    def make(n, noise, seed):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, size=(n, N_FEATURES), dtype=np.uint32)
        y = (2 * (X[:, 0] ^ X[:, 1]) + (X[:, 2] ^ X[:, 3])).astype(np.uint32)
        if noise > 0:
            mask = rng.random(n) < noise
            y[mask] = rng.integers(0, N_CLASSES, size=mask.sum()).astype(np.uint32)
        return X, y

    X_tr, y_tr = make(5000, NOISE, 1)
    X_te, y_te = make(1000, 0.0, 2)

    print(f"\n{'─'*60}")
    print(f"  Python TMU  TMCompositeClassifier vs single")
    print(f"  {N_FEATURES} features · 4 classes · s={S} · {NOISE*100:.0f}% train noise")

    if TMComposite is not None:
        # Use native TMCompositeClassifier
        composite = TMComposite()
        for seed in [10, 20, 30]:
            composite.add(TMClassifier(
                number_of_clauses=CLAUSES_EACH * N_CLASSES,
                T=THRESHOLD,
                s=S,
                number_of_state_bits_ta=8,
                weighted_clauses=True,
                platform="CPU",
            ))
        use_native = True
        print(f"  Composite: 3×{CLAUSES_EACH} clauses/class (native TMCompositeClassifier)")
    else:
        # Fallback: manually aggregate 3 independent classifiers
        classifiers = [
            TMClassifier(
                number_of_clauses=CLAUSES_EACH * N_CLASSES,
                T=THRESHOLD,
                s=S,
                number_of_state_bits_ta=8,
                weighted_clauses=True,
                platform="CPU",
            )
            for _ in range(3)
        ]
        use_native = False
        print(f"  Composite: 3×{CLAUSES_EACH} clauses/class (manual ensemble — TMCompositeClassifier not found)")

    single = TMClassifier(
        number_of_clauses=TOTAL_CLAUSES * N_CLASSES,
        T=THRESHOLD,
        s=S,
        number_of_state_bits_ta=8,
        weighted_clauses=True,
        platform="CPU",
    )

    print(f"  Single:    {TOTAL_CLAUSES} clauses/class")
    print(f"{'─'*60}")
    print(f"{'epoch':>5}  {'composite acc':>14}  {'single acc':>12}")

    def composite_predict(X):
        if use_native:
            return composite.predict(X)
        votes = np.zeros((len(X), N_CLASSES), dtype=int)
        for clf in classifiers:
            preds = clf.predict(X)
            votes[np.arange(len(X)), preds] += 1
        return votes.argmax(axis=1)

    def composite_fit(X, y):
        if use_native:
            composite.fit(X, y)
        else:
            for clf in classifiers:
                clf.fit(X, y)

    for epoch in range(1, N_EPOCHS + 1):
        composite_fit(X_tr, y_tr)
        single.fit(X_tr, y_tr)
        if epoch % 5 == 0 or epoch == 1:
            ca = float(np.mean(composite_predict(X_te) == y_te))
            sa = float(np.mean(single.predict(X_te) == y_te))
            print(f"{epoch:>5}  {ca:>14.4f}  {sa:>12.4f}")

    ca = float(np.mean(composite_predict(X_te) == y_te))
    sa = float(np.mean(single.predict(X_te) == y_te))
    print(f"\nFinal composite accuracy: {ca:.4f}")
    print(f"Final single accuracy:    {sa:.4f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare Python TMU vs Rust tmu-rs for new model types.",
    )
    p.add_argument("--regressor",  action="store_true", help="TMRegressor only")
    p.add_argument("--conv",       action="store_true", help="ConvolutionalTM only")
    p.add_argument("--composite",  action="store_true", help="TMCompositeClassifier only")
    args = p.parse_args()

    run_all = not (args.regressor or args.conv or args.composite)

    if run_all or args.regressor:
        run_regressor()
    if run_all or args.conv:
        run_convolutional()
    if run_all or args.composite:
        run_composite()


if __name__ == "__main__":
    main()
