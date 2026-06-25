#!/usr/bin/env python3
"""
Generate shared binary data files used by both the Rust examples and
scripts/compare_new_models.py so that both sides train and test on
bit-identical samples.

Run once before any comparison:
  python scripts/gen_shared_data.py

Produces 14 files in data/:
  data/cmp_regressor_X_{train,test}.bin   uint8  5000×20 / 1000×20
  data/cmp_regressor_y_{train,test}.bin   float64 (little-endian)
  data/cmp_conv_X_{train,test}.bin        uint8  5000×12 / 1000×12
  data/cmp_conv_y_{train,test}.bin        uint32 (little-endian)
  data/cmp_conv2d_X_{train,test}.bin      uint8  5000×8 / 1000×8
  data/cmp_conv2d_y_{train,test}.bin      uint32 (little-endian)
  data/cmp_composite_X_{train,test}.bin   uint8  5000×8 / 1000×8
  data/cmp_composite_y_{train,test}.bin   uint32 (little-endian)

Format: raw little-endian bytes, row-major, no header.
"""
import pathlib
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy required — pip install numpy")

DATA = pathlib.Path("data")
DATA.mkdir(exist_ok=True)


def _save(path, arr):
    """Write arr to path with native-dtype preservation; always little-endian."""
    arr.astype(arr.dtype.newbyteorder("<")).tofile(path)
    print(f"  wrote {path}  shape={arr.shape}  dtype={arr.dtype}")


# ── TMRegressor ───────────────────────────────────────────────────────────────
# Dataset: 20 binary features; target y = (count of 1s in features 0..4) × 20
# y ∈ {0, 20, 40, 60, 80, 100}; threshold = 100
def _make_regressor(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 20), dtype=np.uint8)
    scale = 100.0 / 5.0   # 5 counting features → max count = 5
    y = (X[:, :5].sum(axis=1) * scale).astype(np.float64)
    return X, y


print("── TMRegressor ──────────────────────────────────────────────────────")
X_tr, y_tr = _make_regressor(5000, seed=1)
X_te, y_te = _make_regressor(1000, seed=2)
_save(DATA / "cmp_regressor_X_train.bin", X_tr)
_save(DATA / "cmp_regressor_y_train.bin", y_tr)
_save(DATA / "cmp_regressor_X_test.bin",  X_te)
_save(DATA / "cmp_regressor_y_test.bin",  y_te)


# ── ConvolutionalTM (1-D) ─────────────────────────────────────────────────────
# Dataset: 4 binary features; XOR pattern fixed at patch 0 (features 0 and 1).
# Kernel=2, stride=1 → 3 patch positions ([0,1], [1,2], [2,3]).
# Label = X[:,0] XOR X[:,1]; features 2 and 3 are independent noise.
#
# Structured design: repeat all 16 possible 4-bit patterns equally so that both
# the Rust CTM (OR semantics) and the Python TMU (patch-expansion) see each
# feature combination the same number of times. This guarantees convergence for
# both approaches regardless of random-seed differences in the RNGs.
def _make_conv1d(n, seed):
    rng = np.random.default_rng(seed)
    # All 16 patterns: shape (16, 4)
    patterns = np.array(
        [[int(b) for b in format(i, '04b')] for i in range(16)], dtype=np.uint8
    )
    # Repeat to fill n rows, then shuffle
    reps = n // 16
    rem  = n % 16
    X = np.concatenate([
        np.tile(patterns, (reps, 1)),
        patterns[:rem]
    ], axis=0).astype(np.uint8)
    rng.shuffle(X)
    y = (X[:, 0] ^ X[:, 1]).astype(np.uint32)
    return X, y


print("\n── ConvolutionalTM 1-D ──────────────────────────────────────────────")
X_tr, y_tr = _make_conv1d(5000, seed=1)
X_te, y_te = _make_conv1d(1000, seed=2)
_save(DATA / "cmp_conv_X_train.bin", X_tr)
_save(DATA / "cmp_conv_y_train.bin", y_tr)
_save(DATA / "cmp_conv_X_test.bin",  X_te)
_save(DATA / "cmp_conv_y_test.bin",  y_te)


# ── ConvolutionalTM (2-D) ─────────────────────────────────────────────────────
# Dataset: 2×4 binary image (8 features, row-major), 2×2 patches, stride=1
# → 1×3 = 3 patch positions: (0,0), (0,1), (0,2)
# Signal at patch (0,0): label = x[0,0] XOR x[1,0]  (vertical XOR of first column).
# All 16 combinations of the first 2×2 block [x[0,0],x[0,1],x[1,0],x[1,1]] are
# equally represented (structured); the remaining 4 pixels are independent noise.
# This gives both OR-semantics and patch-expansion ~75% accuracy (same as 1-D case).
def _make_conv2d(n, seed):
    INPUT_ROWS, INPUT_COLS = 2, 4
    rng = np.random.default_rng(seed)

    # Structured top-left 2×2 block: all 16 patterns equally represented
    # Patch (0,0) covers flat indices [0, 1, 4, 5] = [x[0,0], x[0,1], x[1,0], x[1,1]]
    block_patterns = np.array(
        [[int(b) for b in format(i, '04b')] for i in range(16)], dtype=np.uint8
    )
    reps = n // 16
    rem  = n % 16
    block = np.concatenate([
        np.tile(block_patterns, (reps, 1)),
        block_patterns[:rem]
    ], axis=0).astype(np.uint8)
    rng.shuffle(block)   # block[:, k] = [x[0,0], x[0,1], x[1,0], x[1,1]]

    # label = x[0,0] XOR x[1,0]  (patch-relative: px0 XOR px2)
    y = (block[:, 0] ^ block[:, 2]).astype(np.uint32)

    # Build full 2×4 image; fill noise for non-signal pixels
    X = rng.integers(0, 2, size=(n, INPUT_ROWS * INPUT_COLS), dtype=np.uint8)
    X[:, 0] = block[:, 0]   # x[0,0]
    X[:, 1] = block[:, 1]   # x[0,1]
    X[:, 4] = block[:, 2]   # x[1,0]
    X[:, 5] = block[:, 3]   # x[1,1]
    return X, y


print("\n── ConvolutionalTM 2-D ──────────────────────────────────────────────")
X_tr, y_tr = _make_conv2d(5000, seed=1)
X_te, y_te = _make_conv2d(1000, seed=2)
_save(DATA / "cmp_conv2d_X_train.bin", X_tr)
_save(DATA / "cmp_conv2d_y_train.bin", y_tr)
_save(DATA / "cmp_conv2d_X_test.bin",  X_te)
_save(DATA / "cmp_conv2d_y_test.bin",  y_te)


# ── TMCompositeClassifier ─────────────────────────────────────────────────────
# Dataset: 8 binary features; 4-class XOR problem
# y = 2*(x[0]^x[1]) + (x[2]^x[3])  ∈ {0, 1, 2, 3}
# 5% label noise in training, 0% in test.
def _make_composite(n, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 8), dtype=np.uint8)
    y = (2 * (X[:, 0].astype(np.int32) ^ X[:, 1]) +
             (X[:, 2].astype(np.int32) ^ X[:, 3])).astype(np.uint32)
    if noise > 0.0:
        mask = rng.random(n) < noise
        y[mask] = rng.integers(0, 4, size=int(mask.sum())).astype(np.uint32)
    return X, y


print("\n── TMCompositeClassifier ────────────────────────────────────────────")
X_tr, y_tr = _make_composite(5000, noise=0.05, seed=1)
X_te, y_te = _make_composite(1000, noise=0.0,  seed=2)
_save(DATA / "cmp_composite_X_train.bin", X_tr)
_save(DATA / "cmp_composite_y_train.bin", y_tr)
_save(DATA / "cmp_composite_X_test.bin",  X_te)
_save(DATA / "cmp_composite_y_test.bin",  y_te)

print("\nDone — 14 files written to data/")
