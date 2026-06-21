"""
Same benchmark as examples/absorb_timing.rs but run against TMU (Python/C).
Identical config: n_features=30, n_train=2000, n_clauses=200, T=20, s=3.0.
"""
import time
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier


def absorbed_include_fraction(clause_bank, cps):
    """Fraction of literals at max TA state (all state-bit planes = 1)."""
    cb = clause_bank
    # flat layout: (n_clauses, n_ta_chunks, n_state_bits_ta) → reshape
    state = cb.clause_bank.reshape(
        cb.number_of_clauses, cb.number_of_ta_chunks, cb.number_of_state_bits_ta
    )
    # AND-reduce over state-bit axis → 1-bit = at max state
    at_max = np.bitwise_and.reduce(state[:cps], axis=2)   # (cps, n_ta_chunks) uint32
    absorb = int(np.sum(np.vectorize(lambda x: bin(x).count('1'))(at_max)))
    total  = cps * cb.number_of_ta_chunks * 32
    return absorb / total if total else 0.0


def run(label: str, state_bits: int, n_epochs: int):
    n_features = 30
    n_train    = 2000
    n_clauses  = 200
    threshold  = 20
    s          = 3.0

    rng = np.random.default_rng(42)
    xs = rng.integers(0, 2, size=(n_train, n_features), dtype=np.uint32)
    ys = (xs[:, 0] ^ xs[:, 1]).astype(np.uint32)

    tm = TMClassifier(
        number_of_clauses=n_clauses,
        T=threshold,
        s=s,
        number_of_state_bits_ta=state_bits,
        weighted_clauses=True,
        platform="CPU",
    )

    n_classes = 2
    cps = n_clauses // n_classes

    print(f"\n── {label} (state_bits={state_bits}) ──")
    print(f"{'epoch':>5}  {'time':>9}  {'acc%':>6}  {'abs_inc%':>8}")

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        tm.fit(xs, ys)
        us = int((time.perf_counter() - t0) * 1_000_000)

        acc = 100.0 * float(np.mean(tm.predict(xs) == ys))
        abs_pct = absorbed_include_fraction(tm.clause_banks[0], cps) * 100.0
        print(f"{epoch:>5}  {us:>7}µs  {acc:>5.1f}%  {abs_pct:>7.1f}%")


if __name__ == "__main__":
    run("fast absorb", state_bits=2, n_epochs=200)
    run("slow absorb", state_bits=8, n_epochs=15)
