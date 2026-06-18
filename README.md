# Examples — ports of the TMU classification demos

These reproduce the [`cair/tmu`](https://github.com/cair/tmu) demos that use the
plain multiclass `TMClassifier` (the model this crate ports). TMU's
convolutional, coalesced, regression, autoencoder, and composite demos use
different machine types and are **not** included.

| TMU demo                       | this crate            | data needed | run |
|--------------------------------|-----------------------|-------------|-----|
| `XORDemo`                      | `xor`                 | none        | `cargo run --release --example xor` |
| `NoisyXORDemo`                 | `noisy_xor`           | none        | `cargo run --release --example noisy_xor` |
| `InterpretabilityDemo`         | `interpretability`    | none        | `cargo run --release --example interpretability` |
| `BreastCancerDemo`             | `breast_cancer`       | sklearn     | `python scripts/prepare_breast_cancer.py` then `cargo run --release --example breast_cancer` |
| `MNISTDemo` / `…WeightedClauses` | `mnist`             | MNIST       | `python scripts/prepare_mnist.py` then `cargo run --release --features parallel --example mnist` |
| `IMDbTextCategorizationDemo`   | `imdb`                | Keras IMDb  | `python scripts/prepare_imdb.py` then `cargo run --release --features parallel --example imdb` |

Plus `ndr_flows` (not a TMU demo): a synthetic network-flow detection example
showing the booleanizer + interpretable rules.

## Data prep

The three dataset-backed demos read files produced by the Python scripts in
`scripts/`. They run on your machine

* `prepare_breast_cancer.py` — uses the dataset bundled with scikit-learn
  (`pip install scikit-learn`), no download. Writes `data/breast_cancer.csv`.
* `prepare_mnist.py` — Keras (`tensorflow`) if present, else scikit-learn's
  OpenML fetch (one-time download). Writes binarized `data/mnist_*_bin.csv`
  (pixel > 75 → 1, i.e. ~0.3·255, matching the classic TM binarization).
* `prepare_imdb.py` — Keras IMDb (`tensorflow`). Writes sparse bag-of-words
  `data/imdb_*.txt`; `N_FEATURES` there must match `examples/imdb.rs` (5000).

## Validation status

* `xor`, `noisy_xor`, `interpretability` and the **breast_cancer** pipeline were
  validated on real/generated data via the Python mirror (breast cancer reaches
  ~99–100% test accuracy). The multiclass binary pipeline used by `mnist`/`imdb`
  was validated on scikit-learn's 8×8 digits (~93% with few clauses).
* Hyperparameters mirror the spirit of the TMU demos (e.g. MNIST: 2000 clauses,
  T=50, s=10.0; IMDb: 2000 clauses, T=80, s=10.0). Lower the clause counts for
  faster runs. These large configs benefit a lot from `--features parallel`.
* As with the rest of the crate, the Rust here has **not** been compiled in the
  authoring environment — run `cargo build --examples` first.
