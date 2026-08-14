//! Mini-batch data-parallel training: the primary multicore path.
//!
//! # Why this exists
//!
//! The crate has two older ways to use more than one core during training, and
//! neither scales well:
//!
//! * **Clause-parallel** (inside each model's `update_class`) enters a Rayon region
//!   *once per class update*, i.e. twice per training sample. Dispatch overhead
//!   dominates, so it is gated off below
//!   [`DENSE_TRAIN_PARALLEL_MIN`](crate::clause_bank::dense::DENSE_TRAIN_PARALLEL_MIN)
//!   = 4096 clauses/class, and even above it a 4-core box measures only ~1.3×.
//!   It stays as the fallback (and is the only multicore option under Type III),
//!   because it is bit-identical to scalar training.
//! * **Full-epoch sharding** split the whole epoch across threads, trained a private
//!   replica per shard, then averaged once. One dispatch per epoch, but each replica
//!   drifted over `n / threads` samples before any synchronisation, so accuracy
//!   suffered — and the model was `clone()`d per shard every epoch.
//!
//! This module keeps the replica idea and fixes the cadence: shard **mini-batches**
//! of `plan.batch` samples, merge, repeat. Dispatches drop from `2n` per epoch to
//! `~2n/batch`, replicas resynchronise many times per epoch instead of once, and
//! replicas are cached and refreshed in place instead of reallocated.
//!
//! # What to expect from it
//!
//! Do not expect large speedups. Dense TM training is memory-bandwidth bound: on a
//! 4-core Xeon this path measures within ~10% of the exact clause-parallel path
//! across model sizes from 0.03 MiB to 33.6 MiB of TA state. That is the same
//! ceiling the exact path runs into (~1.3–1.4×, see `BENCHMARKS.md`), and it is a
//! property of the workload rather than of the scheduling strategy. The merge
//! cadence is the one setting that moves the number materially, and mostly
//! downwards if set too small — see [`DP_SYNC_OVERHEAD_DIVISOR`].
//!
//! What this path *does* buy is a multicore route that applies at any model size
//! (the exact path is gated off below 4096 clauses/class), needs no opt-in, and
//! costs one dispatch per mini-batch rather than two per sample.
//!
//! # Determinism
//!
//! The sample order is drawn from the master RNG, the shard seeds are drawn in a
//! fixed order, each replica trains a fixed contiguous slice with its own streams,
//! and the merge reduces replicas in index order and is chunk-order independent.
//! So data-parallel output is **bit-reproducible for a fixed
//! `(seed, shards, batch, rule)`** and independent of Rayon scheduling. It is *not*
//! the same as exact training, and it changes with the shard count — pin
//! `data_parallel_shards` (or `TMU_RS_DP_SHARDS`) to reproduce across machines with
//! different core counts.
//!
//! # Rejected alternative
//!
//! Persistent per-shard threads with a [`std::sync::Barrier`] per mini-batch would
//! cut dispatch to `k` spawns per epoch, but it needs simultaneous `&mut` into every
//! replica (not expressible without an unsafe aliasing helper) and buys nothing
//! measurable: at `batch = 128` a 3 s epoch does ~32 fan-outs, i.e. tens of
//! microseconds total. Worse, a Rayon worker blocked on a barrier cannot steal, and
//! with fewer pool threads than shards it deadlocks. Mini-batching is the fix.

/// Cached per-shard training replicas.
///
/// Lives on the model as a `serde(skip)` field so it never reaches disk, and is
/// rebuilt lazily by [`fit_epoch_dp`]. Memory cost while populated is
/// `shards × ~1.13 × n_clauses × n_literals` bytes — call the model's
/// `clear_replica_cache()` to release it.
pub(crate) struct ReplicaPool<M> {
    pub(crate) replicas: Vec<M>,
}

impl<M> Default for ReplicaPool<M> {
    fn default() -> Self {
        Self {
            replicas: Vec::new(),
        }
    }
}

/// Cloning a model yields a model with an **empty** pool.
///
/// This is not a shortcut, it is load-bearing: models derive `Clone`, and
/// [`DataParallel::dp_fork`] is itself built on `clone()`. A derived `Clone` here
/// would deep-copy every cached replica, making a single model clone cost
/// `shards × model_bytes` and making forking quadratic. A fresh clone lazily
/// re-forks its own replicas instead.
impl<M> Clone for ReplicaPool<M> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<M> std::fmt::Debug for ReplicaPool<M> {
    /// Prints the replica *count* only — never recurses into the cached models,
    /// which would make `{:?}` on a model dump the whole pool.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReplicaPool {{ replicas: <{} cached> }}",
            self.replicas.len()
        )
    }
}

/// Overwrite `dst` with `src`, reusing `dst`'s allocation whenever the lengths match.
///
/// This is what makes replica refresh allocation-free in steady state. The
/// length-mismatch branch also makes a stale replica *self-heal* after the master's
/// geometry changes (e.g. `grow_features`), so refresh doubles as invalidation.
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn sync_vec<T: Copy>(dst: &mut Vec<T>, src: &[T]) {
    if dst.len() == src.len() {
        dst.copy_from_slice(src);
    } else {
        dst.clear();
        dst.extend_from_slice(src);
    }
}

#[cfg(feature = "parallel")]
pub(crate) use imp::*;

#[cfg(feature = "parallel")]
mod imp {
    use super::ReplicaPool;
    use crate::clause_bank::dense::PARALLEL_MIN;
    use crate::rng::Rng;
    use rayon::prelude::*;
    use std::sync::OnceLock;

    // ---- tuning constants -------------------------------------------------
    //
    // Calibrated on a 4-core box with `examples/parallel_scaling.rs` (batch sweep)
    // and `examples/bench_training.rs`. They select a code path, and — unlike
    // `use_parallel` — they also select *semantics*, so they are deliberately
    // conservative: when in doubt, stay exact.

    /// Sample-count floor before **auto** considers data-parallel training.
    ///
    /// An epoch needs enough samples to give every shard a meaningful slice —
    /// `shards × DP_MIN_SAMPLES_PER_SHARD` is 128 at four shards — and to amortise
    /// the one-time replica fork. Small per-call sites (e.g.
    /// `examples/mordor_autoencoder.rs`, which calls `fit_epoch` on mini-batches)
    /// stay below this and keep training exactly.
    ///
    /// Keep this genuinely low. At 1024 it silently disabled auto for the entire
    /// `examples/parallel_scaling.rs` sweep, which runs 1000 samples/epoch, so the
    /// feature measured as completely inert.
    pub(crate) const DP_AUTO_MIN_SAMPLES: usize = 256;

    /// Per-sample TA traffic floor (bytes) before **auto** engages.
    ///
    /// This — not the model size — is the lower gate, because what has to beat thread
    /// dispatch is the work *one sample* does. An 8-clause/class model over 12
    /// features touches 384 B per sample, far too little to be worth a fan-out; a
    /// 16-clause/class model over 512 features touches 32 KiB.
    ///
    /// Deliberately not [`use_parallel`]: that predicate counts items and words, which
    /// does not distinguish those two cases.
    pub(crate) const DP_AUTO_MIN_SAMPLE_BYTES: usize = 1 << 13;

    /// Total replica memory **auto** will allocate without being asked (1 GiB).
    ///
    /// This is a *memory* guard, not a performance one. Replica training costs
    /// `shards × ~1.13 × model TA bytes` of resident memory, which for a large model
    /// is substantial (a 40 MiB model on 4 shards adds ~180 MiB), and silently
    /// allocating an unbounded amount on the user's behalf is not acceptable. Past
    /// this bound the user must ask for it with `.data_parallel(true)`.
    ///
    /// There is deliberately **no cache-residency ceiling** here, though the intuition
    /// that replicas must lose once they outgrow last-level cache is appealing. On a
    /// 4-core Xeon with 33 MiB of L3, `parallel_scaling` Sweep A (min of 9 epochs)
    /// puts data-parallel within ~10% of the exact path at every model size from
    /// 0.03 MiB to 33.6 MiB — no cliff, and no large win either.
    ///
    /// The reason there is not much in it either way: dense TM training is
    /// **memory-bandwidth bound**, and one core already saturates a good fraction of
    /// the available bandwidth streaming TA counters. Extra cores mostly wait on
    /// memory whichever way the work is divided, which is the same reason the exact
    /// clause-parallel path only reaches ~1.3–1.4× (see `BENCHMARKS.md`). What
    /// data-parallel reliably improves is the *merge cadence* knob — see
    /// [`DP_SYNC_OVERHEAD_DIVISOR`], where the spread is 0.46× to 1.14×.
    ///
    /// Treat single measurements on shared hardware with suspicion. An earlier
    /// revision of this gate declined above half the LLC, on the strength of a
    /// median-of-3 benchmark that appeared to show a 2–3× win collapsing to 0.9×.
    /// Re-running with min-of-9 on an idle box showed both the win and the collapse
    /// were noise: the same configuration had been spanning ±20% between runs.
    pub(crate) const DP_AUTO_MAX_REPLICA_BYTES: usize = 1 << 30;

    /// Below this model size, use a single end-of-epoch merge (`batch == n`).
    ///
    /// Keeps existing `.data_parallel(true)` callers on exactly the semantics they
    /// have today, so this change is strictly additive for them.
    pub(crate) const DP_MINIBATCH_MIN_MODEL_BYTES: usize = 1 << 16;

    /// Fidelity floor for [`MergeRule::Average`]: samples each shard must see between
    /// merges.
    ///
    /// Rounded averaging of `u8` TA counters discards updates smaller than the shard
    /// count (with `k = 4`, a counter only one replica moved by +1 averages to
    /// `0.25 → 0`). Feedback is ±1 per literal per sample, so per-counter drift is
    /// `O(samples_per_shard)`; averaging is only faithful once that drift is
    /// comfortably larger than `k`.
    pub(crate) const DP_MIN_SAMPLES_PER_SHARD: usize = 32;

    /// Target synchronisation overhead: merge+refresh traffic ≤ `1/N` of batch work.
    ///
    /// Calibrated against `parallel_scaling`'s Sweep C (2.1 MiB model, 4 shards,
    /// min-of-9 epochs), which is markedly asymmetric — merging too often is far more
    /// expensive than merging too rarely:
    ///
    /// | samples between merges | vs exact |
    /// |---|---|
    /// | 32  | 0.51× |
    /// | 64  | 0.76× |
    /// | 128 | 1.06× |
    /// | 256 | **1.14×** |
    /// | 512 | 1.09× |
    /// | 1000 (whole epoch) | 1.05× |
    ///
    /// A divisor of 10 landed on 128 for that shape — on the wrong side of the knee.
    /// 20 lands on ~256. Erring towards larger batches is also the safer direction:
    /// the downside is replica drift (bounded by [`DP_BATCH_MAX`]), whereas the
    /// downside of erring small is halving throughput.
    pub(crate) const DP_SYNC_OVERHEAD_DIVISOR: usize = 20;

    /// Upper bound on samples between merges (keeps replica drift bounded).
    pub(crate) const DP_BATCH_MAX: usize = 8192;

    /// Shard cap. Also what makes the `u16` merge accumulator safe:
    /// `255 × 64 = 16 320 < u16::MAX`.
    pub(crate) const DP_MAX_SHARDS: usize = 64;

    /// TA bytes per merge chunk — sized to stay L2-resident so the fused
    /// `rebuild_include` re-reads the counters it just wrote from cache.
    pub(crate) const MERGE_CHUNK_BYTES: usize = 1 << 15;

    /// Below this model size, merge serially rather than paying another dispatch.
    pub(crate) const MERGE_PARALLEL_MIN_BYTES: usize = 1 << 16;

    // ---- plan types --------------------------------------------------------

    /// The geometry a model reports so the planner can price a merge against an epoch.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct DpShape {
        /// Total clauses in the model.
        pub n_clauses: usize,
        /// TA counter bytes per clause (`2 × n_features`).
        pub n_literals: usize,
        /// Include-bitset words per clause.
        pub words: usize,
        /// Clause rows touched by **one** training sample. For a vanilla classifier
        /// this is `2 × clauses_per_class` (only the true and one negative class are
        /// updated), *not* the whole bank — which is why the batch size ends up
        /// depending on the class count rather than the clause count.
        pub clauses_per_sample: usize,
    }

    impl DpShape {
        pub(crate) fn model_bytes(&self) -> usize {
            self.n_clauses.saturating_mul(self.n_literals)
        }
        pub(crate) fn sample_bytes(&self) -> usize {
            self.clauses_per_sample.saturating_mul(self.n_literals)
        }
        /// TA + include + per-clause weight/RNG overhead for one replica.
        pub(crate) fn replica_bytes(&self) -> usize {
            self.model_bytes()
                .saturating_add(self.n_clauses.saturating_mul(self.words).saturating_mul(8))
                .saturating_add(self.n_clauses.saturating_mul(12))
        }
    }

    /// How replica state is folded back into the master.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum MergeRule {
        /// Rounded mean across replicas. Loses updates smaller than the shard count,
        /// so it needs [`DP_MIN_SAMPLES_PER_SHARD`] samples per shard to be faithful.
        /// This is the rule full-epoch sharding has always used.
        Average,
        /// `master + Σ(replica − master)`, saturating. Computed as
        /// `Σ replicas − (k−1)·master`, which needs no snapshot buffer because the
        /// master is untouched during a mini-batch. Loses nothing at small batch
        /// sizes but multiplies drift by `k` and saturates at large ones.
        SumDelta,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct DpPlan {
        pub shards: usize,
        /// Samples between merges. `batch >= n` means one merge per epoch.
        pub batch: usize,
        pub merge_parallel: bool,
        pub rule: MergeRule,
        /// Plan came from the auto policy rather than an explicit opt-in — drives the
        /// one-time "training is now approximate" notice.
        pub from_auto: bool,
    }

    /// Environment overrides, parsed once.
    ///
    /// Passed to [`plan_epoch`] as an argument rather than read inside it, so the
    /// policy is unit-testable without mutating process environment.
    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct DpEnv {
        /// `TMU_RS_DATA_PARALLEL=0|1` — hard global off/on switch.
        pub force: Option<bool>,
        /// `TMU_RS_DP_SHARDS` — pin the shard count for reproducibility.
        pub shards: Option<usize>,
        /// `TMU_RS_DP_BATCH` — pin the samples-between-merges.
        pub batch: Option<usize>,
        /// `TMU_RS_DP_MERGE=avg|delta` — experimental merge-rule override.
        pub rule: Option<MergeRule>,
        /// `TMU_RS_DP_QUIET=1` — suppress the one-time auto-enable notice.
        pub quiet: bool,
    }

    pub(crate) fn dp_env() -> &'static DpEnv {
        static ENV: OnceLock<DpEnv> = OnceLock::new();
        ENV.get_or_init(|| {
            let flag = |k: &str| match std::env::var(k).ok().as_deref() {
                Some("0") | Some("false") => Some(false),
                Some("1") | Some("true") => Some(true),
                _ => None,
            };
            let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
            DpEnv {
                force: flag("TMU_RS_DATA_PARALLEL"),
                shards: num("TMU_RS_DP_SHARDS"),
                batch: num("TMU_RS_DP_BATCH"),
                rule: match std::env::var("TMU_RS_DP_MERGE").ok().as_deref() {
                    Some("avg") | Some("average") => Some(MergeRule::Average),
                    Some("delta") | Some("sumdelta") => Some(MergeRule::SumDelta),
                    _ => None,
                },
                quiet: flag("TMU_RS_DP_QUIET").unwrap_or(false),
            }
        })
    }

    // ---- the policy --------------------------------------------------------

    /// Decide whether — and how — to run this epoch data-parallel.
    ///
    /// Returns `None` when the caller should run its exact loop. In that case the
    /// model must be left **completely untouched** (no RNG draws, no allocation), so
    /// that the exact path's RNG stream is identical to a build where this module
    /// does not exist.
    ///
    /// `mode`: `None` = auto, `Some(true)` = force, `Some(false)` = exact.
    /// `pins`: `(shards, batch)` overrides from the model's builder.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_epoch(
        shape: &DpShape,
        n: usize,
        mode: Option<bool>,
        pins: (Option<usize>, Option<usize>),
        supported: bool,
        threads: usize,
        env: &DpEnv,
    ) -> Option<DpPlan> {
        if n == 0 || threads <= 1 {
            return None;
        }
        // Type III (and anything else a model rejects) never data-parallelises, and
        // never panics for it either — auto declines silently.
        if !supported {
            return None;
        }
        if env.force == Some(false) || mode == Some(false) {
            return None;
        }

        let forced = mode == Some(true) || env.force == Some(true);
        if forced {
            // An explicit opt-in is honoured at any model size — only `n` has to be
            // large enough to split. The old code additionally required
            // `use_parallel(n_clauses, words)`, which silently ignored
            // `.data_parallel(true)` for every small model: `data_parallel_learns_xor`
            // (80 clauses, 1 word) has never actually taken the data-parallel path.
            // There is no degenerate-plan risk in dropping that floor, because
            // `DP_MINIBATCH_MIN_MODEL_BYTES` already pins small models to a single
            // end-of-epoch merge and `shards >= 2` is checked below.
            if n < PARALLEL_MIN {
                return None;
            }
        } else {
            if n < DP_AUTO_MIN_SAMPLES || shape.sample_bytes() < DP_AUTO_MIN_SAMPLE_BYTES {
                return None;
            }
            let k = threads.min(DP_MAX_SHARDS).min(n);
            if shape.replica_bytes().saturating_mul(k) > DP_AUTO_MAX_REPLICA_BYTES {
                return None;
            }
        }

        let shards = env
            .shards
            .or(pins.0)
            .unwrap_or(threads)
            .min(DP_MAX_SHARDS)
            .min(n);
        if shards < 2 {
            return None;
        }

        let batch = if shape.model_bytes() < DP_MINIBATCH_MIN_MODEL_BYTES {
            n
        } else {
            // Sync traffic per merge point: read k replicas, write master, re-read
            // master TA for the fused rebuild, write k replicas on refresh.
            let sync = shape.model_bytes().saturating_mul(2 * shards + 2);
            (DP_SYNC_OVERHEAD_DIVISOR.saturating_mul(sync) / shape.sample_bytes().max(1))
                .max(DP_MIN_SAMPLES_PER_SHARD * shards)
                .min(DP_BATCH_MAX)
                .min(n)
        };
        let batch = env.batch.or(pins.1).unwrap_or(batch).clamp(shards, n);

        Some(DpPlan {
            shards,
            batch,
            merge_parallel: shape.model_bytes() >= MERGE_PARALLEL_MIN_BYTES,
            // `Average` for both cadences: it is the rule full-epoch sharding has
            // always used, and the `DP_MIN_SAMPLES_PER_SHARD` floor above keeps the
            // mini-batch cadence inside its faithful regime. `SumDelta` is implemented
            // and unit-tested but stays opt-in via `TMU_RS_DP_MERGE=delta` until a
            // batch-size sweep shows it strictly better.
            rule: env.rule.unwrap_or(MergeRule::Average),
            from_auto: !forced,
        })
    }

    // ---- merge kernels -----------------------------------------------------

    /// Fold one chunk of replica TA counters into the master's chunk.
    ///
    /// Replicas are the **outer** loop so each one's memory is read as a single
    /// sequential run — the fix for the strided
    /// `replicas.iter().map(|r| r.ta[i])` gather this replaces, which walked `k`
    /// arrays with a whole-array stride per output byte.
    ///
    /// `acc` must be at least `master.len()` long; its contents are overwritten.
    pub(crate) fn merge_ta_chunk<'a, I>(
        master: &mut [u8],
        rows: I,
        k: usize,
        acc: &mut [u16],
        rule: MergeRule,
        max_state: u8,
    ) where
        I: Iterator<Item = &'a [u8]>,
    {
        let len = master.len();
        let acc = &mut acc[..len];
        acc.fill(0);
        for row in rows {
            debug_assert_eq!(row.len(), len);
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += v as u16;
            }
        }
        match rule {
            MergeRule::Average => {
                let k32 = k as u32;
                for (t, &a) in master.iter_mut().zip(acc.iter()) {
                    *t = ((a as u32 + k32 / 2) / k32) as u8;
                }
            }
            MergeRule::SumDelta => {
                let km1 = k as i32 - 1;
                let hi = max_state as i32;
                for (t, &a) in master.iter_mut().zip(acc.iter()) {
                    // master still holds the pre-mini-batch value here.
                    *t = (a as i32 - km1 * (*t as i32)).clamp(0, hi) as u8;
                }
            }
        }
    }

    /// [`merge_ta_chunk`] for `i32` per-clause weights.
    ///
    /// `Average` rounds half **away from zero** so signed weight banks (coalesced
    /// models) behave symmetrically, matching the pre-existing merge. `lo`/`hi` are
    /// the model's weight bounds: `[1, threshold]` for vanilla-style banks,
    /// `[0, threshold]` for the regressor, `i32::MIN..=i32::MAX` for signed banks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn merge_i32_chunk<'a, I>(
        master: &mut [i32],
        rows: I,
        k: usize,
        acc: &mut [i64],
        rule: MergeRule,
        lo: i32,
        hi: i32,
    ) where
        I: Iterator<Item = &'a [i32]>,
    {
        let len = master.len();
        let acc = &mut acc[..len];
        acc.fill(0);
        for row in rows {
            debug_assert_eq!(row.len(), len);
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += v as i64;
            }
        }
        let ki = k as i64;
        match rule {
            MergeRule::Average => {
                for (w, &a) in master.iter_mut().zip(acc.iter()) {
                    let avg = (a + (ki / 2) * a.signum()) / ki;
                    *w = (avg as i32).clamp(lo, hi);
                }
            }
            MergeRule::SumDelta => {
                let km1 = ki - 1;
                for (w, &a) in master.iter_mut().zip(acc.iter()) {
                    let v = a - km1 * (*w as i64);
                    *w = v.clamp(lo as i64, hi as i64) as i32;
                }
            }
        }
    }

    /// Clauses per merge chunk, sized so one chunk of TA stays L2-resident.
    pub(crate) fn merge_group(n_literals: usize) -> usize {
        (MERGE_CHUNK_BYTES / n_literals.max(1)).max(1)
    }

    // ---- the driver --------------------------------------------------------

    /// A model that can be replicated across shards and merged back.
    ///
    /// The per-sample training step is *not* on this trait: it is passed to
    /// [`fit_epoch_dp`] as a closure, which is how the four different `fit_epoch`
    /// signatures in this crate (`EncodedBatch + &[usize]`, `+ &[f64]`,
    /// `EncodedBatch` alone, and `&[&[u8]] + &[usize]`) are all handled without the
    /// driver ever naming an input type.
    pub(crate) trait DataParallel: Clone + Send + Sync + Sized {
        fn dp_shape(&self) -> DpShape;
        /// `false` when the current configuration forbids replica training (Type III).
        fn dp_supported(&self) -> bool;
        /// `None` = auto, `Some(true)` = force, `Some(false)` = exact.
        fn dp_mode(&self) -> Option<bool>;
        /// `(shards, batch)` pins from the builder.
        fn dp_pins(&self) -> (Option<usize>, Option<usize>);
        fn dp_rng(&mut self) -> &mut Rng;
        fn dp_pool_mut(&mut self) -> &mut ReplicaPool<Self>;
        /// One-time notice that auto silently made training approximate.
        fn dp_notice(&mut self, plan: &DpPlan);

        /// A replica of `self` that will never nest Rayon and never fork replicas of
        /// its own. Implementations should also drop state a replica cannot need
        /// (e.g. the Type III `ind`/`cat` arrays, since DP implies `!type_iii`).
        fn dp_fork(&self) -> Self;
        /// Make `self` identical to `master` except for RNG streams and the pool,
        /// reusing allocations (see [`sync_vec`]).
        fn dp_sync_from(&mut self, master: &Self);
        /// Re-derive every RNG stream from `seed`. Called once per epoch per replica,
        /// not per mini-batch, so streams keep advancing across merges instead of
        /// re-correlating.
        fn dp_reseed(&mut self, seed: u64);
        /// Fold replica state back into `self`, restoring every derived invariant
        /// (include bitsets, weight bounds).
        fn dp_merge(&mut self, replicas: &[Self], plan: &DpPlan);
    }

    /// Run one epoch data-parallel. Returns `false` — having left `model` untouched —
    /// when the planner declines, so the caller runs its exact loop.
    ///
    /// `train(replica, i)` trains `replica` on sample index `i` of the caller's data.
    pub(crate) fn fit_epoch_dp<M, F>(model: &mut M, n: usize, train: F) -> bool
    where
        M: DataParallel,
        F: Fn(&mut M, usize) + Sync,
    {
        let shape = model.dp_shape();
        let plan = match plan_epoch(
            &shape,
            n,
            model.dp_mode(),
            model.dp_pins(),
            model.dp_supported(),
            rayon::current_num_threads(),
            dp_env(),
        ) {
            Some(p) => p,
            None => return false,
        };
        if plan.from_auto {
            model.dp_notice(&plan);
        }

        // Deterministic shuffle, drawn at the same RNG stream position the exact
        // path would use.
        let mut order: Vec<usize> = (0..n).collect();
        {
            let rng = model.dp_rng();
            for i in (1..n).rev() {
                let k = rng.below(i + 1);
                order.swap(i, k);
            }
        }
        let seeds: Vec<u64> = {
            let rng = model.dp_rng();
            (0..plan.shards).map(|_| rng.next_u64()).collect()
        };

        // Take the pool out so the model is no longer mutably borrowed below.
        let mut reps = std::mem::take(&mut model.dp_pool_mut().replicas);
        reps.truncate(plan.shards);
        while reps.len() < plan.shards {
            reps.push(model.dp_fork());
        }
        for (r, &sd) in reps.iter_mut().zip(&seeds) {
            r.dp_reseed(sd);
        }

        // Even batch sizes: no runt final mini-batch, so no degenerate final merge.
        let n_batches = (n / plan.batch).max(1);
        let bsz = n.div_ceil(n_batches);

        for mini in order.chunks(bsz) {
            let per = mini.len().div_ceil(plan.shards);
            {
                let master: &M = model;
                reps.par_iter_mut().enumerate().for_each(|(s, r)| {
                    // Refresh is fused into the training region: it is ordered after
                    // the previous merge by program order, and it warms this core's
                    // cache with the state it is about to update.
                    r.dp_sync_from(master);
                    let lo = (s * per).min(mini.len());
                    let hi = (lo + per).min(mini.len());
                    for &i in &mini[lo..hi] {
                        train(r, i);
                    }
                });
            }
            model.dp_merge(&reps, &plan);
        }

        model.dp_pool_mut().replicas = reps;
        true
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Shapes of the real in-tree models whose determinism tests must stay green.
        /// If auto ever starts firing for one of these, the named test breaks.
        fn shape(n_clauses: usize, n_features: usize, cps: usize) -> DpShape {
            let n_literals = 2 * n_features;
            DpShape {
                n_clauses,
                n_literals,
                words: n_literals.div_ceil(64),
                clauses_per_sample: 2 * cps,
            }
        }

        const AUTO: Option<bool> = None;
        const NOPINS: (Option<usize>, Option<usize>) = (None, None);

        #[test]
        fn auto_declines_for_tiny_per_sample_work() {
            let env = DpEnv::default();
            // Models whose per-sample work is too small to beat thread dispatch.
            // (name, shape, n)
            let cases = [
                // vanilla_classifier::same_seed_same_result — 384 B per sample.
                ("vanilla same_seed", shape(16, 12, 8), 500),
                // coalesced_classifier::same_seed_same_result
                ("coalesced same_seed", shape(16, 12, 16), 500),
                // coalesced_classifier::large_shared_bank_learns — 160 clauses but
                // only 8 features, so 2.5 KiB per sample.
                ("large shared bank", shape(160, 8, 160), 3000),
            ];
            for (name, sh, n) in cases {
                assert!(
                    plan_epoch(&sh, n, AUTO, NOPINS, true, 4, &env).is_none(),
                    "auto must not fire for {name} ({} B/sample)",
                    sh.sample_bytes()
                );
            }
        }

        #[test]
        fn auto_fires_for_large_models_but_respects_the_memory_budget() {
            let env = DpEnv::default();
            // examples/bench_training.rs: 2 classes x 10_000 clauses/class, 1000
            // features = 40 MiB of TA. Replicas do not fit L3, but data-parallel is
            // not slower there either, so auto should still fire.
            let sh = shape(20_000, 1_000, 10_000);
            assert!(plan_epoch(&sh, 2000, AUTO, NOPINS, true, 4, &env).is_some());

            // But a model whose replicas would blow the memory budget is left alone
            // unless explicitly asked for.
            let huge = shape(4_000_000, 1_000, 2_000_000);
            assert!(huge.replica_bytes() * 4 > DP_AUTO_MAX_REPLICA_BYTES);
            assert!(plan_epoch(&huge, 2000, AUTO, NOPINS, true, 4, &env).is_none());
            assert!(plan_epoch(&huge, 2000, Some(true), NOPINS, true, 4, &env).is_some());
        }

        #[test]
        fn auto_fires_for_the_measured_winning_shapes() {
            let env = DpEnv::default();
            // parallel_scaling Sweep A shapes: these are the ones auto exists to
            // catch. Note n = 1000 here, the sweep's epoch size — DP_AUTO_MIN_SAMPLES
            // must stay below it or the whole sweep silently measures the exact path.
            for (name, cpc) in [("16 cpc", 16usize), ("256 cpc", 256), ("1024 cpc", 1024)] {
                let sh = shape(2 * cpc, 512, cpc);
                let plan = plan_epoch(&sh, 2000, AUTO, NOPINS, true, 4, &env)
                    .unwrap_or_else(|| panic!("auto should fire for {name}"));
                assert_eq!(plan.shards, 4);
                assert!(plan.from_auto);
                assert!(plan.batch >= DP_MIN_SAMPLES_PER_SHARD * plan.shards);
            }
        }

        #[test]
        fn declines_single_thread_and_unsupported_and_off() {
            let env = DpEnv::default();
            let sh = shape(20_000, 1_000, 10_000);
            assert!(plan_epoch(&sh, 2000, AUTO, NOPINS, true, 1, &env).is_none());
            // Type III: declines rather than panicking, even when forced.
            assert!(plan_epoch(&sh, 2000, Some(true), NOPINS, false, 4, &env).is_none());
            assert!(plan_epoch(&sh, 2000, Some(false), NOPINS, true, 4, &env).is_none());
            assert!(plan_epoch(&sh, 0, Some(true), NOPINS, true, 4, &env).is_none());
        }

        #[test]
        fn env_off_overrides_explicit_opt_in() {
            let env = DpEnv {
                force: Some(false),
                ..Default::default()
            };
            let sh = shape(20_000, 1_000, 10_000);
            assert!(plan_epoch(&sh, 2000, Some(true), NOPINS, true, 4, &env).is_none());
        }

        #[test]
        fn pins_and_env_override_the_plan() {
            let sh = shape(2048, 512, 1024);
            let plan = plan_epoch(
                &sh,
                2000,
                AUTO,
                (Some(3), Some(256)),
                true,
                4,
                &DpEnv::default(),
            )
            .unwrap();
            assert_eq!((plan.shards, plan.batch), (3, 256));
            let env = DpEnv {
                shards: Some(2),
                batch: Some(512),
                rule: Some(MergeRule::SumDelta),
                ..Default::default()
            };
            let plan = plan_epoch(&sh, 2000, AUTO, (Some(3), Some(256)), true, 4, &env).unwrap();
            assert_eq!(
                (plan.shards, plan.batch, plan.rule),
                (2, 512, MergeRule::SumDelta)
            );
        }

        #[test]
        fn tiny_models_merge_once_per_epoch() {
            let env = DpEnv::default();
            // vanilla_classifier::data_parallel_learns_xor: 80 clauses x 24 literals
            // = 1920 B, well under DP_MINIBATCH_MIN_MODEL_BYTES. Forced DP must keep
            // today's full-epoch Average semantics.
            let sh = shape(80, 12, 40);
            let plan = plan_epoch(&sh, 6000, Some(true), NOPINS, true, 4, &env).unwrap();
            assert_eq!(plan.batch, 6000);
            assert_eq!(plan.rule, MergeRule::Average);
            assert!(!plan.from_auto);
        }

        #[test]
        fn batch_grows_with_classes_not_clause_count() {
            let env = DpEnv::default();
            // Same total clauses and width; 2 classes vs 10. A merge touches the whole
            // bank either way, but a sample touches 1/n_classes of it, so the 10-class
            // model needs proportionally more samples between merges.
            let two = shape(2048, 512, 1024);
            let ten = shape(2048, 512, 205);
            let b2 = plan_epoch(&two, 8000, AUTO, NOPINS, true, 4, &env)
                .unwrap()
                .batch;
            let b10 = plan_epoch(&ten, 8000, AUTO, NOPINS, true, 4, &env)
                .unwrap()
                .batch;
            assert!(
                b10 > b2,
                "10-class batch {b10} should exceed 2-class batch {b2}"
            );
        }

        #[test]
        fn average_loses_a_single_shard_delta_and_sum_delta_keeps_it() {
            // One replica moved a counter by +3, three left it alone.
            let base = 100u8;
            let rows: Vec<Vec<u8>> = vec![vec![base + 3], vec![base], vec![base], vec![base]];
            let mut acc = [0u16; 1];

            let mut avg = [base];
            merge_ta_chunk(
                &mut avg,
                rows.iter().map(|r| r.as_slice()),
                4,
                &mut acc,
                MergeRule::Average,
                255,
            );
            assert_eq!(avg[0], base + 1, "Average rounds 3/4 of a step down to 1");

            let mut delta = [base];
            merge_ta_chunk(
                &mut delta,
                rows.iter().map(|r| r.as_slice()),
                4,
                &mut acc,
                MergeRule::SumDelta,
                255,
            );
            assert_eq!(delta[0], base + 3, "SumDelta preserves the whole update");
        }

        #[test]
        fn sum_delta_saturates_at_both_ends() {
            let mut acc = [0u16; 1];
            let rows: Vec<Vec<u8>> = vec![vec![250], vec![250], vec![250], vec![250]];
            let mut hi = [200u8];
            merge_ta_chunk(
                &mut hi,
                rows.iter().map(|r| r.as_slice()),
                4,
                &mut acc,
                MergeRule::SumDelta,
                255,
            );
            assert_eq!(hi[0], 255);

            let rows: Vec<Vec<u8>> = vec![vec![0], vec![0], vec![0], vec![0]];
            let mut lo = [200u8];
            merge_ta_chunk(
                &mut lo,
                rows.iter().map(|r| r.as_slice()),
                4,
                &mut acc,
                MergeRule::SumDelta,
                255,
            );
            assert_eq!(lo[0], 0);
        }

        #[test]
        fn weight_merge_respects_bounds_and_rounds_symmetrically() {
            let mut acc = [0i64; 1];
            // Unbounded signed bank: -5, -5, -6, -6 -> -5.5 -> -6 (away from zero).
            let rows: Vec<Vec<i32>> = vec![vec![-5], vec![-5], vec![-6], vec![-6]];
            let mut signed = [0i32];
            merge_i32_chunk(
                &mut signed,
                rows.iter().map(|r| r.as_slice()),
                4,
                &mut acc,
                MergeRule::Average,
                i32::MIN,
                i32::MAX,
            );
            assert_eq!(signed[0], -6);

            // Vanilla bank clamps to [1, threshold].
            let rows: Vec<Vec<i32>> = vec![vec![0], vec![0], vec![0], vec![0]];
            let mut clamped = [4i32];
            merge_i32_chunk(
                &mut clamped,
                rows.iter().map(|r| r.as_slice()),
                4,
                &mut acc,
                MergeRule::Average,
                1,
                50,
            );
            assert_eq!(clamped[0], 1);
        }

        #[test]
        fn merge_handles_a_short_final_chunk() {
            // 3 clauses of 2 literals with a group of 2 leaves a 1-clause tail.
            let master_src = [10u8, 10, 10, 10, 10, 10];
            let rows: Vec<Vec<u8>> = vec![vec![12u8; 6], vec![14u8; 6]];
            let mut acc = vec![0u16; 4];
            let mut master = master_src;
            for (g, chunk) in master.chunks_mut(4).enumerate() {
                let off = g * 4;
                // `len` must be hoisted: capturing `chunk` in the row closure would
                // borrow it immutably while it is also passed as `&mut`.
                let len = chunk.len();
                merge_ta_chunk(
                    chunk,
                    rows.iter().map(|r| &r[off..off + len]),
                    2,
                    &mut acc,
                    MergeRule::Average,
                    255,
                );
            }
            assert_eq!(master, [13u8; 6]);
        }
    }
}
