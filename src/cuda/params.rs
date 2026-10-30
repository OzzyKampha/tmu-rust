//! `#[repr(C)]` mirrors of the kernel parameter structs in `kernels.cu`.
//!
//! Field order and types must match the C structs exactly (device pointers are
//! `u64`). The emulator verifies the struct sizes on every launch.

#![allow(dead_code)]

macro_rules! params {
    ($($(#[$m:meta])* $name:ident { $($f:ident : $t:ty),* $(,)? })*) => {
        $(
            $(#[$m])*
            #[repr(C)]
            #[derive(Clone, Copy, Default)]
            pub(crate) struct $name { $(pub $f: $t),* }
        )*
    };
}

params! {
    GroupedFireParams {
        include: u64, valid: u64, lit: u64, act: u64, weights: u64, sums: u64, fire: u64,
        group_ids: u64, group_rngs: u64, drop_s0: u64, inv: u64, keep: u64,
        drop_p: f64,
        n_groups: u32, cpg: u32, words: u32, n_literals: u32,
        dig_inv: u32, dig_keep: u32, n_digits: u32, n_fire_blocks: u32,
        ae_nf: i32, _pad: u32,
    }
    GroupedUpdateParams {
        ta: u64, include: u64, weights: u64, rngs: u64, ind: u64, cat: u64, valid: u64,
        lit: u64, act: u64, sums: u64, fire: u64, group_ids: u64, drop_s0: u64, inv: u64,
        keep: u64, class_weights: u64,
        drop_p: f64, d: f64, max_inc: u64,
        n_groups: u32, cpg: u32, words: u32, n_literals: u32,
        threshold: i32, ae_nf: i32,
        boost: u32, half: u32, max_state: u32, type_iii: u32,
    }
    ConvFireParams {
        include: u64, valid: u64, patches: u64, act: u64, weights: u64, sums: u64,
        group_ids: u64, group_rngs: u64, drop_s0: u64, inv: u64, keep: u64,
        drop_p: f64,
        n_groups: u32, cpg: u32, words: u32, n_literals: u32,
        n_patches: u32, dig_inv: u32, dig_keep: u32, n_digits: u32,
        n_fire_blocks: u32, _pad: u32,
    }
    ConvUpdateParams {
        ta: u64, include: u64, weights: u64, rngs: u64, ind: u64, cat: u64, valid: u64,
        patches: u64, act: u64, sums: u64, group_ids: u64, drop_s0: u64, inv: u64, keep: u64,
        drop_p: f64, d: f64, max_inc: u64,
        n_groups: u32, cpg: u32, words: u32, n_literals: u32, n_patches: u32,
        threshold: i32,
        boost: u32, half: u32, max_state: u32, type_iii: u32,
    }
    SharedFireParams {
        include: u64, valid: u64, lit: u64, act: u64, w_out: u64, sum: u64, fire: u64,
        out_rng: u64, drop_s0: u64, inv: u64, keep: u64,
        drop_p: f64,
        n_clauses: u32, words: u32, n_fire_blocks: u32,
        dig_inv: u32, dig_keep: u32, n_digits: u32,
        ex_a: i32, ex_b: i32,
    }
    CoPlanParams {
        master_rng: u64, class_rngs: u64, weights: u64, fire: u64, active: u64, plan: u64,
        inv: u64, keep: u64,
        drop_p: f64,
        n_clauses: u32, n_classes: u32, words: u32, y: u32,
        threshold: i32,
        focused: u32, dig_inv: u32, dig_keep: u32, n_digits: u32, _pad: u32,
    }
    CoUpdateParams {
        ta: u64, include: u64, weights: u64, rngs: u64, ind: u64, cat: u64, valid: u64,
        lit: u64, act: u64, fire: u64, active: u64, plan: u64, inv: u64, keep: u64,
        d: f64, max_inc: u64,
        n_clauses: u32, words: u32, n_literals: u32,
        threshold: i32,
        boost: u32, half: u32, max_state: u32, type_iii: u32,
    }
    CaeGateParams {
        weights: u64, gate: u64,
        gate_rng: u64,
        n_clauses: u32, n_features: u32,
        threshold: i32, _pad: u32,
    }
    CaeUpdateParams {
        ta: u64, include: u64, weights: u64, rngs: u64, ind: u64, cat: u64, valid: u64,
        lit: u64, act: u64, fire: u64, gate: u64, sum: u64, drop_s0: u64, inv: u64, keep: u64,
        drop_p: f64, d: f64, max_inc: u64,
        n_clauses: u32, words: u32, n_literals: u32, o: u32,
        threshold: i32, nf: i32,
        boost: u32, half: u32, max_state: u32, type_iii: u32,
    }
    RgFireParams {
        include: u64, valid: u64, lit: u64, weights: u64, pred_sum: u64,
        n_clauses: u32, words: u32,
    }
    RgPlanParams {
        master_rng: u64, literal_rng: u64, pred_sum: u64, plan_p: u64, plan: u64, act: u64,
        drop_s0: u64, inv: u64, keep: u64,
        y: f64, drop_p: f64,
        n_clauses: u32, words: u32,
        threshold: i32,
        dig_inv: u32, dig_keep: u32, dig_lit: u32, n_digits: u32, literal_drop: u32,
    }
    RgUpdateParams {
        ta: u64, include: u64, weights: u64, rngs: u64, ind: u64, cat: u64, valid: u64,
        lit: u64, act: u64, plan: u64, plan_p: u64, drop_s0: u64, inv: u64, keep: u64,
        drop_p: f64, d: f64, max_inc: u64,
        n_clauses: u32, words: u32, n_literals: u32,
        threshold: i32,
        boost: u32, half: u32, max_state: u32, type_iii: u32,
    }
    SparseBank {
        inc_idx: u64, exc_idx: u64, una_idx: u64, inc_st: u64, exc_st: u64,
        n_inc: u64, n_exc: u64, n_una: u64,
    }
    SpFireParams {
        bank: SparseBank,
        lit: u64, act: u64, weights: u64, sums: u64, fire: u64, group_ids: u64,
        class_rngs: u64, drop_s0: u64,
        drop_p: f64,
        n_groups: u32, cpg: u32, n_literals: u32, n_fire_blocks: u32,
    }
    SpUpdateParams {
        bank: SparseBank,
        weights: u64, rngs: u64, lit: u64, act: u64, sums: u64, fire: u64, group_ids: u64,
        drop_s0: u64, class_weights: u64,
        drop_p: f64, inv_p: f64, keep_p: f64, max_inc: u64,
        n_groups: u32, cpg: u32, n_literals: u32,
        threshold: i32,
        boost: u32, half: u32, max_state: u32, _pad: u32,
    }
    ScoresParams {
        include: u64, valid: u64, lits: u64, weights: u64, out: u64,
        n_clauses: u32, words: u32, n_outputs: u32, cpg: u32, n_patches: u32, mode: u32,
        global_acc: u32, _pad: u32,
    }
    SparseScoresParams {
        bank: SparseBank,
        lits: u64, weights: u64, out: u64,
        n_clauses: u32, words: u32, n_classes: u32, cpg: u32, n_literals: u32, _pad: u32,
    }
}

pub(crate) const SCORE_GROUPED: u32 = 0;
pub(crate) const SCORE_SHARED: u32 = 1;
