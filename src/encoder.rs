//! Encoder: transforms raw data into bit-packed literal vectors for [`TsetlinMachine`].
//!
//! Three built-in modes:
//! - [`Encoder::for_binary`] — already-binary `u8` features, just packs
//! - [`Encoder::fit_numeric`] — `f64` features → quantile booleanization → pack
//! - [`Encoder::fit_categorical`] — `"col::val"` token strings → vocabulary → pack
//!
//! Binary and categorical encoders can grow their feature space after fitting
//! ([`Encoder::grow_binary`], [`Encoder::extend_categorical`]) — pair with
//! `TsetlinMachine::grow_features` to expand a trained model onto new data
//! without discarding learned automata. Numeric encoders are fixed-size once fit.
//!
//! The output types [`EncodedSample`] and [`EncodedBatch`] can only be constructed
//! by an encoder; raw slices are not accepted by [`TsetlinMachine`] methods.
//!
//! [`TsetlinMachine`]: crate::TsetlinMachine

use std::collections::HashMap;

use crate::booleanizer::Booleanizer;
use crate::clause_bank::dense::{pack, words_for};

// ── output types ─────────────────────────────────────────────────────────────

/// Bit-packed literal vector for a single sample.
///
/// Produced exclusively by [`Encoder`] — cannot be constructed directly.
pub struct EncodedSample(pub(crate) Vec<u64>);

/// Bit-packed literal vectors for a batch of samples.
///
/// Produced exclusively by [`Encoder`] — cannot be constructed directly.
pub struct EncodedBatch {
    pub(crate) data: Vec<u64>,
    pub(crate) n: usize,
    pub(crate) words: usize,
}

impl EncodedSample {
    /// Build an [`EncodedSample`] from a 0/1 byte slice.
    ///
    /// `bits` must have exactly `n_features` elements (one per binary feature).
    /// Use this to integrate a custom encoder without modifying this crate.
    pub fn from_bits(bits: &[u8], n_features: usize) -> Self {
        assert_eq!(bits.len(), n_features, "bits.len() must equal n_features");
        let mut out = vec![0u64; words_for(2 * n_features)];
        pack(bits, n_features, &mut out);
        Self(out)
    }
}

impl EncodedBatch {
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Build an [`EncodedBatch`] from a flat, row-major word array.
    ///
    /// `data` must have exactly `n * words` elements.
    /// Use this to integrate a custom encoder without modifying this crate.
    pub fn from_words(data: Vec<u64>, n: usize, words: usize) -> Self {
        assert_eq!(data.len(), n * words, "data.len() must equal n * words");
        Self { data, n, words }
    }

    /// Build an [`EncodedBatch`] from row-major 0/1 bit slices (one slice per sample).
    ///
    /// Every row must have exactly `n_features` elements; each is packed into the
    /// TM's literal format internally. This is the simplest way to feed a custom
    /// encoder's output to the TM — you only produce bits, never raw words.
    pub fn from_bit_rows(rows: &[&[u8]], n_features: usize) -> Self {
        let words = words_for(2 * n_features);
        let n = rows.len();
        let mut data = vec![0u64; n * words];
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                n_features,
                "each row must have n_features elements"
            );
            pack(row, n_features, &mut data[i * words..(i + 1) * words]);
        }
        Self { data, n, words }
    }
}

// ── encoder kind ─────────────────────────────────────────────────────────────

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum EncoderKind {
    Binary,
    Numeric {
        booleanizer: Booleanizer,
    },
    Categorical {
        /// token string → bit index
        vocab: HashMap<String, usize>,
        /// bit index → token string (for interpretability)
        index: Vec<String>,
        /// set of column prefixes seen during fit (for UNK fallback)
        known_columns: HashMap<String, usize>,
    },
}

// ── public Encoder ────────────────────────────────────────────────────────────

/// Encodes raw data into bit-packed literal vectors accepted by [`TsetlinMachine`].
///
/// Train the encoder once on your training set, then use it to encode both
/// train and test data before passing to the TM.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Encoder {
    kind: EncoderKind,
    n_features: usize,
    words: usize,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for Encoder {
    const TAG: u8 = crate::serial::TAG_ENCODER;
}

impl Encoder {
    // ── constructors ─────────────────────────────────────────────────────────

    /// Encoder for already-binary (`0`/`1`) feature vectors.
    ///
    /// No fitting required — just packs into the bit-interleaved literal format.
    pub fn for_binary(n_features: usize) -> Self {
        assert!(n_features >= 1, "n_features must be >= 1");
        Self {
            kind: EncoderKind::Binary,
            n_features,
            words: words_for(2 * n_features),
        }
    }

    /// Encoder for continuous (`f64`) feature vectors.
    ///
    /// Fits `n_thresholds` quantile cut-points per feature on `xs`, then
    /// booleanizes: output feature count = `xs[0].len() * n_thresholds`.
    pub fn fit_numeric(xs: &[&[f64]], n_thresholds: usize) -> Self {
        assert!(!xs.is_empty(), "training set must not be empty");
        let n_raw = xs[0].len();
        let booleanizer = Booleanizer::fit(xs, n_raw, n_thresholds);
        let n_features = booleanizer.n_output_features();
        Self {
            kind: EncoderKind::Numeric { booleanizer },
            n_features,
            words: words_for(2 * n_features),
        }
    }

    /// Encoder for categorical data represented as `"col::val"` token strings.
    ///
    /// Builds a vocabulary from `samples` (a training set where each sample is
    /// a slice of token strings).  Three special tokens are added automatically:
    /// - `"col::<UNK>"` for each column prefix seen in training (unseen value)
    /// - `"<OOV>"` global sentinel (unseen column at inference time)
    ///
    /// Token ordering: regular tokens (sorted) → per-column UNK sentinels (sorted) → `"<OOV>"`.
    pub fn fit_categorical(samples: &[&[&str]]) -> Self {
        assert!(!samples.is_empty(), "training set must not be empty");

        // Collect unique regular tokens and unique column prefixes.
        let mut token_set: std::collections::BTreeSet<String> = Default::default();
        let mut col_set: std::collections::BTreeSet<String> = Default::default();

        for sample in samples {
            for &token in *sample {
                token_set.insert(token.to_string());
                if let Some(col) = token.split("::").next() {
                    col_set.insert(col.to_string());
                }
            }
        }

        // Build ordered index: regular tokens first, then "<col>::<UNK>", then "<OOV>".
        let mut index: Vec<String> = token_set.into_iter().collect();
        let n_regular = index.len();

        let mut known_columns: HashMap<String, usize> = HashMap::new();
        for col in &col_set {
            let unk = format!("{col}::<UNK>");
            known_columns.insert(col.clone(), index.len());
            index.push(unk);
        }
        index.push("<OOV>".to_string());

        let n_features = index.len();
        let vocab: HashMap<String, usize> = index
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let _ = n_regular; // used implicitly via index construction order
        Self {
            kind: EncoderKind::Categorical {
                vocab,
                index,
                known_columns,
            },
            n_features,
            words: words_for(2 * n_features),
        }
    }

    // ── growing ──────────────────────────────────────────────────────────────

    /// Extend a categorical encoder with new samples: genuinely unseen tokens are
    /// appended as new features AFTER all existing features, so every existing
    /// index — including per-column `"col::<UNK>"` sentinels and `"<OOV>"` — stays
    /// stable. New column prefixes also get a `"col::<UNK>"` sentinel. Returns the
    /// number of features added (`0` = the vocabulary already covered the samples).
    ///
    /// After a non-zero extension, grow the paired machine to match and re-encode:
    /// `if enc.extend_categorical(&new) > 0 { tm.grow_features(enc.n_features()); }`
    ///
    /// The global sorted-index invariant of [`Encoder::fit_categorical`] is
    /// intentionally dropped after extension; within each extension batch, new
    /// tokens are appended in sorted order for determinism.
    ///
    /// Note: a token that previously fell into a `<UNK>`/`<OOV>` slot and now has
    /// its own feature encodes differently afterwards (UNK bit off, own bit on),
    /// so predictions on *those* samples may legitimately change. Samples whose
    /// tokens were all previously known encode bit-identically.
    ///
    /// # Panics
    /// Panics if the encoder is not categorical.
    pub fn extend_categorical(&mut self, samples: &[&[&str]]) -> usize {
        match &mut self.kind {
            EncoderKind::Categorical {
                vocab,
                index,
                known_columns,
            } => {
                let before = index.len();

                let mut new_tokens: std::collections::BTreeSet<String> = Default::default();
                let mut new_cols: std::collections::BTreeSet<String> = Default::default();
                for sample in samples {
                    for &token in *sample {
                        if !vocab.contains_key(token) {
                            new_tokens.insert(token.to_string());
                        }
                        if let Some(col) = token.split("::").next() {
                            if !known_columns.contains_key(col) {
                                new_cols.insert(col.to_string());
                            }
                        }
                    }
                }

                for token in new_tokens {
                    vocab.insert(token.clone(), index.len());
                    index.push(token);
                }
                for col in new_cols {
                    let unk = format!("{col}::<UNK>");
                    // The UNK string may collide with a literal token appended
                    // above — reuse its index rather than pushing a duplicate.
                    let idx = *vocab.entry(unk.clone()).or_insert(index.len());
                    if idx == index.len() {
                        index.push(unk);
                    }
                    known_columns.insert(col, idx);
                }

                self.n_features = index.len();
                self.words = words_for(2 * self.n_features);
                self.n_features - before
            }
            _ => panic!("extend_categorical requires a categorical encoder"),
        }
    }

    /// Grow a binary encoder to `new_n_features`. Callers must append the new
    /// features at the END of each row so existing feature indices stay stable;
    /// pair with `TsetlinMachine::grow_features(new_n_features)`.
    ///
    /// # Panics
    /// Panics if the encoder is not binary or if `new_n_features` would shrink it.
    pub fn grow_binary(&mut self, new_n_features: usize) {
        assert!(
            matches!(self.kind, EncoderKind::Binary),
            "grow_binary requires a binary encoder"
        );
        assert!(
            new_n_features >= self.n_features,
            "grow_binary cannot shrink: {} -> {new_n_features}",
            self.n_features
        );
        self.n_features = new_n_features;
        self.words = words_for(2 * new_n_features);
    }

    // ── metadata ─────────────────────────────────────────────────────────────

    /// Number of binary features produced by this encoder (= `n_features` for the TM).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// For numeric encoders: return `(feature_index, threshold)` for bit `bit`.
    ///
    /// Useful for printing interpretable clause rules.
    pub fn bit_origin(&self, bit: usize) -> (usize, f64) {
        match &self.kind {
            EncoderKind::Numeric { booleanizer } => booleanizer.bit_origin(bit),
            _ => panic!("bit_origin is only available on numeric encoders"),
        }
    }

    /// For categorical encoders: return the `"col::val"` token string for bit `bit`.
    ///
    /// Useful for printing interpretable clause rules.
    pub fn vocab_token(&self, bit: usize) -> &str {
        match &self.kind {
            EncoderKind::Categorical { index, .. } => index
                .get(bit)
                .map(|s| s.as_str())
                .expect("bit index out of vocab range"),
            _ => panic!("vocab_token is only available on categorical encoders"),
        }
    }

    // ── single-sample encode ──────────────────────────────────────────────────

    /// Encode a binary (`0`/`1`) feature vector.
    pub fn encode_one(&self, x: &[u8]) -> EncodedSample {
        assert!(
            matches!(self.kind, EncoderKind::Binary),
            "encode_one requires a binary encoder; use encode_one_numeric or encode_one_categorical"
        );
        let mut out = vec![0u64; self.words];
        pack(x, self.n_features, &mut out);
        EncodedSample(out)
    }

    /// Encode a continuous (`f64`) feature vector.
    pub fn encode_one_numeric(&self, x: &[f64]) -> EncodedSample {
        match &self.kind {
            EncoderKind::Numeric { booleanizer } => {
                let mut bin = vec![0u8; self.n_features];
                booleanizer.transform_row(x, &mut bin);
                let mut out = vec![0u64; self.words];
                pack(&bin, self.n_features, &mut out);
                EncodedSample(out)
            }
            _ => panic!("encode_one_numeric requires a numeric encoder"),
        }
    }

    /// Encode a sample represented as a slice of `"col::val"` token strings.
    pub fn encode_one_categorical(&self, tokens: &[&str]) -> EncodedSample {
        match &self.kind {
            EncoderKind::Categorical {
                vocab,
                known_columns,
                index: _,
            } => {
                // Looked up rather than assumed last: extend_categorical appends
                // new tokens after the "<OOV>" sentinel.
                let oov_idx = *vocab
                    .get("<OOV>")
                    .expect("categorical vocab missing <OOV>");
                let mut bin = vec![0u8; self.n_features];
                for &token in tokens {
                    let idx = if let Some(&i) = vocab.get(token) {
                        i
                    } else if let Some(col) = token.split("::").next() {
                        if let Some(&unk_i) = known_columns.get(col) {
                            unk_i
                        } else {
                            oov_idx
                        }
                    } else {
                        oov_idx
                    };
                    bin[idx] = 1;
                }
                let mut out = vec![0u64; self.words];
                pack(&bin, self.n_features, &mut out);
                EncodedSample(out)
            }
            _ => panic!("encode_one_categorical requires a categorical encoder"),
        }
    }

    // ── batch encode ──────────────────────────────────────────────────────────

    /// Encode a batch of binary (`0`/`1`) feature vectors.
    pub fn encode_batch(&self, xs: &[&[u8]]) -> EncodedBatch {
        assert!(
            matches!(self.kind, EncoderKind::Binary),
            "encode_batch requires a binary encoder; use encode_batch_numeric or encode_batch_categorical"
        );
        let n = xs.len();
        let w = self.words;
        let mut data = vec![0u64; n * w];
        for (i, x) in xs.iter().enumerate() {
            pack(x, self.n_features, &mut data[i * w..(i + 1) * w]);
        }
        EncodedBatch { data, n, words: w }
    }

    /// Encode a batch of continuous (`f64`) feature vectors.
    pub fn encode_batch_numeric(&self, xs: &[&[f64]]) -> EncodedBatch {
        match &self.kind {
            EncoderKind::Numeric { booleanizer } => {
                let n = xs.len();
                let w = self.words;
                let mut data = vec![0u64; n * w];
                let mut bin = vec![0u8; self.n_features];
                for (i, x) in xs.iter().enumerate() {
                    booleanizer.transform_row(x, &mut bin);
                    pack(&bin, self.n_features, &mut data[i * w..(i + 1) * w]);
                    bin.fill(0);
                }
                EncodedBatch { data, n, words: w }
            }
            _ => panic!("encode_batch_numeric requires a numeric encoder"),
        }
    }

    /// Encode a batch of samples, each represented as `"col::val"` token slices.
    pub fn encode_batch_categorical(&self, samples: &[&[&str]]) -> EncodedBatch {
        let n = samples.len();
        let w = self.words;
        let mut data = vec![0u64; n * w];
        for (i, tokens) in samples.iter().enumerate() {
            let s = self.encode_one_categorical(tokens);
            data[i * w..(i + 1) * w].copy_from_slice(&s.0);
        }
        EncodedBatch { data, n, words: w }
    }
}

#[cfg(test)]
mod grow_tests {
    use super::*;

    /// True if positive-literal bit `i` is set in the encoded sample.
    fn bit(s: &EncodedSample, i: usize) -> bool {
        s.0[i / 64] >> (i % 64) & 1 != 0
    }

    fn fit_ab() -> Encoder {
        let s1: Vec<&str> = vec!["proc::cmd.exe", "user::alice"];
        let s2: Vec<&str> = vec!["proc::powershell.exe", "user::bob"];
        let train: Vec<&[&str]> = vec![s1.as_slice(), s2.as_slice()];
        Encoder::fit_categorical(&train)
    }

    #[test]
    fn extend_categorical_keeps_indices_stable() {
        let mut e = fit_ab();
        let before_n = e.n_features();
        let before_tokens: Vec<String> =
            (0..before_n).map(|i| e.vocab_token(i).to_string()).collect();

        // Two new tokens in known columns plus one new column (adds its UNK too).
        let n1: Vec<&str> = vec!["proc::rundll32.exe", "user::carol", "host::web01"];
        let added = e.extend_categorical(&[n1.as_slice()]);

        // 3 new regular tokens + 1 new column UNK.
        assert_eq!(added, 4);
        assert_eq!(e.n_features(), before_n + added);
        for (i, tok) in before_tokens.iter().enumerate() {
            assert_eq!(e.vocab_token(i), tok, "existing index {i} moved");
        }
    }

    #[test]
    fn extend_categorical_moves_token_out_of_unk() {
        let mut e = fit_ab();
        // Known column, unseen value: encodes to the proc UNK bit before extension.
        let unk_idx = (0..e.n_features())
            .find(|&i| e.vocab_token(i) == "proc::<UNK>")
            .unwrap();
        let s = e.encode_one_categorical(&["proc::rundll32.exe"]);
        assert!(bit(&s, unk_idx));

        let n1: Vec<&str> = vec!["proc::rundll32.exe"];
        e.extend_categorical(&[n1.as_slice()]);

        // Now it has its own appended feature; the UNK bit itself did not move.
        assert_eq!(e.vocab_token(unk_idx), "proc::<UNK>");
        let new_idx = (0..e.n_features())
            .find(|&i| e.vocab_token(i) == "proc::rundll32.exe")
            .unwrap();
        let s = e.encode_one_categorical(&["proc::rundll32.exe"]);
        assert!(bit(&s, new_idx));
        assert!(!bit(&s, unk_idx));
    }

    #[test]
    fn extend_categorical_oov_still_correct() {
        // Regression test: encode_one_categorical must look up "<OOV>" instead of
        // assuming it is the last index (it is not, after an extension).
        let mut e = fit_ab();
        let oov_idx = (0..e.n_features())
            .find(|&i| e.vocab_token(i) == "<OOV>")
            .unwrap();

        let n1: Vec<&str> = vec!["proc::rundll32.exe"];
        e.extend_categorical(&[n1.as_slice()]);
        assert_ne!(oov_idx, e.n_features() - 1, "<OOV> should no longer be last");

        // A token with an unknown column still maps to the original OOV index.
        let s = e.encode_one_categorical(&["zzz"]);
        assert!(bit(&s, oov_idx));
        assert!(!bit(&s, e.n_features() - 1));
    }

    #[test]
    fn extend_categorical_noop_returns_zero() {
        let mut e = fit_ab();
        let before_n = e.n_features();
        let s1: Vec<&str> = vec!["proc::cmd.exe", "user::bob"];
        let added = e.extend_categorical(&[s1.as_slice()]);
        assert_eq!(added, 0);
        assert_eq!(e.n_features(), before_n);
    }

    #[test]
    fn grow_binary_updates_geometry() {
        let mut e = Encoder::for_binary(5);
        let row = [1u8, 0, 1, 1, 0];
        let before = e.encode_one(&row);

        e.grow_binary(8);
        assert_eq!(e.n_features(), 8);
        assert_eq!(e.words, crate::clause_bank::dense::words_for(16));

        // Zero-padded row reproduces the old positive bits in the low positions.
        let padded = [1u8, 0, 1, 1, 0, 0, 0, 0];
        let after = e.encode_one(&padded);
        for i in 0..5 {
            assert_eq!(bit(&before, i), bit(&after, i), "positive bit {i}");
        }
        for i in 5..8 {
            assert!(!bit(&after, i), "new feature bit {i} must be 0");
        }
    }

    #[test]
    #[should_panic(expected = "cannot shrink")]
    fn grow_binary_panics_on_shrink() {
        let mut e = Encoder::for_binary(5);
        e.grow_binary(4);
    }

    #[test]
    #[should_panic(expected = "requires a categorical encoder")]
    fn extend_categorical_panics_on_binary() {
        let mut e = Encoder::for_binary(5);
        let s: Vec<&str> = vec!["a::b"];
        e.extend_categorical(&[s.as_slice()]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn extend_categorical_serde_roundtrip() {
        use crate::serial::SaveLoad;

        let mut e = fit_ab();
        let n1: Vec<&str> = vec!["proc::rundll32.exe", "host::web01"];
        e.extend_categorical(&[n1.as_slice()]);

        let mut buf = Vec::new();
        e.write_to(&mut buf).unwrap();
        let loaded = Encoder::read_from(&mut buf.as_slice()).unwrap();

        assert_eq!(e.n_features(), loaded.n_features());
        let q: Vec<&str> = vec!["proc::rundll32.exe", "user::alice", "zzz"];
        let query: Vec<&[&str]> = vec![q.as_slice()];
        assert_eq!(
            e.encode_batch_categorical(&query).data,
            loaded.encode_batch_categorical(&query).data
        );
        for i in 0..e.n_features() {
            assert_eq!(e.vocab_token(i), loaded.vocab_token(i));
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::serial::{self, SaveLoad};

    fn roundtrip(e: &Encoder) -> Encoder {
        let mut buf = Vec::new();
        e.write_to(&mut buf).unwrap();
        Encoder::read_from(&mut buf.as_slice()).unwrap()
    }

    #[test]
    fn binary_encoder_roundtrip() {
        let e = Encoder::for_binary(5);
        let loaded = roundtrip(&e);
        assert_eq!(e.n_features(), loaded.n_features());

        let rows: Vec<Vec<u8>> = vec![vec![1, 0, 1, 1, 0], vec![0, 0, 1, 0, 1]];
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        assert_eq!(e.encode_batch(&refs).data, loaded.encode_batch(&refs).data);
    }

    #[test]
    fn numeric_encoder_roundtrip() {
        let train: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![i as f64, (i % 7) as f64, (i as f64) * 0.5])
            .collect();
        let train_refs: Vec<&[f64]> = train.iter().map(|r| r.as_slice()).collect();
        let e = Encoder::fit_numeric(&train_refs, 4);
        let loaded = roundtrip(&e);
        assert_eq!(e.n_features(), loaded.n_features());

        let test: Vec<Vec<f64>> = vec![vec![3.0, 2.0, 9.0], vec![40.0, 1.0, 0.0]];
        let test_refs: Vec<&[f64]> = test.iter().map(|r| r.as_slice()).collect();
        assert_eq!(
            e.encode_batch_numeric(&test_refs).data,
            loaded.encode_batch_numeric(&test_refs).data
        );
        // Interpretable metadata must survive the round-trip too.
        assert_eq!(e.bit_origin(0), loaded.bit_origin(0));
    }

    #[test]
    fn categorical_encoder_roundtrip() {
        let s1: Vec<&str> = vec!["proc::cmd.exe", "user::alice"];
        let s2: Vec<&str> = vec!["proc::powershell.exe", "user::bob"];
        let train: Vec<&[&str]> = vec![s1.as_slice(), s2.as_slice()];
        let e = Encoder::fit_categorical(&train);
        let loaded = roundtrip(&e);
        assert_eq!(e.n_features(), loaded.n_features());

        // Include an unseen token to exercise the UNK/OOV fallback paths.
        let q1: Vec<&str> = vec!["proc::cmd.exe", "user::carol"];
        let q2: Vec<&str> = vec!["proc::unknown.exe", "newcol::x"];
        let query: Vec<&[&str]> = vec![q1.as_slice(), q2.as_slice()];
        assert_eq!(
            e.encode_batch_categorical(&query).data,
            loaded.encode_batch_categorical(&query).data
        );
        assert_eq!(e.vocab_token(0), loaded.vocab_token(0));
    }

    #[test]
    fn load_rejects_corrupt_data() {
        assert!(Encoder::read_from(&mut [].as_slice()).is_err());
        assert!(Encoder::read_from(&mut b"XXXXjunkjunk".as_slice()).is_err());
        // Valid header/tag but a truncated (empty) bincode payload.
        let mut buf = Vec::new();
        serial::write_header(&mut buf, serial::TAG_ENCODER).unwrap();
        assert!(Encoder::read_from(&mut buf.as_slice()).is_err());
        // Right magic/version but the wrong artifact type.
        let mut buf = Vec::new();
        serial::write_header(&mut buf, serial::TAG_VANILLA).unwrap();
        assert!(Encoder::read_from(&mut buf.as_slice()).is_err());
    }
}
