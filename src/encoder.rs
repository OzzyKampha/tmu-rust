//! Encoder: transforms raw data into bit-packed literal vectors for [`TsetlinMachine`].
//!
//! Three built-in modes:
//! - [`Encoder::for_binary`] — already-binary `u8` features, just packs
//! - [`Encoder::fit_numeric`] — `f64` features → quantile booleanization → pack
//! - [`Encoder::fit_categorical`] — `"col::val"` token strings → vocabulary → pack
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
}

// ── encoder kind ─────────────────────────────────────────────────────────────

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
pub struct Encoder {
    kind: EncoderKind,
    n_features: usize,
    words: usize,
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
            kind: EncoderKind::Categorical { vocab, index, known_columns },
            n_features,
            words: words_for(2 * n_features),
        }
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
            EncoderKind::Categorical { index, .. } => {
                index.get(bit).map(|s| s.as_str()).expect("bit index out of vocab range")
            }
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
            EncoderKind::Categorical { vocab, known_columns, index } => {
                let oov_idx = index.len() - 1; // "<OOV>" is always last
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
