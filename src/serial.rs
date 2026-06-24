//! Minimal, dependency-free binary (de)serialisation helpers used by the model
//! and encoder `save`/`load` methods.
//!
//! All multi-byte values are written **little-endian**.  Files begin with a
//! 4-byte magic (`TMRS`), a `u32` format version, and a 1-byte type tag so a
//! load can fail cleanly on a mismatched or corrupt file rather than silently
//! producing a garbage model.  Length-prefixed encodings use a `u64` count so
//! the format is stable across 32-/64-bit targets.

use std::io::{self, Read, Write};

/// File magic: identifies a tmu-rs serialised artifact.
pub(crate) const MAGIC: [u8; 4] = *b"TMRS";
/// Current on-disk format version.  Bump on any incompatible layout change.
pub(crate) const VERSION: u32 = 1;

/// Type tags distinguishing the artifact kinds.
pub(crate) const TAG_VANILLA: u8 = 1;
pub(crate) const TAG_COALESCED: u8 = 2;
pub(crate) const TAG_AUTOENCODER: u8 = 3;
pub(crate) const TAG_ENCODER: u8 = 4;

/// Build an `InvalidData` I/O error with a descriptive message.
pub(crate) fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ── writers ────────────────────────────────────────────────────────────────

pub(crate) fn w_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

pub(crate) fn w_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn w_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn w_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn w_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_bits().to_le_bytes())
}

pub(crate) fn w_bool<W: Write>(w: &mut W, v: bool) -> io::Result<()> {
    w_u8(w, v as u8)
}

pub(crate) fn w_usize<W: Write>(w: &mut W, v: usize) -> io::Result<()> {
    w_u64(w, v as u64)
}

/// Write a length-prefixed UTF-8 string.
pub(crate) fn w_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w_usize(w, s.len())?;
    w.write_all(s.as_bytes())
}

pub(crate) fn w_vec_u8<W: Write>(w: &mut W, v: &[u8]) -> io::Result<()> {
    w_usize(w, v.len())?;
    w.write_all(v)
}

pub(crate) fn w_vec_i32<W: Write>(w: &mut W, v: &[i32]) -> io::Result<()> {
    w_usize(w, v.len())?;
    for &x in v {
        w_i32(w, x)?;
    }
    Ok(())
}

pub(crate) fn w_vec_f64<W: Write>(w: &mut W, v: &[f64]) -> io::Result<()> {
    w_usize(w, v.len())?;
    for &x in v {
        w_f64(w, x)?;
    }
    Ok(())
}

// ── readers ────────────────────────────────────────────────────────────────

pub(crate) fn r_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

pub(crate) fn r_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub(crate) fn r_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub(crate) fn r_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

pub(crate) fn r_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_bits(u64::from_le_bytes(b)))
}

pub(crate) fn r_bool<R: Read>(r: &mut R) -> io::Result<bool> {
    Ok(r_u8(r)? != 0)
}

pub(crate) fn r_usize<R: Read>(r: &mut R) -> io::Result<usize> {
    let v = r_u64(r)?;
    usize::try_from(v).map_err(|_| bad("length does not fit in usize on this platform"))
}

pub(crate) fn r_str<R: Read>(r: &mut R) -> io::Result<String> {
    let len = r_usize(r)?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| bad("invalid UTF-8 in serialised string"))
}

pub(crate) fn r_vec_u8<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let len = r_usize(r)?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub(crate) fn r_vec_i32<R: Read>(r: &mut R) -> io::Result<Vec<i32>> {
    let len = r_usize(r)?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(r_i32(r)?);
    }
    Ok(v)
}

pub(crate) fn r_vec_f64<R: Read>(r: &mut R) -> io::Result<Vec<f64>> {
    let len = r_usize(r)?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(r_f64(r)?);
    }
    Ok(v)
}

// ── header ───────────────────────────────────────────────────────────────────

/// Write magic + version + type tag.
pub(crate) fn write_header<W: Write>(w: &mut W, tag: u8) -> io::Result<()> {
    w.write_all(&MAGIC)?;
    w_u32(w, VERSION)?;
    w_u8(w, tag)
}

/// Validate magic + version + type tag.  Errors on any mismatch.
pub(crate) fn read_header<R: Read>(r: &mut R, expected_tag: u8) -> io::Result<()> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(bad("not a tmu-rs file (bad magic)"));
    }
    let version = r_u32(r)?;
    if version != VERSION {
        return Err(bad(format!(
            "unsupported tmu-rs format version {version} (expected {VERSION})"
        )));
    }
    let tag = r_u8(r)?;
    if tag != expected_tag {
        return Err(bad(format!(
            "wrong artifact type: file tag {tag}, expected {expected_tag}"
        )));
    }
    Ok(())
}

/// Write a `Vec<Rng>` as a length-prefixed list of `u64` states.
pub(crate) fn w_rngs<W: Write>(w: &mut W, rngs: &[crate::rng::Rng]) -> io::Result<()> {
    w_usize(w, rngs.len())?;
    for rng in rngs {
        w_u64(w, rng.state())?;
    }
    Ok(())
}

/// Read a `Vec<Rng>` written by [`w_rngs`].
pub(crate) fn r_rngs<R: Read>(r: &mut R) -> io::Result<Vec<crate::rng::Rng>> {
    let len = r_usize(r)?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(crate::rng::Rng::from_state(r_u64(r)?));
    }
    Ok(v)
}
