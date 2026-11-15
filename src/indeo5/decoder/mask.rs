//! The transparency trailer that follows the last band of every frame
//! of a stream with gop_flags bit 3: the rectangles that bound the
//! opaque part of the picture, and a bitmask that says which pixels are
//! opaque.
//!
//! Layout, in read order:
//!
//! | Bits | Field | Condition |
//! |------|-------|-----------|
//! | 24 | trailer size in bytes, from its first byte | frame_flags bit 0 |
//! | 3 | rectangle count n | always |
//! | 4 x 16 | n rectangles: x, y, width, height | |
//! | 1 | row delta: each mask row is XORed with the row above | always |
//! | 1 | a code descriptor follows | always |
//! | 4 + 4k | the descriptor: row count k, k row widths | the bit above |
//!
//! Then, from a byte boundary, the mask: one bit that says the mask is a
//! single value, and that value in the next bit; or else a bit that says
//! a size follows, the size (8 bits, or 0xff and 24 bits: the bytes of
//! the mask data counted from its first byte), an align, the value of
//! the first run in one bit, and run lengths coded with the descriptor's
//! code until the runs cover the picture. A run code of 0 stands for 255
//! pixels and keeps the value for the run after it; any other code is
//! that many pixels and the value flips after the run. Runs go row by
//! row over the width rounded up to a multiple of 32; the samples past
//! the width are dropped. The row delta applies to run-coded masks only:
//! the vendor driver shows a single-value mask with the bit set as one
//! value everywhere.
//!
//! The staged specification has no section for the trailer. Its layout
//! was worked out from the streams and confirmed by measuring the
//! vendor driver's output. Rent-A-Hero has 21 IV50 movies with
//! transparency, 4 150 frames. In every run-coded frame the runs end
//! exactly at the end of the picture and the data ends exactly at the
//! trailer size. On 1 844 of the frames (every frame of 20 movies and
//! the first 600 of the credits), the mask's transparent pixels are
//! exactly the pixels the vendor driver fills with the transparency
//! colour when it converts the frame to RGB (see
//! [`Iv50Transparency`](super::Iv50Transparency)).
//!
//! The descriptor stays in force for later frames that carry none. The
//! rectangles' union limits the part of the picture a frame rebuilds;
//! the mask changes no plane sample.

use super::Iv50Error;
use super::bits::Bits;
use super::huff::{Huff, read_descriptor};

/// One transparency rectangle of a frame, in luma pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Iv50Rect {
    /// Left column.
    pub x: u32,
    /// Top row.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in rows.
    pub height: u32,
}

/// Mask value of an opaque pixel.
pub(crate) const OPAQUE: u8 = 0xff;

/// The trailer's fields before the mask data.
pub(crate) struct TrailerHeader {
    pub rects: Vec<Iv50Rect>,
    pub row_delta: bool,
    pub descriptor: Option<Vec<u8>>,
}

/// Read the trailer up to the mask data, including the align before it.
pub(crate) fn read_header(r: &mut Bits, frame_flags: u32) -> TrailerHeader {
    if frame_flags & 1 != 0 {
        r.read(24);
    }
    let n = r.read(3);
    let rects = (0..n)
        .map(|_| Iv50Rect {
            x: r.read(16),
            y: r.read(16),
            width: r.read(16),
            height: r.read(16),
        })
        .collect();
    let row_delta = r.bit();
    let descriptor = r.bit().then(|| read_descriptor(r));
    r.align();
    TrailerHeader {
        rects,
        row_delta,
        descriptor,
    }
}

/// Row length of the mask in samples: the picture width rounded up to a
/// multiple of 32.
pub(crate) fn stride(width: usize) -> usize {
    (width + 31) & !31
}

/// Decode the mask data into `mask`, `stride(width)` samples per row,
/// [`OPAQUE`] or 0. `code` is the descriptor in force, if any.
pub(crate) fn read_mask(
    r: &mut Bits,
    code: Option<&Huff>,
    row_delta: bool,
    stride: usize,
    mask: &mut [u8],
) -> Result<(), Iv50Error> {
    if r.bit() {
        let v = if r.bit() { OPAQUE } else { 0 };
        mask.fill(v);
        return if r.overrun() {
            Err(Iv50Error::Truncated)
        } else {
            Ok(())
        };
    }
    if r.bit() && r.read(8) == 0xff {
        // The size of the mask data: informational.
        r.read(24);
    }
    r.align();
    let code = code.ok_or(Iv50Error::BadHeader(
        "transparency mask without a code descriptor",
    ))?;
    let mut v = if r.bit() { OPAQUE } else { 0 };
    let mut pos = 0;
    while pos < mask.len() {
        let c = code
            .decode(r)
            .ok_or(Iv50Error::BadHeader("transparency mask code"))?;
        if r.overrun() {
            return Err(Iv50Error::Truncated);
        }
        // A run that would pass the end of the mask is cut there; the
        // streams seen end exactly at it.
        let (n, flip) = if c == 0 {
            (255, false)
        } else {
            (c as usize, true)
        };
        let end = mask.len().min(pos + n);
        mask[pos..end].fill(v);
        pos = end;
        if flip {
            v ^= OPAQUE;
        }
    }
    if row_delta {
        for i in stride..mask.len() {
            mask[i] ^= mask[i - stride];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::bits::PaddedFrame;
    use super::super::testutil::BitWriter;
    use super::*;

    /// Symbol 0 is one bit, 1 to 128 and 129 to 256 nine bits.
    const ROWS: [u8; 3] = [0, 7, 7];

    /// Run-coded mask data without a size: the first value and the run
    /// codes.
    fn runs(first: u32, codes: &[u32]) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put(0, 1);
        w.put(0, 1);
        w.align();
        w.put(first, 1);
        for &c in codes {
            w.sym(&ROWS, c);
        }
        w.bytes
    }

    fn decode(data: &[u8], row_delta: bool, stride: usize, len: usize) -> Vec<u8> {
        let code = Huff::new(&ROWS).unwrap();
        let f = PaddedFrame::new(data);
        let mut mask = vec![7; len];
        read_mask(
            &mut f.reader_at(0),
            Some(&code),
            row_delta,
            stride,
            &mut mask,
        )
        .unwrap();
        mask
    }

    #[test]
    fn stride_is_the_width_rounded_up_to_32() {
        assert_eq!(stride(1), 32);
        assert_eq!(stride(32), 32);
        assert_eq!(stride(33), 64);
        // The largest coded width (13 bits).
        assert_eq!(stride(8191), 8192);
    }

    #[test]
    fn runs_go_on_across_rows() {
        // A picture 33 wide has rows of 64. Runs of 40, 50 and 102
        // cover three rows: 0..40 transparent, 40..90 opaque, then
        // transparent.
        let data = runs(0, &[40, 50, 102]);
        let m = decode(&data, false, 64, 192);
        let want: Vec<u8> = (0..192)
            .map(|i| if (40..90).contains(&i) { OPAQUE } else { 0 })
            .collect();
        assert_eq!(m, want);
        // With the row delta, each row is XORed with the row above as
        // it stands after its own XOR.
        let d = decode(&data, true, 64, 192);
        for i in 0..64 {
            assert_eq!(d[i], want[i]);
            assert_eq!(d[64 + i], want[64 + i] ^ want[i]);
            assert_eq!(d[128 + i], want[128 + i] ^ d[64 + i]);
        }
    }

    #[test]
    fn a_single_value_ignores_the_code_and_the_row_delta() {
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 1);
        let f = PaddedFrame::new(&w.bytes);
        let mut mask = vec![0; 96];
        read_mask(&mut f.reader_at(0), None, true, 32, &mut mask).unwrap();
        assert!(mask.iter().all(|&v| v == OPAQUE));
    }
}
