//! Indeo 5 band reconstruction over synthetic bitstreams: custom
//! picture sizes, the four bands of a wavelet level, inter residuals
//! and motion vectors that reach into the band padding.
//!
//! Each case pins a behaviour of IR50_32.DLL's output as FFmpeg's
//! `indeo5` decoder reproduces it, which the real streams these were
//! found on (Rent-A-Hero's custom-size movies) decode to bit for bit.

use oxideav_indeo::indeo5::{
    Codebook, DecodeError, Indeo5Decoder, PlaneRole, SessionError, decode_intra_picture,
};

/// LSB-first bit packer mirroring the decoder's bit order.
struct BitWriter {
    bits: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { bits: Vec::new() }
    }
    fn put(&mut self, value: u32, n: u32) {
        for i in 0..n {
            self.bits.push(((value >> i) & 1) as u8);
        }
    }
    /// A VLC codeword, MSB first.
    fn put_codeword(&mut self, code: u32, len: u8) {
        for i in (0..len).rev() {
            self.bits.push(((code >> i) & 1) as u8);
        }
    }
    fn align(&mut self) {
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(0);
        }
    }
    fn finish(mut self) -> Vec<u8> {
        self.align();
        let mut out: Vec<u8> = self
            .bits
            .chunks(8)
            .map(|c| c.iter().enumerate().fold(0u8, |b, (i, &v)| b | (v << i)))
            .collect();
        out.resize(out.len() + 8, 0);
        out
    }
}

/// `band_info` of a 16x16 macroblock of 8x8 blocks, full-pel vectors.
const BAND_MB16_BLK8: u32 = 0b000000;
/// `band_info` of an 8x8 macroblock of 4x4 blocks.
const BAND_MB8_BLK4: u32 = 0b000100;
/// Frame flag: the frame carries its own macroblock VLC table.
const FRAME_MB_HUFF: u32 = 0x40;

/// The picture start, GOP header (custom size, YVU9), GOP trailer and
/// frame header of an INTRA frame. `luma_levels` 1 gives four luma
/// bands.
fn intra_header(w: &mut BitWriter, width: u32, height: u32, luma_levels: u32) {
    w.put(0x1f, 5);
    w.put(0, 3); // INTRA
    w.put(0, 8); // frame_number
    w.put(0x00, 8); // gop_flags
    w.put(luma_levels, 3);
    w.put(15, 4); // custom size: height, then width
    w.put(height, 13);
    w.put(width, 13);
    for _ in 0..(luma_levels * 3 + 1) {
        w.put(BAND_MB16_BLK8, 6);
    }
    w.put(BAND_MB8_BLK4, 6);
    // The GOP trailer starts on a byte boundary: 23 fixed bits and the
    // extension flag.
    w.align();
    w.put(0, 24);
    frame_header(w, 0);
}

/// The frame header, with a custom one-row macroblock VLC table of 8
/// extra bits (symbol `n` codes as `n`) when `flags` asks for one.
fn frame_header(w: &mut BitWriter, flags: u32) {
    w.align();
    w.put(flags, 8);
    if flags & FRAME_MB_HUFF != 0 {
        w.put(7, 3); // custom descriptor
        w.put(1, 4); // one row
        w.put(8, 4); // of 8 extra bits
    }
    w.put(0, 3);
    w.align();
}

/// An INTER frame's picture start and frame header.
fn inter_header(w: &mut BitWriter, frame_number: u8, flags: u32) {
    w.put(0x1f, 5);
    w.put(1, 3); // INTER
    w.put(u32::from(frame_number), 8);
    frame_header(w, flags);
}

/// A band header with no optional fields, then the header of its one
/// tile (coded, implicit size).
fn band_and_tile(w: &mut BitWriter, glob_quant: u32) {
    w.put(0x00, 8); // band_flags
    w.put(0, 1); // no checksum
    w.put(glob_quant, 5);
    w.align();
    w.put(0, 1); // tile not empty
    w.put(0, 1); // implicit size
    w.align();
}

fn empty_band(w: &mut BitWriter) {
    w.put(0x01, 8);
    w.align();
}

/// One block of the default block tables carrying a single `+1` at
/// scan position 0, then the end of block.
fn dc_one_block(w: &mut BitWriter) {
    w.put(0, 1);
    w.put(0b000, 3); // (run 0, +1)
    w.put(0, 1);
    w.put(0b001, 3); // EOB
}

/// The symbol of a signed macroblock VLC value (0, 1, -1, 2, ...).
fn mb_symbol(v: i32) -> u32 {
    if v > 0 {
        (2 * v - 1) as u32
    } else {
        (-2 * v) as u32
    }
}

fn put_mb_value(w: &mut BitWriter, v: i32) {
    let cb = Codebook::build(&[8]).unwrap();
    let cw = cb.codeword(mb_symbol(v)).unwrap();
    w.put_codeword(cw.code, cw.length);
}

#[test]
fn custom_size_reads_height_first() {
    // 40 wide, 24 high: the GOP header codes 24, then 40.
    let mut w = BitWriter::new();
    intra_header(&mut w, 40, 24, 0);
    for _ in 0..3 {
        empty_band(&mut w);
    }
    let mut dec = Indeo5Decoder::new();
    let out = dec.decode(&w.finish()).expect("intra");
    assert_eq!(out.dimensions, (40, 24));
    assert_eq!(out.output.plane_bytes(PlaneRole::Luma).len(), 40 * 24);
    assert_eq!(out.output.plane_bytes(PlaneRole::ChromaU).len(), 10 * 6);
}

#[test]
fn wavelet_bands_take_their_own_matrix_and_no_dc_chain() {
    // A 32x32 picture of four 16x16 luma bands, one macroblock each.
    // LL (band 0): block 0 codes +1 at quantiser 23, which matrix
    // group 1 (the LL band of a wavelet level) dequantises to 12: a
    // flat LL of (12 + 1) >> 1 = 6, which the recomposition turns into
    // 16 * 6 >> 6 = 1 above mid-grey. Group 0, the matrix of a plane
    // without wavelet bands, gives 3 and a flat 128.
    let mut w = BitWriter::new();
    intra_header(&mut w, 32, 32, 1);
    band_and_tile(&mut w, 23);
    w.put(0, 1); // MB not empty
    w.put(0b0001, 4); // block 0 coded
    w.align();
    dc_one_block(&mut w);
    w.align();
    // HL (band 1, row Slant): block 0 codes +1 (33 after group 2's
    // dequantisation: row 0 of the block becomes 17). The uncoded
    // blocks 1..3 stay zero: only a 2D Slant band predicts the DC of
    // an intra block from the block before.
    band_and_tile(&mut w, 23);
    w.put(0, 1);
    w.put(0b0001, 4);
    w.align();
    dc_one_block(&mut w);
    w.align();
    // LH, HH and both chroma planes empty.
    for _ in 0..4 {
        empty_band(&mut w);
    }

    let decoded = decode_intra_picture(&w.finish()).expect("decode");
    let out = decoded.output.expect("output");
    let luma = out.plane_bytes(PlaneRole::Luma);
    // HL's row 0 reaches output rows 0..3; below that the picture is
    // the flat LL.
    for y in 4..32 {
        for x in 0..32 {
            assert_eq!(luma[y * 32 + x], 129, "({x}, {y})");
        }
    }
    // (0, 0): LL 16 * 6, plus HL's (17 + 17) * 8, over 64.
    assert_eq!(u32::from(luma[0]), 128 + ((16 * 6 + 34 * 8) >> 6));
    // HL's blocks 1..3 are zero, so the right half of rows 0..3 (band
    // columns 8 and up) is flat too.
    for y in 0..4 {
        for x in 18..32 {
            assert_eq!(luma[y * 32 + x], 129, "({x}, {y})");
        }
    }
}

/// A 32x32 one-band INTRA frame: block 0 of macroblock 0 codes +1 at
/// quantiser 23 (3 after dequantisation, 2 as samples), which the DC
/// chain carries over the whole band: luma 130.
fn flat_intra_32() -> Vec<u8> {
    let mut w = BitWriter::new();
    intra_header(&mut w, 32, 32, 0);
    band_and_tile(&mut w, 23);
    w.put(0, 1);
    w.put(0b0001, 4);
    for _ in 0..3 {
        w.put(0, 1);
        w.put(0b0000, 4);
    }
    w.align();
    dc_one_block(&mut w);
    w.align();
    empty_band(&mut w);
    empty_band(&mut w);
    w.finish()
}

#[test]
fn inter_residual_adds_to_the_prediction() {
    let mut dec = Indeo5Decoder::new();
    let f0 = dec.decode(&flat_intra_32()).expect("intra");
    assert!(
        f0.output
            .plane_bytes(PlaneRole::Luma)
            .iter()
            .all(|&p| p == 130)
    );

    // INTER: macroblock 0 is an inter macroblock with a zero vector
    // whose block 0 codes +1 (1 after the inter matrix at quantiser
    // 0, 1 as samples); the others are empty and copy the reference.
    let mut w = BitWriter::new();
    inter_header(&mut w, 1, FRAME_MB_HUFF);
    band_and_tile(&mut w, 0);
    w.put(0, 1); // not empty
    w.put(1, 1); // inter
    w.put(0b0001, 4);
    put_mb_value(&mut w, 0); // mv_y delta
    put_mb_value(&mut w, 0); // mv_x delta
    for _ in 0..3 {
        w.put(1, 1); // empty
    }
    w.align();
    dc_one_block(&mut w);
    w.align();
    empty_band(&mut w);
    empty_band(&mut w);

    let f1 = dec.decode(&w.finish()).expect("inter");
    let luma = f1.output.plane_bytes(PlaneRole::Luma);
    for y in 0..32 {
        for x in 0..32 {
            let want = if x < 8 && y < 8 { 131 } else { 130 };
            assert_eq!(luma[y * 32 + x], want, "({x}, {y})");
        }
    }
}

/// An INTRA frame of 40x24 (a band buffer of 48x32) followed by an
/// INTER frame whose first macroblock moves by `(mv_x, mv_y)`.
fn decode_with_vector(mv_x: i32, mv_y: i32) -> Result<(), SessionError> {
    let mut dec = Indeo5Decoder::new();
    let mut w = BitWriter::new();
    intra_header(&mut w, 40, 24, 0);
    band_and_tile(&mut w, 0);
    for _ in 0..6 {
        w.put(0, 1);
        w.put(0b0000, 4);
    }
    w.align();
    w.align();
    empty_band(&mut w);
    empty_band(&mut w);
    dec.decode(&w.finish()).expect("intra");

    let mut w = BitWriter::new();
    inter_header(&mut w, 1, FRAME_MB_HUFF);
    band_and_tile(&mut w, 0);
    w.put(0, 1);
    w.put(1, 1); // inter
    w.put(0b0000, 4);
    put_mb_value(&mut w, mv_y);
    put_mb_value(&mut w, mv_x);
    for _ in 1..6 {
        w.put(1, 1);
    }
    w.align();
    w.align();
    empty_band(&mut w);
    empty_band(&mut w);
    dec.decode(&w.finish()).map(|_| ())
}

#[test]
fn motion_vectors_may_point_into_the_band_padding() {
    // The 40x24 band is stored 48x32, padded to whole 16x16
    // macroblocks, and the block at (32, 16) lies in that padding.
    decode_with_vector(32, 16).expect("a vector into the padding is valid");
    // One sample further leaves the buffer.
    let err = decode_with_vector(33, 16).unwrap_err();
    assert!(
        matches!(err, SessionError::Decode(DecodeError::Mc { .. })),
        "{err:?}"
    );
    let err = decode_with_vector(0, 17).unwrap_err();
    assert!(
        matches!(err, SessionError::Decode(DecodeError::Mc { .. })),
        "{err:?}"
    );
}
