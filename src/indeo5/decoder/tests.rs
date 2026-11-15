//! Tests of the whole decoder on synthetic bitstreams, and of the
//! four-band recombination.

use super::band::RvMap;
use super::headers::{BandInfo, GopHeader};
use super::huff::{BLOCK_PRESETS, MB_PRESETS};
use super::synth::{OutPlane, Window, four_band};
use super::testutil::BitWriter;
use super::*;

/// The level index that dequantises to `d` at step 1 (spec/05 §2.3).
fn code_for(d: i32) -> u16 {
    if d > 0 {
        (2 * d - 1) as u16
    } else {
        (-2 * d) as u16
    }
}

/// Write one coded block holding only a DC value `d` (step 1), with the
/// default run/value table and block code.
fn dc_block(w: &mut BitWriter, rv: &RvMap, d: i32) {
    let rows = BLOCK_PRESETS[7];
    w.sym(rows, rv.symbol_for(1, code_for(d)).expect("run 1 symbol"));
    w.sym(rows, rv.eob());
}

/// Picture start, then for an intra frame a GOP header of a custom
/// `width` by `height` picture with one luma band (16x16 macroblocks,
/// 8x8 blocks) and one chroma band (4x4 macroblocks and blocks).
fn picture_start(w: &mut BitWriter, ftype: u32, number: u32, size: Option<(u32, u32)>) {
    w.put(0x1f, 5);
    w.put(ftype, 3);
    w.put(number, 8);
    if let Some((width, height)) = size {
        w.put(0, 8); // gop_flags
        w.put(0, 3); // subsampling: 1 luma band, 1 chroma band
        w.put(15, 4); // custom size
        w.put((width << 13) | height, 26);
        w.put(0, 6); // luma band info: mb 16, blk 8
        w.put(6, 6); // chroma band info: mb 4, blk 4
        w.align();
        w.put(2 << 19, 24); // version 2
        w.align();
    }
}

/// Frame header with `flags` and default tables.
fn frame_header(w: &mut BitWriter, flags: u32) {
    w.put(flags, 8);
    w.put(0, 3); // plane filter bits
    w.align();
}

/// Two empty chroma bands (V, then U).
fn empty_chroma(w: &mut BitWriter) {
    w.put(1, 8);
    w.put(1, 8);
}

/// DC values of the four intra blocks of the test picture. With DC
/// prediction the blocks hold (P + 1) >> 1 for the running sums P = 20,
/// 40, 14, 30: 10, 20, 7 and 15.
const DCS: [i32; 4] = [20, 20, -26, 16];

/// An intra frame of a 16 by 8 picture whose one luma macroblock codes
/// the four blocks of `DCS`.
fn intra_frame() -> Vec<u8> {
    let rv = RvMap::new(8, &[]).unwrap();
    let mut w = BitWriter::new();
    picture_start(&mut w, 0, 0, Some((16, 8)));
    frame_header(&mut w, 0);
    // Luma band header: no flags, no checksum, gq 0.
    w.put(0, 8);
    w.put(0, 1);
    w.put(0, 5);
    w.align();
    // Tile: coded, no size.
    w.put(0, 1);
    w.put(0, 1);
    w.align();
    // Macroblock: coded, all four blocks.
    w.put(0, 1);
    w.put(0xf, 4);
    w.align();
    for d in DCS {
        dc_block(&mut w, &rv, d);
    }
    w.align();
    empty_chroma(&mut w);
    w.bytes
}

/// An inter frame (type 1) whose luma macroblock is inter with vector
/// (0, 8), so its top blocks predict from rows 8 to 15, the band padding
/// below the 8-row picture. Block 0 adds a residual of DC 10, which the
/// transform makes a constant 5. The frame carries band sizes.
fn inter_frame(number: u32) -> Vec<u8> {
    let rv = RvMap::new(8, &[]).unwrap();
    let mut w = BitWriter::new();
    picture_start(&mut w, 1, number, None);
    frame_header(&mut w, 0x80);
    let band_start = w.bytes.len();
    w.put(0, 8);
    w.put(0, 24); // band size, patched below
    w.put(0, 1);
    w.put(0, 5);
    w.align();
    w.put(0, 1);
    w.put(0, 1);
    w.align();
    w.put(0, 1); // coded
    w.put(1, 1); // inter
    w.put(1, 4); // cbp: block 0
    let mb = MB_PRESETS[7];
    w.sym(mb, 15); // dy = +8
    w.sym(mb, 0); // dx = 0
    w.align();
    dc_block(&mut w, &rv, 10);
    w.align();
    let size = (w.bytes.len() - band_start) as u32;
    w.bytes[band_start + 1..band_start + 4].copy_from_slice(&size.to_le_bytes()[..3]);
    empty_chroma(&mut w);
    w.bytes
}

fn rows(p: Iv50Plane) -> Vec<Vec<u8>> {
    (0..p.height).map(|r| p.row(r).to_vec()).collect()
}

#[test]
fn custom_size_is_height_then_width() {
    let mut dec = Iv50Decoder::new();
    assert_eq!(dec.decode(&intra_frame()), Ok(FrameType::Intra));
    // The 26-bit field holds the height in bits 0..12 and the width in
    // bits 13..25 (spec/02 §1.6: height is read first).
    assert_eq!(dec.dimensions(), Some((16, 8)));
    let y = dec.y().unwrap();
    assert_eq!((y.width, y.height), (16, 8));
    let u = dec.u().unwrap();
    assert_eq!((u.width, u.height), (4, 2));
}

#[test]
fn intra_blocks_use_dc_prediction() {
    let mut dec = Iv50Decoder::new();
    dec.decode(&intra_frame()).unwrap();
    for row in rows(dec.y().unwrap()) {
        assert_eq!(&row[..8], &[128 + 10; 8]);
        assert_eq!(&row[8..], &[128 + 20; 8]);
    }
    for p in [dec.u().unwrap(), dec.v().unwrap()] {
        assert!(rows(p).iter().flatten().all(|&v| v == 128));
    }
}

#[test]
fn inter_residual_adds_to_prediction_from_band_padding() {
    let mut dec = Iv50Decoder::new();
    dec.decode(&intra_frame()).unwrap();
    assert_eq!(dec.decode(&inter_frame(1)), Ok(FrameType::Inter));
    // Block 0 = block 2 of the reference (7) + residual 5; block 1 is
    // uncoded and takes block 3 of the reference (15).
    for row in rows(dec.y().unwrap()) {
        assert_eq!(&row[..8], &[128 + 12; 8]);
        assert_eq!(&row[8..], &[128 + 15; 8]);
    }
}

#[test]
fn repeated_frame_number_is_a_null_frame() {
    let mut dec = Iv50Decoder::new();
    dec.decode(&intra_frame()).unwrap();
    dec.decode(&inter_frame(1)).unwrap();
    let before = rows(dec.y().unwrap());
    // Same number as the previous frame: NULL, the picture stays.
    assert_eq!(dec.decode(&inter_frame(1)), Ok(FrameType::Null));
    assert_eq!(rows(dec.y().unwrap()), before);
    // A new number decodes again, from the inter frame's picture.
    dec.decode(&inter_frame(2)).unwrap();
    assert_ne!(rows(dec.y().unwrap()), before);
}

#[test]
fn bad_headers_are_rejected() {
    let mut dec = Iv50Decoder::new();
    assert_eq!(dec.decode(&[0x1e, 0]), Err(Iv50Error::BadPictureStart));
    assert_eq!(
        dec.decode(&[0x1f | (5 << 5), 0]),
        Err(Iv50Error::BadPictureStart)
    );
    assert_eq!(dec.decode(&inter_frame(1)), Err(Iv50Error::NoReference));
    // Subsampling code 3.
    let mut w = BitWriter::new();
    w.put(0x1f, 5);
    w.put(0, 3);
    w.put(0, 8);
    w.put(0, 8);
    w.put(3, 3);
    w.put(0, 32);
    assert_eq!(
        dec.decode(&w.bytes),
        Err(Iv50Error::BadHeader("subsampling code"))
    );
    // Several tiles, a locked stream and transparency with half-size
    // chroma are refused.
    for flags in [0x40, 0x20, 0x0a] {
        let mut w = BitWriter::new();
        w.put(0x1f, 5);
        w.put(0, 3);
        w.put(0, 8);
        w.put(flags, 8);
        w.put(0, 32);
        w.put(0, 32);
        assert!(matches!(
            dec.decode(&w.bytes),
            Err(Iv50Error::Unsupported(_))
        ));
    }
    // Seven luma bands (codes 2 and 6) are refused, also when every band
    // of the frame is empty.
    for sub in [2, 6] {
        let (luma, chroma) = if sub == 2 { (7, 1) } else { (7, 4) };
        let mut w = BitWriter::new();
        w.put(0x1f, 5);
        w.put(0, 3);
        w.put(0, 8);
        w.put(0, 8);
        w.put(sub, 3);
        w.put(15, 4);
        w.put((16 << 13) | 8, 26);
        for _ in 0..luma {
            w.put(0, 6);
        }
        for _ in 0..chroma {
            w.put(6, 6);
        }
        w.align();
        w.put(2 << 19, 24);
        w.align();
        frame_header(&mut w, 0);
        for _ in 0..luma + 2 * chroma {
            w.put(1, 8);
        }
        assert!(matches!(
            Iv50Decoder::new().decode(&w.bytes),
            Err(Iv50Error::Unsupported(_))
        ));
    }
    // Truncated band data.
    let f = intra_frame();
    assert!(dec.decode(&f[..f.len() - 4]).is_err());
}

/// A plane of 64 by 32 with four luma bands of 32 by 16.
fn four_band_layout() -> PlaneLayout {
    let info = BandInfo {
        half_pel: false,
        mb_size: 8,
        blk_size: 8,
        xform: 0,
    };
    let gop = GopHeader {
        flags: 0,
        luma_bands: 4,
        chroma_bands: 1,
        width: 64,
        height: 32,
        luma_info: vec![info; 4],
        chroma_info: vec![info],
        transparency: false,
        colour: None,
        intra_only: false,
    };
    build_layout(&gop).swap_remove(0)
}

/// Recombine the whole 64 by 32 plane with band `b` set by `f(row, col)`
/// over its whole buffer and the other bands zero.
fn synth_one(b: usize, f: impl Fn(usize, usize) -> i16) -> OutPlane {
    let lay = four_band_layout();
    let mut area = vec![0i16; lay.area_len];
    let g = &lay.bands[b];
    for r in 0..16 {
        for c in 0..g.pitch {
            area[g.offset + r * g.pitch + c] = f(r, c);
        }
    }
    let mut out = OutPlane::new(64, 32);
    let win = Window {
        y: 0,
        x: 0,
        h: 32,
        w: 64,
    };
    four_band(&area, &lay.bands, win, &mut out);
    out
}

fn px(out: &OutPlane, r: usize, c: usize) -> u8 {
    out.data[r * out.stride + c]
}

/// The four-band recombination against the synthesis filter pair of
/// spec/06 §3.2, written as a plain upsample-and-filter: a low-pass
/// axis places band sample `k` on plane position `2k` with the taps
/// `{1, 2, 1} / 2`, a high-pass axis on position `2k + 1` with
/// `{1, 2, -6, 2, 1} / 4`; band 1 is high-pass vertically, band 2
/// horizontally, band 3 in both. The plane sample is a quarter of the
/// sum of the four bands, truncated: the accumulator of spec/08 §3.0
/// holds 16 times the sum and is shifted right by 6. Compared away from
/// the plane's edges, where the neighbour rules differ.
#[test]
fn four_band_equals_the_five_three_synthesis_pair() {
    const LOW: [i32; 3] = [1, 2, 1];
    const HIGH: [i32; 5] = [1, 2, -6, 2, 1];
    // The weight of band sample `k` at plane position `pos` along one
    // axis, with the low-pass taps doubled and the high-pass ones
    // quadrupled.
    let weight = |high: bool, k: usize, pos: usize| -> i32 {
        let (centre, taps): (i32, &[i32]) = if high {
            (2 * k as i32 + 1, &HIGH)
        } else {
            (2 * k as i32, &LOW)
        };
        let d = pos as i32 - centre + taps.len() as i32 / 2;
        if (0..taps.len() as i32).contains(&d) {
            taps[d as usize]
        } else {
            0
        }
    };
    let lay = four_band_layout();
    let mut state = 2024u32;
    for round in 0..6 {
        let mut area = vec![0i16; lay.area_len];
        let mut bands = [[[0i32; 32]; 16]; 4];
        let span = [40, 200, 1000][round % 3];
        for (b, band) in bands.iter_mut().enumerate() {
            let g = &lay.bands[b];
            assert_eq!(g.pitch, 32);
            for (r, row) in band.iter_mut().enumerate() {
                for (c, v) in row.iter_mut().enumerate() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *v = ((state >> 8) % (2 * span)) as i32 - span as i32;
                    area[g.offset + r * 32 + c] = *v as i16;
                }
            }
        }
        let mut out = OutPlane::new(64, 32);
        let win = Window {
            y: 0,
            x: 0,
            h: 32,
            w: 64,
        };
        four_band(&area, &lay.bands, win, &mut out);
        for y in 4..28 {
            for x in 4..58 {
                let mut acc = 0i32;
                for (b, band) in bands.iter().enumerate() {
                    let (high_y, high_x) = (b & 1 == 1, b & 2 == 2);
                    // With the doubled and quadrupled taps a band has a
                    // gain of 4, 8, 8 or 16; this brings all to 16.
                    let gain = [4, 2, 2, 1][b];
                    // The band samples within two of the plane sample.
                    let near = |pos: usize| pos / 2 - 2..=pos / 2 + 2;
                    for (r, row) in band.iter().enumerate() {
                        let wy = weight(high_y, r, y);
                        if wy == 0 || !near(y).contains(&r) {
                            continue;
                        }
                        acc += gain
                            * wy
                            * row
                                .iter()
                                .enumerate()
                                .filter(|(k, _)| near(x).contains(k))
                                .map(|(k, &v)| weight(high_x, k, x) * v)
                                .sum::<i32>();
                    }
                }
                let want = (((acc as i16) >> 6) as i32 + 128).clamp(0, 255) as u8;
                assert_eq!(px(&out, y, x), want, "round {round} sample ({x}, {y})");
            }
        }
    }
}

#[test]
fn four_band_low_band_is_upsampled() {
    // L = 16 * c: even columns 16 * L >> 6 = 4c, odd columns average
    // with the next sample.
    let out = synth_one(0, |_, c| 16 * c as i16);
    for r in 0..32 {
        for c in 0..31 {
            assert_eq!(px(&out, r, 2 * c), 128 + 4 * c as u8);
            assert_eq!(px(&out, r, 2 * c + 1), 128 + 4 * c as u8 + 2);
        }
        // The last written column takes its left neighbour's value.
        assert_eq!(px(&out, r, 63), px(&out, r, 62));
    }
}

#[test]
fn four_band_constant_high_bands() {
    // A constant V band of 64 adds +16 to even rows and -16 to odd rows
    // (a V band of 1 gives the same pattern, scaled).
    let out = synth_one(1, |_, _| 64);
    for r in 0..32 {
        let want = if r % 2 == 0 { 144 } else { 112 };
        assert!((0..64).all(|c| px(&out, r, c) == want), "row {r}");
    }
    // A constant D band of 64: ee and oo +16, eo and oe -16.
    let out = synth_one(3, |_, _| 64);
    for r in 0..32 {
        for c in 0..63 {
            let want = if (r + c) % 2 == 0 { 144 } else { 112 };
            assert_eq!(px(&out, r, c), want, "row {r} col {c}");
        }
    }
}

#[test]
fn four_band_first_column_repeats_itself() {
    // H nonzero only in band column 0: the left neighbour of column 0 is
    // column 0 itself, so ee = 8 * (H + H) = 8 * 128.
    let out = synth_one(2, |_, c| if c == 0 { 64 } else { 0 });
    for r in (0..32).step_by(2) {
        assert_eq!(px(&out, r, 0), 128 + 16);
        // Band column 1: s2 = H[1] + H[0] = 64.
        assert_eq!(px(&out, r, 2), 128 + 8);
    }
}

/// An intra frame like [`intra_frame`] in a stream with transparency
/// (gop_flags bit 3) and the transparency colour `colour`, whose
/// trailer carries `rects` as (x, y, w, h) and then whatever `rest`
/// writes: the row-delta bit, the descriptor and the mask data.
fn transparent_intra_frame(
    rects: &[(u32, u32, u32, u32)],
    colour: Option<u32>,
    rest: impl FnOnce(&mut BitWriter),
) -> Vec<u8> {
    let rv = RvMap::new(8, &[]).unwrap();
    let mut w = BitWriter::new();
    w.put(0x1f, 5);
    w.put(0, 3);
    w.put(0, 8);
    w.put(8, 8); // gop_flags: transparency
    w.put(0, 3);
    w.put(15, 4);
    w.put((16 << 13) | 8, 26);
    w.put(0, 6);
    w.put(6, 6);
    match colour {
        Some(c) => {
            w.put(8, 4);
            w.put(c, 24);
        }
        None => w.put(0, 4),
    }
    w.align();
    w.put(2 << 19, 24);
    w.align();
    frame_header(&mut w, 0);
    w.put(0, 8);
    w.put(0, 1);
    w.put(0, 5);
    w.align();
    w.put(0, 1);
    w.put(0, 1);
    w.align();
    w.put(0, 1);
    w.put(0xf, 4);
    w.align();
    for d in DCS {
        dc_block(&mut w, &rv, d);
    }
    w.align();
    empty_chroma(&mut w);
    w.put(rects.len() as u32, 3);
    for &(x, y, rw, rh) in rects {
        w.put(x, 16);
        w.put(y, 16);
        w.put(rw, 16);
        w.put(rh, 16);
    }
    rest(&mut w);
    w.align();
    w.bytes
}

/// Trailer rest: no row delta, no descriptor, a mask that is opaque
/// everywhere.
fn opaque_mask(w: &mut BitWriter) {
    w.put(0, 2);
    w.align();
    w.put(1, 1); // one value
    w.put(1, 1); // opaque
}

/// The descriptor the mask tests use: symbol 0 is one bit, 1 to 128 and
/// 129 to 256 nine bits.
const MASK_ROWS: [u8; 3] = [0, 7, 7];

/// Trailer rest: row delta `delta`, the descriptor when `descriptor`,
/// no size, the first run's value `first` and the run codes `codes`.
fn run_mask(
    delta: bool,
    descriptor: bool,
    first: u32,
    codes: &[u32],
) -> impl FnOnce(&mut BitWriter) {
    let codes = codes.to_vec();
    move |w: &mut BitWriter| {
        w.put(delta as u32, 1);
        w.put(descriptor as u32, 1);
        if descriptor {
            w.put(MASK_ROWS.len() as u32, 4);
            for r in MASK_ROWS {
                w.put(r as u32, 4);
            }
        }
        w.align();
        w.put(0, 1); // run-coded
        w.put(0, 1); // no size
        w.align();
        w.put(first, 1);
        for c in codes {
            w.sym(&MASK_ROWS, c);
        }
    }
}

/// The mask of the current picture, one row per picture row.
fn mask_rows(dec: &Iv50Decoder) -> Vec<Vec<u8>> {
    rows(dec.transparency().unwrap().mask)
}

#[test]
fn transparency_rectangles_outside_the_picture_write_nothing() {
    for rect in [
        (60000, 0, 10, 10),
        (0, 60000, 10, 10),
        (65535, 65535, 65535, 65535),
    ] {
        let mut dec = Iv50Decoder::new();
        assert_eq!(
            dec.decode(&transparent_intra_frame(&[rect], None, opaque_mask)),
            Ok(FrameType::Intra)
        );
        // The window lies outside the picture, so nothing is
        // recombined and the planes keep their initial zeros.
        assert!(rows(dec.y().unwrap()).iter().flatten().all(|&v| v == 0));
    }
}

#[test]
fn transparency_colour_rectangles_and_single_value_masks() {
    let mut dec = Iv50Decoder::new();
    let f = transparent_intra_frame(&[(1, 2, 3, 4), (5, 0, 2, 2)], Some(0x563412), opaque_mask);
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    let t = dec.transparency().unwrap();
    // Red in bits 0..7, green in 8..15, blue in 16..23.
    assert_eq!(t.colour, Some([0x12, 0x34, 0x56]));
    let rect = |x, y, width, height| Iv50Rect {
        x,
        y,
        width,
        height,
    };
    assert_eq!(t.rects, &[rect(1, 2, 3, 4), rect(5, 0, 2, 2)]);
    // The mask covers the picture; its rows are the width rounded up to
    // 32.
    assert_eq!((t.mask.width, t.mask.height, t.mask.stride), (16, 8, 32));
    assert!(mask_rows(&dec).iter().flatten().all(|&v| v == 255));
    // A transparent single-value mask, with the row delta set: the
    // delta applies to run-coded masks only.
    let f = transparent_intra_frame(&[], None, |w| {
        w.put(1, 1); // row delta
        w.put(0, 1); // no descriptor
        w.align();
        w.put(1, 1);
        w.put(0, 1); // transparent
    });
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    let t = dec.transparency().unwrap();
    assert_eq!(t.colour, None);
    assert!(t.rects.is_empty());
    assert!(mask_rows(&dec).iter().flatten().all(|&v| v == 0));
}

#[test]
fn transparency_mask_runs() {
    // Runs cover 32 samples a row (16 visible); 2 transparent, then 254
    // opaque.
    let mut dec = Iv50Decoder::new();
    let f = transparent_intra_frame(&[(2, 0, 14, 8)], None, run_mask(false, true, 0, &[2, 254]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    let m = mask_rows(&dec);
    assert_eq!(m[0][..2], [0, 0]);
    assert!(m[0][2..].iter().all(|&v| v == 255));
    assert!(m[1..].iter().flatten().all(|&v| v == 255));
    // Code 0 is 255 samples and keeps the value for the next run: one
    // opaque run of 256.
    let f = transparent_intra_frame(&[], None, run_mask(false, true, 1, &[0, 1]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    assert!(mask_rows(&dec).iter().flatten().all(|&v| v == 255));
    // Any other code flips the value after its run.
    let f = transparent_intra_frame(&[], None, run_mask(false, true, 1, &[1, 0]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    let m = mask_rows(&dec);
    assert_eq!(m[0][0], 255);
    assert!(m.iter().flatten().skip(1).all(|&v| v == 0));
}

#[test]
fn transparency_mask_row_delta() {
    // The same runs with the row delta: each row is XORed with the
    // decoded row above, so the rows alternate.
    let mut dec = Iv50Decoder::new();
    let f = transparent_intra_frame(&[], None, run_mask(true, true, 0, &[2, 254]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    for (r, row) in mask_rows(&dec).iter().enumerate() {
        let (left, right) = if r % 2 == 0 { (0, 255) } else { (255, 0) };
        assert_eq!(row[..2], [left, left], "row {r}");
        assert!(row[2..].iter().all(|&v| v == right), "row {r}");
    }
}

#[test]
fn transparency_mask_size_fields() {
    // An 8-bit size, and 0xff followed by a 24-bit size: both are read
    // and skipped.
    for long in [false, true] {
        let mut dec = Iv50Decoder::new();
        let f = transparent_intra_frame(&[], None, |w| {
            w.put(0, 1);
            w.put(1, 1);
            w.put(MASK_ROWS.len() as u32, 4);
            for r in MASK_ROWS {
                w.put(r as u32, 4);
            }
            w.align();
            w.put(0, 1);
            w.put(1, 1); // size
            if long {
                w.put(0xff, 8);
                w.put(0x123, 24);
            } else {
                w.put(9, 8);
            }
            w.align();
            w.put(0, 1);
            w.sym(&MASK_ROWS, 2);
            w.sym(&MASK_ROWS, 254);
        });
        assert_eq!(dec.decode(&f), Ok(FrameType::Intra), "long {long}");
        let m = mask_rows(&dec);
        assert_eq!(m[0][..3], [0, 0, 255], "long {long}");
    }
}

#[test]
fn transparency_descriptor_stays_and_null_frames_keep_the_mask() {
    let mut dec = Iv50Decoder::new();
    // No descriptor before the first one: the run-coded mask can't
    // decode.
    let f = transparent_intra_frame(&[], None, run_mask(false, false, 0, &[]));
    assert_eq!(
        dec.decode(&f),
        Err(Iv50Error::BadHeader(
            "transparency mask without a code descriptor"
        ))
    );
    assert!(dec.transparency().is_none());
    let f = transparent_intra_frame(&[(2, 0, 14, 8)], None, run_mask(false, true, 0, &[2, 254]));
    dec.decode(&f).unwrap();
    // A later frame without a descriptor uses the last one.
    let f = transparent_intra_frame(&[(4, 0, 12, 8)], None, run_mask(false, false, 0, &[4, 252]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    let m = mask_rows(&dec);
    assert_eq!(m[0][..5], [0, 0, 0, 0, 255]);
    // A NULL frame keeps the picture and its transparency.
    let mut w = BitWriter::new();
    w.put(0x1f, 5);
    w.put(4, 3);
    w.put(1, 8);
    assert_eq!(dec.decode(&w.bytes), Ok(FrameType::Null));
    assert_eq!(dec.transparency().unwrap().rects[0].x, 4);
    assert_eq!(mask_rows(&dec), m);
    // A frame whose mask data ends early fails and keeps the last mask.
    // The mask data is 19 bits: the first value and two codes of 9 bits.
    let mut f =
        transparent_intra_frame(&[(0, 0, 16, 8)], None, run_mask(false, false, 0, &[2, 254]));
    f.truncate(f.len() - 2);
    assert_eq!(dec.decode(&f), Err(Iv50Error::Truncated));
    assert_eq!(dec.transparency().unwrap().rects[0].x, 4);
    assert_eq!(mask_rows(&dec), m);
}

#[test]
fn no_transparency_without_gop_flag() {
    let mut dec = Iv50Decoder::new();
    dec.decode(&intra_frame()).unwrap();
    assert!(dec.transparency().is_none());
}

/// The trailer that `rest` completes after `rects`, as the bytes that
/// follow a frame's last band.
fn trailer(rects: &[(u32, u32, u32, u32)], rest: impl FnOnce(&mut BitWriter)) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put(rects.len() as u32, 3);
    for &(x, y, rw, rh) in rects {
        w.put(x, 16);
        w.put(y, 16);
        w.put(rw, 16);
        w.put(rh, 16);
    }
    rest(&mut w);
    w.bytes
}

/// An inter frame like [`inter_frame`] in a stream with transparency:
/// the trailer follows its bands.
fn transparent_inter_frame(number: u32, trailer: &[u8]) -> Vec<u8> {
    let mut f = inter_frame(number);
    f.extend_from_slice(trailer);
    f
}

#[test]
fn transparency_mask_of_inter_frames() {
    let mut dec = Iv50Decoder::new();
    let f = transparent_intra_frame(&[(0, 0, 16, 8)], None, run_mask(false, true, 0, &[2, 254]));
    dec.decode(&f).unwrap();
    // An inter frame carries a mask of its own, here with the
    // descriptor of the intra frame: 3 opaque, then transparent.
    let t = trailer(&[(0, 0, 16, 8)], run_mask(false, false, 1, &[3, 253]));
    let f = transparent_inter_frame(1, &t);
    assert_eq!(dec.decode(&f), Ok(FrameType::Inter));
    let m = mask_rows(&dec);
    assert_eq!(m[0][..4], [255, 255, 255, 0]);
    assert!(
        m[0][4..]
            .iter()
            .chain(m[1..].iter().flatten())
            .all(|&v| v == 0)
    );
    // An inter frame whose mask is cut short fails and keeps it.
    let f = transparent_inter_frame(2, &t[..t.len() - 2]);
    assert_eq!(dec.decode(&f), Err(Iv50Error::Truncated));
    assert_eq!(mask_rows(&dec), m);
}

#[test]
fn transparency_runs_past_the_end_are_cut() {
    let mut dec = Iv50Decoder::new();
    // 2 + 255 samples for a mask of 256.
    let f = transparent_intra_frame(&[], None, run_mask(false, true, 0, &[2, 255]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    let m = mask_rows(&dec);
    assert_eq!(m[0][..3], [0, 0, 255]);
    assert!(m[1..].iter().flatten().all(|&v| v == 255));
    // Two runs of 255 that keep the value: the second ends past the
    // mask.
    let f = transparent_intra_frame(&[], None, run_mask(false, true, 0, &[0, 0]));
    assert_eq!(dec.decode(&f), Ok(FrameType::Intra));
    assert!(mask_rows(&dec).iter().flatten().all(|&v| v == 0));
}

#[test]
fn transparency_descriptors_that_code_nothing_fail() {
    let mut dec = Iv50Decoder::new();
    let f = transparent_intra_frame(&[], None, run_mask(false, true, 0, &[2, 254]));
    dec.decode(&f).unwrap();
    let m = mask_rows(&dec);
    // A descriptor of no rows, even for a single-value mask.
    let f = transparent_intra_frame(&[], None, |w| {
        w.put(0, 1);
        w.put(1, 1);
        w.put(0, 4);
        w.align();
        w.put(1, 1);
        w.put(1, 1);
    });
    assert_eq!(
        dec.decode(&f),
        Err(Iv50Error::BadHeader("transparency mask descriptor"))
    );
    // One row of no bits: a code of no length, which decodes nothing
    // and must not run without end.
    let f = transparent_intra_frame(&[], None, |w| {
        w.put(0, 1);
        w.put(1, 1);
        w.put(1, 4);
        w.put(0, 4);
        w.align();
        w.put(0, 1);
        w.put(0, 1);
        w.align();
        w.put(0, 1);
    });
    assert_eq!(
        dec.decode(&f),
        Err(Iv50Error::BadHeader("transparency mask code"))
    );
    assert_eq!(mask_rows(&dec), m);
}

#[test]
fn a_frame_without_transparency_drops_the_mask() {
    let mut dec = Iv50Decoder::new();
    // A transparent picture, then one of the same size in a GOP without
    // transparency.
    let f = transparent_intra_frame(&[], None, |w| {
        w.put(0, 2);
        w.align();
        w.put(1, 1);
        w.put(0, 1);
    });
    dec.decode(&f).unwrap();
    dec.decode(&intra_frame()).unwrap();
    assert!(dec.transparency().is_none());
    // A transparency frame that fails keeps the picture without
    // transparency, so there is still no mask to give.
    let mut f = transparent_intra_frame(&[], None, run_mask(false, true, 0, &[2, 254]));
    f.truncate(f.len() - 2);
    assert_eq!(dec.decode(&f), Err(Iv50Error::Truncated));
    assert!(dec.y().is_some());
    assert!(dec.transparency().is_none());
}

/// Every cut and every one-bit change of frames with transparency
/// decodes or fails without a panic, and leaves a mask that covers the
/// picture.
#[test]
fn damaged_transparency_frames_never_panic() {
    let good = transparent_intra_frame(&[(0, 0, 16, 8)], None, run_mask(false, true, 0, &[2, 254]));
    let mut frames = vec![
        good.clone(),
        transparent_intra_frame(
            &[(1, 2, 3, 4)],
            Some(0x10203),
            run_mask(true, true, 1, &[7, 0, 3]),
        ),
        transparent_intra_frame(&[], None, opaque_mask),
    ];
    for long in [false, true] {
        frames.push(transparent_intra_frame(&[], None, move |w| {
            w.put(1, 1);
            w.put(1, 1);
            w.put(MASK_ROWS.len() as u32, 4);
            for r in MASK_ROWS {
                w.put(r as u32, 4);
            }
            w.align();
            w.put(0, 1);
            w.put(1, 1);
            if long {
                w.put(0xff, 8);
                w.put(0x123, 24);
            } else {
                w.put(9, 8);
            }
            w.align();
            w.put(0, 1);
            w.sym(&MASK_ROWS, 30);
            w.sym(&MASK_ROWS, 0);
        }));
    }
    let inter = transparent_inter_frame(
        1,
        &trailer(&[(0, 0, 16, 8)], run_mask(true, false, 1, &[3, 253])),
    );
    let check = |dec: &Iv50Decoder| {
        if let Some(t) = dec.transparency() {
            let m = t.mask;
            assert!(m.stride >= m.width && m.data.len() >= m.stride * m.height);
            assert!(m.data.iter().all(|&v| v == 0 || v == 255));
        }
    };
    let damaged = |f: &[u8]| {
        let mut out: Vec<Vec<u8>> = (0..f.len()).map(|n| f[..n].to_vec()).collect();
        for bit in 0..f.len() * 8 {
            let mut g = f.to_vec();
            g[bit / 8] ^= 1 << (bit % 8);
            out.push(g);
        }
        out
    };
    for f in &frames {
        for g in damaged(f) {
            // On its own, and after a good picture.
            let mut dec = Iv50Decoder::new();
            let _ = dec.decode(&g);
            check(&dec);
            let mut dec = Iv50Decoder::new();
            dec.decode(&good).unwrap();
            let _ = dec.decode(&g);
            check(&dec);
        }
    }
    for g in damaged(&inter) {
        let mut dec = Iv50Decoder::new();
        dec.decode(&good).unwrap();
        let _ = dec.decode(&g);
        check(&dec);
    }
}

#[test]
fn failed_intra_frame_with_a_new_size_leaves_no_picture() {
    let mut dec = Iv50Decoder::new();
    dec.decode(&intra_frame()).unwrap();
    // A different size whose band data is missing: the GOP header is
    // taken, which clears the buffers, and then the frame fails.
    let mut w = BitWriter::new();
    picture_start(&mut w, 0, 1, Some((32, 8)));
    frame_header(&mut w, 0);
    assert_eq!(dec.decode(&w.bytes), Err(Iv50Error::Truncated));
    assert!(dec.y().is_none());
    assert_eq!(dec.decode(&inter_frame(2)), Err(Iv50Error::NoReference));
    // A failed frame of the same size keeps the picture.
    dec.decode(&intra_frame()).unwrap();
    let f = intra_frame();
    assert!(dec.decode(&f[..f.len() - 4]).is_err());
    assert!(dec.y().is_some());
}
