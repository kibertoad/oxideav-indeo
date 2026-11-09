//! The two vendored IV50 keyframes decoded with [`Iv50Decoder`]: luma
//! must equal the vendor decoder's YUY2 output sample for sample.

use oxideav_indeo::indeo5::{FrameType, Iv50Decoder};

const EDUC: &[u8] = include_bytes!("data/intra-240x180-educ.iv50");
const EDUC_YUY2: &[u8] = include_bytes!("data/intra-240x180-educ.expected.yuy2");
const INDEO5: &[u8] = include_bytes!("data/intra-320x240-indeo5.iv50");
const INDEO5_YUY2: &[u8] = include_bytes!("data/intra-320x240-indeo5.expected.yuy2");

fn check_luma(frame: &[u8], yuy2: &[u8], w: usize, h: usize) {
    let mut dec = Iv50Decoder::new();
    assert_eq!(dec.decode(frame), Ok(FrameType::Intra));
    assert_eq!(dec.dimensions(), Some((w, h)));
    let y = dec.y().unwrap();
    for r in 0..h {
        let want: Vec<u8> = (0..w).map(|c| yuy2[(r * w + c) * 2]).collect();
        assert_eq!(y.row(r), &want[..], "row {r}");
    }
}

#[test]
fn educ_240x180_luma_matches_vendor() {
    check_luma(EDUC, EDUC_YUY2, 240, 180);
}

#[test]
fn indeo5_320x240_luma_matches_vendor() {
    check_luma(INDEO5, INDEO5_YUY2, 320, 240);
}

#[test]
fn truncated_and_corrupted_frames_never_panic() {
    for len in 0..INDEO5.len() {
        let _ = Iv50Decoder::new().decode(&INDEO5[..len]);
    }
    let mut dec = Iv50Decoder::new();
    for i in 0..INDEO5.len() {
        let mut f = INDEO5.to_vec();
        f[i] ^= 0x5a;
        let _ = dec.decode(&f);
        // Inter frames built from the keyframe exercise the reference
        // paths on the corrupted state.
        f[0] = 0x1f | (1 << 5);
        f[1] = i as u8;
        let _ = dec.decode(&f);
    }
}
