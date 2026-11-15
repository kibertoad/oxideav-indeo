//! Picture start (`spec/01-file-header.md` §3), GOP header, frame header
//! and band header (`spec/02-gop-and-band-layer.md` §1 to §3).

use super::Iv50Error;
use super::bits::Bits;
use super::huff::read_descriptor;

/// Picture sizes of ids 0 to 14 (spec/02 §1.6; `tables/region_10088c90`
/// and `tables/region_10088cd0`), as (width, height).
const PICTURE_SIZES: [(u32, u32); 15] = [
    (640, 480),
    (320, 240),
    (160, 120),
    (704, 480),
    (352, 240),
    (352, 288),
    (176, 144),
    (240, 180),
    (640, 240),
    (704, 240),
    (80, 60),
    (88, 72),
    (0, 0),
    (0, 0),
    (0, 0),
];

/// The picture start fields (spec/01 §3.2 to §3.4).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PictureStart {
    pub frame_type: u32,
    pub frame_number: u32,
}

/// Read the picture start: a 5-bit start code 0x1f, a 3-bit frame type
/// and an 8-bit frame number.
pub(crate) fn read_picture_start(r: &mut Bits) -> Result<PictureStart, Iv50Error> {
    let code = r.read(5);
    let frame_type = r.read(3);
    let frame_number = r.read(8);
    if code != 0x1f || frame_type > 4 {
        return Err(Iv50Error::BadPictureStart);
    }
    Ok(PictureStart {
        frame_type,
        frame_number,
    })
}

/// One band's entry of the GOP header's band info (spec/02 §1.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BandInfo {
    /// Motion vectors in half samples.
    pub half_pel: bool,
    /// Macroblock size in samples.
    pub mb_size: u32,
    /// Block size in samples.
    pub blk_size: u32,
    /// Transform type: the transform id of spec/02 §1.7, 0 to 3 for a
    /// luma band (2D, row, column, none) and 4 to 7 for a chroma band
    /// (spec/06 §2.3 lists the kernels).
    pub xform: u32,
}

/// The GOP header (spec/02 §1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GopHeader {
    pub flags: u32,
    pub luma_bands: usize,
    pub chroma_bands: usize,
    pub width: u32,
    pub height: u32,
    /// Band info for the luma bands, then the chroma bands.
    pub luma_info: Vec<BandInfo>,
    pub chroma_info: Vec<BandInfo>,
    /// gop_flags bit 3: frames carry a transparency trailer.
    pub transparency: bool,
    /// The transparency colour (spec/02 §1.8), when the GOP header
    /// carries one: red in bits 0..7, green in 8..15, blue in 16..23.
    pub colour: Option<u32>,
    /// gop_flags bit 2: the stream has intra frames only (a predicted
    /// frame under such a GOP header is an error, spec/07 §1.1).
    pub intra_only: bool,
}

impl GopHeader {
    /// gop_flags bit 1: the chroma planes are half the luma size in each
    /// direction, not a quarter (spec/02 §1.1, §1.6).
    pub fn half_chroma(&self) -> bool {
        self.flags & 2 != 0
    }

    /// Base-2 logarithm of the luma to chroma size ratio: 1 or 2.
    pub fn chroma_shift(&self) -> u32 {
        if self.half_chroma() { 1 } else { 2 }
    }

    /// Chroma plane size (spec/02 §1.6): `(size + 1) >> 1` with half-size
    /// chroma, `(size + 3) >> 2` otherwise.
    pub fn chroma_size(&self) -> (u32, u32) {
        let s = self.chroma_shift();
        let r = (1 << s) - 1;
        ((self.width + r) >> s, (self.height + r) >> s)
    }
}

/// Read the band info of one band (spec/02 §1.7): bit 0 `mv_res`, bit 1
/// `mb_size_id`, bit 2 `blk_size_id`, bit 3 `trans_flg`; with `trans_flg`
/// set, bits 4 and 5 are `ext_trans` and a 2-bit `end_marker` follows,
/// else bits 4 and 5 are the `end_marker`. `p` is 0 for luma and 1 for
/// chroma.
fn read_band_info(r: &mut Bits, index: usize, p: u32) -> Result<BandInfo, Iv50Error> {
    let v = r.read(6);
    let s = (v >> 1) & 3;
    let class = if v >= 8 {
        // The end marker after `ext_trans` must be 0 (spec/02 §1.7).
        if r.read(2) != 0 {
            return Err(Iv50Error::BadHeader("band info guard bits"));
        }
        ((v >> 4) & 3) + 4 * p
    } else {
        index as u32 + 4 * p
    };
    Ok(BandInfo {
        half_pel: v & 1 != 0,
        mb_size: [16, 8, 8, 4][s as usize],
        blk_size: [8, 8, 4, 4][s as usize],
        xform: class,
    })
}

/// Read the GOP header that follows the picture start of a type 0
/// frame, up to and including the align before the frame header.
pub(crate) fn read_gop(r: &mut Bits) -> Result<GopHeader, Iv50Error> {
    let flags = r.read(8);
    if flags & 1 != 0 {
        // `gop_hdr_size`: informational (spec/02 §1.2).
        r.read(16);
    }
    if flags & 0x20 != 0 {
        // `lock_word`: spec/02 §1.3 and §6 item 1 leave its position
        // and the key check open.
        return Err(Iv50Error::Unsupported("locked stream (gop_flags bit 5)"));
    }
    if flags & 0x40 != 0 {
        // `slice_size_id` (spec/02 §1.4): spec/02 §4.1 and spec/03 §1.1
        // give different tile counts for a decomposed band and neither
        // gives a chroma band's tile size.
        return Err(Iv50Error::Unsupported("multiple tiles (gop_flags bit 6)"));
    }
    if flags & 0x0a == 0x0a {
        // Half-size chroma (spec/02 §1.6) with transparency: how the
        // window of a transparency frame maps to half-size planes is
        // not in the staged specification.
        return Err(Iv50Error::Unsupported(
            "transparency with half-size chroma (gop_flags bits 1 and 3)",
        ));
    }
    let sub = r.read(3);
    if !matches!(sub, 0 | 1 | 2 | 6) {
        return Err(Iv50Error::BadHeader("subsampling code"));
    }
    if sub & 2 != 0 {
        // Codes 2 and 6: seven luma bands (and four chroma bands for 6).
        // spec/06 §3.3 and §3.4 leave the synthesis arithmetic of a
        // second decomposition level, and spec/06 §5.1 the matrix group
        // of a four-band chroma plane, unestablished.
        return Err(Iv50Error::Unsupported(
            "seven-band luma or four-band chroma (subsampling codes 2 and 6)",
        ));
    }
    let luma_bands = ((sub & 3) * 3 + 1) as usize;
    let chroma_bands = ((sub >> 2) * 3 + 1) as usize;
    let size_id = r.read(4);
    let (width, height) = if size_id == 15 {
        let v = r.read(26);
        (v >> 13, v & 0x1fff)
    } else {
        PICTURE_SIZES[size_id as usize]
    };
    if width == 0 || height == 0 {
        return Err(Iv50Error::Unsupported("picture size 0x0"));
    }
    let mut luma_info = Vec::with_capacity(luma_bands);
    for b in 0..luma_bands {
        luma_info.push(read_band_info(r, b, 0)?);
    }
    let mut chroma_info = Vec::with_capacity(chroma_bands);
    for b in 0..chroma_bands {
        chroma_info.push(read_band_info(r, b, 1)?);
    }
    let transparency = flags & 8 != 0;
    let mut colour = None;
    if transparency {
        let t = r.read(4);
        if t & 7 != 0 {
            return Err(Iv50Error::BadHeader("transparency flags"));
        }
        if t & 8 != 0 {
            colour = Some(r.read(24));
        }
    }
    r.align();
    let version_word = r.read(24);
    let version = (version_word >> 19) & 0xf;
    if version != 2 {
        // `value4` of spec/02 §1.9: every staged stream carries 2. The
        // section gives the field no meaning, so other values are
        // refused rather than guessed at.
        return Err(Iv50Error::Unsupported("bitstream version other than 2"));
    }
    if version_word & (1 << 23) != 0 {
        loop {
            let w = r.read(16);
            if w & 0x8000 == 0 || r.overrun() {
                break;
            }
        }
    }
    r.align();
    Ok(GopHeader {
        flags,
        luma_bands,
        chroma_bands,
        width,
        height,
        luma_info,
        chroma_info,
        transparency,
        colour,
        intra_only: flags & 4 != 0,
    })
}

/// The frame header (spec/02 §2).
#[derive(Debug, Clone)]
pub(crate) struct FrameHeader {
    pub flags: u32,
    /// Macroblock Huffman descriptor (preset or custom).
    pub mb_rows: Vec<u8>,
}

/// Read the frame header, up to and including its final align.
pub(crate) fn read_frame_header(r: &mut Bits) -> Result<FrameHeader, Iv50Error> {
    let flags = r.read(8);
    if flags & 1 != 0 {
        // `pic_hdr_size`: not range-checked (spec/02 §2.3).
        r.read(24);
    }
    if flags & 0x10 != 0 {
        // `frm_checksum`: parsed and stored only (spec/08 §7.1).
        r.read(16);
    }
    if flags & 0x20 != 0 {
        // `frm_hdr_ext` (spec/02 §2.5): chunks of an 8-bit length and
        // that many bytes, ended by a length of 0.
        loop {
            let n = r.read(8);
            if n == 0 || r.overrun() {
                break;
            }
            r.skip(n * 8);
        }
    }
    let mut mb_rows = super::huff::MB_PRESETS[7].to_vec();
    if flags & 0x40 != 0 {
        let sel = r.read(3);
        mb_rows = if sel == 7 {
            read_descriptor(r)
        } else {
            super::huff::MB_PRESETS[sel as usize].to_vec()
        };
    }
    // `value5` (spec/02 §2.7).
    r.read(3);
    r.align();
    if r.overrun() {
        return Err(Iv50Error::Truncated);
    }
    Ok(FrameHeader { flags, mb_rows })
}

/// The band header (spec/02 §3).
#[derive(Debug, Clone)]
pub(crate) struct BandHeader {
    pub flags: u32,
    /// Byte length of the whole band, if the frame carries band sizes.
    pub size: Option<u32>,
    /// Symbol swap pairs `(a, b)` (spec/02 §3.4; spec/05 §2.4).
    pub corrections: Vec<(u8, u8)>,
    /// Run/value table slot.
    pub rv_table: usize,
    /// Block Huffman descriptor.
    pub blk_rows: Vec<u8>,
    /// Global quantiser.
    pub gq: u32,
}

impl BandHeader {
    pub fn empty(&self) -> bool {
        self.flags & 1 != 0
    }
    pub fn inherit(&self) -> bool {
        self.flags & 2 != 0
    }
    pub fn has_delta(&self) -> bool {
        self.flags & 4 != 0
    }
    /// The band reads quantiser deltas of its own (bit 2 set, bit 3
    /// clear).
    pub fn reads_delta(&self) -> bool {
        self.flags & 0xc == 4
    }
}

/// Read a band header starting at the reader's (byte-aligned) position,
/// up to and including the align before the first tile.
pub(crate) fn read_band_header(r: &mut Bits, frame_flags: u32) -> Result<BandHeader, Iv50Error> {
    let flags = r.read(8);
    let mut h = BandHeader {
        flags,
        size: None,
        corrections: Vec::new(),
        rv_table: 8,
        blk_rows: super::huff::BLOCK_PRESETS[7].to_vec(),
        gq: 0,
    };
    if flags & 1 != 0 {
        // Empty band: the header is this one byte.
        return Ok(h);
    }
    if frame_flags & 0x80 != 0 {
        h.size = Some(r.read(24));
    }
    if flags & 0x10 != 0 {
        let n = r.read(8);
        if n > 61 {
            return Err(Iv50Error::BadHeader("correction count above 61"));
        }
        for _ in 0..n {
            let a = r.read(8) as u8;
            let b = r.read(8) as u8;
            h.corrections.push((a, b));
        }
    }
    if flags & 0x40 != 0 {
        h.rv_table = r.read(3) as usize;
    }
    if flags & 0x80 != 0 {
        let sel = r.read(3);
        h.blk_rows = if sel == 7 {
            read_descriptor(r)
        } else {
            super::huff::BLOCK_PRESETS[sel as usize].to_vec()
        };
    }
    if r.bit() {
        // `band_checksum`: parsed and stored only (spec/02 §3.7;
        // spec/08 §7.2).
        r.read(16);
    }
    h.gq = r.read(5);
    if flags & 0x20 != 0 {
        // `band_hdr_ext` (spec/02 §3.9): align, then chunks of an 8-bit
        // length, that many bytes and a bit that says whether another
        // chunk follows.
        r.align();
        loop {
            let n = r.read(8);
            r.skip(n * 8);
            if !r.bit() || r.overrun() {
                break;
            }
        }
    }
    r.align();
    if r.overrun() {
        return Err(Iv50Error::Truncated);
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::super::bits::PaddedFrame;
    use super::*;

    /// Fields written least significant bit first.
    #[derive(Default)]
    struct W {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl W {
        fn put(&mut self, value: u32, n: u32) -> &mut Self {
            for i in 0..n {
                if self.bit.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                *self.bytes.last_mut().unwrap() |= (((value >> i) & 1) as u8) << (self.bit % 8);
                self.bit += 1;
            }
            self
        }
        fn align(&mut self) -> &mut Self {
            self.bit = self.bit.next_multiple_of(8);
            self
        }
    }

    fn gop(w: &W) -> Result<GopHeader, Iv50Error> {
        let f = PaddedFrame::new(&w.bytes);
        read_gop(&mut f.reader_at(0))
    }

    /// The tail every GOP header ends with: align, `value1` to `value3`,
    /// `value4 = 2`, no extension, align (spec/02 §1.9).
    fn gop_tail(w: &mut W) {
        w.align().put(2 << 19, 24).align();
    }

    /// The picture sizes equal the staged width and height tables
    /// (spec/02 §1.6), including the 704x480 of id 3.
    #[test]
    fn picture_sizes_equal_the_staged_tables() {
        let col = |csv: &str| -> Vec<u32> {
            csv.lines()
                .skip(1)
                .map(|l| l.split(',').nth(3).unwrap().parse().unwrap())
                .collect()
        };
        let widths = col(include_str!(
            "../../../tests/data/iv50-tables/region_10088c90_u32.csv"
        ));
        let heights = col(include_str!(
            "../../../tests/data/iv50-tables/region_10088cd0_u32.csv"
        ));
        for (id, &(w, h)) in PICTURE_SIZES.iter().enumerate() {
            assert_eq!((w, h), (widths[id], heights[id]), "id {id}");
        }
        assert_eq!(PICTURE_SIZES[3], (704, 480));
    }

    #[test]
    fn picture_start_fields() {
        // 0x1f, type 3, number 0xa5.
        let f = PaddedFrame::new(&[0x1f | (3 << 5), 0xa5]);
        let s = read_picture_start(&mut f.reader_at(0)).unwrap();
        assert_eq!((s.frame_type, s.frame_number), (3, 0xa5));
        for bad in [[0x1e, 0], [0x1f | (5 << 5), 0], [0x1f | (7 << 5), 0]] {
            let f = PaddedFrame::new(&bad);
            assert_eq!(
                read_picture_start(&mut f.reader_at(0)).unwrap_err(),
                Iv50Error::BadPictureStart
            );
        }
    }

    /// Every standard size id, with the header size field present, one
    /// luma and one chroma band.
    #[test]
    fn gop_header_standard_sizes() {
        for id in 0..12u32 {
            let mut w = W::default();
            w.put(1, 8).put(0xbeef, 16).put(0, 3).put(id, 4);
            w.put(0, 6).put(6, 6);
            gop_tail(&mut w);
            let g = gop(&w).unwrap();
            assert_eq!((g.width, g.height), PICTURE_SIZES[id as usize]);
            assert_eq!((g.luma_bands, g.chroma_bands), (1, 1));
            assert_eq!(g.chroma_size(), (g.width.div_ceil(4), g.height.div_ceil(4)));
            assert!(!g.transparency && !g.intra_only);
        }
        // Ids 12 to 14 are 0x0.
        for id in 12..15u32 {
            let mut w = W::default();
            w.put(0, 8).put(0, 3).put(id, 4).put(0, 6).put(6, 6);
            gop_tail(&mut w);
            assert!(matches!(gop(&w), Err(Iv50Error::Unsupported(_))));
        }
    }

    /// Band info (spec/02 §1.7): the size tables, `mv_res`, the default
    /// transform of each band position and an explicit `ext_trans`.
    #[test]
    fn band_info_fields() {
        // Four luma bands with the four size combinations, band 2 with
        // half-sample vectors; one chroma band.
        let mut w = W::default();
        w.put(0, 8).put(1, 3).put(2, 4);
        w.put(0b0000, 6)
            .put(0b0010, 6)
            .put(0b0101, 6)
            .put(0b0110, 6);
        w.put(0b0110, 6);
        gop_tail(&mut w);
        let g = gop(&w).unwrap();
        assert_eq!((g.width, g.height), (160, 120));
        let sizes: Vec<(u32, u32)> = g
            .luma_info
            .iter()
            .map(|b| (b.mb_size, b.blk_size))
            .collect();
        assert_eq!(sizes, [(16, 8), (8, 8), (8, 4), (4, 4)]);
        let half: Vec<bool> = g.luma_info.iter().map(|b| b.half_pel).collect();
        assert_eq!(half, [false, false, true, false]);
        let xf: Vec<u32> = g.luma_info.iter().map(|b| b.xform).collect();
        assert_eq!(xf, [0, 1, 2, 3]);
        assert_eq!(g.chroma_info[0].xform, 4);
        // `trans_flg` with `ext_trans` 2 on the luma band: 8 bits.
        let mut w = W::default();
        w.put(0, 8).put(0, 3).put(2, 4);
        w.put(0b10_1000, 6).put(0, 2).put(6, 6);
        gop_tail(&mut w);
        let g = gop(&w).unwrap();
        assert_eq!(g.luma_info[0].xform, 2);
        assert_eq!(
            g.chroma_info[0],
            BandInfo {
                half_pel: false,
                mb_size: 4,
                blk_size: 4,
                xform: 4
            }
        );
        // A non-zero end marker after `ext_trans`.
        let mut w = W::default();
        w.put(0, 8)
            .put(0, 3)
            .put(2, 4)
            .put(0b10_1000, 6)
            .put(1, 2)
            .put(6, 6);
        gop_tail(&mut w);
        assert_eq!(gop(&w), Err(Iv50Error::BadHeader("band info guard bits")));
    }

    /// `decomp_levels` (spec/02 §1.5) accepts 0, 1, 2 and 6 only.
    #[test]
    fn decomposition_codes() {
        for code in [3u32, 4, 5, 7] {
            let mut w = W::default();
            w.put(0, 8).put(code, 3).put(2, 4);
            assert_eq!(gop(&w), Err(Iv50Error::BadHeader("subsampling code")));
        }
        for code in [2u32, 6] {
            let mut w = W::default();
            w.put(0, 8).put(code, 3).put(2, 4);
            assert!(matches!(gop(&w), Err(Iv50Error::Unsupported(_))));
        }
    }

    /// The transparency block (spec/02 §1.8): three zero bits, a colour
    /// flag and an optional 24-bit colour; then the trailing fields with
    /// an extension of 16-bit words that ends at a clear top bit.
    #[test]
    fn transparency_block_and_extension() {
        let mut w = W::default();
        w.put(8 | 4, 8).put(0, 3).put(2, 4).put(0, 6).put(6, 6);
        w.put(8, 4).put(0x123456, 24);
        w.align().put((2 << 19) | (1 << 23), 24);
        w.put(0x8001, 16).put(0xffff, 16).put(0x7fff, 16);
        w.align();
        // The frame header follows on the next byte.
        w.put(0x40, 8).put(3, 3).put(5, 3).align();
        let f = PaddedFrame::new(&w.bytes);
        let mut r = f.reader_at(0);
        let g = read_gop(&mut r).unwrap();
        assert!(g.transparency && g.intra_only);
        let fh = read_frame_header(&mut r).unwrap();
        assert_eq!(fh.flags, 0x40);
        assert_eq!(fh.mb_rows, super::super::huff::MB_PRESETS[3]);
        assert_eq!(r.byte_pos(), w.bytes.len());
        // Non-zero alignment bits in the transparency block.
        let mut w = W::default();
        w.put(8, 8)
            .put(0, 3)
            .put(2, 4)
            .put(0, 6)
            .put(6, 6)
            .put(1, 4);
        assert_eq!(gop(&w), Err(Iv50Error::BadHeader("transparency flags")));
    }

    /// The frame header (spec/02 §2): every optional field at once.
    #[test]
    fn frame_header_optional_fields() {
        let mut w = W::default();
        w.put(0x01 | 0x10 | 0x20 | 0x40 | 0x80, 8);
        w.put(0x00abcd, 24); // pic_hdr_size
        w.put(0x55aa, 16); // frm_checksum
        w.put(2, 8).put(0x1111, 16).put(1, 8).put(0x22, 8).put(0, 8); // frm_hdr_ext
        w.put(7, 3).put(3, 4).put(1, 4).put(2, 4).put(8, 4); // custom code
        w.put(0, 3).align();
        let f = PaddedFrame::new(&w.bytes);
        let mut r = f.reader_at(0);
        let fh = read_frame_header(&mut r).unwrap();
        assert_eq!(fh.flags, 0xf1);
        assert_eq!(fh.mb_rows, [1, 2, 8]);
        assert_eq!(r.byte_pos(), w.bytes.len());
        // Without bit 6 the macroblock code is preset 7 (spec/04 §1.1).
        let f = PaddedFrame::new(&[0, 0]);
        let fh = read_frame_header(&mut f.reader_at(0)).unwrap();
        assert_eq!(fh.mb_rows, super::super::huff::MB_PRESETS[7]);
        // A header that runs off the frame.
        let f = PaddedFrame::new(&[0x01, 0]);
        assert_eq!(
            read_frame_header(&mut f.reader_at(0)).unwrap_err(),
            Iv50Error::Truncated
        );
    }

    /// The band header (spec/02 §3): every optional field at once, in
    /// order, and the defaults when none is present.
    #[test]
    fn band_header_fields() {
        let mut w = W::default();
        w.put(0x02 | 0x04 | 0x10 | 0x20 | 0x40 | 0x80, 8);
        w.put(0x000123, 24); // band_data_size
        w.put(2, 8).put(4, 8).put(220, 8).put(4, 8).put(6, 8); // swap pairs
        w.put(5, 3); // rv_tab_sel
        w.put(2, 3); // block code preset 2
        w.put(1, 1).put(0xee60, 16); // band_checksum
        w.put(19, 5); // band_glob_quant
        w.align()
            .put(1, 8)
            .put(0x77, 8)
            .put(1, 1)
            .put(0, 8)
            .put(0, 1); // band_hdr_ext
        w.align();
        let f = PaddedFrame::new(&w.bytes);
        let mut r = f.reader_at(0);
        let h = read_band_header(&mut r, 0x80).unwrap();
        assert_eq!(h.size, Some(0x123));
        assert_eq!(h.corrections, [(4, 220), (4, 6)]);
        assert_eq!(h.rv_table, 5);
        assert_eq!(h.blk_rows, super::super::huff::BLOCK_PRESETS[2]);
        assert_eq!(h.gq, 19);
        assert!(h.inherit() && h.has_delta() && h.reads_delta() && !h.empty());
        assert_eq!(r.byte_pos(), w.bytes.len());
        // No optional field: default tables, slot 8 (spec/02 §3.5),
        // block code preset 7 (spec/02 §3.6), two bytes in all.
        let mut w = W::default();
        w.put(0x0c, 8).put(0, 1).put(31, 5).align();
        let f = PaddedFrame::new(&w.bytes);
        let mut r = f.reader_at(0);
        let h = read_band_header(&mut r, 0).unwrap();
        assert_eq!((h.size, h.rv_table, h.gq), (None, 8, 31));
        assert_eq!(h.blk_rows, super::super::huff::BLOCK_PRESETS[7]);
        assert!(h.has_delta() && !h.reads_delta());
        assert_eq!(r.byte_pos(), 2);
        // An empty band is its flag byte alone (spec/02 §3.3), also in a
        // frame that carries band sizes.
        let f = PaddedFrame::new(&[0x01, 0xff, 0xff, 0xff]);
        let mut r = f.reader_at(0);
        assert!(read_band_header(&mut r, 0x80).unwrap().empty());
        assert_eq!(r.bit_pos(), 8);
        // More than 61 swap pairs (spec/02 §3.4).
        let mut w = W::default();
        w.put(0x10, 8).put(62, 8);
        let f = PaddedFrame::new(&w.bytes);
        assert_eq!(
            read_band_header(&mut f.reader_at(0), 0).unwrap_err(),
            Iv50Error::BadHeader("correction count above 61")
        );
        // 61 is accepted.
        let mut w = W::default();
        w.put(0x10, 8).put(61, 8);
        for i in 0..61 {
            w.put(i, 8).put(255 - i, 8);
        }
        w.put(0, 1).put(0, 5).align();
        let f = PaddedFrame::new(&w.bytes);
        let h = read_band_header(&mut f.reader_at(0), 0).unwrap();
        assert_eq!(h.corrections.len(), 61);
        assert_eq!(h.corrections[60], (60, 195));
    }

    /// The header fields of the staged 320x240 keyframe that its notes
    /// and spec/05 §2.4 record.
    #[test]
    fn staged_keyframe_headers() {
        let data: &[u8] = include_bytes!("../../../tests/data/intra-320x240-indeo5.iv50");
        let f = PaddedFrame::new(data);
        let mut r = f.reader_at(0);
        let s = read_picture_start(&mut r).unwrap();
        assert_eq!((s.frame_type, s.frame_number), (0, 0));
        let g = read_gop(&mut r).unwrap();
        assert_eq!(
            (g.width, g.height, g.luma_bands, g.chroma_bands),
            (320, 240, 1, 1)
        );
        let fh = read_frame_header(&mut r).unwrap();
        assert_eq!(fh.flags, 0xdd);
        // Band chain: Y at 20 (1064 bytes), V at 1084 (242), U at 1326.
        assert_eq!(r.byte_pos(), 20);
        let y = read_band_header(&mut r, fh.flags).unwrap();
        assert_eq!(y.size, Some(1064));
        assert_eq!((y.rv_table, y.gq), (4, 9));
        assert_eq!(
            y.corrections,
            [
                (4, 220),
                (4, 6),
                (8, 61),
                (7, 80),
                (5, 23),
                (12, 59),
                (13, 171),
                (15, 100),
                (18, 103)
            ]
        );
        let v = read_band_header(&mut f.reader_at(1084), fh.flags).unwrap();
        assert_eq!(
            (v.size, &v.corrections[..]),
            (Some(242), &[(5u8, 48u8)][..])
        );
        let u = read_band_header(&mut f.reader_at(1326), fh.flags).unwrap();
        assert_eq!(
            (u.size, &u.corrections[..]),
            (Some(298), &[(5u8, 12u8)][..])
        );
        assert_eq!(1326 + 298 + 8, data.len());
    }
}
