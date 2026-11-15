//! IV50 picture decoder that reproduces the vendor driver's internal
//! 8-bit planes.
//!
//! [`Iv50Decoder`] takes one compressed frame at a time (one AVI video
//! chunk) and keeps the state that spans frames: the GOP header, three
//! sets of band buffers that rotate by frame type (spec/07 §1.2, §1.3), the
//! luma band 0 macroblock records that later bands inherit from (spec/06
//! §5.2), the transparency bounding box, mask and mask code, and the
//! output planes, which persist because a transparency frame only
//! rebuilds its bounding box.
//!
//! A frame is decoded in this order: picture start (spec/01 §3), GOP
//! header for intra frames and frame header (spec/02 §1, §2), every band
//! in bitstream order (luma bands, then the bands of chroma plane 1 (V),
//! then those of chroma plane 2 (U); spec/02 §4.4), the transparency
//! trailer with its rectangles and mask, the buffer rotation and the
//! recombination of the bands into the 8-bit planes (spec/08 §3).
//!
//! # Specification
//!
//! Section numbers in this module's comments (`spec/NN §x.y`) refer to
//! the staged clean-room specification, `docs/video/indeo/indeo5/spec/`
//! in the OxideAV workspace, and `tables/...` to its extracted tables.
//! The staged streams of its `fixtures/` (two intra keyframes and a
//! vendor-encoded sequence of one intra and seven inter frames) decode
//! byte-exact against the vendor driver's output
//! (`tests/indeo5_fixtures.rs`).
//!
//! ## Where the decoder follows the fixtures, not the text
//!
//! The staged text states these differently; the streams only decode
//! the way the code reads them.
//!
//! * **Code descriptors** (spec/04 §1.3, §3.2 call the entries canonical
//!   Huffman code lengths): entry `i` is the width of row `i`; a code is
//!   `i` one-bits, a zero-bit unless the row is the last, then that many
//!   bits, most significant first.
//! * **Macroblock header order** (spec/03 §4.5): coded flag, type bit
//!   (inter frames, band not inheriting), block pattern, quantiser
//!   delta, vertical then horizontal vector delta. The delta is read
//!   only when the pattern is non-zero or frame_flags bit 3 is set.
//! * **Macroblock symbols** use the frame header's macroblock code
//!   (spec/03 §4.2, spec/04 §5.2 and §5.3 name the band's block code).
//! * **Signed symbols**: symbol `s` is entry `s + 1` of
//!   `tables/level_unmap_1009e438` (spec/04 §3.4 indexes it by `s`).
//! * **Block end and escape** are the two sentinel symbols of spec/05
//!   §2.2: rank 0 ends the block, rank 1 escapes to a run symbol and two
//!   6-bit halves of the level index (spec/05 §4.1 and §4.3 describe a
//!   counter-driven end instead).
//!
//! One reading no staged stream arbitrates: spec/02 §1.6 lists the
//! custom picture size as height then width, 13 bits each, and then
//! places the height in the high half of the 26 bits. The decoder
//! follows the field list (and the staged wiki page): the first 13 bits
//! are the height.
//!
//! ## Beyond the staged specification
//!
//! Carried over from the decoder's first implementation and kept;
//! the staged text has no section for them. The staged fixtures
//! exercise the first group, nothing staged exercises the second.
//!
//! * Exercised: intra DC prediction (a running sum per tile, uncoded
//!   intra blocks filled with half of it), the vector and type
//!   inheritance of a chroma band from luma band 0 with its rounding,
//!   the clamp of the macroblock quantiser to 0..23, `value4 == 2` in
//!   the GOP header.
//! * Not exercised: the four-point transform (the `(5, 2)` stage alone,
//!   as spec/06 §2.3 words it), the fill values of an empty band, the
//!   transparency window, the extra block of an odd row of one-block
//!   macroblocks, half-sample vectors, empty tiles, and the edge rules
//!   and rounding of the four-band synthesis. Away from the
//!   edges that synthesis is the filter pair of spec/06 §3.2 applied as
//!   a plain upsample-and-filter, a quarter of the bands' sum truncated
//!   once (a unit test compares the two); frame types 2 and 3 follow
//!   spec/07 §1.5 (also under test).
//!
//! ## Transparency mask
//!
//! The staged text has no section for the frame trailer that carries
//! the rectangles and the mask, and no staged fixture has transparency.
//! The layout (`mask.rs`) was worked out from the Rent-A-Hero movies
//! with transparency and confirmed by measuring the vendor driver's
//! output: on those movies every run-coded mask ends exactly at the end
//! of the picture and at the trailer's size, and the mask equals the
//! pixels that the vendor driver, converting to RGB, fills with the
//! transparency colour ([`Iv50Transparency`]).
//!
//! ## Refused
//!
//! [`Iv50Error::Unsupported`] for what the staged text leaves open:
//! several tiles (spec/02 §4.1 against spec/03 §1.1), a locked stream
//! (spec/02 §1.3), seven-band luma and four-band chroma (spec/06 §3.3,
//! §5.1), a transform whose size differs from the block size and the
//! one-axis and identity transforms of a chroma band (spec/06 §6 item
//! 3), `value4` other than 2 (spec/02 §1.9), transparency together with
//! half-size chroma, a half-size chroma band that inherits from luma
//! band 0 while its macroblocks are not half the size of that band's,
//! and a band quantiser above 23 on an intra macroblock (spec/06 §5.1).

mod band;
mod bits;
mod headers;
mod huff;
mod mask;
mod synth;
mod tables;
mod xform;
mod yuy2;

pub use mask::Iv50Rect;

use band::{B0Rec, BandGeom, BandJob, Bank, TileGeom, build_bank, fill_empty_band};
use bits::PaddedFrame;
use headers::GopHeader;
use huff::Huff;
use synth::{OutPlane, Window};

/// Largest picture area, in luma pixels, the decoder accepts. A custom
/// picture size can ask for up to 8191 by 8191; larger pictures than this
/// are refused before any buffer is allocated.
/// [`Iv50Decoder::with_max_pixels`] lowers the bound.
pub const MAX_PICTURE_PIXELS: u64 = 4096 * 4096;

/// The type of a decoded frame (spec/01 §3.3; spec/07 §1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameType {
    /// Type 0: every macroblock is intra; carries the GOP header. Becomes
    /// the reference of the following type 1 frames.
    Intra,
    /// Type 1: predicted from the last type 0 or 1 frame, and becomes the
    /// reference of the following frames in its place.
    Inter,
    /// Type 2: predicted from the last type 2 frame since the last type 0
    /// or 1 frame (else from that frame). Later type 2 and 3 frames
    /// predict from it, type 0 and 1 frames never do.
    DroppableInterScalability,
    /// Type 3: predicted like a type 2 frame and never used as a
    /// reference.
    DroppableInter,
    /// Type 4, or a predicted frame that repeats the previous frame's
    /// number: no picture data; the picture stays as it was.
    Null,
}

/// Errors of [`Iv50Decoder::decode`]. Every one of them means the frame
/// was not decoded, and the planes keep the previous picture. The one
/// exception is an intra frame whose GOP header changes the picture size
/// or the band counts: the decoder takes the new layout before it decodes
/// the bands, which clears the picture, so after such a frame fails
/// there is none until the next intra frame decodes.
///
/// An intra frame decodes in place into the reference buffers, as the
/// vendor driver's does (spec/07 §1.2). When one fails, the planes keep the previous picture, but
/// the bands it decoded before the error stay in the reference, so the
/// inter frames that follow predict from that mix until the next intra
/// frame decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Iv50Error {
    /// The start code is not 0x1f or the frame type is 5 to 7.
    BadPictureStart,
    /// A header field holds a value the format doesn't allow.
    BadHeader(&'static str),
    /// The frame ends before its data does.
    Truncated,
    /// A macroblock header doesn't decode.
    BadMbData,
    /// Block data doesn't decode.
    BadBlockData(&'static str),
    /// An inter or NULL frame arrived before any intra frame.
    NoReference,
    /// An inter frame in a stream that declares intra frames only.
    InterInIntraOnlyStream,
    /// A feature the specification leaves undetermined.
    Unsupported(&'static str),
}

impl core::fmt::Display for Iv50Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Iv50Error::BadPictureStart => f.write_str("bad picture start code or frame type"),
            Iv50Error::BadHeader(what) => write!(f, "bad header field: {what}"),
            Iv50Error::Truncated => f.write_str("frame data ends early"),
            Iv50Error::BadMbData => f.write_str("bad macroblock data"),
            Iv50Error::BadBlockData(what) => write!(f, "bad block data: {what}"),
            Iv50Error::NoReference => f.write_str("predicted frame before the first intra frame"),
            Iv50Error::InterInIntraOnlyStream => f.write_str("inter frame in an intra-only stream"),
            Iv50Error::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for Iv50Error {}

/// One plane of the decoded picture.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Iv50Plane<'a> {
    /// Visible width in samples.
    pub width: usize,
    /// Visible height in samples.
    pub height: usize,
    /// Distance between rows in `data`.
    pub stride: usize,
    /// The samples, row `r` starting at `r * stride`.
    pub data: &'a [u8],
}

impl Iv50Plane<'_> {
    /// Row `r` of the visible picture, `width` samples; `r` must be below
    /// `height`.
    pub fn row(&self, r: usize) -> &[u8] {
        &self.data[r * self.stride..r * self.stride + self.width]
    }
}

/// The transparency of the current picture, for a stream whose GOP
/// header sets gop_flags bit 3 (spec/02 §1.8 for the colour).
///
/// The planes hold the decoded picture everywhere, transparent pixels
/// included; the mask says which pixels to show. The vendor driver,
/// asked for RGB output (`ICM_DECOMPRESS` to a 16 or 24-bit DIB),
/// writes every pixel: the picture where the mask is opaque and
/// [`colour`](Self::colour) where it is transparent, black when the
/// stream gives no colour. An application keys that colour out, or
/// takes the mask.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Iv50Transparency<'a> {
    /// The GOP header's transparency colour as `[red, green, blue]`, or
    /// `None` when it carries none.
    pub colour: Option<[u8; 3]>,
    /// The rectangles of the last frame's trailer. A frame with none
    /// leaves the part of the picture it rebuilds as it was.
    pub rects: &'a [Iv50Rect],
    /// One sample per luma pixel: 255 where the picture is opaque, 0
    /// where it is transparent.
    pub mask: Iv50Plane<'a>,
}

/// What a frame's transparency trailer gives besides its mask: the
/// rectangles, and the run code when the trailer carries a descriptor.
struct Trailer {
    rects: Vec<Iv50Rect>,
    code: Option<Huff>,
}

/// The geometry of one plane: its size and its bands.
#[derive(Debug, Clone)]
struct PlaneLayout {
    width: usize,
    height: usize,
    bands: Vec<BandGeom>,
    /// Samples in the plane's band area, margins included.
    area_len: usize,
}

/// Samples before the first band and after the last one of a plane
/// area. Band reads that leave the bands (motion vectors, spec/07 §5.4;
/// the neighbour taps of the four-band synthesis) land here and read
/// zero.
const MARGIN: usize = 8192 + 512;
/// Samples kept between the bands of a plane.
const BAND_GAP: usize = 0x200;

/// Stateful IV50 decoder.
pub struct Iv50Decoder {
    gop: Option<GopHeader>,
    layout: Vec<PlaneLayout>,
    /// Three buffer sets, each one plane area per plane.
    sets: [Vec<Vec<i16>>; 3],
    /// Indices into `sets`: W (decoded into), R1, R2, and whether R2
    /// holds a usable frame (spec/07 §1.2, §1.3 and §4.3).
    w: usize,
    r1: usize,
    r2: usize,
    r2_valid: bool,
    b0: Vec<B0Rec>,
    prev_number: Option<u32>,
    /// Output planes in bitstream order: Y, V, U.
    out: Vec<OutPlane>,
    /// Transparency bounding box (y, x, h, w).
    bbox: Option<(usize, usize, usize, usize)>,
    /// The last frame's transparency rectangles and mask
    /// (`mask::stride(width)` samples per row), whether the mask holds
    /// a decoded one, the buffer the next frame's mask decodes into, and
    /// the mask's run code in force.
    rects: Vec<Iv50Rect>,
    mask: Vec<u8>,
    have_mask: bool,
    mask_next: Vec<u8>,
    mask_code: Option<Huff>,
    bank: Box<Bank>,
    /// Largest picture area accepted, in luma pixels.
    max_pixels: u64,
    /// Huffman tables of the macroblock and block layers.
    mb_huffs: Vec<Huff>,
    blk_huffs: Vec<Huff>,
    have_picture: bool,
}

impl Default for Iv50Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for Iv50Decoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Iv50Decoder")
            .field("gop", &self.gop)
            .field("have_picture", &self.have_picture)
            .finish_non_exhaustive()
    }
}

impl Iv50Decoder {
    /// A decoder waiting for its first intra frame.
    pub fn new() -> Self {
        Iv50Decoder {
            gop: None,
            layout: Vec::new(),
            sets: [Vec::new(), Vec::new(), Vec::new()],
            w: 0,
            r1: 1,
            r2: 2,
            r2_valid: false,
            b0: Vec::new(),
            prev_number: None,
            out: Vec::new(),
            bbox: None,
            rects: Vec::new(),
            mask: Vec::new(),
            have_mask: false,
            mask_next: Vec::new(),
            mask_code: None,
            bank: build_bank(),
            max_pixels: MAX_PICTURE_PIXELS,
            mb_huffs: Vec::new(),
            blk_huffs: Vec::new(),
            have_picture: false,
        }
    }

    /// Lower the largest picture area the decoder accepts to `pixels`
    /// luma pixels (never above [`MAX_PICTURE_PIXELS`]). A GOP header
    /// that asks for more is refused before any buffer is allocated, so
    /// this bounds the memory a stream can make the decoder take: about
    /// 8 bytes per luma pixel (three sets of 16-bit band buffers and the
    /// 8-bit planes) plus fixed margins.
    pub fn with_max_pixels(mut self, pixels: u64) -> Self {
        self.max_pixels = pixels.min(MAX_PICTURE_PIXELS);
        self
    }

    /// Picture size from the last GOP header, or `None` before the first
    /// intra frame.
    pub fn dimensions(&self) -> Option<(usize, usize)> {
        self.gop
            .as_ref()
            .map(|g| (g.width as usize, g.height as usize))
    }

    /// The luma plane of the current picture.
    pub fn y(&self) -> Option<Iv50Plane<'_>> {
        self.plane(0)
    }

    /// The U (Cb) plane of the current picture.
    pub fn u(&self) -> Option<Iv50Plane<'_>> {
        self.plane(2)
    }

    /// The V (Cr) plane of the current picture.
    pub fn v(&self) -> Option<Iv50Plane<'_>> {
        self.plane(1)
    }

    /// The current picture as packed `YUY2` (`Y0 U Y1 V`, top row
    /// first, `4 * ceil(width / 2)` bytes per row), the layout the
    /// vendor driver hands a `YUY2` host: luma unchanged, chroma widened
    /// by two in each direction with a truncating two-tap average,
    /// horizontally first (`spec/08-output-reconstruction.md` §3.4a).
    ///
    /// `None` before the first picture, and for a picture with half-size
    /// chroma, whose `YUY2` conversion that section does not cover.
    pub fn yuy2(&self) -> Option<Vec<u8>> {
        let (y, u, v) = (self.y()?, self.u()?, self.v()?);
        if self.gop.as_ref()?.half_chroma() {
            return None;
        }
        Some(yuy2::pack(&y, &u, &v))
    }

    /// How many luma samples one chroma sample spans in each direction:
    /// 4 (the planes of [`u`](Self::u) and [`v`](Self::v) are
    /// `ceil(size / 4)`; gop_flags bit 1 clear) or 2 (`ceil(size / 2)`;
    /// bit 1 set, spec/02 §1.6). `None` before the first GOP header.
    pub fn chroma_ratio(&self) -> Option<usize> {
        self.gop.as_ref().map(|g| 1 << g.chroma_shift())
    }

    /// The transparency of the current picture: `None` when the stream
    /// has none (gop_flags bit 3 clear) or before the first picture. A
    /// NULL frame and a frame that fails keep the last frame's; a frame
    /// of a GOP without transparency drops it.
    pub fn transparency(&self) -> Option<Iv50Transparency<'_>> {
        let gop = self.gop.as_ref()?;
        if !gop.transparency || !self.have_picture || !self.have_mask {
            return None;
        }
        let (w, h) = (gop.width as usize, gop.height as usize);
        Some(Iv50Transparency {
            colour: gop
                .colour
                .map(|c| [c as u8, (c >> 8) as u8, (c >> 16) as u8]),
            rects: &self.rects,
            mask: Iv50Plane {
                width: w,
                height: h,
                stride: mask::stride(w),
                data: &self.mask,
            },
        })
    }

    fn plane(&self, i: usize) -> Option<Iv50Plane<'_>> {
        if !self.have_picture {
            return None;
        }
        let p = &self.out[i];
        Some(Iv50Plane {
            width: p.width,
            height: p.height,
            stride: p.stride,
            data: &p.data,
        })
    }

    /// Decode one compressed frame. Returns the frame's type; the
    /// picture is then in [`y`](Self::y), [`u`](Self::u) and
    /// [`v`](Self::v). A NULL frame leaves the picture as it was.
    pub fn decode(&mut self, data: &[u8]) -> Result<FrameType, Iv50Error> {
        if data.is_empty() {
            // An empty chunk carries no frame: the picture stays.
            if !self.have_picture {
                return Err(Iv50Error::NoReference);
            }
            return Ok(FrameType::Null);
        }
        let frame = PaddedFrame::new(data);
        let mut r = frame.reader_at(0);
        let start = headers::read_picture_start(&mut r)?;
        let mut ftype = start.frame_type;
        // The repeated-number rule (spec/01 §3.4): a non-intra frame with the
        // previous frame's number is a NULL frame.
        if ftype != 0 && self.prev_number == Some(start.frame_number) {
            ftype = 4;
        }
        self.prev_number = Some(start.frame_number);
        if ftype == 4 {
            if !self.have_picture {
                return Err(Iv50Error::NoReference);
            }
            return Ok(FrameType::Null);
        }
        if ftype == 0 {
            let gop = headers::read_gop(&mut r)?;
            if gop.width as u64 * gop.height as u64 > self.max_pixels {
                return Err(Iv50Error::Unsupported(
                    "picture area above the decoder's limit",
                ));
            }
            self.set_gop(gop);
        } else {
            let Some(gop) = &self.gop else {
                return Err(Iv50Error::NoReference);
            };
            if gop.intra_only {
                return Err(Iv50Error::InterInIntraOnlyStream);
            }
            if !self.have_picture {
                return Err(Iv50Error::NoReference);
            }
        }
        let fh = headers::read_frame_header(&mut r)?;
        let saved = (self.w, self.r1, self.r2, self.r2_valid);
        // Buffer rotation before decoding (spec/07 §1.2).
        match ftype {
            0 => {
                std::mem::swap(&mut self.w, &mut self.r1);
                self.r2_valid = false;
            }
            1 => self.r2_valid = false,
            _ => {
                if self.r2_valid {
                    std::mem::swap(&mut self.r1, &mut self.r2);
                }
            }
        }
        let decoded = self.w;
        match self.decode_bands(&frame, r.byte_pos(), ftype, &fh) {
            Ok(Some(t)) => {
                if !t.rects.is_empty() {
                    self.set_bbox(&t.rects);
                }
                self.rects = t.rects;
                if t.code.is_some() {
                    self.mask_code = t.code;
                }
                std::mem::swap(&mut self.mask, &mut self.mask_next);
                self.have_mask = true;
            }
            Ok(None) => {
                // A frame of a GOP without transparency: the last mask
                // doesn't describe its picture, nor the one a later
                // frame that fails leaves in place.
                self.rects.clear();
                self.have_mask = false;
            }
            Err(e) => {
                (self.w, self.r1, self.r2, self.r2_valid) = saved;
                return Err(e);
            }
        }
        // Rotation after decoding (spec/07 §1.3).
        match ftype {
            0 | 1 => std::mem::swap(&mut self.w, &mut self.r1),
            2 => {
                if self.r2_valid {
                    std::mem::swap(&mut self.r1, &mut self.r2);
                }
                std::mem::swap(&mut self.w, &mut self.r2);
                self.r2_valid = true;
            }
            _ => {
                if self.r2_valid {
                    std::mem::swap(&mut self.r1, &mut self.r2);
                }
            }
        }
        self.recombine(decoded);
        self.have_picture = true;
        Ok(match ftype {
            0 => FrameType::Intra,
            1 => FrameType::Inter,
            2 => FrameType::DroppableInterScalability,
            _ => FrameType::DroppableInter,
        })
    }

    /// Take a new GOP header. A change of picture size or band counts
    /// rebuilds and clears every buffer (spec/02 §1.6: the GOP header
    /// overrides the picture size per GOP).
    fn set_gop(&mut self, gop: GopHeader) {
        let rebuild = match &self.gop {
            Some(old) => {
                old.width != gop.width
                    || old.height != gop.height
                    || old.luma_bands != gop.luma_bands
                    || old.chroma_bands != gop.chroma_bands
                    || old.half_chroma() != gop.half_chroma()
            }
            None => true,
        };
        self.layout = build_layout(&gop);
        let b0_count = self.layout[0].bands[0]
            .tiles
            .iter()
            .map(|t| t.cols * t.rows)
            .sum();
        if rebuild {
            for set in &mut self.sets {
                *set = self.layout.iter().map(|p| vec![0i16; p.area_len]).collect();
            }
            (self.w, self.r1, self.r2, self.r2_valid) = (0, 1, 2, false);
            self.b0 = vec![B0Rec::default(); b0_count];
            self.out = self
                .layout
                .iter()
                .map(|p| OutPlane::new(p.width, p.height))
                .collect();
            self.bbox = None;
            self.rects.clear();
            self.have_mask = false;
            self.have_picture = false;
        } else {
            self.b0.resize(b0_count, B0Rec::default());
        }
        self.gop = Some(gop);
    }

    /// A cached Huffman table for a descriptor, from `cache`. Only one
    /// index of a cache is in use at a time, so any entry can make room.
    fn huff_index(cache: &mut Vec<Huff>, rows: &[u8]) -> Result<usize, Iv50Error> {
        if let Some(i) = cache.iter().position(|h| h.rows() == rows) {
            return Ok(i);
        }
        let h = Huff::new(rows).ok_or(Iv50Error::BadHeader("Huffman descriptor"))?;
        if cache.len() >= 16 {
            cache.swap_remove(0);
        }
        cache.push(h);
        Ok(cache.len() - 1)
    }

    /// Decode every band of the frame into buffer set W, starting at
    /// byte `pos`, and in a stream with transparency the trailer after
    /// them, whose mask goes to `mask_next`.
    fn decode_bands(
        &mut self,
        frame: &PaddedFrame,
        mut pos: usize,
        ftype: u32,
        fh: &headers::FrameHeader,
    ) -> Result<Option<Trailer>, Iv50Error> {
        let mb_idx = Self::huff_index(&mut self.mb_huffs, &fh.mb_rows)?;
        let gop = self.gop.as_ref().expect("GOP checked by the caller");
        let b0_mb_size = gop.luma_info[0].mb_size;
        let half_chroma = gop.half_chroma();
        let transparency = gop.transparency;
        let b0 = &self.layout[0].bands[0];
        let b0_drifts =
            b0.info.mb_size == b0.info.blk_size && b0.tiles.iter().any(|t| t.cols % 2 == 1);
        for plane in 0..3 {
            for bi in 0..self.layout[plane].bands.len() {
                if pos >= frame.len() {
                    return Err(Iv50Error::Truncated);
                }
                let mut r = frame.reader_at(pos);
                let hdr = headers::read_band_header(&mut r, fh.flags)?;
                if hdr.empty() {
                    let g = &self.layout[plane].bands[bi];
                    fill_empty_band(&mut self.sets[self.w][plane], g, bi);
                    pos += 1;
                    continue;
                }
                let blk_idx = Self::huff_index(&mut self.blk_huffs, &hdr.blk_rows)?;
                // Split the sets so the current one is mutable and the
                // reference readable.
                let (cur, refp) = split_sets(&mut self.sets, self.w, self.r1);
                let geom = &self.layout[plane].bands[bi];
                let job = BandJob {
                    geom,
                    hdr: &hdr,
                    // A half-size chroma macroblock covers one luma band
                    // 0 macroblock only when it is half its size.
                    foreign_grid: plane != 0 && half_chroma && geom.info.mb_size * 2 != b0_mb_size,
                    is_b0: plane == 0 && bi == 0,
                    frame_type: ftype,
                    frame_flags: fh.flags,
                    b0_mb_size,
                    b0_drifts,
                    mb_huff: &self.mb_huffs[mb_idx],
                    blk_huff: &self.blk_huffs[blk_idx],
                    bank: &self.bank,
                    cur: &mut cur[plane],
                    refp: &refp[plane],
                    b0: &mut self.b0,
                };
                job.decode(&mut r)?;
                pos = match hdr.size {
                    Some(s) => pos + s as usize,
                    None => r.byte_pos(),
                };
            }
        }
        if !transparency {
            return Ok(None);
        }
        let (w, h) = (gop.width as usize, gop.height as usize);
        let mut r = frame.reader_at(pos);
        let t = mask::read_header(&mut r, fh.flags);
        if r.overrun() {
            return Err(Iv50Error::Truncated);
        }
        let code = match &t.descriptor {
            Some(rows) => {
                Some(Huff::new(rows).ok_or(Iv50Error::BadHeader("transparency mask descriptor"))?)
            }
            None => None,
        };
        let stride = mask::stride(w);
        self.mask_next.resize(stride * h, 0);
        mask::read_mask(
            &mut r,
            code.as_ref().or(self.mask_code.as_ref()),
            t.row_delta,
            stride,
            &mut self.mask_next,
        )?;
        Ok(Some(Trailer {
            rects: t.rects,
            code,
        }))
    }

    /// Store the union of the frame's transparency rectangles.
    fn set_bbox(&mut self, rects: &[Iv50Rect]) {
        let x0 = rects.iter().map(|r| r.x).min().unwrap_or(0) as usize;
        let y0 = rects.iter().map(|r| r.y).min().unwrap_or(0) as usize;
        let x1 = rects.iter().map(|r| r.x + r.width).max().unwrap_or(0) as usize;
        let y1 = rects.iter().map(|r| r.y + r.height).max().unwrap_or(0) as usize;
        self.bbox = Some((y0, x0, y1 - y0, x1 - x0));
    }

    /// Recombine the bands of buffer set `set` into the output planes
    /// over the frame's window (spec/08 §1.3, §3).
    fn recombine(&mut self, set: usize) {
        let gop = self.gop.as_ref().expect("GOP present");
        let (pw, ph) = (gop.width as usize, gop.height as usize);
        let (mut y0, mut x0, mut h, mut w) = (0usize, 0usize, ph, pw);
        if gop.transparency
            && let Some((by, bx, bh, bw)) = self.bbox
        {
            let (wy, wx, wh, ww) = (0usize, 0usize, ph, pw);
            y0 = wy.max(by);
            x0 = wx.max(bx);
            h = if wy + wh < by + bh {
                (wh + wy).saturating_sub(y0)
            } else {
                (by + bh).saturating_sub(y0)
            };
            w = if wx + ww < bx + bw {
                (ww + wx).saturating_sub(x0)
            } else {
                (bx + bw).saturating_sub(x0)
            };
            h = h.max(1);
            w = w.max(1);
            // A bounding box that starts below or right of the picture
            // gives a window outside the planes. The vendor driver would write
            // past its planes there; this decoder writes nothing.
            if y0 >= ph || x0 >= pw {
                return;
            }
        }
        for plane in 0..3 {
            let win = if plane == 0 {
                Window { y: y0, x: x0, h, w }
            } else {
                // The chroma samples the luma window covers. (With
                // half-size chroma the window is always the picture:
                // the GOP header reader refuses transparency there.)
                let s = gop.chroma_shift();
                let r = (1usize << s) - 1;
                let (yc, xc) = (y0 >> s, x0 >> s);
                Window {
                    y: yc,
                    x: xc,
                    h: ((y0 + r + h) >> s) - yc,
                    w: ((x0 + r + w) >> s) - xc,
                }
            };
            let lay = &self.layout[plane];
            let area = &self.sets[set][plane];
            // One or four bands: the GOP header reader refuses seven.
            if lay.bands.len() == 1 {
                synth::one_band(area, &lay.bands[0], win, &mut self.out[plane]);
            } else {
                synth::four_band(area, &lay.bands, win, &mut self.out[plane]);
            }
        }
    }
}

/// Whether `data` starts like an IV50 frame: a valid picture start and,
/// for an intra frame, a GOP header and frame header that parse (a
/// header that asks for an unsupported feature still counts), or for a
/// predicted frame a frame header that fits in `data`.
pub(crate) fn looks_like_iv50(data: &[u8]) -> bool {
    let frame = PaddedFrame::new(data);
    let mut r = frame.reader_at(0);
    let Ok(start) = headers::read_picture_start(&mut r) else {
        return false;
    };
    if start.frame_type == 4 {
        return true;
    }
    if start.frame_type == 0 {
        match headers::read_gop(&mut r) {
            Ok(_) => {}
            Err(Iv50Error::Unsupported(_)) => return true,
            Err(_) => return false,
        }
    }
    headers::read_frame_header(&mut r).is_ok()
}

/// Borrow set `a` mutably and set `b` immutably.
fn split_sets(
    sets: &mut [Vec<Vec<i16>>; 3],
    a: usize,
    b: usize,
) -> (&mut Vec<Vec<i16>>, &Vec<Vec<i16>>) {
    assert_ne!(a, b);
    let [s0, s1, s2] = sets;
    match (a, b) {
        (0, 1) => (s0, s1),
        (0, 2) => (s0, s2),
        (1, 0) => (s1, s0),
        (1, 2) => (s1, s2),
        (2, 0) => (s2, s0),
        _ => (s2, s1),
    }
}

/// Band sizes, buffer layout and tiles for a GOP header (spec/02 §1.6,
/// §4; spec/06 §4.2 and §5.1).
fn build_layout(gop: &GopHeader) -> Vec<PlaneLayout> {
    let (lw, lh) = (gop.width as usize, gop.height as usize);
    let (cw, ch) = gop.chroma_size();
    let (cw, ch) = (cw as usize, ch as usize);
    let mut layouts = Vec::with_capacity(3);
    for plane in 0..3 {
        let (pw, ph, infos) = if plane == 0 {
            (lw, lh, &gop.luma_info)
        } else {
            (cw, ch, &gop.chroma_info)
        };
        let n = infos.len();
        let mut bands = Vec::with_capacity(n);
        let mut off = MARGIN;
        for (bi, info) in infos.iter().enumerate() {
            // One or four bands (seven-band planes are refused by the GOP
            // header reader).
            let (bw, bh) = if n == 1 { (pw, ph) } else { (pw >> 1, ph >> 1) };
            let (bw, bh) = (bw.max(1), bh.max(1));
            let pitch = ((2 * bw + 31) & !31) / 2;
            let rows = bh.next_multiple_of(16);
            let matrix = if plane == 0 {
                if n == 1 { 0 } else { bi + 1 }
            } else if n == 1 {
                5
            } else {
                bi + 6
            };
            // One tile covering the band (multi-tile streams are refused
            // by the GOP header reader).
            let mb = info.mb_size as usize;
            let tiles = vec![TileGeom {
                x: 0,
                y: 0,
                cols: bw.div_ceil(mb),
                rows: bh.div_ceil(mb),
                copy_width: pitch,
                rec_base: 0,
            }];
            bands.push(BandGeom {
                height: bh,
                pitch,
                offset: off,
                info: *info,
                matrix,
                tiles,
            });
            off += rows * pitch + BAND_GAP;
        }
        layouts.push(PlaneLayout {
            width: pw,
            height: ph,
            bands,
            area_len: off + MARGIN,
        });
    }
    layouts
}

#[cfg(test)]
mod stream_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod testutil;
