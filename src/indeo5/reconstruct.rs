//! Indeo 5 band reconstruction: macroblock info, coefficient blocks,
//! motion compensation, the 5/3 wavelet recomposition and the buffer
//! rotation across frames.
//!
//! The decoder keeps every band of every plane in a buffer padded to
//! whole macroblocks: the band's width and height rounded up to 16
//! samples for luma and 8 for chroma (the largest macroblock of the
//! plane). Each macroblock writes all of its blocks, also those that
//! reach into the padding, and a motion vector may point into the
//! padding of the reference buffer. Three buffer sets rotate between
//! frames: an INTRA or INTER frame decodes into one of the first two
//! and predicts from the other, and an INTER_SCAL frame decodes into
//! the third, so that the frame after it predicts from it.
//!
//! The behaviour follows what IR50_32.DLL produces as reproduced by
//! FFmpeg's `indeo5` decoder (`libavcodec/indeo5.c`, `ivi.c` and
//! `ivi_dsp.c`), which served as the reference for checking it frame
//! by frame on real streams:
//!
//! * A plane has one band, or (luma only) four bands of one wavelet
//!   level, each `(size + 1) / 2` of the plane.
//! * The four luma bands use their own transforms: band 0 the 2D
//!   Slant, band 1 the row Slant, band 2 the column Slant and band 3
//!   none. Each also has its own dequantisation matrix, and only the
//!   2D Slant bands predict the DC of an intra block from the block
//!   before.
//! * An inter macroblock adds its coded residual to the prediction
//!   from the reference band, and the bands after the first luma band
//!   may take the motion vectors and quantiser deltas of the first luma
//!   band's macroblocks (scaled for the smaller chroma macroblocks).

use super::band::BandHeader;
use super::bitreader::BitReader;
use super::codebook::{Codebook, HuffContext};
use super::decode::{
    BandBlockSet, BandTrace, BlockCoding, BlockRecord, DecodeError, DecodeStats,
    decode_block_stream, resolve_transform, skip_to_byte,
};
use super::frame::FrameHeader;
use super::gop::{BandInfo, GopHeader, TransformId};
use super::header::FrameType;
use super::mc::McError;
use super::output::{ReconstructionPlane, plane_stride};
use super::quant::{GROUP_4X4, QUANT_BASE, QUANT_LEVELS, QUANT_SCALE};
use super::rv_table::RvTable;
use super::tile_header::{TileDataSize, TileHeader};
use super::transform::{
    SCAN_COLUMN_8X8, SCAN_RASTER_4X4, SCAN_RASTER_8X8, SCAN_ZIGZAG_4X4, SCAN_ZIGZAG_8X8,
};

/// The column-major 4x4 scan, the 4-point counterpart of
/// [`SCAN_COLUMN_8X8`] for a row-Slant band of 4x4 blocks.
const SCAN_COLUMN_4X4: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// Quantiser class of the inter matrices in [`QUANT_BASE`] /
/// [`QUANT_SCALE`].
const CLASS_INTER: usize = 0;
/// Quantiser class of the intra matrices.
const CLASS_INTRA: usize = 1;

/// The inverse transform of a band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Xform {
    /// 2D Slant (rows and columns).
    Slant2d,
    /// Slant along each row.
    SlantRow,
    /// Slant along each column.
    SlantCol,
    /// No transform: the coefficients are the samples.
    Put,
}

impl Xform {
    fn from_transform(t: TransformId) -> Self {
        match t {
            TransformId::Slant2d | TransformId::Standard => Xform::Slant2d,
            TransformId::SlantRow => Xform::SlantRow,
            TransformId::SlantColumn => Xform::SlantCol,
            TransformId::None => Xform::Put,
        }
    }

    /// The scan that places a block's coefficients: the zig-zag for
    /// the 2D Slant, column by column for the row Slant (whose
    /// coefficients run down a column first), row by row otherwise.
    fn scan(self, blk_size: usize) -> &'static [u8] {
        match (self, blk_size) {
            (Xform::Slant2d, 8) => &SCAN_ZIGZAG_8X8,
            (Xform::SlantRow, 8) => &SCAN_COLUMN_8X8,
            (_, 8) => &SCAN_RASTER_8X8,
            (Xform::Slant2d, _) => &SCAN_ZIGZAG_4X4,
            (Xform::SlantRow, _) => &SCAN_COLUMN_4X4,
            _ => &SCAN_RASTER_4X4,
        }
    }
}

/// The geometry of one tile in band coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TileGeom {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

/// The decoded parameters of one macroblock. They persist in the
/// decoder across frames, as the reference the later bands of a frame
/// read them from is the first luma band's array.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MbInfo {
    x: usize,
    y: usize,
    /// Offset of the top-left sample in the band buffer.
    offs: usize,
    /// 0 = intra, 1 = inter.
    mb_type: u8,
    /// Coded-block pattern, bit `n` for block `n`.
    cbp: u8,
    q_delta: i32,
    mv_x: i32,
    mv_y: i32,
}

/// The layout of one band.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BandGeom {
    width: usize,
    height: usize,
    /// Row pitch of the band buffer (the width rounded up to the
    /// plane's largest macroblock).
    pitch: usize,
    /// Rows of the band buffer (the height rounded up the same way).
    aheight: usize,
    mb_size: usize,
    blk_size: usize,
    halfpel: bool,
    xform: Xform,
    /// The matrix group of the band's 8x8 or 4x4 blocks.
    quant_group: usize,
    tiles: Vec<TileGeom>,
}

impl BandGeom {
    fn blocks_per_mb(&self) -> usize {
        if self.mb_size != self.blk_size { 4 } else { 1 }
    }

    fn buf_len(&self) -> usize {
        self.pitch * self.aheight
    }
}

/// What the picture layout depends on: when a GOP header changes it,
/// the buffers start over from zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PicConf {
    width: u32,
    height: u32,
    luma_bands: u32,
    chroma_bands: u32,
    tile_size: Option<u32>,
}

/// The decoder state that carries from frame to frame: the band
/// buffers, the macroblock parameters and the buffer rotation.
#[derive(Debug, Clone)]
pub(crate) struct BandState {
    conf: Option<PicConf>,
    /// `[plane][band]`.
    geom: Vec<Vec<BandGeom>>,
    /// `[slot][plane][band]`; an empty buffer is one not used yet,
    /// which reads as zeros.
    bufs: [Vec<Vec<Vec<i16>>>; 3],
    /// `[plane][band][tile][mb]`.
    mbs: Vec<Vec<Vec<Vec<MbInfo>>>>,
    buf_switch: usize,
    dst_buf: usize,
    ref_buf: usize,
    ref2_buf: usize,
    inter_scal: bool,
    prev_frame_type: FrameType,
}

impl Default for BandState {
    fn default() -> Self {
        BandState {
            conf: None,
            geom: Vec::new(),
            bufs: [Vec::new(), Vec::new(), Vec::new()],
            mbs: Vec::new(),
            buf_switch: 0,
            dst_buf: 0,
            ref_buf: 1,
            ref2_buf: 2,
            inter_scal: false,
            prev_frame_type: FrameType::Intra,
        }
    }
}

/// What one frame's band decode produced.
pub(crate) struct BandsOutcome {
    pub recon: Vec<ReconstructionPlane>,
    pub stats: DecodeStats,
    pub band_traces: Vec<BandTrace>,
    pub blocks: Vec<BandBlockSet>,
    /// The checksum of each band's samples, in [`Self::blocks`] order.
    pub band_sums: Vec<u16>,
}

/// `IVI_TOSIGNED`: the sign is in the low bit (0, 1, -1, 2, -2, ...).
#[inline]
fn to_signed(v: u32) -> i32 {
    let v = v as i32;
    if v & 1 != 0 { (v >> 1) + 1 } else { -(v >> 1) }
}

/// Scale a luma macroblock's vector for a band with `scale` times
/// smaller macroblocks, rounding away from zero.
#[inline]
fn scale_mv(mv: i32, scale: u32) -> i32 {
    (mv + i32::from(mv > 0) + (scale as i32 - 1)) >> scale
}

impl BandState {
    /// Take a GOP header's layout. A layout other than the previous
    /// one discards every buffer; a change of block sizes alone
    /// rebuilds the macroblock arrays.
    pub(crate) fn configure(&mut self, gop: &GopHeader) -> Result<(), DecodeError> {
        let luma_bands = gop.decomp.luma_bands;
        let chroma_bands = gop.decomp.chroma_bands;
        if !matches!((luma_bands, chroma_bands), (1, 1) | (4, 1)) {
            return Err(DecodeError::UnsupportedLayout {
                luma_bands,
                chroma_bands,
            });
        }
        let conf = PicConf {
            width: gop.width,
            height: gop.height,
            luma_bands,
            chroma_bands,
            tile_size: gop.slice_size(),
        };
        let mut geom = Vec::with_capacity(3);
        for p in 0..3 {
            let (pw, ph, infos) = if p == 0 {
                (gop.width, gop.height, &gop.luma_band_info)
            } else {
                (gop.chroma_width, gop.chroma_height, &gop.chroma_band_info)
            };
            let (pw, ph) = (pw as usize, ph as usize);
            let n = infos.len();
            let (bw, bh) = if n == 1 {
                (pw, ph)
            } else {
                ((pw + 1) >> 1, (ph + 1) >> 1)
            };
            let align = if p == 0 { 16 } else { 8 };
            // Tile size: the GOP's for luma, a quarter for chroma,
            // and half again for the bands of a wavelet level.
            let (mut tw, mut th) = match conf.tile_size {
                None => (gop.width as usize, gop.height as usize),
                Some(s) => (s as usize, s as usize),
            };
            if p != 0 {
                tw = (tw + 3) >> 2;
                th = (th + 3) >> 2;
            } else if n == 4 {
                tw >>= 1;
                th >>= 1;
            }
            let (tw, th) = (tw.max(1), th.max(1));
            let mut bands = Vec::with_capacity(n);
            for (b, info) in infos.iter().enumerate() {
                bands.push(band_geom(p, b, n, info, (bw, bh), align, (tw, th))?);
            }
            geom.push(bands);
        }

        if self.conf != Some(conf) {
            *self = BandState {
                conf: Some(conf),
                ..BandState::default()
            };
        }
        if self.geom != geom {
            self.mbs = geom
                .iter()
                .map(|bands| {
                    bands
                        .iter()
                        .map(|g| {
                            g.tiles
                                .iter()
                                .map(|t| vec![MbInfo::default(); mbs_per_tile(t, g.mb_size)])
                                .collect()
                        })
                        .collect()
                })
                .collect();
            self.geom = geom;
        }
        Ok(())
    }

    /// Pick the buffers the frame decodes into and predicts from.
    fn switch_buffers(&mut self, frame_type: FrameType) {
        match self.prev_frame_type {
            FrameType::Intra | FrameType::Inter => {
                self.buf_switch ^= 1;
                self.dst_buf = self.buf_switch;
                self.ref_buf = self.buf_switch ^ 1;
            }
            FrameType::DroppableInterScalability => {
                if !self.inter_scal {
                    self.ref2_buf = 2;
                    self.inter_scal = true;
                }
                core::mem::swap(&mut self.dst_buf, &mut self.ref2_buf);
                self.ref_buf = self.ref2_buf;
            }
            FrameType::DroppableInter | FrameType::Null => {}
        }
        match frame_type {
            FrameType::Intra | FrameType::Inter => {
                if frame_type == FrameType::Intra {
                    self.buf_switch = 0;
                }
                self.inter_scal = false;
                self.dst_buf = self.buf_switch;
                self.ref_buf = self.buf_switch ^ 1;
            }
            _ => {}
        }
        self.prev_frame_type = frame_type;
    }

    /// Note a NULL frame, which decodes nothing but takes part in the
    /// buffer rotation.
    pub(crate) fn null_frame(&mut self) {
        self.switch_buffers(FrameType::Null);
    }

    /// Decode one picture-carrying frame's bands and recompose its
    /// planes. The reader is at the first band header.
    pub(crate) fn decode_frame(
        &mut self,
        r: &mut BitReader<'_>,
        frame_type: FrameType,
        frame: &FrameHeader,
    ) -> Result<BandsOutcome, DecodeError> {
        if self.geom.is_empty() {
            return Err(DecodeError::NotIntra { found: frame_type });
        }
        self.switch_buffers(frame_type);
        let (dst, rf) = (self.dst_buf, self.ref_buf);
        if dst == 2 && self.conf.is_some_and(|c| c.luma_bands <= 1) {
            // FFmpeg keeps no third buffer for a picture without
            // wavelet bands, where an INTER_SCAL frame can't occur.
            return Err(DecodeError::UnsupportedLayout {
                luma_bands: 1,
                chroma_bands: 1,
            });
        }
        let mb_cb = Codebook::from_huff_desc(HuffContext::Mb, frame.mb_huff_desc.as_ref())?;
        let band_size_present = frame.flags.band_data_size_present();
        let intra = frame_type == FrameType::Intra;

        let mut stats = DecodeStats::default();
        let mut band_traces = Vec::new();
        let mut blocks = Vec::new();
        let mut band_sums = Vec::new();

        for p in 0..3 {
            for b in 0..self.geom[p].len() {
                let len = self.geom[p][b].buf_len();
                for slot in [dst, rf] {
                    let set = &mut self.bufs[slot];
                    set.resize_with(3, Vec::new);
                    set[p].resize_with(self.geom[p].len(), Vec::new);
                    if set[p][b].len() != len {
                        set[p][b] = vec![0; len];
                    }
                }
                let mut buf = core::mem::take(&mut self.bufs[dst][p][b]);
                let result = self.decode_band(
                    r,
                    (p, b),
                    &mut buf,
                    rf,
                    intra,
                    frame.flags.raw,
                    &mb_cb,
                    band_size_present,
                    &mut stats,
                    &mut band_traces,
                );
                let geom = &self.geom[p][b];
                let sum = band_sum(&buf, geom);
                self.bufs[dst][p][b] = buf;
                let set = result?;
                blocks.push(set);
                band_sums.push(sum);
            }
        }

        let recon = (0..3)
            .map(|p| self.output_plane(p, dst))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BandsOutcome {
            recon,
            stats,
            band_traces,
            blocks,
            band_sums,
        })
    }

    /// The reconstructed plane `p` from buffer set `slot`: the band
    /// itself for a one-band plane, the recomposed wavelet level for a
    /// four-band one. Values are centred on zero.
    fn output_plane(&self, p: usize, slot: usize) -> Result<ReconstructionPlane, DecodeError> {
        let geom = &self.geom[p];
        let bufs = &self.bufs[slot][p];
        let (w, h) = if geom.len() == 1 {
            (geom[0].width, geom[0].height)
        } else {
            let conf = self.conf.expect("configured");
            if p == 0 {
                (conf.width as usize, conf.height as usize)
            } else {
                unreachable!("chroma planes have one band")
            }
        };
        let stride = plane_stride(w as u32) as usize;
        let mut data = vec![0i32; stride * h];
        if geom.len() == 1 {
            let g = &geom[0];
            for y in 0..h {
                for x in 0..w {
                    data[y * stride + x] = i32::from(bufs[0][y * g.pitch + x]);
                }
            }
        } else {
            recompose53(
                [&bufs[0], &bufs[1], &bufs[2], &bufs[3]],
                geom[0].pitch,
                (geom[0].width, geom[0].height),
                (w, h),
                &mut data,
                stride,
            );
        }
        Ok(ReconstructionPlane::new(
            w as u32,
            h as u32,
            stride as u32,
            data,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_band(
        &mut self,
        r: &mut BitReader<'_>,
        (p, b): (usize, usize),
        buf: &mut [i16],
        rf: usize,
        intra: bool,
        frame_flags: u8,
        mb_cb: &Codebook,
        band_size_present: bool,
        stats: &mut DecodeStats,
        band_traces: &mut Vec<BandTrace>,
    ) -> Result<BandBlockSet, DecodeError> {
        let geom = &self.geom[p][b];
        let refb: &[i16] = &self.bufs[rf][p][b];
        r.align()?;
        let band_start = r.byte_pos();
        let header =
            BandHeader::parse(&mut *r, band_size_present).map_err(|error| DecodeError::Band {
                plane_idx: p,
                band_idx: b,
                error,
            })?;
        stats.bands += 1;
        let mut set = BandBlockSet {
            plane_idx: p,
            band_idx: b,
            glob_quant: header.band_glob_quant.unwrap_or(0),
            records: Vec::new(),
            stored_checksum: header.band_checksum,
        };
        if header.empty {
            // FFmpeg refuses an empty band; repeating the reference
            // band is the lenient reading.
            stats.empty_bands += 1;
            buf.copy_from_slice(refb);
            return Ok(set);
        }
        let band_end = header.band_data_size.map(|s| band_start + u64::from(s));
        let blk_cb = Codebook::from_huff_desc(HuffContext::Block, header.blk_huff_desc.as_ref())?;
        let rv = RvTable::for_band(header.rv_tab_sel, &header.rv_tab_corr)?;
        let flags = header.flags;
        let inherit_mv = flags.mv_inherit();
        let qdelta_present = flags.qdelta_present();
        // Without coded deltas a band takes the first luma band's.
        let inherit_qdelta = flags.qdelta_inherit() || !qdelta_present;
        let band = BandCtx {
            p,
            b,
            geom,
            glob_quant: i32::from(set.glob_quant),
            inherit_mv,
            inherit_qdelta,
            qdelta_present,
            intra,
            frame_flags,
            mv_scale: ((self.geom[0][0].mb_size >> 3) as u32)
                .saturating_sub((geom.mb_size >> 3) as u32),
        };

        r.align()?;
        let mut pos = r.bits_read();
        for t in 0..geom.tiles.len() {
            let tile = geom.tiles[t];
            let ref_mbs: Option<Vec<MbInfo>> = if p == 0 && b == 0 {
                None
            } else {
                self.mbs[0][0]
                    .get(t)
                    .filter(|m| m.len() == self.mbs[p][b][t].len())
                    .cloned()
            };
            let mbs = &mut self.mbs[p][b][t];
            let th = TileHeader::parse(&mut *r, false)?;
            stats.tiles += 1;
            match th.size {
                TileDataSize::Empty => {
                    stats.empty_tiles += 1;
                    band.empty_tile(&tile, mbs, ref_mbs.as_deref(), buf, refb, t)?;
                }
                TileDataSize::Implicit | TileDataSize::Explicit(_) => {
                    r.align()?;
                    band.mb_info(r, &tile, mbs, ref_mbs.as_deref(), mb_cb, stats, t)?;
                    band.blocks(r, mbs, buf, refb, &blk_cb, &rv, stats, t, &mut set.records)?;
                    r.align()?;
                    if let TileDataSize::Explicit(n) = th.size {
                        // The size counts the bytes from the tile's
                        // start. FFmpeg refuses a tile that doesn't
                        // fill it exactly; a shorter one is skipped
                        // past as encoder padding.
                        let target = pos + u64::from(n) * 8;
                        skip_to_byte(&mut *r, target / 8)?;
                        pos = target;
                    } else {
                        pos = r.bits_read();
                    }
                }
            }
        }
        r.align()?;
        band_traces.push(BandTrace {
            plane_idx: p,
            band_idx: b,
            consumed: r.byte_pos() - band_start,
            declared: header.band_data_size,
        });
        if let Some(end) = band_end
            && r.byte_pos() < end
        {
            skip_to_byte(&mut *r, end)?;
        }
        Ok(set)
    }
}

/// The geometry of band `b` of plane `p`, with its transform and
/// matrix group.
fn band_geom(
    p: usize,
    b: usize,
    num_bands: usize,
    info: &BandInfo,
    (bw, bh): (usize, usize),
    align: usize,
    (tw, th): (usize, usize),
) -> Result<BandGeom, DecodeError> {
    let blk_size = info.blk_size as usize;
    let mb_size = info.mb_size as usize;
    let xform = Xform::from_transform(resolve_transform(info, b));
    // Luma bands of a wavelet level each have their own 8x8 matrix
    // (groups 1..4), a single luma band group 0. FFmpeg has no matrix
    // for 8x8 chroma blocks; group 0 is the lenient choice.
    let quant_group = if blk_size == 4 {
        GROUP_4X4
    } else if p == 0 && num_bands > 1 {
        b + 1
    } else {
        0
    };
    // FFmpeg pads luma bands to 16 and chroma bands to 8, the largest
    // macroblock it allows in the plane; a larger chroma macroblock,
    // which it refuses, gets padding to its own size.
    let align = align.max(mb_size);
    let mut tiles = Vec::new();
    let mut y = 0;
    while y < bh {
        let mut x = 0;
        while x < bw {
            tiles.push(TileGeom {
                x,
                y,
                width: tw.min(bw - x),
                height: th.min(bh - y),
            });
            x += tw;
        }
        y += th;
    }
    Ok(BandGeom {
        width: bw,
        height: bh,
        pitch: bw.div_ceil(align) * align,
        aheight: bh.div_ceil(align) * align,
        mb_size,
        blk_size,
        halfpel: info.mv_halfpel,
        xform,
        quant_group,
        tiles,
    })
}

fn mbs_per_tile(t: &TileGeom, mb_size: usize) -> usize {
    t.width.div_ceil(mb_size) * t.height.div_ceil(mb_size)
}

/// The band checksum: the sum of the band's samples, modulo 2^16.
fn band_sum(buf: &[i16], g: &BandGeom) -> u16 {
    let mut sum = 0u16;
    for y in 0..g.height {
        for &v in &buf[y * g.pitch..y * g.pitch + g.width] {
            sum = sum.wrapping_add(v as u16);
        }
    }
    sum
}

/// The parameters one band's tiles decode with.
struct BandCtx<'g> {
    p: usize,
    b: usize,
    geom: &'g BandGeom,
    glob_quant: i32,
    inherit_mv: bool,
    inherit_qdelta: bool,
    qdelta_present: bool,
    intra: bool,
    frame_flags: u8,
    mv_scale: u32,
}

impl BandCtx<'_> {
    fn is_first_luma(&self) -> bool {
        self.p == 0 && self.b == 0
    }

    fn inherited_mv(&self, rm: &MbInfo) -> (i32, i32) {
        if self.mv_scale > 0 {
            (
                scale_mv(rm.mv_x, self.mv_scale),
                scale_mv(rm.mv_y, self.mv_scale),
            )
        } else {
            (rm.mv_x, rm.mv_y)
        }
    }

    fn mv_error(&self, t: usize) -> DecodeError {
        DecodeError::Mc {
            plane_idx: self.p,
            band_idx: self.b,
            tile_idx: t,
            error: McError::SourceOutOfBounds,
        }
    }

    /// Whether `mb`'s motion vector keeps its block inside the
    /// reference buffer, the half-sample neighbour included.
    fn mv_in_bounds(&self, mb: &MbInfo) -> bool {
        let g = self.geom;
        let s = i32::from(g.halfpel);
        let (dx, dy) = (mb.mv_x >> s, mb.mv_y >> s);
        let (cx, cy) = (mb.mv_x & s, mb.mv_y & s);
        let (x, y, size) = (mb.x as i64, mb.y as i64, g.mb_size as i64);
        x + i64::from(dx) >= 0
            && x + i64::from(dx) + size + i64::from(cx) <= g.pitch as i64
            && y + i64::from(dy) >= 0
            && y + i64::from(dy) + size + i64::from(cy) <= g.aheight as i64
    }

    /// Parse the macroblock info of a coded tile.
    #[allow(clippy::too_many_arguments)]
    fn mb_info(
        &self,
        r: &mut BitReader<'_>,
        tile: &TileGeom,
        mbs: &mut [MbInfo],
        ref_mbs: Option<&[MbInfo]>,
        mb_cb: &Codebook,
        stats: &mut DecodeStats,
        t: usize,
    ) -> Result<(), DecodeError> {
        let g = self.geom;
        if ref_mbs.is_none() && ((self.qdelta_present && self.inherit_qdelta) || self.inherit_mv) {
            return Err(DecodeError::MissingReferenceMbs {
                plane_idx: self.p,
                band_idx: self.b,
            });
        }
        let vlc =
            |r: &mut BitReader<'_>| -> Result<i32, DecodeError> { Ok(to_signed(mb_cb.decode(r)?)) };
        let (mut mv_x, mut mv_y) = (0i32, 0i32);
        let mut i = 0;
        let mut y = tile.y;
        while y < tile.y + tile.height {
            let mut x = tile.x;
            while x < tile.x + tile.width {
                let rm = ref_mbs.map(|m| m[i]);
                let mut mb = MbInfo {
                    x,
                    y,
                    offs: y * g.pitch + x,
                    ..MbInfo::default()
                };
                stats.mbs += 1;
                if r.read_bit()? == 1 {
                    // An empty macroblock: inter, nothing coded. FFmpeg
                    // refuses one in an intra picture; there it repeats
                    // the running DC like an intra macroblock without
                    // coded blocks.
                    stats.mbs_skipped += 1;
                    mb.mb_type = u8::from(!self.intra);
                    mb.cbp = 0;
                    if self.is_first_luma() && self.frame_flags & 8 != 0 {
                        mb.q_delta = vlc(r)?;
                    }
                    if self.inherit_mv
                        && let Some(rm) = rm
                    {
                        (mb.mv_x, mb.mv_y) = self.inherited_mv(&rm);
                    }
                } else {
                    mb.mb_type = match rm {
                        Some(rm) if self.inherit_mv => rm.mb_type,
                        _ if self.intra => 0,
                        _ => r.read_bit()?,
                    };
                    mb.cbp = r.read(g.blocks_per_mb() as u32)? as u8;
                    if mb.cbp == 0 {
                        stats.mbs_coded_no_ac += 1;
                    }
                    if self.qdelta_present {
                        if self.inherit_qdelta {
                            if let Some(rm) = rm {
                                mb.q_delta = rm.q_delta;
                            }
                        } else if mb.cbp != 0 || (self.is_first_luma() && self.frame_flags & 8 != 0)
                        {
                            mb.q_delta = vlc(r)?;
                        }
                    }
                    if mb.mb_type != 0 {
                        if let (true, Some(rm)) = (self.inherit_mv, rm) {
                            (mb.mv_x, mb.mv_y) = self.inherited_mv(&rm);
                        } else {
                            mv_y += vlc(r)?;
                            mv_x += vlc(r)?;
                            mb.mv_x = mv_x;
                            mb.mv_y = mv_y;
                        }
                    }
                }
                if mb.mb_type != 0 && !self.mv_in_bounds(&mb) {
                    return Err(self.mv_error(t));
                }
                mbs[i] = mb;
                i += 1;
                x += g.mb_size;
            }
            y += g.mb_size;
        }
        r.align()?;
        Ok(())
    }

    /// Decode the blocks of a coded tile into `buf`.
    #[allow(clippy::too_many_arguments)]
    fn blocks(
        &self,
        r: &mut BitReader<'_>,
        mbs: &[MbInfo],
        buf: &mut [i16],
        refb: &[i16],
        blk_cb: &Codebook,
        rv: &RvTable,
        stats: &mut DecodeStats,
        t: usize,
        records: &mut Vec<BlockRecord>,
    ) -> Result<(), DecodeError> {
        let g = self.geom;
        let blk = g.blk_size;
        let n = blk * blk;
        let scan = g.xform.scan(blk);
        let is_2d = g.xform == Xform::Slant2d;
        let mut prev_dc: i32 = 0;
        for mb in mbs {
            let is_intra = mb.mb_type == 0;
            let quant = (self.glob_quant + mb.q_delta).clamp(0, QUANT_LEVELS as i32 - 1) as usize;
            let class = if is_intra { CLASS_INTRA } else { CLASS_INTER };
            let scale = QUANT_SCALE[g.quant_group][class][quant];
            let base = &QUANT_BASE[g.quant_group][class];
            let (mc_type, mv_x, mv_y) = if g.halfpel {
                (
                    ((mb.mv_y & 1) << 1) | (mb.mv_x & 1),
                    mb.mv_x >> 1,
                    mb.mv_y >> 1,
                )
            } else {
                (0, mb.mv_x, mb.mv_y)
            };
            if !is_intra && !self.mv_in_bounds(mb) {
                return Err(self.mv_error(t));
            }
            let mut cbp = mb.cbp;
            for k in 0..g.blocks_per_mb() {
                let offs = mb.offs + blk * ((k & 1) + usize::from(k & 2 != 0) * g.pitch);
                if offs + (blk - 1) * g.pitch + blk > buf.len() {
                    return Err(self.mv_error(t));
                }
                let coded = cbp & 1 != 0;
                cbp >>= 1;
                let mut coeffs = [0i16; 64];
                if coded {
                    stats.coded_blocks += 1;
                    decode_block_stream(
                        r,
                        blk_cb,
                        rv,
                        n as u32,
                        stats,
                        (self.p, self.b, t),
                        &mut coeffs,
                    )?;
                    let mut trvec = [0i32; 64];
                    for (s, &level) in coeffs.iter().enumerate().take(n) {
                        if level == 0 {
                            continue;
                        }
                        let pos = scan[s] as usize;
                        let q = (u32::from(base[pos]) * scale) >> 8;
                        trvec[pos] = dequant(i32::from(level), q as i32);
                    }
                    if is_intra && is_2d {
                        prev_dc = prev_dc.wrapping_add(trvec[0]);
                        trvec[0] = prev_dc;
                    }
                    inverse_transform(g.xform, blk, &trvec, &mut buf[offs..], g.pitch);
                    if !is_intra {
                        mc(
                            buf,
                            offs,
                            refb,
                            offs,
                            (mv_x, mv_y),
                            g.pitch,
                            blk,
                            mc_type,
                            true,
                        )
                        .map_err(|_| self.mv_error(t))?;
                    }
                } else if is_intra {
                    dc_transform(g.xform, blk, prev_dc, &mut buf[offs..], g.pitch);
                } else {
                    mc(
                        buf,
                        offs,
                        refb,
                        offs,
                        (mv_x, mv_y),
                        g.pitch,
                        blk,
                        mc_type,
                        false,
                    )
                    .map_err(|_| self.mv_error(t))?;
                }
                let (bx, by) = ((offs % g.pitch) as u32, (offs / g.pitch) as u32);
                records.push(BlockRecord {
                    x: bx,
                    y: by,
                    blk_size: blk as u32,
                    quant: quant as u8,
                    coding: if coded {
                        BlockCoding::Coded
                    } else if mb.mb_type != 0 && mb.cbp == 0 && mb.mv_x == 0 && mb.mv_y == 0 {
                        BlockCoding::Skipped
                    } else {
                        BlockCoding::DcOnly
                    },
                    coeffs,
                });
            }
        }
        r.align()?;
        Ok(())
    }

    /// An empty tile: its macroblocks predict from the reference band
    /// without residual, with the inherited vectors where the band
    /// takes them, and copy it where none moves.
    fn empty_tile(
        &self,
        tile: &TileGeom,
        mbs: &mut [MbInfo],
        ref_mbs: Option<&[MbInfo]>,
        buf: &mut [i16],
        refb: &[i16],
        t: usize,
    ) -> Result<(), DecodeError> {
        let g = self.geom;
        let clear_first = !self.qdelta_present && self.is_first_luma();
        let mut need_mc = false;
        let mut i = 0;
        let mut y = tile.y;
        while y < tile.y + tile.height {
            let mut x = tile.x;
            while x < tile.x + tile.width {
                let mb = &mut mbs[i];
                mb.x = x;
                mb.y = y;
                mb.offs = y * g.pitch + x;
                mb.mb_type = 1;
                mb.cbp = 0;
                if clear_first {
                    mb.q_delta = self.glob_quant;
                    mb.mv_x = 0;
                    mb.mv_y = 0;
                }
                if let Some(rm) = ref_mbs.map(|m| m[i]) {
                    if self.inherit_qdelta {
                        mb.q_delta = rm.q_delta;
                    }
                    if self.inherit_mv {
                        (mb.mv_x, mb.mv_y) = self.inherited_mv(&rm);
                        need_mc |= mb.mv_x != 0 || mb.mv_y != 0;
                        let mb = *mb;
                        if !self.mv_in_bounds(&mb) {
                            return Err(self.mv_error(t));
                        }
                    }
                }
                i += 1;
                x += g.mb_size;
            }
            y += g.mb_size;
        }
        if self.inherit_mv && need_mc {
            let blk = g.blk_size;
            for mb in mbs.iter() {
                let (mc_type, mv_x, mv_y) = if g.halfpel {
                    (
                        ((mb.mv_y & 1) << 1) | (mb.mv_x & 1),
                        mb.mv_x >> 1,
                        mb.mv_y >> 1,
                    )
                } else {
                    (0, mb.mv_x, mb.mv_y)
                };
                for k in 0..g.blocks_per_mb() {
                    let offs = mb.offs + blk * ((k & 1) + usize::from(k & 2 != 0) * g.pitch);
                    mc(
                        buf,
                        offs,
                        refb,
                        offs,
                        (mv_x, mv_y),
                        g.pitch,
                        blk,
                        mc_type,
                        false,
                    )
                    .map_err(|_| self.mv_error(t))?;
                }
            }
        } else {
            for row in tile.y..tile.y + tile.height {
                let at = row * g.pitch + tile.x;
                buf[at..at + tile.width].copy_from_slice(&refb[at..at + tile.width]);
            }
        }
        Ok(())
    }
}

/// Dequantise a level at step `q`: `level * q` moved half a step away
/// from zero, less one for an odd step. Steps 0 and 1 leave the level
/// as it is.
#[inline]
fn dequant(level: i32, q: i32) -> i32 {
    if q > 1 {
        level
            .wrapping_mul(q)
            .wrapping_add(level.signum() * (((q ^ 1) - 1) >> 1))
    } else {
        level
    }
}

/// One 8-point inverse Slant over `s` (natural coefficient order),
/// each output passed through `fin`.
#[inline]
fn slant8(s: [i32; 8], fin: impl Fn(i32) -> i32) -> [i32; 8] {
    let [s1, s4, s8, s5, s2, s6, s3, s7] = s;
    let t4 = s5.wrapping_add((s4.wrapping_mul(4).wrapping_sub(s5).wrapping_add(4)) >> 3);
    let t5 = s4.wrapping_add(
        (s4.wrapping_neg()
            .wrapping_sub(s5.wrapping_mul(4))
            .wrapping_add(4))
            >> 3,
    );
    let (t1, t5) = (s1.wrapping_add(t5), s1.wrapping_sub(t5));
    let (t2, t6) = (s2.wrapping_add(s6), s2.wrapping_sub(s6));
    let (t7, t3) = (s7.wrapping_add(s3), s7.wrapping_sub(s3));
    let (t4, t8) = (t4.wrapping_add(s8), t4.wrapping_sub(s8));
    let (t1, t2) = (t1.wrapping_add(t2), t1.wrapping_sub(t2));
    let (t4, t3) = reflect(t4, t3);
    let (t5, t6) = (t5.wrapping_add(t6), t5.wrapping_sub(t6));
    let (t8, t7) = reflect(t8, t7);
    let (t1, t4) = (t1.wrapping_add(t4), t1.wrapping_sub(t4));
    let (t2, t3) = (t2.wrapping_add(t3), t2.wrapping_sub(t3));
    let (t5, t8) = (t5.wrapping_add(t8), t5.wrapping_sub(t8));
    let (t6, t7) = (t6.wrapping_add(t7), t6.wrapping_sub(t7));
    [t1, t2, t3, t4, t5, t6, t7, t8].map(fin)
}

/// One 4-point inverse Slant.
#[inline]
fn slant4(s: [i32; 4], fin: impl Fn(i32) -> i32) -> [i32; 4] {
    let [s1, s4, s2, s3] = s;
    let (t1, t2) = (s1.wrapping_add(s2), s1.wrapping_sub(s2));
    let (t4, t3) = reflect(s4, s3);
    let (t1, t4) = (t1.wrapping_add(t4), t1.wrapping_sub(t4));
    let (t2, t3) = (t2.wrapping_add(t3), t2.wrapping_sub(t3));
    [t1, t2, t3, t4].map(fin)
}

/// The (1/2, 5/4) reflection of the Slant butterflies.
#[inline]
fn reflect(a: i32, b: i32) -> (i32, i32) {
    let o1 = ((a.wrapping_add(b.wrapping_mul(2)).wrapping_add(2)) >> 2).wrapping_add(a);
    let o2 = ((a.wrapping_mul(2).wrapping_sub(b).wrapping_add(2)) >> 2).wrapping_sub(b);
    (o1, o2)
}

#[inline]
fn halve(x: i32) -> i32 {
    x.wrapping_add(1) >> 1
}

/// A 1D inverse Slant of `n` points.
fn slant_n(n: usize, v: &[i32], fin: fn(i32) -> i32) -> [i32; 8] {
    let mut out = [0i32; 8];
    if n == 8 {
        out = slant8(v[..8].try_into().expect("8 points"), fin);
    } else {
        out[..4].copy_from_slice(&slant4(v[..4].try_into().expect("4 points"), fin));
    }
    out
}

/// Inverse-transform an `n`x`n` block of dequantised coefficients
/// (raster order, stride `n`) into `out` at row pitch `pitch`.
fn inverse_transform(x: Xform, n: usize, coef: &[i32; 64], out: &mut [i16], pitch: usize) {
    let mut res = [0i32; 64];
    let col = |src: &[i32; 64], c: usize| -> [i32; 8] {
        let mut v = [0i32; 8];
        for (k, e) in v.iter_mut().enumerate().take(n) {
            *e = src[k * n + c];
        }
        v
    };
    match x {
        Xform::Slant2d => {
            let mut tmp = [0i32; 64];
            for c in 0..n {
                let o = slant_n(n, &col(coef, c), |v| v);
                for k in 0..n {
                    tmp[k * n + c] = o[k];
                }
            }
            for row in 0..n {
                let o = slant_n(n, &tmp[row * n..row * n + n], halve);
                res[row * n..row * n + n].copy_from_slice(&o[..n]);
            }
        }
        Xform::SlantRow => {
            for row in 0..n {
                let o = slant_n(n, &coef[row * n..row * n + n], halve);
                res[row * n..row * n + n].copy_from_slice(&o[..n]);
            }
        }
        Xform::SlantCol => {
            for c in 0..n {
                let o = slant_n(n, &col(coef, c), halve);
                for k in 0..n {
                    res[k * n + c] = o[k];
                }
            }
        }
        Xform::Put => res[..n * n].copy_from_slice(&coef[..n * n]),
    }
    for row in 0..n {
        for c in 0..n {
            out[row * pitch + c] = res[row * n + c] as i16;
        }
    }
}

/// The block of an intra block without coded coefficients: the
/// transform of a lone DC coefficient `dc`, which is zero in every
/// band but a 2D Slant one.
fn dc_transform(x: Xform, n: usize, dc: i32, out: &mut [i16], pitch: usize) {
    let half = halve(dc) as i16;
    for row in 0..n {
        for c in 0..n {
            out[row * pitch + c] = match x {
                Xform::Slant2d => half,
                Xform::SlantRow if row == 0 => half,
                Xform::SlantCol if c == 0 => half,
                Xform::Put if row == 0 && c == 0 => dc as i16,
                _ => 0,
            };
        }
    }
}

/// Motion-compensate an `n`x`n` block: the reference block displaced
/// by `(mv_x, mv_y)` whole samples plus the half-sample `mc_type`
/// (bit 0 horizontal, bit 1 vertical), stored or added at `offs`.
#[allow(clippy::too_many_arguments)]
fn mc(
    buf: &mut [i16],
    offs: usize,
    refb: &[i16],
    base: usize,
    (mv_x, mv_y): (i32, i32),
    pitch: usize,
    n: usize,
    mc_type: i32,
    add: bool,
) -> Result<(), ()> {
    let ref_offs = base as i64 + i64::from(mv_y) * pitch as i64 + i64::from(mv_x);
    let extra = usize::from(mc_type > 1) * pitch + usize::from(mc_type & 1 != 0);
    if ref_offs < 0 || ref_offs as usize + (n - 1) * pitch + n + extra > refb.len() {
        return Err(());
    }
    let ro = ref_offs as usize;
    for row in 0..n {
        for c in 0..n {
            let a = ro + row * pitch + c;
            let v = match mc_type {
                0 => i32::from(refb[a]),
                1 => (i32::from(refb[a]) + i32::from(refb[a + 1])) >> 1,
                2 => (i32::from(refb[a]) + i32::from(refb[a + pitch])) >> 1,
                _ => {
                    (i32::from(refb[a])
                        + i32::from(refb[a + 1])
                        + i32::from(refb[a + pitch])
                        + i32::from(refb[a + pitch + 1]))
                        >> 2
                }
            };
            let d = &mut buf[offs + row * pitch + c];
            *d = if add {
                (i32::from(*d) + v) as i16
            } else {
                v as i16
            };
        }
    }
    Ok(())
}

/// Recompose a plane from the four bands of one 5/3 wavelet level:
/// LL low-passed both ways, HL high-passed vertically, LH horizontally
/// and HH both ways. Each 2x2 output block comes from the band samples
/// at its position and their right, lower and upper neighbours; at the
/// band's edges a missing neighbour repeats the edge sample. The
/// output is `sum >> 6`, centred on zero.
fn recompose53(
    bands: [&[i16]; 4],
    pitch: usize,
    (bw, bh): (usize, usize),
    (w, h): (usize, usize),
    out: &mut [i32],
    stride: usize,
) {
    let at = |b: usize, x: usize, y: usize| -> i32 { i32::from(bands[b][y * pitch + x]) };
    for j in 0..h.div_ceil(2) {
        let y = 2 * j;
        let row = j.min(bh - 1);
        let up = if j == 0 { row } else { row - 1 };
        let down = if y + 2 >= h {
            row
        } else {
            (row + 1).min(bh - 1)
        };

        // The values carried from column to column.
        let mut b0_1 = at(0, 0, row);
        let mut b0_2 = at(0, 0, down);
        let mut b1_1 = at(1, 0, up);
        let mut b1_2 = at(1, 0, row);
        let mut b1_3 = b1_1 - b1_2 * 6 + at(1, 0, down);
        let mut b2_2 = at(2, 0, row);
        let mut b2_3 = b2_2;
        let mut b2_5 = at(2, 0, down);
        let mut b2_6 = b2_5;
        let mut b3_2 = at(3, 0, up);
        let mut b3_3 = b3_2;
        let mut b3_5 = at(3, 0, row);
        let mut b3_6 = b3_5;
        let mut b3_8 = b3_2 - b3_5 * 6 + at(3, 0, down);
        let mut b3_9 = b3_8;

        for i in 0..w.div_ceil(2) {
            let x = 2 * i;
            // The next column, or this one at the right edge.
            let nx = if x + 2 >= w { i } else { i + 1 }.min(bw - 1);

            let b2_1 = b2_2;
            b2_2 = b2_3;
            let b2_4 = b2_5;
            b2_5 = b2_6;
            let b3_1 = b3_2;
            b3_2 = b3_3;
            let b3_4 = b3_5;
            b3_5 = b3_6;
            let b3_7 = b3_8;
            b3_8 = b3_9;

            // LL: low-pass both ways.
            let tmp0 = b0_1;
            let tmp2 = b0_2;
            b0_1 = at(0, nx, row);
            b0_2 = at(0, nx, down);
            let tmp1 = tmp0 + b0_1;
            let mut p0 = tmp0 * 16;
            let mut p1 = tmp1 * 8;
            let mut p2 = (tmp0 + tmp2) * 8;
            let mut p3 = (tmp1 + tmp2 + b0_2) * 4;

            // HL: high-pass vertically, low-pass horizontally.
            let tmp0 = b1_2;
            let tmp1 = b1_1;
            b1_2 = at(1, nx, row);
            b1_1 = at(1, nx, up);
            let tmp2 = tmp1 - tmp0 * 6 + b1_3;
            b1_3 = b1_1 - b1_2 * 6 + at(1, nx, down);
            p0 += (tmp0 + tmp1) * 8;
            p1 += (tmp0 + tmp1 + b1_1 + b1_2) * 4;
            p2 += tmp2 * 4;
            p3 += (tmp2 + b1_3) * 2;

            // LH: low-pass vertically, high-pass horizontally.
            b2_3 = at(2, nx, row);
            b2_6 = at(2, nx, down);
            let tmp0 = b2_1 + b2_2;
            let tmp1 = b2_1 - b2_2 * 6 + b2_3;
            p0 += tmp0 * 8;
            p1 += tmp1 * 4;
            p2 += (tmp0 + b2_4 + b2_5) * 4;
            p3 += (tmp1 + b2_4 - b2_5 * 6 + b2_6) * 2;

            // HH: high-pass both ways.
            b3_6 = at(3, nx, row);
            b3_3 = at(3, nx, up);
            let tmp0 = b3_1 + b3_4;
            let tmp1 = b3_2 + b3_5;
            let tmp2 = b3_3 + b3_6;
            b3_9 = b3_3 - b3_6 * 6 + at(3, nx, down);
            p0 += (tmp0 + tmp1) * 4;
            p1 += (tmp0 - tmp1 * 6 + tmp2) * 2;
            p2 += (b3_7 + b3_8) * 2;
            p3 += b3_7 - b3_8 * 6 + b3_9;

            out[y * stride + x] = p0 >> 6;
            if x + 1 < w {
                out[y * stride + x + 1] = p1 >> 6;
            }
            if y + 1 < h {
                out[(y + 1) * stride + x] = p2 >> 6;
                if x + 1 < w {
                    out[(y + 1) * stride + x + 1] = p3 >> 6;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_steps_leave_the_level() {
        assert_eq!(dequant(5, 0), 5);
        assert_eq!(dequant(-5, 1), -5);
        // Even step: level * q + q / 2 away from zero.
        assert_eq!(dequant(3, 4), 14);
        assert_eq!(dequant(-3, 4), -14);
        // Odd step: one less.
        assert_eq!(dequant(1, 5), 6);
        assert_eq!(dequant(-1, 5), -6);
    }

    #[test]
    fn chroma_vectors_scale_by_halves() {
        assert_eq!(scale_mv(5, 1), 3);
        assert_eq!(scale_mv(4, 1), 2);
        assert_eq!(scale_mv(-5, 1), -3);
        assert_eq!(scale_mv(-4, 1), -2);
        assert_eq!(scale_mv(0, 1), 0);
        assert_eq!(scale_mv(6, 2), 2);
    }

    #[test]
    fn buffers_rotate_as_the_reference_decoder_does() {
        // (frame type, buffer decoded into, buffer predicted from):
        // INTRA and INTER alternate between buffers 0 and 1; an
        // INTER_SCAL frame borrows buffer 2 for the frame after it.
        let mut s = BandState::default();
        let seq = [
            (FrameType::Intra, 0, 1),
            (FrameType::Inter, 1, 0),
            (FrameType::Inter, 0, 1),
            (FrameType::DroppableInter, 1, 0),
            (FrameType::Inter, 1, 0),
            (FrameType::DroppableInterScalability, 0, 1),
            (FrameType::DroppableInterScalability, 2, 0),
            (FrameType::Inter, 0, 1),
            (FrameType::Null, 1, 0),
            (FrameType::Inter, 1, 0),
            (FrameType::Intra, 0, 1),
        ];
        for (i, (ft, dst, rf)) in seq.into_iter().enumerate() {
            s.switch_buffers(ft);
            assert_eq!((s.dst_buf, s.ref_buf), (dst, rf), "frame {i} {ft:?}");
        }
    }

    #[test]
    fn flat_ll_recomposes_to_a_quarter() {
        // A flat LL band of 40 and empty high bands give 40 * 16 >> 6
        // = 10 everywhere, odd picture sizes included.
        for (w, h) in [(7usize, 5usize), (8, 8), (1, 1), (13, 2)] {
            let (bw, bh) = (w.div_ceil(2), h.div_ceil(2));
            let pitch = 16;
            let ll = vec![40i16; pitch * bh.max(1) + pitch];
            let zero = vec![0i16; ll.len()];
            let stride = plane_stride(w as u32) as usize;
            let mut out = vec![0i32; stride * h];
            recompose53(
                [&ll, &zero, &zero, &zero],
                pitch,
                (bw, bh),
                (w, h),
                &mut out,
                stride,
            );
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(out[y * stride + x], 10, "{w}x{h} ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn slant_of_a_lone_dc_is_flat() {
        // The 2D Slant of DC d alone fills the block with (d + 1) >> 1,
        // what an uncoded intra block of a 2D band repeats.
        for n in [4usize, 8] {
            let mut coef = [0i32; 64];
            coef[0] = 9;
            let mut out = vec![0i16; 8 * 8];
            inverse_transform(Xform::Slant2d, n, &coef, &mut out, 8);
            let mut dc = vec![0i16; 8 * 8];
            dc_transform(Xform::Slant2d, n, 9, &mut dc, 8);
            assert_eq!(out, dc);
            assert!(
                out.chunks(8)
                    .take(n)
                    .all(|r| r[..n].iter().all(|&v| v == 5))
            );
        }
    }
}
