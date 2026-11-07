//! Indeo 5 whole-frame decode driver.
//!
//! Spec source: `docs/video/indeo/indeo5/spec/01`..`spec/08` — the
//! per-frame thread of `spec/02 §4.4`:
//!
//! ```text
//! picture header (spec/01 + spec/02 §1/§2)
//!   for each plane band chain:
//!     for each band:
//!       band header (spec/02 §3)          — empty band -> zeros
//!       for each tile in raster order:
//!         per-tile size header (spec/03 §2) — empty tile -> zeros
//!         per-MB header phase (spec/03 §3/§4)
//!         per-block coefficient phase (spec/05)
//!     wavelet recompose (spec/06 §3)
//!   bias-and-clamp + planar pack (spec/08) -> HostBuffer
//! ```
//!
//! ## Tile phase split (fixture-arbitrated, r388)
//!
//! Within one coded tile **all per-MB headers come first, then all
//! coded blocks' coefficient streams** — the Indeo 4 wiki "Bitstream
//! organization" split ("Macroblocks info data" then "Blocks data"),
//! shared by Indeo 5. This two-phase layout (with the CBP-before-
//! qdelta field order, [`super::MbHeader`]) is the only arrangement
//! under which both staged `IV50` INTRA fixtures decode every band to
//! byte-exact exhaustion; an interleaved per-MB reading does not
//! decode the fixtures.
//!
//! The per-block `(run, val)` streams decode through the band's
//! block-Huffman codebook ([`super::Codebook`], prefix form) and the
//! band's rv-table ([`super::RvTable`], the r338 static slots +
//! swap corrections). §2.8 explicit tile sizes are confirmed to span
//! the whole tile from its first byte (three independent byte-exact
//! chains in the 320x240 fixture), and the driver repositions on them.
//!
//! **Reconstruction.** The bands decode into the padded band buffers
//! of [`super::reconstruct::BandState`], which follows FFmpeg's
//! `indeo5` decoder (`libavcodec/indeo5.c`, `ivi.c`, `ivi_dsp.c`): the
//! per-band transform and scan, the quantiser matrix of each band, the
//! dequantisation, the DC chain of intra blocks in 2D Slant bands,
//! inter residuals added to the motion-compensated prediction, motion
//! vectors inherited from the first luma band, the three-buffer
//! rotation and the 5/3 wavelet recomposition. The output matches
//! FFmpeg's on every frame of the real `IV50` streams it was checked
//! against, custom picture sizes and four-band luma included.

use super::assemble::{AssembleError, assemble_frame};
use super::band::BandError;
use super::bitreader::{BitReader, BitReaderError};
use super::codebook::{Codebook, CodebookError};
use super::format::OutputFormat;
use super::frame::FrameHeader;
use super::gop::{BandInfo, GopHeader, Subsampling, TransformId};
use super::header::FrameType;
use super::mb_header::MbHeaderError;
use super::mc::McError;
use super::output::{OutputError, ReconstructionPlane};
use super::pack::HostBuffer;
use super::picture::{PictureError, PictureHeader};
use super::reconstruct::BandState;
use super::rv_table::{RvEntry, RvTable, RvTableError, escape_lindex, escape_value, run_advance};

/// The largest picture area the driver allocates band buffers for
/// (4096 × 4096 samples): a resource guard against hostile headers,
/// far above any Indeo 5 stream's real size.
pub const MAX_PICTURE_PIXELS: u64 = 4096 * 4096;

/// Errors raised by the whole-frame driver.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Fault in the spec/01/spec/02 picture-header stack.
    Picture(PictureError),
    /// The driver currently decodes INTRA (and NULL) frames only;
    /// INTER frames need the session-carried GOP state + reference
    /// planes (`spec/07`).
    NotIntra {
        /// The frame type found.
        found: FrameType,
    },
    /// Fault in a band header.
    Band {
        /// Plane index (0 = Y; 1 = the V chain, 2 = the U chain —
        /// `YVU9` coded order, `spec/02 §4.4`).
        plane_idx: usize,
        /// Band index within the plane.
        band_idx: usize,
        /// The underlying error.
        error: BandError,
    },
    /// A codebook descriptor failed to build (`spec/04 §1`).
    Codebook(CodebookError),
    /// A band's `rv_tab_sel` was invalid (`spec/02 §3.5`).
    RvTable(RvTableError),
    /// Fault in a per-MB header.
    MbHeader {
        /// Plane index.
        plane_idx: usize,
        /// Band index.
        band_idx: usize,
        /// Tile index in raster order.
        tile_idx: usize,
        /// The underlying error.
        error: MbHeaderError,
    },
    /// A per-block coefficient stream faulted (`spec/05`).
    BlockStream {
        /// Plane index.
        plane_idx: usize,
        /// Band index.
        band_idx: usize,
        /// Tile index in raster order.
        tile_idx: usize,
        /// Description of the fault.
        reason: BlockStreamFault,
    },
    /// An explicit tile size (`spec/03 §2.8`) or `band_data_size`
    /// (`spec/02 §3.2`) placed the next-payload target *behind* the
    /// parser's position — the stream and its size fields disagree.
    SizeFieldBehindCursor {
        /// The parser's byte position.
        at_byte: u64,
        /// The size field's target byte.
        target_byte: u64,
    },
    /// An INTER frame's reference band buffers did not match the
    /// GOP-derived band geometry (`spec/07 §1.2` workspace contract).
    ReferenceMismatch {
        /// Plane index.
        plane_idx: usize,
        /// Band index.
        band_idx: usize,
    },
    /// A motion-compensated fetch fell outside the reference band
    /// (`spec/07 §5.4` padding contract violated by the MV).
    Mc {
        /// Plane index.
        plane_idx: usize,
        /// Band index.
        band_idx: usize,
        /// Tile index in raster order.
        tile_idx: usize,
        /// The underlying error.
        error: McError,
    },
    /// The picture area exceeds [`MAX_PICTURE_PIXELS`] — a decoder
    /// resource guard, not a format limit (the custom-dimension fields
    /// can name 8191×8191, which would cost gigabytes of band buffers).
    PictureTooLarge {
        /// The picture width.
        width: u32,
        /// The picture height.
        height: u32,
    },
    /// The GOP header asks for a band layout the decoder doesn't
    /// reconstruct: it takes one band per plane, or four luma bands
    /// (one wavelet level) with one band per chroma plane.
    UnsupportedLayout {
        /// Luma bands.
        luma_bands: u32,
        /// Bands of each chroma plane.
        chroma_bands: u32,
    },
    /// A band takes macroblock vectors or quantiser deltas from the
    /// first luma band, whose tiles don't have the same macroblocks.
    MissingReferenceMbs {
        /// Plane index.
        plane_idx: usize,
        /// Band index.
        band_idx: usize,
    },
    /// Underlying bit-reader fault.
    BitReader(BitReaderError),
    /// Reconstruction-plane geometry fault (`spec/08`).
    Output(OutputError),
    /// Output-assembly fault (`spec/08`).
    Assemble(AssembleError),
}

/// The ways a per-block coefficient stream can fault (`spec/05`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStreamFault {
    /// The scan position advanced past the block's coefficient count.
    PositionOverflow {
        /// The overflowing position.
        position: i32,
        /// The block's coefficient budget (16 or 64).
        budget: u32,
    },
    /// A decoded symbol had no rv-table mapping (the over-256
    /// custom-codebook tail — a reported docs-gap).
    UnmappedSymbol {
        /// The decoded symbol.
        symbol: u32,
    },
}

impl From<PictureError> for DecodeError {
    fn from(e: PictureError) -> Self {
        DecodeError::Picture(e)
    }
}
impl From<BitReaderError> for DecodeError {
    fn from(e: BitReaderError) -> Self {
        DecodeError::BitReader(e)
    }
}
impl From<CodebookError> for DecodeError {
    fn from(e: CodebookError) -> Self {
        DecodeError::Codebook(e)
    }
}
impl From<RvTableError> for DecodeError {
    fn from(e: RvTableError) -> Self {
        DecodeError::RvTable(e)
    }
}
impl From<OutputError> for DecodeError {
    fn from(e: OutputError) -> Self {
        DecodeError::Output(e)
    }
}
impl From<AssembleError> for DecodeError {
    fn from(e: AssembleError) -> Self {
        DecodeError::Assemble(e)
    }
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::Picture(e) => write!(f, "indeo5 decode: {e}"),
            DecodeError::NotIntra { found } => write!(
                f,
                "indeo5 decode: frame type {found:?} needs session state (INTRA-only driver)"
            ),
            DecodeError::Band {
                plane_idx,
                band_idx,
                error,
            } => write!(
                f,
                "indeo5 decode: plane {plane_idx} band {band_idx}: {error}"
            ),
            DecodeError::Codebook(e) => write!(f, "indeo5 decode: {e}"),
            DecodeError::RvTable(e) => write!(f, "indeo5 decode: {e}"),
            DecodeError::MbHeader {
                plane_idx,
                band_idx,
                tile_idx,
                error,
            } => write!(
                f,
                "indeo5 decode: plane {plane_idx} band {band_idx} tile {tile_idx}: {error}"
            ),
            DecodeError::BlockStream {
                plane_idx,
                band_idx,
                tile_idx,
                reason,
            } => write!(
                f,
                "indeo5 decode: plane {plane_idx} band {band_idx} tile {tile_idx}: block stream {reason:?} (spec/05)"
            ),
            DecodeError::SizeFieldBehindCursor {
                at_byte,
                target_byte,
            } => write!(
                f,
                "indeo5 decode: size field targets byte {target_byte} behind cursor byte {at_byte} (spec/03 §2.8)"
            ),
            DecodeError::ReferenceMismatch {
                plane_idx,
                band_idx,
            } => write!(
                f,
                "indeo5 decode: plane {plane_idx} band {band_idx}: reference band geometry mismatch (spec/07 §1.2)"
            ),
            DecodeError::Mc {
                plane_idx,
                band_idx,
                tile_idx,
                error,
            } => write!(
                f,
                "indeo5 decode: plane {plane_idx} band {band_idx} tile {tile_idx}: {error}"
            ),
            DecodeError::PictureTooLarge { width, height } => write!(
                f,
                "indeo5 decode: picture {width}x{height} exceeds the {MAX_PICTURE_PIXELS}-pixel decoder guard"
            ),
            DecodeError::UnsupportedLayout {
                luma_bands,
                chroma_bands,
            } => write!(
                f,
                "indeo5 decode: unsupported band layout ({luma_bands} luma, {chroma_bands} chroma bands)"
            ),
            DecodeError::MissingReferenceMbs {
                plane_idx,
                band_idx,
            } => write!(
                f,
                "indeo5 decode: plane {plane_idx} band {band_idx} inherits from first-luma-band macroblocks that don't match its own"
            ),
            DecodeError::BitReader(e) => write!(f, "indeo5 decode: {e}"),
            DecodeError::Output(e) => write!(f, "indeo5 decode: {e}"),
            DecodeError::Assemble(e) => write!(f, "indeo5 decode: {e}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Why a walk stopped short of full reconstruction at some position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontierReason {
    /// A coded tile in an inter band with the MV-inheritance flag
    /// set — gated on the `spec/07 §3.4`/`§3.5` per-band
    /// `0x3604`/`0x3664` inheritance-MV tables docs-gap.
    MvInheritance,
}

/// One gated element the driver encountered (and, where possible,
/// skipped past via an explicit size field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeFrontier {
    /// Plane index (0 = Y; 1 = V chain, 2 = U chain).
    pub plane_idx: usize,
    /// Band index within the plane.
    pub band_idx: usize,
    /// Tile index in raster order.
    pub tile_idx: usize,
    /// The gate.
    pub reason: FrontierReason,
    /// `true` when an explicit tile size / `band_data_size` allowed
    /// the driver to reposition and keep parsing; `false` when the
    /// parse stopped here (`parse_complete == false`).
    pub skipped_past: bool,
}

/// Aggregate structural counts over one decoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct DecodeStats {
    /// Bands walked (including empty ones).
    pub bands: u32,
    /// Bands taking the `spec/02 §3.3` empty fast path.
    pub empty_bands: u32,
    /// Tiles walked.
    pub tiles: u32,
    /// Tiles with the `spec/03 §2.2` empty flag.
    pub empty_tiles: u32,
    /// Macroblocks walked.
    pub mbs: u32,
    /// Macroblocks with the `spec/03 §4.1` skip flag.
    pub mbs_skipped: u32,
    /// Coded macroblocks whose CBP carried no AC data.
    pub mbs_coded_no_ac: u32,
    /// Coded blocks whose `(run, val)` stream was decoded (`spec/05`).
    pub coded_blocks: u32,
    /// Non-zero coefficients decoded across all coded blocks.
    pub coefficients: u64,
    /// Escape-path emissions (`spec/05 §4.2`).
    pub escapes: u32,
}

/// Per-band byte-consumption trace: how many bytes of the band's
/// payload the structural walk consumed (before size-field
/// reconciliation), against the declared `band_data_size`. The staged
/// fixtures decode with `consumed <= declared` and a small trailing
/// tail (encoder padding inside the last tile's explicit byte count).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandTrace {
    /// Plane index.
    pub plane_idx: usize,
    /// Band index within the plane.
    pub band_idx: usize,
    /// Bytes consumed by the walk from the band header's first byte
    /// (after end-of-band byte alignment).
    pub consumed: u64,
    /// The declared `band_data_size` (`None` when the frame does not
    /// carry per-band sizes).
    pub declared: Option<u32>,
}

/// One band's decoded coefficient work list plus its checksum status:
/// the input the (docs-gapped) coefficient→pixel transform stage
/// consumes, and the `spec/08 §7` reconstruction oracle for the band.
#[derive(Debug, Clone)]
pub struct BandReconstruction {
    /// Plane index (0 = Y; 1 = V chain, 2 = U chain).
    pub plane_idx: usize,
    /// Band index within the plane.
    pub band_idx: usize,
    /// The band's global quantiser (`spec/06 §5.1`, `band_glob_quant`).
    pub glob_quant: u8,
    /// Every walked block of the band, in decode order, carrying its
    /// scan-ordered decoded coefficients (`spec/05` stream) plus the
    /// block's effective per-MB quantiser (`spec/06 §5.2`). Empty when
    /// the band took a gated / empty path.
    pub blocks: Vec<BlockRecord>,
    /// The band's stored `band_checksum` (`spec/08 §7.2`), if present.
    pub stored_checksum: Option<u16>,
    /// The band checksum recomputed from the reconstructed pixels vs
    /// the stored value (`spec/08 §7.2`, formula recovered by black-box
    /// validation): the sum of the band's samples before the wavelet
    /// recomposition and the output clamp.
    pub checksum: super::ChecksumStatus,
}

/// One decoded INTRA frame: the parsed header stack, the assembled
/// host buffer, and the structural coverage report.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedPicture {
    /// The parsed picture header.
    pub header: PictureHeader,
    /// The host output format the GOP subsampling selects
    /// (`None` for a NULL frame).
    pub format: Option<OutputFormat>,
    /// The assembled planar host buffer (`None` for a NULL frame —
    /// `spec/08 §6.4` no-output path).
    pub output: Option<HostBuffer>,
    /// Every gated element encountered, in decode order.
    pub frontiers: Vec<DecodeFrontier>,
    /// Structural coverage counts.
    pub stats: DecodeStats,
    /// Per-band consumption traces.
    pub band_traces: Vec<BandTrace>,
    /// Per-band decoded coefficients + reconstruction-checksum status
    /// (the coefficient→pixel transform's input work list and its
    /// `spec/08 §7` oracle).
    pub bands: Vec<BandReconstruction>,
    /// The recomputed-vs-stored **frame** checksum (`spec/08 §7.1`,
    /// formula recovered by black-box validation).
    pub frame_checksum: super::ChecksumStatus,
    /// `false` when a frontier could not be skipped past and parsing
    /// stopped early (later bands / planes reconstruct as zeros).
    pub parse_complete: bool,
}

impl DecodedPicture {
    /// `true` when every walked structural element was decoded without
    /// hitting a gate.
    pub fn fully_reconstructed(&self) -> bool {
        self.parse_complete && self.frontiers.is_empty()
    }

    /// The count of bands whose recomputed checksum matched the stored
    /// value (`spec/08 §7.2`) — bands reconstructed byte-sum-exactly.
    pub fn bands_verified(&self) -> usize {
        self.bands.iter().filter(|b| b.checksum.verified()).count()
    }
}

pub(crate) fn skip_to_byte(r: &mut BitReader<'_>, target_byte: u64) -> Result<(), DecodeError> {
    let target_bits = target_byte * 8;
    let at = r.bits_read();
    if target_bits < at {
        return Err(DecodeError::SizeFieldBehindCursor {
            at_byte: at / 8,
            target_byte,
        });
    }
    let mut remaining = target_bits - at;
    while remaining > 0 {
        let take = remaining.min(u64::from(super::bitreader::MAX_READ_BITS));
        r.skip(take as u32)?;
        remaining -= take;
    }
    Ok(())
}

/// How one block of a coded macroblock is signalled (`spec/03 §4.3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockCoding {
    /// CBP bit set — a `(run, val)` coefficient stream follows.
    Coded,
    /// CBP bit clear — DC-only / no coefficient stream.
    DcOnly,
    /// The whole MB carried the skip flag — no header fields either.
    Skipped,
}

/// One block of a walked tile, in band coordinates, carrying its
/// decoded scan-ordered coefficients (`spec/05` stream through the
/// band codebook + rv-table; `coeffs[pos]` is the value emitted at
/// scan position `pos`, `0..budget`).
#[derive(Debug, Clone)]
pub struct BlockRecord {
    /// Band-relative top-left x of the block.
    pub x: u32,
    /// Band-relative top-left y of the block.
    pub y: u32,
    /// Block side (4 or 8).
    pub blk_size: u32,
    /// Effective per-MB quantiser (`spec/06 §5.2`).
    pub quant: u8,
    /// How the block was signalled.
    pub coding: BlockCoding,
    /// Scan-position-ordered coefficient values (first
    /// `blk_size * blk_size` entries meaningful).
    pub coeffs: [i16; 64],
}

/// Decode one coded block's `(run, val)` coefficient stream
/// (`spec/05` + the wiki "Block data" annex): symbols through the
/// block codebook, `(run, val)` through the rv-table, `pos += run + 1`
/// scan advance from `pos = -1`, EOB-terminated. Every symbol carries
/// a non-zero level (the composite space has no level-0 entry — the
/// r451 "stuffing" reading is withdrawn, see [`super::RvTable`]);
/// placement and dequantisation happen in `reconstruct_intra_band`.
pub(crate) fn decode_block_stream(
    r: &mut BitReader<'_>,
    blk_cb: &Codebook,
    rv: &RvTable,
    budget: u32,
    stats: &mut DecodeStats,
    at: (usize, usize, usize),
    coeffs: &mut [i16; 64],
) -> Result<(), DecodeError> {
    let (plane_idx, band_idx, tile_idx) = at;
    let mut pos: i32 = -1;
    loop {
        let symbol = blk_cb.decode(r).map_err(DecodeError::Codebook)?;
        let entry = rv.lookup(symbol).ok_or(DecodeError::BlockStream {
            plane_idx,
            band_idx,
            tile_idx,
            reason: BlockStreamFault::UnmappedSymbol { symbol },
        })?;
        let (run, val) = match entry {
            RvEntry::Eob => return Ok(()),
            RvEntry::Esc => {
                stats.escapes += 1;
                // spec/05 §4.2 — three further VLCs: run, lindex_lo,
                // lindex_hi.
                let run = blk_cb.decode(r).map_err(DecodeError::Codebook)?;
                let lo = blk_cb.decode(r).map_err(DecodeError::Codebook)?;
                let hi = blk_cb.decode(r).map_err(DecodeError::Codebook)?;
                let val = escape_value(escape_lindex(lo, hi));
                (run.min(255) as u8, val)
            }
            RvEntry::Val { run, val } => (run, val),
        };
        pos = run_advance(pos, run);
        if pos >= budget as i32 {
            return Err(DecodeError::BlockStream {
                plane_idx,
                band_idx,
                tile_idx,
                reason: BlockStreamFault::PositionOverflow {
                    position: pos,
                    budget,
                },
            });
        }
        // The level stays quantised here; `reconstruct_intra_band`
        // reverses the quantiser by table once the block's raster
        // slot (and so its step) is known (spec/05 §2.3, spec/06 §5.4).
        coeffs[pos as usize] = val;
        stats.coefficients += 1;
    }
}

/// Resolve a band's effective transform (`spec/02 §1.7`): explicit
/// `ext_trans` wins; the `Standard` fallback maps the band's
/// frequency position — LL → 2D Slant, HL → row Slant, LH → column
/// Slant, HH → no transform (`spec/06 §1.1` wiki-annex confirmation).
pub(crate) fn resolve_transform(binfo: &BandInfo, band_idx: usize) -> TransformId {
    match binfo.transform_id {
        TransformId::Standard => {
            if band_idx == 0 {
                TransformId::Slant2d
            } else {
                match (band_idx - 1) % 3 {
                    0 => TransformId::SlantRow,
                    1 => TransformId::SlantColumn,
                    _ => TransformId::None,
                }
            }
        }
        explicit => explicit,
    }
}

/// The decoded payload of one picture-carrying frame: the
/// reconstructed planes and what the walk saw.
#[derive(Debug, Clone)]
pub(crate) struct PayloadOutcome {
    pub recon: Vec<ReconstructionPlane>,
    pub frontiers: Vec<DecodeFrontier>,
    pub stats: DecodeStats,
    pub band_traces: Vec<BandTrace>,
    pub parse_complete: bool,
    pub blocks: Vec<BandBlockSet>,
    /// Each band's sample checksum, in [`Self::blocks`] order.
    pub band_sums: Vec<u16>,
}

/// All walked blocks of one band, in decode order (the reconstruction
/// work list the coefficient stage hands the transform stage).
#[derive(Debug, Clone)]
pub(crate) struct BandBlockSet {
    pub plane_idx: usize,
    pub band_idx: usize,
    pub glob_quant: u8,
    pub records: Vec<BlockRecord>,
    pub stored_checksum: Option<u16>,
}

/// Decode one INTRA frame to pixels.
///
/// Threads the `spec/02 §4.4` per-plane / per-band / per-tile walk
/// over the structural layers, decodes the per-block coefficient
/// streams, recomposes each plane's bands (`spec/06 §3`), and
/// assembles the `spec/08` host buffer. For multi-frame sequences
/// (INTER / NULL-repeat) use [`super::Indeo5Decoder`].
pub fn decode_intra_picture(bitstream: &[u8]) -> Result<DecodedPicture, DecodeError> {
    let (header, mut r) = PictureHeader::parse_with_reader(bitstream, None)?;

    // NULL frame: repeat-previous, no coded payload (spec/08 §6.4).
    if header.is_null() {
        return Ok(DecodedPicture {
            header,
            format: None,
            output: None,
            frontiers: Vec::new(),
            stats: DecodeStats::default(),
            band_traces: Vec::new(),
            bands: Vec::new(),
            frame_checksum: super::ChecksumStatus::Absent,
            parse_complete: true,
        });
    }
    if header.frame_type() != FrameType::Intra {
        return Err(DecodeError::NotIntra {
            found: header.frame_type(),
        });
    }

    let gop = header.gop.as_ref().expect("INTRA frame carries a GOP");
    let frame = header
        .frame
        .as_ref()
        .expect("INTRA frame carries a frame header");

    let mut state = BandState::default();
    let payload = decode_payload(&mut r, gop, FrameType::Intra, frame, &mut state)?;
    let format = output_format(gop);
    let output = assemble_frame(
        &payload.recon[0],
        &payload.recon[1],
        &payload.recon[2],
        format,
    )?;

    let (bands, frame_ck) = reconstruction_checksums(
        &output,
        frame.frm_checksum,
        &payload.blocks,
        &payload.band_sums,
    );

    Ok(DecodedPicture {
        header,
        format: Some(format),
        output: Some(output),
        frontiers: payload.frontiers,
        stats: payload.stats,
        band_traces: payload.band_traces,
        bands,
        frame_checksum: frame_ck,
        parse_complete: payload.parse_complete,
    })
}

/// `spec/08 §7` reconstruction oracle: compare the per-band and
/// per-frame checksums recomputed from the reconstruction against the
/// stream's stored values (formulas recovered by black-box
/// validation; see [`super::verify`]). A band's checksum is the sum of
/// its own samples, which for a plane of one band is the plane's
/// pixels less 128.
pub(crate) fn reconstruction_checksums(
    output: &HostBuffer,
    stored_frame_checksum: Option<u16>,
    blocks: &[BandBlockSet],
    band_sums: &[u16],
) -> (Vec<BandReconstruction>, super::ChecksumStatus) {
    let luma = output.plane_bytes(super::PlaneRole::Luma);
    let cu = output.plane_bytes(super::PlaneRole::ChromaU);
    let cv = output.plane_bytes(super::PlaneRole::ChromaV);
    let frame_ck =
        super::ChecksumStatus::compare(stored_frame_checksum, super::frame_checksum(luma, cu, cv));
    let bands = blocks
        .iter()
        .zip(band_sums)
        .map(|(set, &sum)| BandReconstruction {
            plane_idx: set.plane_idx,
            band_idx: set.band_idx,
            glob_quant: set.glob_quant,
            blocks: set.records.clone(),
            stored_checksum: set.stored_checksum,
            checksum: super::ChecksumStatus::compare(set.stored_checksum, sum),
        })
        .collect();
    (bands, frame_ck)
}

/// The host output format the GOP subsampling selects
/// (`spec/08 §2.2`).
pub(crate) fn output_format(gop: &GopHeader) -> OutputFormat {
    match gop.flags.subsampling() {
        Subsampling::Yvu9 => OutputFormat::Yvu9,
        Subsampling::Yv12 => OutputFormat::Yv12,
    }
}

/// Decode a picture-carrying frame's bands into `state`'s buffers and
/// recompose its planes. An INTRA frame first takes `gop`'s layout.
pub(crate) fn decode_payload(
    r: &mut BitReader<'_>,
    gop: &GopHeader,
    frame_type: FrameType,
    frame: &FrameHeader,
    state: &mut BandState,
) -> Result<PayloadOutcome, DecodeError> {
    if u64::from(gop.width) * u64::from(gop.height) > MAX_PICTURE_PIXELS {
        return Err(DecodeError::PictureTooLarge {
            width: gop.width,
            height: gop.height,
        });
    }
    if frame_type == FrameType::Intra {
        state.configure(gop)?;
    }
    let out = state.decode_frame(r, frame_type, frame)?;
    Ok(PayloadOutcome {
        recon: out.recon,
        frontiers: Vec::new(),
        stats: out.stats,
        band_traces: out.band_traces,
        parse_complete: true,
        blocks: out.blocks,
        band_sums: out.band_sums,
    })
}
