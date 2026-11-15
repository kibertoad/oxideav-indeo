# oxideav-indeo

[![CI](https://github.com/OxideAV/oxideav-indeo/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-indeo/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-indeo.svg)](https://crates.io/crates/oxideav-indeo) [![docs.rs](https://docs.rs/oxideav-indeo/badge.svg)](https://docs.rs/oxideav-indeo) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust Intel Indeo (IV2 / IV3 / IV4 / IV5) video codec family for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework. Built
from clean-room specification and behavioural-trace documents under
`docs/video/indeo/` only — no external codec source is consulted.

## Status

Two of the four Indeo generations decode real streams, both validated
pixel-for-pixel against black-box reference decodes of the fixtures
staged under `docs/video/indeo/`:

| Codec | FourCC | State |
| ----- | ------ | ----- |
| Indeo 3 | `IV31` / `IV32` | **Decodes.** Both staged corpora (160×120 all-intra, 176×144 with a 4-frame intra period) reproduce the reference decode **pixel-exact on every frame and every plane** — 16 / 16 frames: intra (families A / E), inter (motion compensation with full- and half-pel vectors, families B / F), VQ_NULL cells, NULL-frame repeat, the two-bank reference ping-pong. |
| Indeo 5 | `IV50` | **Decodes.** Intra, inter and NULL frames, one-band and four-band luma, quarter- and half-size chroma, transparency windows. All ten staged frames (two intra keyframes, a vendor-encoded intra + 7 inter sequence) are **byte-exact against the vendor driver's `YUY2` output**, luma and chroma. On the 40 IV50 movies of Rent-A-Hero (5 245 frames) the 8-bit planes equal the vendor driver's on every frame of 22 movies; in the other 18, all with transparency, the only difference is one run of at most 31 luma pixels that the driver never initialises. |
| Indeo 2 / 4 | `RT21` / `IV41` | Wiki-snapshot documentation only under `docs/video/indeo/indeoN/wiki/` (no formal `spec/`); scaffold pending docs work. |

All truth is drawn from the staged spec / trace docs — no external
codec source is consulted; where the spec text leaves a value-level
choice open, the fixtures arbitrate it and the choice is recorded in
the module docs and the CHANGELOG.

### Indeo 3 (`IV31` / `IV32`)

Entry points: `indeo3::Indeo3PictureDecoder` (stateful; `decode(frame)
-> DecodedPicture` with native 4:1:0 planes), the `oxideav_core`
bridge `indeo3::Indeo3RegistryDecoder` / `decode_video_frame`
(`PixelFormat::Yuv444P`, chroma box-replicated 4×4 per `spec/07
§5.5`), and the building blocks `indeo3::decode_plane` /
`PlaneBuffers` / `PlaneBanks`.

How a frame decodes (r459):

- `spec/01` / `spec/02` — the 64-byte combined header, the per-plane
  preludes (motion-vector tables) and the per-frame codebook arena
  rebuilt from `alt_quant[]` over the codec-init staging image
  (`spec/04 §5.2` / `§6`, reproduced word-for-word from the seed area).
- `spec/04 §5.3` — the **cell-geometry banks**: five sub-tables per
  plane, regenerated from the picture size by the vendor's populator
  rule (4 080 / 4 080 staged entries, `tests/indeo3_geometry_banks.rs`);
  every leaf of the binary tree is positioned through them (full-strip
  or last-strip bank by strip slot).
- `spec/03` — the MSB-first binary-tree walk with the vertical /
  horizontal heap indices; the tree bits and the byte-level reads share
  one cursor whose accumulator keeps a tree byte's leftover bits across
  a cell's byte stream.
- `spec/06` / `spec/07` — a coded cell is one mode byte (nibble tables
  → family, codebook base, LUT rewrite of the row above for band
  nibbles ≥ 8, which also selects the even seed sets, staging blocks
  8..15) followed by codes per 4×4 block (A / B) or 8×8 block (E / F),
  four per block, row-group major, with the eight `0xF8..0xFF` escapes
  at their block positions (`0xFB` runs count blocks); a code's word is
  applied dyad by dyad, the raw `pred + word` bit 31 selecting the
  two-byte form. E doubles rows horizontally and stores `(avg(above,
  row), row)` pairs (repeating at the strip top); F adds the doubled
  deltas per pixel to both rows over the motion-compensated content.
- `spec/05` — INTER leaves fetch the reference cell by the packed
  byte-offset vector (`176 · vert + horiz`, bits 0 / 1 the half-pel
  filters, truncating averages), then take their residual through B /
  F; a VQ_NULL on an INTER cell keeps the fetched content.
- Output: 7-bit samples upshifted (`spec/07 §4.3`), chroma at
  `ceil(w / 4) × ceil(h / 4)`.

Not exercised by the fixtures (implemented per spec, unvalidated or
reported): families C / D (8-row × 4-pixel cells — reported as
`CellDecodeError::UnsupportedFamily`, not guessed), pictures wider
than 320 (three-strip wide flag), the `0xF8` literal fill, RGB host
output (the `spec/07 §5.4` LUT is unstaged; not needed for YUV
output).

### Indeo 5 (`IV50`)

Entry points: `indeo5::Iv50Decoder` (stateful; `decode(frame) ->
FrameType`, then the `y()`, `u()` and `v()` planes, or `yuy2()` for the
packed layout the vendor driver hands a `YUY2` host) and the
`oxideav_core` bridge `indeo5::Indeo5RegistryDecoder` /
`decode_video_frame`.

The decoder reproduces the vendor driver's 8-bit planes: picture start
and the repeated-number NULL rule, GOP, frame and band headers, the
row-width prefix codes, macroblock headers with quantiser deltas and
inherited vectors, run/value tables with swap pairs and escapes,
dequantisation, DC prediction, the inverse Slant transforms,
half-sample motion compensation from the rotating reference buffers,
the four-band synthesis filter, half-size as well as quarter-size
chroma, and the transparency bounding box that limits which part of
the picture a frame rebuilds.

**Transparency.** For a stream with transparency (gop_flags bit 3),
`Iv50Decoder::transparency()` gives the GOP header's transparency
colour, the frame's rectangles and a per-pixel mask decoded from the
frame trailer's run-length code. The planes keep the whole picture;
the mask says which pixels are opaque. The staged specification has no
section for the trailer: its layout was worked out from the streams.
The vendor driver, decoding to an RGB DIB, writes the transparency
colour (black when the stream has none) to every transparent pixel,
and on the 21 Rent-A-Hero movies with transparency the decoded mask
equals the pixels it fills on every frame compared (1 844 frames).

**Specification.** Comments cite the staged clean-room specification
(`docs/video/indeo/indeo5/spec/` by chapter and section, and its
`tables/`). Every decoder table is compared with the staged table files
in unit tests, the transforms with the recurrence of `spec/06` §1.2,
and all ten staged frames (two intra keyframes, and one intra plus
seven inter frames from the vendor encoder) decode **byte-exact against
the vendor driver's `YUY2` output, luma and chroma**. The module docs of
`indeo5::Iv50Decoder`'s module list where the decoder follows the
fixtures rather than the staged text, and what it carries that the text
does not cover.

**Output.** Planes are Y, U, V. Size and pixel format are stream
metadata (`Decoder::output_params`, with a `Frame::StreamParams` event
and a moved generation counter when an intra frame changes them):

| Stream chroma | Decoder planes | Registry frames |
| ------------- | -------------- | --------------- |
| quarter size (4:1:0, every staged stream) | `ceil(w / 4)` × `ceil(h / 4)` | `PixelFormat::Yuv444P`, each chroma sample repeated over its 4×4 luma pixels (the framework has no 4:1:0 format) |
| half size (4:2:0, gop_flags bit 1) | `ceil(w / 2)` × `ceil(h / 2)` | `PixelFormat::Yuv420P`, as decoded |

**Speed.** Single thread, release build, Apple M-series, best of 300
passes over each staged sequence (three runs agree within 3 %; the
harness is `benches/indeo5_decode.rs`, the stage shares come from a
sampling profile of the same loop):

| Sequence | `Iv50Decoder::decode` | Registry decoder (with the `VideoFrame` copy) |
| -------- | --------------------- | --------------------------------------------- |
| 176×144, 1 intra + 7 inter (vendor encoder) | 63.7 → 38.1 µs/frame (15.7 k → 26.2 k fps) | 103.7 → 47.1 µs/frame (9.6 k → 21.2 k fps) |
| 320×240 intra, 1 105 coded blocks | 65.7 → 34.2 µs/frame (15.2 k → 29.3 k fps) | 162.5 → 58.7 µs/frame (6.2 k → 17.0 k fps) |
| 240×180 intra, flat | 20.9 → 12.9 µs/frame (47.8 k → 77.4 k fps) | 82.1 → 27.6 µs/frame (12.2 k → 36.2 k fps) |

Before, the inter sequence spent 44 % in motion compensation (a
per-sample closure with a bounds check), 28 % in entropy decoding and
macroblock headers, 26 % in the inverse transforms and 2 % in the plane
writer; the intra keyframe 52 % in the transforms. Motion compensation
now works on fixed-width rows, the transforms skip the passes that a
block's occupied rows and columns make trivial (flat blocks, first-row
blocks, empty columns), uncoded blocks are filled with fixed-width
stores, the run/value map is built without allocating, and the registry
bridge widens a chroma row once and copies rows whole. Output is
bit-identical (the fixture digests did not move).

**Refused** with `Iv50Error::Unsupported`, because the staged
specification leaves them open: several tiles per band, locked streams
(access key), seven-band luma and four-band chroma, a transform whose
size differs from the band's block size, one-axis transforms of a
chroma band, a GOP header `value4` other than 2, transparency combined
with half-size chroma, a half-size chroma band inheriting from luma
macroblocks that are not twice its own, and an intra macroblock at a
band quantiser above 23.

## Selected public API

- `indeo3::Indeo3PictureDecoder::decode` → `DecodedPicture` (`luma`,
  `chroma_v`, `chroma_u`, `repeated_previous`, per-plane `PlaneStats`;
  `to_yuv444_planes()`).
- `indeo3::decode_plane` / `PlaneBuffers` / `PlaneContext` — one plane
  through the cell decoder over its strip buffers; `PlaneBanks` /
  `GeometryBank` / `split_extent` / `chroma_plane_dims` — the geometry
  banks.
- `indeo3::FrameHeader::parse`, `PictureLayer::parse`, `StagingImage`,
  `VqArena::apply_alt_quant`, `DyadDeltaTable` — the header, picture
  layer and codec-init tables.
- `indeo3::DecodeSession` — the inter-frame sequencer (first-frame /
  seek INTRA gate, NULL repeat, reference bank).
- `indeo5::Iv50Decoder::decode` → `FrameType`; `y()` / `u()` / `v()`
  → `Iv50Plane` (width, height, stride, samples); `chroma_ratio()`;
  `yuy2()`; `transparency()` → `Iv50Transparency` (colour,
  `Iv50Rect`s, mask as an `Iv50Plane`); `Iv50Error`.
- Registry: `register` (crate root), `indeo3::register_codecs` /
  `indeo5::register`, `codec_id_for_fourcc`, `probe`.

The crate forbids `unsafe` (`#![forbid(unsafe_code)]`).

## License

MIT.
