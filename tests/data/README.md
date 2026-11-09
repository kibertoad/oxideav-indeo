# Indeo test fixtures

## `intra-240x180-educ.iv50` / `intra-320x240-indeo5.iv50`

Real Intel/Ligos-encoded `IV50` INTRA keyframes (raw codec bitstream,
no AVI container), vendored from the clean-room docs staging at
`docs/video/indeo/indeo5/fixtures/` (r338, univdreams sandbox). They
are **media**, not source: the first `movi` chunk of
`Educ_Movie_DeadlyForce.avi` (240x180) and `indeo5.avi` (320x240) from
the project sample mirror. Full provenance (source URLs, container
SHAs, black-box reproduce commands against the vendor decoder) lives
in each fixture's `notes.md` under the docs staging directory.

| File | Bytes | SHA-256 |
| ---- | ----- | ------- |
| `intra-240x180-educ.iv50` | 260 | `c481126de9e74149f02f28e74693ddbd483e92e43e44c6b96ca7e54bf3922c2f` |
| `intra-320x240-indeo5.iv50` | 1632 | `f617e936dcdcdf76ccf18bf6219a175be8a798c3cea8627dfd47d21c6494e7bd` |

Both frames are `YVU9`, `decomp_levels = 0` (one band per plane),
single-tile-per-band. The 240x180 frame is a black frame (the vendor
decoder reproduces `Y=16, U=V=128` for it); the 320x240 frame carries
1 105 coded blocks across its three bands.

`*.expected.yuy2` are the fixtures' staged `expected.yuv` reference
decodes (the vendor decoder's packed `YUY2` host buffer, `Y0 U Y1 V`
per 4-byte unit, `width*height*2` bytes; chroma at 4:2:2 as the
vendor's own writer upsamples it). Luma is compared sample-for-sample
in `tests/indeo5_decoder.rs`.

| File | Bytes | SHA-256 |
| ---- | ----- | ------- |
| `intra-240x180-educ.expected.yuy2` | 86400 | `2ff24b741d9577e1b8b22d88b3a67902842347b2e878a094249fdf3904f3ec45` |
| `intra-320x240-indeo5.expected.yuy2` | 153600 | `e531aa42393bcaf455616e87c2bcf003f90991248963e5e9804d5878bd20fdff` |

`iv50-quant-matrices-1007b000.csv` is the staged regenerated runtime
quantiser bank (`docs/video/indeo/indeo5/tables/quant_matrices_1007b000.csv`,
Extractor round 15, live-verified in Validator round 16), the oracle
for the quantiser bank test in `src/indeo5/decoder/band.rs`.

## `iv32-160x120-all-intra/` / `iv32-176x144-4frame-intra-period/`

Real `IV32` (Indeo 3) coded access units with black-box reference
decodes, vendored from the clean-room docs staging at
`docs/video/indeo/indeo3/fixtures/` (r451; staged 2026-07-31 from the
public multimedia sample archive, MD5-verified against the archive's
own manifest — full provenance in each fixture's `notes.md` there).

Each directory carries `samples.bin` (the coded access units,
byte-exact from the source container), `samples-index.csv`
(`frame,offset,size` rows), and `expected.yuv` (the reference decode,
planar **4:1:0** `yuv410p`, 8 frames at coded size).

| Fixture | Coded | Frames | Focus |
| ------- | ----- | -----: | ----- |
| `iv32-160x120-all-intra` | 160×120 | 8 | every frame intra |
| `iv32-176x144-4frame-intra-period` | 176×144 | 8 | frames 0/4 intra, rest inter |

| File | SHA-256 |
| ---- | ------- |
| `iv32-160x120-all-intra/samples.bin` | `da5d7a6147f85a12be6edcf5d973f7a3d156a740ef5d06a1b19be197f667b0be` |
| `iv32-160x120-all-intra/expected.yuv` | `3d8ca7a27542ed8d2d4c7a174265a1ef0f47519a5108d69e7a9ed158c2395dab` |
| `iv32-176x144-4frame-intra-period/samples.bin` | `a95f388c8886a6cf1dec8609ed6e4ab3e0fdc6b38f0c3dfe1fd3977acb072807` |
| `iv32-176x144-4frame-intra-period/expected.yuv` | `d95603969ef93dc8ee97baf437f189a741409c2cb48b5493a0e5c185ac26beb3` |
