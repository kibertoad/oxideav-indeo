#![no_main]
use libfuzzer_sys::fuzz_target;
use oxideav_indeo::indeo5::Iv50Decoder;

// A sequence of frames through one decoder: the input is cut into
// frames at a 16-bit length prefix, so inter and NULL frames follow an
// intra one and decode against the state it left. Must return, never
// panic.
fuzz_target!(|data: &[u8]| {
    let mut dec = Iv50Decoder::new();
    let mut rest = data;
    let mut frames = 0;
    while rest.len() >= 2 && frames < 6 {
        let len = usize::from(u16::from_le_bytes([rest[0], rest[1]])) % 4096;
        rest = &rest[2..];
        let take = len.min(rest.len());
        let (frame, tail) = rest.split_at(take);
        if dec.decode(frame).is_ok() {
            let _ = dec.y();
        }
        rest = tail;
        frames += 1;
    }
});
