use std::io::BufRead;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ctclib::{BeamSearchDecoder, BeamSearchDecoderOptions, Decoder, GreedyDecoder, ZeroLM};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn load_logits() -> (usize, usize, Vec<f32>) {
    let file = std::io::BufReader::new(std::fs::File::open("data/logit.txt").unwrap());
    let lines = file.lines().collect::<Result<Vec<_>, _>>().unwrap();
    let step = lines.len();
    let mut logits = Vec::new();
    for line in lines {
        let iter = line.split(" ").map(|x| x.parse::<f32>().unwrap());
        logits.extend(iter);
    }
    let vocab = logits.len() / step;
    (step, vocab, logits)
}

fn decoder_options() -> BeamSearchDecoderOptions {
    BeamSearchDecoderOptions {
        beam_size: 100,
        beam_size_token: 2000000,
        beam_threshold: f32::MAX,
        lm_weight: 0.0,
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let (steps, n_vocab, data) = load_logits();
    let blank = (n_vocab - 1) as i32;
    let mut decoder = GreedyDecoder;
    c.bench_function("GreedyDecoder", |b| {
        b.iter(|| decoder.decode(black_box(&data), black_box(steps), n_vocab))
    });
    let mut decoder = BeamSearchDecoder::new(decoder_options(), blank, ZeroLM);
    c.bench_function("ZeroLM", |b| {
        #[cfg(feature = "dhat-heap")]
        let _profiler = dhat::Profiler::new_heap();
        b.iter(|| decoder.decode(black_box(&data), black_box(steps), n_vocab))
    });
}

#[cfg(feature = "kenlm")]
fn criterion_benchmark_kenlm(c: &mut Criterion) {
    use ctclib::{Dict, KenLM};

    let (steps, n_vocab, data) = load_logits();
    let blank = (n_vocab - 1) as i32;
    let dict = Dict::read("data/letter.dict").unwrap();
    let mut decoder = BeamSearchDecoder::new(
        BeamSearchDecoderOptions {
            lm_weight: 0.5,
            ..decoder_options()
        },
        blank,
        KenLM::new("data/overfit.arpa", &dict),
    );
    c.bench_function("KenLM", |b| {
        b.iter(|| decoder.decode(black_box(&data), black_box(steps), n_vocab))
    });
}

#[cfg(not(feature = "kenlm"))]
criterion_group!(benches, criterion_benchmark);
#[cfg(feature = "kenlm")]
criterion_group!(benches, criterion_benchmark, criterion_benchmark_kenlm);
criterion_main!(benches);
