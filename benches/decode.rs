use std::io::BufRead;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ctclib::{Decoder, DecoderOptions, Dict, KenLM, ZeroLM};

fn load_logits() -> (usize, usize, Vec<f32>) {
    let file = std::io::BufReader::new(std::fs::File::open("data/logit.txt").unwrap());
    let mut lines = file.lines();
    let step_and_vocab = lines
        .next()
        .unwrap()
        .unwrap()
        .split(" ")
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let step = step_and_vocab[0];
    let vocab = step_and_vocab[1];
    let logits = lines
        .map(|x| x.unwrap().parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    (step, vocab, logits)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (steps, n_vocab, data) = load_logits();
    let blank = (n_vocab - 1) as i32;
    let option = DecoderOptions {
        beam_size: 100,
        beam_size_token: 2000000,
        beam_threshold: f32::MAX,
        lm_weight: 0.0,
    };
    let mut decoder = Decoder::new(option.clone(), blank, ZeroLM::new(n_vocab));
    c.bench_function("ZeroLM", |b| {
        b.iter(|| decoder.decode(black_box(&data), black_box(steps), n_vocab))
    });
    let dict = Dict::read("data/letter.dict").unwrap();
    let mut decoder = Decoder::new(
        DecoderOptions {
            lm_weight: 0.5,
            ..option.clone()
        },
        blank,
        KenLM::new("data/overfit.arpa", &dict),
    );
    c.bench_function("KenLM", |b| {
        b.iter(|| decoder.decode(black_box(&data), black_box(steps), n_vocab))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
