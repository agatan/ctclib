use std::io::BufRead;

use ctclib::{
    BeamSearchDecoder, BeamSearchDecoderOptions, Decoder, Dict, GreedyDecoder, KenLM, ZeroLM,
};

fn load_logits() -> (usize, usize, Vec<f32>) {
    let file = std::io::BufReader::new(std::fs::File::open("data/logit.txt").unwrap());
    let lines = file.lines().collect::<Result<Vec<_>, _>>().unwrap();
    let step = lines.len();
    let mut logits = Vec::new();
    for line in lines {
        let iter = line.split(' ').map(|x| x.parse::<f32>().unwrap());
        logits.extend(iter);
    }
    let vocab = logits.len() / step;
    (step, vocab, logits)
}

fn load_letter_dicts() -> Vec<String> {
    let file = std::io::BufReader::new(std::fs::File::open("data/letter.dict").unwrap());
    file.lines().map(|x| x.unwrap()).collect::<Vec<_>>()
}

#[test]
fn greedy_decoder_decodes_sequence_greedy() {
    let (steps, n_vocab, data) = load_logits();
    let vocab = load_letter_dicts();
    let blank = (n_vocab - 1) as i32;
    let mut decoder = GreedyDecoder;
    let outputs = decoder.decode(&data, steps, n_vocab,blank);
    let output = &outputs[0];
    let tokens = output.reduced_tokens(blank);
    let text = tokens
        .into_iter()
        .map(|i| vocab[i as usize].as_str())
        .collect::<Vec<&str>>()
        .join("");
    assert_eq!(text, "MISTE|QUILTER|T|IS|TH|E|APOSTLESR|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLADB|TO|WELCOME|HIS|GOSUPEL|N|");
}

#[test]
fn beam_search_decoder_decodes_sequence() {
    let (steps, n_vocab, data) = load_logits();
    let vocab = load_letter_dicts();
    let blank = (n_vocab - 1) as i32;
    let mut decoder = BeamSearchDecoder::new(
        BeamSearchDecoderOptions {
            beam_size: 100,
            beam_size_token: 2000000,
            beam_threshold: f32::MAX,
            lm_weight: 0.0,
        },
        ZeroLM,
    );
    let outputs = decoder.decode(&data, steps, n_vocab, blank);
    let output = &outputs[0];
    let tokens = output.reduced_tokens(blank);
    let text = tokens
        .into_iter()
        .map(|i| vocab[i as usize].as_str())
        .collect::<Vec<&str>>()
        .join("");
    assert_eq!(text, "MISTE|QUILTER|T|IS|TH|E|APOSTLESR|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLADB|TO|WELCOME|HIS|GOSPEL|N|");
}

#[test]
fn beam_search_decoder_decodes_sequence_with_kenlm() {
    let (steps, n_vocab, data) = load_logits();
    let vocab = load_letter_dicts();
    let blank = (n_vocab - 1) as i32;
    let dict = Dict::read("data/letter.dict").unwrap();
    let mut decoder = BeamSearchDecoder::new(
        BeamSearchDecoderOptions {
            beam_size: 100,
            beam_size_token: 2000000,
            beam_threshold: f32::MAX,
            lm_weight: 0.5,
        },
        KenLM::new("data/overfit.arpa", &dict),
    );
    let outputs = decoder.decode(&data, steps, n_vocab, blank);
    let output = &outputs[0];
    let tokens = output.reduced_tokens(blank);
    let text = tokens
        .into_iter()
        .map(|i| vocab[i as usize].as_str())
        .collect::<Vec<&str>>()
        .join("");
    assert_eq!(text, "MISTE|QUILTER|T|IS|THE|APOSTLES|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLAD|TO|WECOME|HIS|GOSPEL|");
}
