mod beamsearch;
mod greedy;

pub use beamsearch::{BeamSearchDecoder, BeamSearchDecoderOptions};
pub use greedy::GreedyDecoder;

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderOutput {
    pub score: f32,
    pub am_score: f32,
    pub lm_score: f32,
    pub tokens: Vec<i32>,
}

impl DecoderOutput {
    fn reserved(len: usize) -> Self {
        Self {
            score: 0.0,
            am_score: 0.0,
            lm_score: 0.0,
            tokens: vec![0; len],
        }
    }

    /// Returns the token sequence where blank and consecutive tokens have been resolved.
    pub fn reduced_tokens(&self, blank: i32) -> Vec<i32> {
        let mut output = Vec::new();
        let mut last_token = blank;
        for &tok in self.tokens.iter() {
            if last_token != tok && tok != blank {
                output.push(tok);
            }
            last_token = tok;
        }
        output
    }
}

/// Decoder is a trait for decoding a ctc sequence of tokens.
pub trait Decoder {
    fn decode(&mut self, data: &[f32], steps: usize, tokens: usize) -> Vec<DecoderOutput>;
}
