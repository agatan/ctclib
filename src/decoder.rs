mod beamsearch;
mod greedy;

pub use beamsearch::{BeamSearchDecoder, BeamSearchDecoderOptions};
pub use greedy::GreedyDecoder;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecoderOutput {
    /// Score of this beam.
    pub score: f32,
    /// A sequence of tokens. Note that the sequence is always shorter than the original sequence.
    pub tokens: Vec<i32>,
    /// Timesteps of each token.
    pub timesteps: Vec<usize>,
    /// Acoustic model scores of each token.
    pub am_scores: Vec<f32>,
    /// Language model scores of each token.
    pub lm_scores: Vec<f32>,
}

impl DecoderOutput {
    fn new() -> Self {
        Self::default()
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
    fn decode(
        &mut self,
        data: &[f32],
        steps: usize,
        tokens: usize,
        blank_id: i32,
    ) -> Vec<DecoderOutput>;
}
