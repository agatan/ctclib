use super::{Decoder, DecoderOutput};

#[derive(Debug, Clone)]
pub struct GreedyDecoder;

impl Decoder for GreedyDecoder {
    fn decode(&mut self, data: &[f32], steps: usize, tokens: usize) -> Vec<DecoderOutput> {
        let mut output = DecoderOutput {
            score: 0.0,
            am_score: 0.0,
            lm_score: 0.0,
            tokens: vec![0; steps],
        };
        for step in 0..steps {
            let target = &data[step * tokens..(step + 1) * tokens];
            let (score, token) = find_max_index(target);
            output.score += score;
            output.am_score += score;
            output.tokens[step] = token as i32;
        }
        vec![output]
    }
}

fn find_max_index(vs: &[f32]) -> (f32, usize) {
    let mut max_index = 0usize;
    let mut max_value = vs[0];
    for (i, v) in vs.iter().enumerate() {
        if *v > max_value {
            max_index = i;
            max_value = *v;
        }
    }
    (max_value, max_index)
}
