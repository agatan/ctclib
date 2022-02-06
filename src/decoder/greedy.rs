use super::{Decoder, DecoderOutput};

#[derive(Debug, Clone)]
pub struct GreedyDecoder;

impl Decoder for GreedyDecoder {
    fn decode(
        &mut self,
        data: &[f32],
        steps: usize,
        tokens: usize,
        blank_id: i32,
    ) -> Vec<DecoderOutput> {
        let mut output = DecoderOutput::new();
        let mut last_token = blank_id;
        for step in 0..steps {
            let target = &data[step * tokens..(step + 1) * tokens];
            let (score, token) = find_max_index(target);
            if last_token != token && token != blank_id {
                output.tokens.push(token);
                output.timesteps.push(step);
                output.am_scores.push(score);
                output.score += score;
            }
            last_token = token;
        }
        vec![output]
    }
}

fn find_max_index(vs: &[f32]) -> (f32, i32) {
    let mut max_index = 0usize;
    let mut max_value = vs[0];
    for (i, v) in vs.iter().enumerate() {
        if *v > max_value {
            max_index = i;
            max_value = *v;
        }
    }
    (max_value, max_index as i32)
}
