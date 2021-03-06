use ordered_float::OrderedFloat;

use super::{Decoder, DecoderOutput};
use crate::lm::{LMStateRef, LM};

#[derive(Debug, PartialEq)]
struct DecoderState<T> {
    score: f32,
    token: i32,
    prev_blank: bool,
    am_score: f32,
    lm_score: f32,
    parent_index: isize,
    lm_state: LMStateRef<T>,
}

impl<T> Clone for DecoderState<T> {
    fn clone(&self) -> Self {
        Self {
            score: self.score,
            token: self.token,
            prev_blank: self.prev_blank,
            am_score: self.am_score,
            lm_score: self.lm_score,
            parent_index: self.parent_index,
            lm_state: self.lm_state.clone(),
        }
    }
}

impl<T> DecoderState<T> {
    /// Compare two states by their internal conditions ignoring the scores.
    fn is_same_lm_state(&self, other: &DecoderState<T>) -> bool {
        self.lm_state == other.lm_state
            && self.token == other.token
            && self.prev_blank == other.prev_blank
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeamSearchDecoderOptions {
    pub beam_size: usize,
    pub beam_size_token: usize,
    /// the decoder will ignore paths whose score is more than this value lower than the best score.
    pub beam_threshold: f32,
    /// weight of the language model score.
    pub lm_weight: f32,
}

pub struct BeamSearchDecoder<T: LM> {
    options: BeamSearchDecoderOptions,
    /// All the new candidates that proposed based on the previous step.
    current_candidates: Vec<DecoderState<T::State>>,
    current_best_score: f32,
    current_candidate_pointers: Vec<usize>,
    /// hypothesis for each time step.
    hypothesis: Vec<Vec<DecoderState<T::State>>>,
    /// The language model.
    lm: T,
}

impl<T: LM> Decoder for BeamSearchDecoder<T> {
    fn decode(
        &mut self,
        data: &[f32],
        steps: usize,
        tokens: usize,
        blank_id: i32,
    ) -> Vec<DecoderOutput> {
        self.decode_begin(blank_id);
        self.decode_step(data, steps, tokens, blank_id);
        self.decode_end(steps, blank_id);
        let mut outputs = self.get_all_hypothesis(steps, blank_id);
        outputs.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
        outputs
    }
}

impl<T: LM> BeamSearchDecoder<T> {
    pub fn new(options: BeamSearchDecoderOptions, lm: T) -> Self {
        Self {
            options,
            current_candidates: Vec::new(),
            current_best_score: f32::MIN,
            current_candidate_pointers: Vec::new(),
            hypothesis: Vec::new(),
            lm,
        }
    }

    fn decode_begin(&mut self, blank_id: i32) {
        self.reset_candidate();
        let initial_state = self.lm.start();
        self.hypothesis.clear();
        self.hypothesis.push(Vec::new());
        self.hypothesis[0].push(DecoderState {
            score: 0.0,
            token: blank_id,
            prev_blank: false,
            am_score: 0.0,
            lm_score: 0.0,
            parent_index: -1, /* ROOT */
            lm_state: initial_state,
        });
    }

    fn decode_step(&mut self, data: &[f32], steps: usize, n_vocab: usize, blank_id: i32) {
        // Reserve hypothesis buffer.
        while self.hypothesis.len() < steps + 2 {
            self.hypothesis
                .push(Vec::with_capacity(self.options.beam_size));
        }

        // Loop over time steps.
        let mut target_index = (0..n_vocab).collect::<Vec<_>>();
        for t in 0..steps {
            if n_vocab > self.options.beam_size_token {
                // Collect tokens with the high score at the top `beam_size_token`.
                pdqselect::select_by(&mut target_index, self.options.beam_size_token, |&a, &b| {
                    data[t * n_vocab + a]
                        .partial_cmp(&data[t * n_vocab + b])
                        .unwrap()
                        .reverse()
                });
            }
            self.reset_candidate();
            for (prev_hyp_idx, prev_hyp) in self.hypothesis[t].iter().enumerate() {
                let prev_token = prev_hyp.token;
                let prev_lm_state = &prev_hyp.lm_state;
                let states =
                    target_index
                        .iter()
                        .take(self.options.beam_size_token)
                        .map(|&target| {
                            let token = target as i32;
                            let am_score = data[t * n_vocab + target];
                            let score = prev_hyp.score + am_score;

                            if token != blank_id && (token != prev_token || prev_hyp.prev_blank) {
                                // New token
                                let (lm_state, lm_score) =
                                    self.lm.score(prev_lm_state, token, n_vocab);
                                DecoderState {
                                    score: score + self.options.lm_weight * lm_score,
                                    token,
                                    prev_blank: false,
                                    am_score,
                                    lm_score,
                                    parent_index: prev_hyp_idx as isize,
                                    lm_state,
                                }
                            } else if token == blank_id {
                                // Blank
                                DecoderState {
                                    score,
                                    token,
                                    prev_blank: true,
                                    am_score,
                                    lm_score: prev_hyp.lm_score,
                                    parent_index: prev_hyp_idx as isize,
                                    lm_state: prev_lm_state.clone(),
                                }
                            } else {
                                // Extend
                                DecoderState {
                                    score,
                                    token,
                                    prev_blank: false,
                                    am_score,
                                    lm_score: prev_hyp.lm_score,
                                    parent_index: prev_hyp_idx as isize,
                                    lm_state: prev_lm_state.clone(),
                                }
                            }
                        });
                for state in states {
                    add_candidate(
                        &mut self.current_candidates,
                        &mut self.current_best_score,
                        self.options.beam_threshold,
                        state,
                    )
                }
            }
            // Finalize candidates.
            self.finalize_candidate(t);
        }
    }

    fn decode_end(&mut self, steps: usize, blank_id: i32) {
        self.reset_candidate();
        for (prev_hyp_idx, prev_hyp) in self.hypothesis[steps].iter().enumerate() {
            let prev_lm_state = &prev_hyp.lm_state;
            let (lm_state, lm_score) = self.lm.finish(prev_lm_state);
            add_candidate(
                &mut self.current_candidates,
                &mut self.current_best_score,
                self.options.beam_threshold,
                DecoderState {
                    score: prev_hyp.score + self.options.lm_weight * lm_score,
                    token: blank_id,
                    prev_blank: false,
                    am_score: prev_hyp.am_score,
                    lm_score: prev_hyp.lm_score + lm_score,
                    parent_index: prev_hyp_idx as isize,
                    lm_state,
                },
            );
        }
        self.finalize_candidate(steps);
    }

    fn reset_candidate(&mut self) {
        self.current_best_score = f32::MIN;
        self.current_candidates.clear();
        self.current_candidate_pointers.clear();
    }

    /// Finalize candidates at the current time step.
    /// This prunes the candidates, sort them by score and store them into hyp[t + 1].
    fn finalize_candidate(&mut self, t: usize) {
        // 1. Gather valid candidates.
        // ================================================================
        for (i, candidate) in self.current_candidates.iter().enumerate() {
            if candidate.score > self.current_best_score - self.options.beam_threshold {
                self.current_candidate_pointers.push(i);
            }
        }

        // 2. Merge same patterns.
        // ================================================================
        // Sort candidates so that the same patterns are consecutive.
        self.current_candidate_pointers.sort_by_key(|a| {
            let x = &self.current_candidates[*a];
            (&x.lm_state, x.token, x.prev_blank, OrderedFloat(x.score))
        });
        let mut n_candidates_after_merged = 1;
        let mut last_ptr = self.current_candidate_pointers[0];
        for i in 1..self.current_candidate_pointers.len() {
            let ptr = self.current_candidate_pointers[i];
            if !self.current_candidates[ptr].is_same_lm_state(&self.current_candidates[last_ptr]) {
                // Distinct pattern.
                self.current_candidate_pointers[n_candidates_after_merged] = ptr;
                n_candidates_after_merged += 1;
                last_ptr = ptr;
            } else {
                // Same pattern.
                let (last, current) = {
                    if last_ptr < ptr {
                        let (head, tail) = self.current_candidates.split_at_mut(ptr);
                        (&mut head[last_ptr], &mut tail[0])
                    } else {
                        let (head, tail) = self.current_candidates.split_at_mut(last_ptr);
                        (&mut tail[0], &mut head[ptr])
                    }
                };
                let max_score = last.score.max(current.score);
                let min_score = last.score.min(current.score);
                last.score =
                    max_score + libm::log1p(libm::exp(min_score as f64 - max_score as f64)) as f32;
            }
        }
        self.current_candidate_pointers
            .truncate(n_candidates_after_merged);

        // 3. Sort candidates.
        if self.current_candidate_pointers.len() > self.options.beam_size {
            pdqselect::select_by_key(
                &mut self.current_candidate_pointers,
                self.options.beam_size,
                |&x| OrderedFloat(-self.current_candidates[x].score),
            );
        }

        // 4. Copy candidates to output.
        let output = &mut self.hypothesis[t + 1];
        output.clear();
        for &ptr in self
            .current_candidate_pointers
            .iter()
            .take(self.options.beam_size)
        {
            output.push(self.current_candidates[ptr].clone());
        }
    }

    fn get_all_hypothesis(&self, final_step: usize, blank_id: i32) -> Vec<DecoderOutput> {
        self.hypothesis[final_step + 1]
            .iter()
            .map(|hyp| {
                let mut output = DecoderOutput::new();
                output.score = hyp.score;
                let mut hyps = Vec::with_capacity(final_step + 1);
                let mut hyp_ = hyp;
                for i in (0..final_step + 1).rev() {
                    hyps.push(hyp_.clone());
                    hyp_ = &self.hypothesis[i][hyp_.parent_index as usize];
                    if hyp_.parent_index == -1 {
                        break;
                    }
                }
                let mut last_token = blank_id;
                for (step, hyp) in hyps.into_iter().rev().enumerate() {
                    let token = hyp.token;
                    if last_token != token && token != blank_id {
                        output.tokens.push(token);
                        output.timesteps.push(step);
                        output.am_scores.push(hyp.am_score);
                        output.lm_scores.push(hyp.lm_score);
                    }
                    last_token = token;
                }
                output
            })
            .collect()
    }
}

fn add_candidate<T>(
    output: &mut Vec<DecoderState<T>>,
    current_best_score: &mut f32,
    beam_threshold: f32,
    state: DecoderState<T>,
) {
    if state.score > *current_best_score {
        *current_best_score = state.score;
    }
    if state.score > *current_best_score - beam_threshold {
        output.push(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::{lm::ZeroLM, BeamSearchDecoder, BeamSearchDecoderOptions, Decoder, DecoderOutput};

    #[test]
    fn it_works() {
        let options = BeamSearchDecoderOptions {
            beam_size: 1,
            beam_size_token: 10,
            beam_threshold: f32::MAX,
            lm_weight: 0.0,
        };
        let mut decoder = BeamSearchDecoder::new(options, ZeroLM);
        let steps = 3;
        let tokens = 4;
        #[rustfmt::skip]
        let data = &[
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
        ];
        let outputs = decoder.decode(data, steps, tokens, 3);
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0],
            DecoderOutput {
                score: 4.0,
                tokens: vec![0, 1],
                timesteps: vec![0, 2],
                am_scores: vec![1.0, 2.0],
                lm_scores: vec![0.0, 0.0],
            }
        )
    }
}
