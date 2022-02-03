use std::cmp::Ordering;

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

impl <T> Clone for DecoderState<T> {
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
    /// This is used to sort states so that the same states will be consecutive.
    fn cmp_without_score(&self, other: &DecoderState<T>) -> Ordering {
        let lm_cmp = self.lm_state.partial_cmp(&other.lm_state).unwrap();
        if lm_cmp != Ordering::Equal {
            return lm_cmp;
        }
        if self.token != other.token {
            self.token.cmp(&other.token)
        } else if self.prev_blank != other.prev_blank {
            self.prev_blank.cmp(&other.prev_blank)
        } else {
            Ordering::Equal
        }
    }

    fn cmp_without_score_then_score(&self, other: &DecoderState<T>) -> Ordering {
        let without_score = self.cmp_without_score(other);
        if without_score != Ordering::Equal {
            without_score
        } else {
            self.cmp_by_score(other)
        }
    }

    fn cmp_by_score(&self, other: &DecoderState<T>) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

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

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderOptions {
    pub beam_size: usize,
    pub beam_size_token: usize,
    /// the decoder will ignore paths whose score is more than this value lower than the best score.
    pub beam_threshold: f32,
    /// weight of the language model score.
    pub lm_weight: f32,
}

pub struct Decoder<T: LM> {
    options: DecoderOptions,
    /// All the new candidates that proposed based on the previous step.
    current_candidates: Vec<DecoderState<T::State>>,
    current_best_score: f32,
    current_candidate_pointers: Vec<usize>,
    /// blank_index is the index of the blank token.
    blank: i32,
    /// hypothesis for each time step.
    hypothesis: Vec<Vec<DecoderState<T::State>>>,
    /// The language model.
    lm: T,
}

impl<T: LM> Decoder<T> {
    pub fn new(options: DecoderOptions, blank: i32, lm: T) -> Self {
        Self {
            options,
            current_candidates: Vec::new(),
            current_best_score: f32::MIN,
            current_candidate_pointers: Vec::new(),
            blank,
            hypothesis: Vec::new(),
            lm,
        }
    }

    pub fn decode(&mut self, data: &[f32], steps: usize, tokens: usize) -> Vec<DecoderOutput> {
        self.decode_begin();
        self.decode_step(data, steps, tokens);
        self.decode_end(steps);
        let mut outputs = self.get_all_hypothesis(steps);
        outputs.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
        outputs
    }

    fn decode_begin(&mut self) {
        self.reset_candidate();
        let initial_state = self.lm.start();
        self.hypothesis.clear();
        self.hypothesis.push(Vec::new());
        self.hypothesis[0].push(DecoderState {
            score: 0.0,
            token: self.blank,
            prev_blank: false,
            am_score: 0.0,
            lm_score: 0.0,
            parent_index: -1, /* ROOT */
            lm_state: initial_state,
        });
    }

    fn decode_step(&mut self, data: &[f32], steps: usize, tokens: usize) {
        // Reserve hypothesis buffer.
        while self.hypothesis.len() < steps + 2 {
            self.hypothesis.push(Vec::new());
        }

        // Loop over time steps.
        let mut target_index = (0..tokens).collect::<Vec<_>>();
        for t in 0..steps {
            if tokens > self.options.beam_size_token {
                // Collect tokens with the high score at the top `beam_size_token`.
                pdqselect::select_by(&mut target_index, self.options.beam_size_token, |&a, &b| {
                    data[t * tokens + a]
                        .partial_cmp(&data[t * tokens + b])
                        .unwrap()
                        .reverse()
                });
            }
            self.reset_candidate();
            for (prev_hyp_idx, prev_hyp) in self.hypothesis[t].iter().enumerate() {
                let prev_token = prev_hyp.token;
                let prev_lm_state = &prev_hyp.lm_state;
                for &target in target_index.iter().take(self.options.beam_size_token) {
                    let token = target as i32;
                    let am_score = data[t * tokens + target];
                    let score = prev_hyp.score + am_score;

                    if token != self.blank && (token != prev_token || prev_hyp.prev_blank) {
                        // New token
                        let (lm_state, lm_score) = self.lm.score(prev_lm_state, token);
                        add_candidate(
                            &mut self.current_candidates,
                            &mut self.current_best_score,
                            self.options.beam_threshold,
                            DecoderState {
                                score: score + self.options.lm_weight * lm_score,
                                token,
                                prev_blank: false,
                                am_score: prev_hyp.am_score + am_score,
                                lm_score: prev_hyp.lm_score + lm_score,
                                parent_index: prev_hyp_idx as isize,
                                lm_state,
                            },
                        );
                    } else if token == self.blank {
                        // Blank
                        add_candidate(
                            &mut self.current_candidates,
                            &mut self.current_best_score,
                            self.options.beam_threshold,
                            DecoderState {
                                score,
                                token,
                                prev_blank: true,
                                am_score: prev_hyp.am_score + am_score,
                                lm_score: prev_hyp.lm_score,
                                parent_index: prev_hyp_idx as isize,
                                lm_state: prev_lm_state.clone(),
                            },
                        );
                    } else {
                        // Extend
                        add_candidate(
                            &mut self.current_candidates,
                            &mut self.current_best_score,
                            self.options.beam_threshold,
                            DecoderState {
                                score,
                                token,
                                prev_blank: false,
                                am_score: prev_hyp.am_score + am_score,
                                lm_score: prev_hyp.lm_score,
                                parent_index: prev_hyp_idx as isize,
                                lm_state: prev_lm_state.clone(),
                            },
                        );
                    }
                }
            }
            // Finalize candidates.
            self.finalize_candidate(t);
        }
    }

    fn decode_end(&mut self, steps: usize) {
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
                    token: self.blank,
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
    /// This prunes the candidates and sort them by score.
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
        self.current_candidate_pointers.sort_by(|a, b| {
            self.current_candidates[*a].cmp_without_score_then_score(&self.current_candidates[*b])
        });
        let mut n_candidates_after_merged = 1;
        let mut last_ptr = self.current_candidate_pointers[0];
        for i in 1..self.current_candidate_pointers.len() {
            let ptr = self.current_candidate_pointers[i];
            if self.current_candidates[ptr].cmp_without_score(&self.current_candidates[last_ptr])
                != Ordering::Equal
            {
                // Distinct pattern.
                self.current_candidate_pointers[n_candidates_after_merged] = ptr;
                n_candidates_after_merged += 1;
                last_ptr = ptr;
            } else {
                // Same pattern.
                let max_score = self.current_candidates[last_ptr]
                    .score
                    .max(self.current_candidates[ptr].score);
                let min_score = self.current_candidates[last_ptr]
                    .score
                    .min(self.current_candidates[ptr].score);
                self.current_candidates[last_ptr].score =
                    max_score + libm::log1p(libm::exp(min_score as f64 - max_score as f64)) as f32;
            }
        }
        self.current_candidate_pointers
            .truncate(n_candidates_after_merged);
        self.current_candidate_pointers.shrink_to_fit();

        // 3. Sort candidates.
        if self.current_candidate_pointers.len() > self.options.beam_size {
            pdqselect::select_by(
                &mut self.current_candidate_pointers,
                self.options.beam_size,
                |&a, &b| {
                    self.current_candidates[a]
                        .cmp_by_score(&self.current_candidates[b])
                        .reverse()
                },
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

    fn get_all_hypothesis(&self, final_step: usize) -> Vec<DecoderOutput> {
        self.hypothesis[final_step + 1]
            .iter()
            .map(|hyp| {
                let mut output = DecoderOutput::reserved(final_step + 1);
                output.score = hyp.score;
                output.am_score = hyp.am_score;
                output.lm_score = hyp.lm_score;
                let mut hyp_ = hyp;
                for i in (0..final_step + 1).rev() {
                    output.tokens[i] = hyp_.token;
                    hyp_ = &self.hypothesis[i][hyp_.parent_index as usize];
                    if hyp_.parent_index == -1 {
                        break;
                    }
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
    use crate::{lm::ZeroLM, Decoder, DecoderOptions, DecoderOutput};

    #[test]
    fn it_works() {
        let options = DecoderOptions {
            beam_size: 1,
            beam_size_token: 10,
            beam_threshold: f32::MAX,
            lm_weight: 0.0,
        };
        let mut decoder = Decoder::new(options, 4, ZeroLM);
        let steps = 3;
        let tokens = 4;
        #[rustfmt::skip]
        let data = &[
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
        ];
        let outputs = decoder.decode(data, steps, tokens);
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0],
            DecoderOutput {
                score: 4.0,
                am_score: 4.0,
                lm_score: 0.0,
                tokens: vec![0, 0, 1, 4],
            }
        )
    }
}
