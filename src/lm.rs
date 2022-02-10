#[cfg(feature = "kenlm")]
pub mod kenlm;

use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug, Default)]
pub struct SequenceState {
    children: Vec<Option<SequenceStateRef>>,
}

/// A reference to a LM state.
/// LMStateRef holds the information of the token sequence being decoded, and the identity of the token sequence can be confirmed by comparing LMStateRef.
#[derive(Default)]
pub struct SequenceStateRef(Rc<RefCell<SequenceState>>);

impl Clone for SequenceStateRef {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl std::hash::Hash for SequenceStateRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state)
    }
}

impl SequenceStateRef {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn child(&self, token: i32, n_vocab: usize) -> Self {
        if token == -1 {
            return self.clone();
        }
        let mut self_state = self.0.borrow_mut();
        // Allocate spaces lazily.
        if self_state.children.is_empty() {
            self_state.children.resize(n_vocab + 1 /* EOS */, None);
        }
        let child = &mut self_state.children[token as usize];
        match child {
            // If the child is already allocated, return it.
            Some(ref child) => child.clone(),
            // If not, allocate it and return it.
            None => {
                let new_child = SequenceStateRef::new();
                *child = Some(new_child.clone());
                new_child
            }
        }
    }
}

impl Debug for SequenceStateRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:p}", self.0.as_ptr())
    }
}

impl PartialEq for SequenceStateRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}
impl Eq for SequenceStateRef {}

impl PartialOrd for SequenceStateRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SequenceStateRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_ptr().cmp(&other.0.as_ptr())
    }
}

/// LM is a wrapper of a language model for decoding.
pub trait LM {
    type State;
    /// Initializes the LM, then returns the root LMStateRef.
    fn start(&mut self) -> Self::State;
    // Returns the new state and the score when a token comes as a continuation of LMStateRef.
    fn score(&mut self, state: &Self::State, token: i32) -> f32;
    // Returns the new state and the score when a token comes as a continuation of LMStateRef.
    fn next_state(&mut self, state: &Self::State, token: i32) -> Self::State;
    // batched version of `next_state`.
    fn batch_next_state(&mut self, states: &[&Self::State], tokens: &[i32]) -> Vec<Self::State> {
        debug_assert_eq!(states.len(), tokens.len());
        states
            .iter()
            .zip(tokens)
            .map(|(&s, &t)| self.next_state(s, t))
            .collect()
    }
    // Returns the new state and the score of the final state.
    fn finish(&mut self, state: &Self::State) -> f32;
}

/// ZeroLM is a language model that always returns 0.
/// This is a stub implementation of LM for interface consistency.
#[derive(Debug)]
pub struct ZeroLM;

impl LM for ZeroLM {
    type State = ();

    fn start(&mut self) -> Self::State {}

    fn score(&mut self, _state: &Self::State, _token: i32) -> f32 {
        0.0
    }

    fn next_state(&mut self, _state: &Self::State, _token: i32) -> Self::State {}

    fn finish(&mut self, _state: &Self::State) -> f32 {
        0.0
    }
}
