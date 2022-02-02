pub mod kenlm;
pub use kenlm::KenLM;

use std::{cell::RefCell, collections::HashMap, fmt::Debug, rc::Rc};

#[derive(Debug, Default)]
pub struct LMState {
    children: HashMap<i32, LMStateRef>,
}

/// A reference to a LM state.
/// LMStateRef holds the information of the token sequence being decoded, and the identity of the token sequence can be confirmed by comparing LMStateRef.
#[derive(Default)]
pub struct LMStateRef(Rc<RefCell<LMState>>);

impl Clone for LMStateRef {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl std::hash::Hash for LMStateRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state)
    }
}

impl LMStateRef {
    fn new() -> Self {
        Self::default()
    }

    fn child(&self, token: i32) -> Self {
        let mut state = self.0.borrow_mut();
        let child = state
            .children
            .entry(token)
            .or_insert_with(LMStateRef::default);
        child.clone()
    }
}

impl Debug for LMStateRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:p}", self.0.as_ptr())
    }
}

impl PartialEq for LMStateRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}
impl Eq for LMStateRef {}

impl PartialOrd for LMStateRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.as_ptr().partial_cmp(&other.0.as_ptr())
    }
}

/// LM is a wrapper of a language model for decoding.
pub trait LM {
    /// Initializes the LM, then returns the root LMStateRef.
    fn start(&mut self) -> LMStateRef;
    // Returns the new state and the score when a token comes as a continuation of LMStateRef.
    fn score(&mut self, state: &LMStateRef, token: i32) -> (LMStateRef, f32);
    // Returns the new state and the score of the final state.
    fn finish(&mut self, state: &LMStateRef) -> (LMStateRef, f32);
}

/// ZeroLM is a language model that always returns 0.
/// This is a stub implementation of LM for interface consistency.
#[derive(Debug)]
pub struct ZeroLM;

impl LM for ZeroLM {
    fn start(&mut self) -> LMStateRef {
        LMStateRef::default()
    }

    fn score(&mut self, state: &LMStateRef, token: i32) -> (LMStateRef, f32) {
        (state.child(token), 0.0)
    }

    fn finish(&mut self, state: &LMStateRef) -> (LMStateRef, f32) {
        (state.clone(), 0.0)
    }
}
