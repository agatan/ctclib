use std::{cell::RefCell, collections::HashMap, fmt::Debug, rc::Rc};

#[derive(Debug, Default)]
struct LMState {
    children: HashMap<i32, LMStateRef>,
}

/// A reference to a LM state.
/// LMStateRef holds the information of the token sequence being decoded, and the identity of the token sequence can be confirmed by comparing LMStateRef.
#[derive(Clone, Default)]
pub struct LMStateRef(Rc<RefCell<LMState>>);

impl LMStateRef {
    /// Create a root LMStateRef.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a LMStateRef with a given LMState and a token.
    /// If the LMStateRef already has a child state with the same token, the child state will be returned.
    /// If not, a new child state will be created and returned.
    pub fn child(&self, token: i32) -> LMStateRef {
        let mut state = self.0.borrow_mut();
        let child = state.children.entry(token).or_insert_with(LMStateRef::new);
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
    fn score(&self, state: &LMStateRef, token: i32) -> (LMStateRef, f32);
    // Returns the new state and the score of the final state.
    fn finish(&self, state: &LMStateRef) -> (LMStateRef, f32);
}

/// ZeroLM is a language model that always returns 0.
/// This is a stub implementation of LM for interface consistency.
#[derive(Debug)]
pub struct ZeroLM;

impl LM for ZeroLM {
    fn start(&mut self) -> LMStateRef {
        LMStateRef::new()
    }

    fn score(&self, state: &LMStateRef, token: i32) -> (LMStateRef, f32) {
        (state.child(token), 0.0)
    }

    fn finish(&self, state: &LMStateRef) -> (LMStateRef, f32) {
        (state.clone(), 0.0)
    }
}
