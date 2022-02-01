pub mod kenlm;

use std::{cell::RefCell, collections::HashMap, fmt::Debug, rc::Rc};

pub trait LMState: PartialOrd + PartialEq + Clone + Sized + 'static {
    /// Create a LMStateRef with a given LMState and a token.
    /// If the LMStateRef already has a child state with the same token, the child state will be returned.
    /// If not, a new child state will be created and returned.
    fn child(&self, token: i32) -> Self;
}

#[derive(Debug, Default)]
pub struct DefaultLMState {
    children: HashMap<i32, LMStateRef<DefaultLMState>>,
}

/// A reference to a LM state.
/// LMStateRef holds the information of the token sequence being decoded, and the identity of the token sequence can be confirmed by comparing LMStateRef.
#[derive(Default)]
pub struct LMStateRef<T>(Rc<RefCell<T>>);

impl<T: Default> LMStateRef<T> {
    /// Create a root LMStateRef.
    pub fn new() -> Self {
        Self::default()
    }
}

impl <T> Clone for LMStateRef<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl LMState for LMStateRef<DefaultLMState> {
    fn child(&self, token: i32) -> Self {
        let mut state = self.0.borrow_mut();
        let child = state.children.entry(token).or_insert_with(LMStateRef::new);
        child.clone()
    }
}

impl<T> Debug for LMStateRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:p}", self.0.as_ptr())
    }
}

impl<T> PartialEq for LMStateRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

impl<T> PartialOrd for LMStateRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.as_ptr().partial_cmp(&other.0.as_ptr())
    }
}

/// LM is a wrapper of a language model for decoding.
pub trait LM {
    type State: LMState;
    /// Initializes the LM, then returns the root LMStateRef.
    fn start(&mut self) -> Self::State;
    // Returns the new state and the score when a token comes as a continuation of LMStateRef.
    fn score(&self, state: &Self::State, token: i32) -> (Self::State, f32);
    // Returns the new state and the score of the final state.
    fn finish(&self, state: &Self::State) -> (Self::State, f32);
}

/// ZeroLM is a language model that always returns 0.
/// This is a stub implementation of LM for interface consistency.
#[derive(Debug)]
pub struct ZeroLM;

impl LM for ZeroLM {
    type State = LMStateRef<DefaultLMState>;

    fn start(&mut self) -> LMStateRef<DefaultLMState> {
        LMStateRef::new()
    }

    fn score(&self, state: &LMStateRef<DefaultLMState>, token: i32) -> (LMStateRef<DefaultLMState>, f32) {
        (state.child(token), 0.0)
    }

    fn finish(&self, state: &LMStateRef<DefaultLMState>) -> (LMStateRef<DefaultLMState>, f32) {
        (state.clone(), 0.0)
    }
}
