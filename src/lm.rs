#[cfg(feature = "kenlm")]
pub mod kenlm;

use std::{
    cell::{Ref, RefCell},
    fmt::Debug,
    rc::Rc,
};

#[derive(Debug, Default)]
pub struct LMState<T> {
    children: Vec<Option<LMStateRef<T>>>,
    #[allow(dead_code)]
    state: T,
}

/// A reference to a LM state.
/// LMStateRef holds the information of the token sequence being decoded, and the identity of the token sequence can be confirmed by comparing LMStateRef.
#[derive(Default)]
pub struct LMStateRef<T>(Rc<RefCell<LMState<T>>>);

impl<T> Clone for LMStateRef<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> std::hash::Hash for LMStateRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state)
    }
}

impl<T> LMStateRef<T> {
    pub fn new(state: T) -> Self {
        Self(Rc::new(RefCell::new(LMState {
            children: Vec::new(),
            state,
        })))
    }

    pub fn child(&self, token: i32, n_vocab: usize, state: T) -> Self {
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
                let new_child = LMStateRef::new(state);
                *child = Some(new_child.clone());
                new_child
            }
        }
    }

    pub fn borrow_internal_state(&self) -> Ref<'_, T> {
        let r = self.0.borrow();
        Ref::map(r, |s| &s.state)
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
impl<T> Eq for LMStateRef<T> {}

impl<T> PartialOrd for LMStateRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for LMStateRef<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_ptr().cmp(&other.0.as_ptr())
    }
}

/// LM is a wrapper of a language model for decoding.
pub trait LM {
    type State;
    /// Initializes the LM, then returns the root LMStateRef.
    fn start(&mut self) -> LMStateRef<Self::State>;
    // Returns the new state and the score when a token comes as a continuation of LMStateRef.
    fn score(
        &mut self,
        state: &LMStateRef<Self::State>,
        token: i32,
        n_vocab: usize,
    ) -> (LMStateRef<Self::State>, f32);
    // Returns the new state and the score of the final state.
    fn finish(&mut self, state: &LMStateRef<Self::State>) -> (LMStateRef<Self::State>, f32);
}

/// ZeroLM is a language model that always returns 0.
/// This is a stub implementation of LM for interface consistency.
#[derive(Debug)]
pub struct ZeroLM;

impl LM for ZeroLM {
    type State = ();

    fn start(&mut self) -> LMStateRef<Self::State> {
        LMStateRef::new(())
    }

    fn score(
        &mut self,
        state: &LMStateRef<Self::State>,
        token: i32,
        n_vocab: usize,
    ) -> (LMStateRef<Self::State>, f32) {
        (state.child(token, n_vocab, ()), 0.0)
    }

    fn finish(&mut self, state: &LMStateRef<Self::State>) -> (LMStateRef<Self::State>, f32) {
        (state.clone(), 0.0)
    }
}
