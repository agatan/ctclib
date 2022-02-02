use std::{cell::RefCell, collections::HashMap, ffi::CString, rc::Rc};

use crate::LMStateRef;

use super::LM;

#[derive(Debug, Clone)]
struct KenLMState(Rc<RefCell<kenlm_sys::lm_ngram_State>>);

impl KenLMState {
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(unsafe { std::mem::zeroed() })))
    }
}

/// A wrapper of a KenLM for decoding.
pub struct KenLM {
    model: *mut kenlm_sys::lm_base_Model,
    vocab: *const kenlm_sys::lm_base_Vocabulary,
    kenlm_states: HashMap<LMStateRef, KenLMState>,
}

impl KenLM {
    pub fn new<T: AsRef<str>>(path: T) -> Self {
        // TODO: convert user vocabulary to KenLM's vocabulary
        let x = CString::new(path.as_ref()).unwrap();
        let model = unsafe { kenlm_sys::lm_ngram_LoadVirtualWithDefaultConfig(x.as_ptr()) };
        let vocab = unsafe { kenlm_sys::lm_base_Model_BaseVocabulary(model) };
        Self {
            model,
            vocab,
            kenlm_states: HashMap::new(),
        }
    }
}

impl LM for KenLM {
    fn start(&mut self) -> LMStateRef {
        let outstate = LMStateRef::new();
        let internal_state = KenLMState::new();
        unsafe {
            kenlm_sys::lm_base_Model_NullContextWrite(
                self.model,
                internal_state.0.as_ptr() as *mut _,
            );
        }
        self.kenlm_states.insert(outstate.clone(), internal_state);
        outstate
    }

    fn score(&mut self, state: &LMStateRef, token: i32) -> (LMStateRef, f32) {
        let outstate = state.child(token);
        let out_kenlm_state = KenLMState::new();
        self.kenlm_states
            .entry(outstate.clone())
            .or_insert_with(|| out_kenlm_state.clone());
        let score = unsafe {
            kenlm_sys::lm_base_Model_BaseScore(
                self.model,
                state.0.as_ptr() as *const _,
                token as u32,
                out_kenlm_state.0.as_ptr() as *mut _,
            )
        };
        (outstate, score)
    }

    fn finish(&mut self, state: &LMStateRef) -> (LMStateRef, f32) {
        let eos = unsafe { kenlm_sys::lm_base_Vocabulary_EndSentence(self.vocab) };
        let outstate = state.child(-1);
        let out_kenlm_state = KenLMState::new();
        self.kenlm_states
            .entry(outstate.clone())
            .or_insert_with(|| out_kenlm_state.clone());
        let score = unsafe {
            kenlm_sys::lm_base_Model_BaseScore(
                self.model,
                state.0.as_ptr() as *const _,
                eos,
                out_kenlm_state.0.as_ptr() as *mut _,
            )
        };
        (outstate, score)
    }
}
