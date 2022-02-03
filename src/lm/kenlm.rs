use std::{collections::HashMap, ffi::CString};

use crate::{Dict, LMStateRef};

use super::LM;

pub type KenLMWordIndex = kenlm_sys::lm_WordIndex;

#[derive(Debug, Clone)]
struct KenLMState(kenlm_sys::lm_ngram_State);

impl KenLMState {
    pub fn new() -> Self {
        Self(unsafe { std::mem::zeroed() })
    }

    pub fn with_ptr<T: 'static>(&self, f: impl FnOnce(*const kenlm_sys::lm_ngram_State) -> T) -> T {
        f(&self.0 as *const _)
    }

    pub fn with_mut_ptr<T: 'static>(
        &mut self,
        f: impl FnOnce(*mut kenlm_sys::lm_ngram_State) -> T,
    ) -> T {
        let ptr = &mut self.0 as *mut kenlm_sys::lm_ngram_State;
        f(ptr)
    }
}

/// A wrapper of a KenLM Model.
struct Model(*mut kenlm_sys::lm_base_Model);

impl Model {
    fn new<T: AsRef<str>>(path: T) -> Self {
        let x = CString::new(path.as_ref()).unwrap();
        let model = unsafe { kenlm_sys::lm_ngram_LoadVirtualWithDefaultConfig(x.as_ptr()) };
        Self(model)
    }

    fn vocab(&self) -> Vocabulary {
        Vocabulary(
            unsafe { kenlm_sys::lm_base_Model_BaseVocabulary(self.0) },
            self,
        )
    }

    fn null_context(&self) -> KenLMState {
        let mut state = KenLMState::new();
        state.with_mut_ptr(|ptr| unsafe {
            kenlm_sys::lm_base_Model_NullContextWrite(self.0, ptr as *mut _)
        });
        state
    }

    fn base_score(&self, state: &KenLMState, token: KenLMWordIndex) -> (KenLMState, f32) {
        state.with_ptr(|state_ptr| {
            let mut outstate = KenLMState::new();
            let score = outstate.with_mut_ptr(|out| unsafe {
                kenlm_sys::lm_base_Model_BaseScore(
                    self.0,
                    state_ptr as *const _,
                    token as u32,
                    out as *mut _,
                )
            });
            (outstate, score)
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            kenlm_sys::lm_base_Model_delete(self.0);
        }
    }
}

/// A wrapper of a reference to KenLM Vocabulary
struct Vocabulary<'a>(*const kenlm_sys::lm_base_Vocabulary, &'a Model);

impl<'a> Vocabulary<'a> {
    fn end_sentence(&self) -> KenLMWordIndex {
        unsafe { kenlm_sys::lm_base_Vocabulary_EndSentence(self.0) }
    }

    fn index(&self, x: &str) -> KenLMWordIndex {
        unsafe {
            kenlm_sys::lm_base_Vocabulary_Index(
                self.0,
                x.as_ptr() as *const _,
                x.as_bytes().len() as u64,
            )
        }
    }
}

#[test]
fn load_model_and_get_vocab() {
    let model = Model::new("data/overfit.arpa");
    let vocab = model.vocab();
    assert_eq!(vocab.begin_sentence(), 1);
    assert_eq!(vocab.end_sentence(), 2);
    assert_eq!(vocab.index("M"), 3);
    assert_eq!(vocab.index("I"), 4);

    let null_context = model.null_context();
    let (next_context, score) = model.base_score(&null_context, vocab.index("M"));
    assert_eq!(score, -1.3728311);
    let (_, score) = model.base_score(&next_context, model.vocab().index("I"));
    assert_eq!(score, -0.77655447);

    // Drop explictly.
    std::mem::drop(model);
}

/// A wrapper of a KenLM for decoding.
pub struct KenLM {
    model: Model,
    kenlm_states: HashMap<LMStateRef, KenLMState>,
    idx_to_kenlm_idx: HashMap<i32, kenlm_sys::lm_WordIndex>,
}

impl KenLM {
    pub fn new<T: AsRef<str>>(path: T, dict: &Dict) -> Self {
        // TODO: convert user vocabulary to KenLM's vocabulary
        let model = Model::new(path);
        let vocab = model.vocab();

        let idx_to_kenlm_idx = dict
            .iter()
            .map(|(word, idx)| {
                let kenlm_idx = vocab.index(word);
                (*idx, kenlm_idx)
            })
            .collect::<HashMap<_, _>>();

        Self {
            model,
            kenlm_states: HashMap::new(),
            idx_to_kenlm_idx,
        }
    }
}

impl LM for KenLM {
    fn start(&mut self) -> LMStateRef {
        let outstate = LMStateRef::new();
        let initial_state = self.model.null_context();
        self.kenlm_states.insert(outstate.clone(), initial_state);
        outstate
    }

    fn score(&mut self, state: &LMStateRef, token: i32) -> (LMStateRef, f32) {
        let kenlm_idx = self.idx_to_kenlm_idx[&token];
        let outstate = state.child(token);
        let (next_kenlm_state, score) = {
            let kenlm_state = &self.kenlm_states[state];
            self.model.base_score(kenlm_state, kenlm_idx)
        };
        self.kenlm_states.insert(outstate.clone(), next_kenlm_state);
        (outstate, score)
    }

    fn finish(&mut self, state: &LMStateRef) -> (LMStateRef, f32) {
        let eos = self.model.vocab().end_sentence();
        let outstate = state.child(-1);
        let (next_kenlm_state, score) = {
            let kenlm_state = &self.kenlm_states[state];
            self.model.base_score(kenlm_state, eos)
        };
        self.kenlm_states.insert(outstate.clone(), next_kenlm_state);
        (outstate, score)
    }
}
