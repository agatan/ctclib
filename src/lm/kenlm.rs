use std::ffi::CString;

use crate::{Dict, LMStateRef};

use super::LM;

pub type KenLMWordIndex = kenlm_sys::lm_WordIndex;

#[derive(Debug, Clone)]
pub struct KenLMState(kenlm_sys::lm_ngram_State);

impl KenLMState {
    fn new() -> Self {
        Self(unsafe { std::mem::zeroed() })
    }

    fn with_ptr<T: 'static>(&self, f: impl FnOnce(*const kenlm_sys::lm_ngram_State) -> T) -> T {
        f(&self.0 as *const _)
    }

    fn with_mut_ptr<T: 'static>(
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
    idx_to_kenlm_idx: Vec<KenLMWordIndex>,
    n_vocab: usize,
}

impl KenLM {
    pub fn new<T: AsRef<str>>(path: T, dict: &Dict) -> Self {
        // TODO: convert user vocabulary to KenLM's vocabulary
        let model = Model::new(path);
        let vocab = model.vocab();

        let mut idx_to_kenlm_idx = vec![0; dict.len()];

        for (word, &idx) in dict.iter() {
            let kenlm_idx = vocab.index(word);
            idx_to_kenlm_idx[idx as usize] = kenlm_idx;
        }

        Self {
            model,
            idx_to_kenlm_idx,
            n_vocab: dict.len(),
        }
    }
}

impl LM for KenLM {
    type State = KenLMState;

    fn start(&mut self) -> LMStateRef<Self::State> {
        let initial_state = self.model.null_context();
        LMStateRef::new(initial_state)
    }

    fn score(
        &mut self,
        state: &LMStateRef<Self::State>,
        token: i32,
    ) -> (LMStateRef<Self::State>, f32) {
        let kenlm_idx = self.idx_to_kenlm_idx[token as usize];
        let (next_kenlm_state, score) = {
            self.model
                .base_score(&state.borrow_internal_state(), kenlm_idx)
        };
        let outstate = state.child(token, self.n_vocab, next_kenlm_state);
        (outstate, score)
    }

    fn finish(&mut self, state: &LMStateRef<Self::State>) -> (LMStateRef<Self::State>, f32) {
        let eos = self.model.vocab().end_sentence();
        let (next_kenlm_state, score) =
            { self.model.base_score(&state.borrow_internal_state(), eos) };
        let outstate = state.child(-1, self.n_vocab, next_kenlm_state);
        (outstate, score)
    }
}
