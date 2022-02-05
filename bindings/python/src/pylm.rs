use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub(crate) struct PyLMState(PyObject);

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct PyLM(pub PyObject);

impl ctclib::LM for PyLM {
    type State = PyLMState;

    fn start(&mut self) -> ctclib::LMStateRef<Self::State> {
        let state = Python::with_gil(|py| self.0.call_method1(py, "start", ()));
        state
            .map(|state| ctclib::LMStateRef::new(PyLMState(state)))
            .unwrap()
    }

    fn score(
        &mut self,
        state: &ctclib::LMStateRef<Self::State>,
        token: i32,
        n_vocab: usize,
    ) -> (ctclib::LMStateRef<Self::State>, f32) {
        let pystate = state.borrow_internal_state().0.clone();
        let (next_pystate, score): (PyObject, f32) = Python::with_gil(|py| {
            self.0
                .call_method1(py, "score", (pystate, token, n_vocab))
                .unwrap()
                .extract::<(PyObject, f32)>(py)
                .unwrap()
        });
        (state.child(token, n_vocab, PyLMState(next_pystate)), score)
    }

    fn finish(
        &mut self,
        state: &ctclib::LMStateRef<Self::State>,
    ) -> (ctclib::LMStateRef<Self::State>, f32) {
        let pystate = state.borrow_internal_state().0.clone();
        let (_, score): (PyObject, f32) = Python::with_gil(|py| {
            self.0
                .call_method1(py, "finish", (pystate,))
                .unwrap()
                .extract::<(PyObject, f32)>(py)
                .unwrap()
        });
        (state.clone(), score)
    }
}
