use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub(crate) struct PyLMState(PyObject);

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct PyLM(pub PyObject);

impl ctclib::LM for PyLM {
    type State = PyLMState;

    fn start(&mut self) -> Self::State {
        let state = Python::with_gil(|py| self.0.call_method1(py, "start", ()));
        state.map(PyLMState).unwrap()
    }

    fn score(&mut self, state: &Self::State, token: i32) -> f32 {
        Python::with_gil(|py| {
            self.0
                .call_method1(py, "score", (state.0.clone(), token))
                .unwrap()
                .extract::<f32>(py)
                .unwrap()
        })
    }

    fn next_state(&mut self, state: &Self::State, token: i32) -> Self::State {
        let next_pystate = Python::with_gil(|py| {
            self.0
                .call_method1(py, "next_state", (state.0.clone(), token))
                .unwrap()
                .extract::<PyObject>(py)
                .unwrap()
        });
        PyLMState(next_pystate)
    }

    fn finish(&mut self, state: &Self::State) -> f32 {
        Python::with_gil(|py| {
            self.0
                .call_method1(py, "finish", (state.0.clone(),))
                .unwrap()
                .extract::<f32>(py)
                .unwrap()
        })
    }
}
