use std::sync::Mutex;

use numpy::{array::PyArray2, ndarray::Dimension};
use pyo3::{exceptions, prelude::*, PyObjectProtocol};

mod pylm;

#[pyclass]
#[derive(Clone)]
struct BeamSearchDecoderOptions(ctclib::BeamSearchDecoderOptions);

#[pymethods]
impl BeamSearchDecoderOptions {
    #[new]
    #[args(lm_weight = "0.0")]
    fn new(beam_size: usize, beam_size_token: usize, beam_threshold: f32, lm_weight: f32) -> Self {
        Self(ctclib::BeamSearchDecoderOptions {
            beam_size,
            beam_size_token,
            beam_threshold,
            lm_weight,
        })
    }
}

#[pyproto]
impl PyObjectProtocol for BeamSearchDecoderOptions {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.0,).into())
    }
}

#[pyclass]
struct DecoderOutput(ctclib::DecoderOutput);

#[pymethods]
impl DecoderOutput {
    #[getter]
    fn score(&self) -> f32 {
        self.0.score
    }

    #[getter]
    fn am_score(&self) -> f32 {
        self.0.am_score
    }

    #[getter]
    fn tokens(&self) -> Vec<i32> {
        self.0.tokens.clone()
    }
}

#[pyproto]
impl PyObjectProtocol for DecoderOutput {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.0,).into())
    }
}

#[pyclass(subclass)]
struct Decoder(Box<dyn ctclib::Decoder + 'static + Send>);

#[pymethods]
impl Decoder {
    #[staticmethod]
    fn greedy() -> Self {
        Decoder(Box::new(ctclib::GreedyDecoder))
    }

    fn decode(&mut self, data: &PyArray2<f32>) -> PyResult<Vec<DecoderOutput>> {
        let (steps, tokens) = data.dims().into_pattern();
        let data = data.readonly();
        let data = data.as_slice()?;
        let outputs = self
            .0
            .decode(data, steps, tokens)
            .into_iter()
            .map(|o| DecoderOutput(o))
            .collect::<Vec<_>>();
        Ok(outputs)
    }
}

#[pyclass(extends=Decoder)]
struct GreedyDecoder;

#[pymethods]
impl GreedyDecoder {
    #[new]
    fn new() -> (Self, Decoder) {
        (GreedyDecoder, Decoder(Box::new(ctclib::GreedyDecoder)))
    }
}

struct BeamSearchDecoderWrapper<T: ctclib::LM>(Mutex<ctclib::BeamSearchDecoder<T>>);

impl<T: ctclib::LM> BeamSearchDecoderWrapper<T> {
    fn new(decoder: ctclib::BeamSearchDecoder<T>) -> Self {
        Self(Mutex::new(decoder))
    }
}

impl<T: ctclib::LM> ctclib::Decoder for BeamSearchDecoderWrapper<T> {
    fn decode(&mut self, data: &[f32], steps: usize, tokens: usize) -> Vec<ctclib::DecoderOutput> {
        self.0.lock().unwrap().decode(data, steps, tokens)
    }
}

unsafe impl<T: ctclib::LM> Send for BeamSearchDecoderWrapper<T> {}

#[pyclass(extends=Decoder)]
struct BeamSearchDecoder;

#[pymethods]
impl BeamSearchDecoder {
    #[new]
    fn new(options: BeamSearchDecoderOptions, blank_id: i32) -> (Self, Decoder) {
        (
            BeamSearchDecoder,
            Decoder(Box::new(BeamSearchDecoderWrapper::new(
                ctclib::BeamSearchDecoder::new(options.0, blank_id, ctclib::ZeroLM),
            ))),
        )
    }
}

#[pyclass(extends=Decoder)]
struct BeamSearchDecoderWithKenLM;

#[pymethods]
impl BeamSearchDecoderWithKenLM {
    #[new]
    fn new(
        options: BeamSearchDecoderOptions,
        blank_id: i32,
        model_path: &str,
        labels: Vec<String>,
    ) -> PyResult<(Self, Decoder)> {
        let dict = ctclib::Dict::from_entries(labels)
            .map_err(|err| exceptions::PyRuntimeError::new_err(format!("{}", err)))?;
        let kenlm = ctclib::KenLM::new(model_path, &dict);
        Ok((
            BeamSearchDecoderWithKenLM,
            Decoder(Box::new(BeamSearchDecoderWrapper::new(
                ctclib::BeamSearchDecoder::new(options.0, blank_id, kenlm),
            ))),
        ))
    }
}

#[pyclass(extends=Decoder)]
struct BeamSearchDecoderWithPyLM;

#[pymethods]
impl BeamSearchDecoderWithPyLM {
    #[new]
    fn new(
        options: BeamSearchDecoderOptions,
        blank_id: i32,
        lm: PyObject,
    ) -> PyResult<(Self, Decoder)> {
        Ok((
            BeamSearchDecoderWithPyLM,
            Decoder(Box::new(BeamSearchDecoderWrapper::new(
                ctclib::BeamSearchDecoder::new(options.0, blank_id, pylm::PyLM(lm)),
            ))),
        ))
    }
}

#[pymodule]
fn pyctclib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Decoder>()?;
    m.add_class::<DecoderOutput>()?;
    m.add_class::<GreedyDecoder>()?;
    m.add_class::<BeamSearchDecoder>()?;
    m.add_class::<BeamSearchDecoderOptions>()?;
    m.add_class::<BeamSearchDecoderWithKenLM>()?;
    m.add_class::<BeamSearchDecoderWithPyLM>()?;
    Ok(())
}
