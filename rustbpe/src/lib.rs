use ahash::AHashMap;
use compact_str::CompactString;
use fancy_regex::Regex;
use pyo3::{buffer, prelude::*};
use pyo3::{pyclass, pymethods, pymodule};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cell::Ref;
use std::collections::HashMap as StdHashMap;

const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
type Pair = (u32, u32);

#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Regex,
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(pattern: &str) -> Self {
        let compiled = Regex::new(pattern).unwrap();
        Tokenizer {
            merges: StdHashMap::new(),
            pattern: pattern.to_string(),
            compiled_pattern: compiled,
        }
    }

    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid regex patter:{}", err))
        })?;
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };
        //
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);
        log::info!("process (buffersize:{}", buffer_size);
        let mut total_sequences = 0u64;
        //fill the python string to the buf
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        pyo3::Bound::from_borrowed_ptr_or_opt(
                            py,
                            pyo3::ffi::PyIter_Next(it.as_ptr()),
                        )
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            })
        };

        loop {
            let exhuasted = refill(&mut buf)?;
            if buf.is_empty() && exhuasted {
                break;
            }
            total_sequences += buf.len() as u64;
            let pattern = self.compiled_pattern.clone();
            let local: AHashMap<CompactString, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in pattern.find_iter(s) {
                            let piece = mat.expect("regex match failed").as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }
            if exhuasted {
                break;
            }
        }

        log::info!(
            "processed {} sequences total,{} unique",
            total_sequences,
            counts.len()
        );

        Ok(())
    }
}

#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // rust log to python log
    m.add_class::<Tokenizer>()?;
    Ok(())
}
