use rustc_hash::FxHashMap;
use std::{
    borrow::Borrow,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DictError {
    #[error("duplicate entry in dictionary: {0}")]
    DuplicateEntry(String),
    #[error("missing index in dictionary: {0}")]
    MissingIndex(i32),
    #[error("missing entry in dictionary: {0}")]
    MissingEntry(String),
    #[error("failed to load dictionary")]
    Load(#[from] std::io::Error),
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Dict {
    entry2idx: FxHashMap<String, i32>,
    idx2entry: FxHashMap<i32, String>,
}

impl Dict {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read<P: AsRef<Path>>(p: P) -> Result<Self, DictError> {
        let file = File::open(p)?;
        Self::parse(file)
    }

    pub fn parse(p: impl Read) -> Result<Self, DictError> {
        let mut dict = Self::new();
        let buf = BufReader::new(p);
        for line in buf.lines() {
            let line = line?;
            dict.add_entry(line.trim().to_owned())?;
        }
        Ok(dict)
    }

    pub fn len(&self) -> usize {
        debug_assert!(self.entry2idx.len() == self.idx2entry.len());
        self.entry2idx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add_entry(&mut self, entry: String) -> Result<i32, DictError> {
        let mut idx = self.entry2idx.len() as i32;
        while self.idx2entry.contains_key(&idx) {
            idx += 1;
        }
        self.add_entry_at(entry, idx)?;
        Ok(idx)
    }

    pub fn add_entry_at(&mut self, entry: String, idx: i32) -> Result<(), DictError> {
        if self.entry2idx.contains_key(&entry) {
            return Err(DictError::DuplicateEntry(entry));
        }
        self.entry2idx.insert(entry.clone(), idx);
        self.idx2entry.insert(idx, entry);
        Ok(())
    }

    pub fn entry(&self, idx: i32) -> Result<&String, DictError> {
        match self.idx2entry.get(&idx) {
            Some(entry) => Ok(entry),
            None => Err(DictError::MissingIndex(idx)),
        }
    }

    pub fn index(&self, entry: &str) -> Result<i32, DictError> {
        match self.entry2idx.get(entry.borrow()) {
            Some(&idx) => Ok(idx),
            None => Err(DictError::MissingEntry(entry.to_owned())),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &i32)> {
        self.entry2idx.iter()
    }
}
