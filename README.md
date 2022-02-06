# ctclib

[![ctclib at crates.io](https://img.shields.io/crates/v/ctclib.svg)](https://crates.io/crates/ctclib)
[![.github/workflows/ci.yml](https://github.com/agatan/ctclib/actions/workflows/ci.yml/badge.svg)](https://github.com/agatan/ctclib/actions/workflows/ci.yml)

**NOTE: This is currently under development.**

A collection of utilities related to CTC, with the goal of being fast and highly flexible.  

## Features

- CTC Decode
  - [x] Greedy Decoder
  - [x] Beam Search Decoder
  - [x] Beam Search Decoder with KenLM
  - [x] Beam Search Decoder with user-defined LM
  - [x] Python bindings

## Installation

`ctclib` depends on [kpu/kenlm](https://github.com/kpu/kenlm).
You must install the following libraries as KenLM dependencies.

- Boost
- Eigen3

For example, if you are using Ubuntu (or some Debian based Linux), you can install them by running the following command:

```sh
apt install libboost-all-dev libeigen3-dev
```

### Use ctclib from Rust

Currently, `ctclib` isn't available on crates.io, but you can use this as git dependencies.

```toml
[dependencies]
ctclib = { version = "*", git = "https://github.com/agatan/ctclib" }
```

### Use ctclib from Python

`ctclib` provides python interfaces, named `pyctclib`.
Currently, `pyctclib` isn't available on PyPI, but you can install this as git dependency.
Ensure that you have installed `cargo` and `libclang-dev`.

```sh
pip install 'git+https://github.com/agatan/ctclib.git#egg=pyctclib&subdirectory=bindings/python'
```

## Example

- [ctc decoder in python](./bindings/python/test.py)
- [ctc decoder in rust](./benches/decode.rs)

```python
import pyctclib

decoder = pyctclib.BeamSearchDecoderWithKenLM(
    pyctclib.BeamSearchDecoderOptions(
      beam_size=100,
      beam_size_token=1000,
      beam_threshold=1,
      lm_weight=0.5,
    ),
    "/path/to/model.arpa",
    ["a", "b", "c", "_"],
)
decode.decode(log_probs)

# or you can use user-defined LM
# See pyctclib.LMProtocol
```