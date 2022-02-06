import time
import contextlib
import os

import numpy as np

import pyctclib


@contextlib.contextmanager
def timer(name):
    print("Start {}".format(name))
    print("=" * 80)
    start = time.time()
    yield
    end = time.time()
    print("Time: {:.3f}s ({})".format(end - start, name))
    print()


class LM:
    def __init__(self):
        self.score_called = 0

    def start(self):
        return None

    def score(self, prev_state, token, _):
        self.score_called += 1
        return prev_state, 0.0

    def finish(self, prev_state):
        return prev_state, 0.0


def read_sample():
    return np.loadtxt(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "logit.txt"
        )
    ).astype(np.float32)


def read_vocab():
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "data", "letter.dict")) as fp:
        return [x.strip() for x in fp.readlines()]

data = read_sample()
n_vocab = data.shape[-1]
vocab = read_vocab()
blank = n_vocab - 1
assert n_vocab == len(vocab) + 1

decoder: pyctclib.Decoder = pyctclib.GreedyDecoder()
with timer("GreedyDecoder"):
    output = decoder.decode(data, blank)[0]
result = "".join([vocab[i] for i in output.tokens])
print(result)
assert result == "MISTE|QUILTER|T|IS|TH|E|APOSTLESR|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLADB|TO|WELCOME|HIS|GOSUPEL|N|"

decoder = pyctclib.BeamSearchDecoder(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
)
with timer("BeamSearchDecoder"):
    output = decoder.decode(data, blank)[0]
result = "".join([vocab[i] for i in output.tokens])
print(result)
assert result == "MISTE|QUILTER|T|IS|TH|E|APOSTLESR|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLADB|TO|WELCOME|HIS|GOSPEL|N|"


lm = LM()
decoder = pyctclib.BeamSearchDecoderWithPyLM(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    lm,
)
with timer("BeamSearchDecoderWithPyLM"):
    output = decoder.decode(data, blank)[0]
result = "".join([vocab[i] for i in output.tokens])
print(result)
assert result == "MISTE|QUILTER|T|IS|TH|E|APOSTLESR|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLADB|TO|WELCOME|HIS|GOSPEL|N|"

decoder = pyctclib.BeamSearchDecoderWithKenLM(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "overfit.arpa"),
    vocab,
)
with timer("BeamSearchDecoderWithKenLM"):
    output = decoder.decode(data, blank)[0]
result = "".join([vocab[i] for i in output.tokens])
print(result)
assert result == "MISTE|QUILTER|T|IS|THE|APOSTLES|OF|THE|RIDDLE|CLASHES|AND|WEHARE|GOLAD|TO|WECOME|HIS|GOSPEL|"