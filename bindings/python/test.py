import time
import contextlib

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
    return np.loadtxt("../../data/logit.txt").astype(np.float32)


def read_vocab():
    with open("../../data/letter.dict", "r") as fp:
        return [x.strip() for x in fp.readlines()]


def decode_ctc(tokens, blank):
    prev = blank
    for token in tokens:
        if token != blank and token != prev:
            yield token
        prev = token

data = read_sample()
n_vocab = data.shape[-1]
vocab = read_vocab()
assert n_vocab == len(vocab) + 1

decoder = pyctclib.GreedyDecoder()
with timer("GreedyDecoder"):
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data)[0].tokens, n_vocab - 1)]))

decoder = pyctclib.BeamSearchDecoder(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    n_vocab - 1,
)
with timer("BeamSearchDecoder"):
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data)[0].tokens, n_vocab - 1)]))


lm = LM()
decoder = pyctclib.BeamSearchDecoderWithPyLM(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    n_vocab - 1,
    lm,
)
with timer("BeamSearchDecoderWithPyLM"):
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data)[0].tokens, n_vocab - 1)]))

decoder = pyctclib.BeamSearchDecoderWithKenLM(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    n_vocab - 1,
    "../../data/overfit.arpa",
    vocab,
)
with timer("BeamSearchDecoderWithKenLM"):
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data)[0].tokens, n_vocab - 1)]))