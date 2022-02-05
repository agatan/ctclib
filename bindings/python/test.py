import time
import contextlib

import pyctclib


@contextlib.contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print("Time: {:.3f}s".format(end - start))


class LM:
    def __init__(self):
        self.score_called = 0

    def start(self):
        return None

    def score(self, prev_state, token, _):
        self.score_called += 1
        return prev_state, 1.0

    def finish(self, prev_state):
        return prev_state, 1.0


def read_sample():
    with open("../../data/logit.txt", "r") as fp:
        steps, n_vocab = [int(x) for x in fp.readline().strip().split(" ")]
        logits = [float(x.strip()) for x in fp.readlines() if x.strip()]
        return logits, steps, n_vocab


def read_vocab():
    with open("../../data/letter.dict", "r") as fp:
        return [x.strip() for x in fp.readlines()]


def decode_ctc(tokens, blank):
    prev = blank
    for token in tokens:
        if token != blank and token != prev:
            yield token
        prev = token

data, steps, n_vocab = read_sample()
vocab = read_vocab()

decoder = pyctclib.GreedyDecoder()
with timer():
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data, steps, n_vocab)[0].tokens, n_vocab - 1)]))

decoder = pyctclib.BeamSearchDecoder(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    n_vocab - 1,
)
with timer():
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data, steps, n_vocab)[0].tokens, n_vocab - 1)]))

lm = LM()
decoder = pyctclib.BeamSearchDecoderWithPyLM(
    pyctclib.BeamSearchDecoderOptions(100, 1000, 1000, 0.5),
    n_vocab - 1,
    lm,
)
with timer():
    print("".join([vocab[i] for i in decode_ctc(decoder.decode(data, steps, n_vocab)[0].tokens, n_vocab - 1)]))
