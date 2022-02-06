import abc
from typing import List, Generic, TypeVar, Tuple
from typing_extensions import Protocol

import numpy as np


class DecoderOutput:
    score: float
    tokens: List[int]
    timesteps: List[int]
    am_scores: List[float]
    lm_scores: List[float]


class Decoder:
    def decode(
        self, 
        data: np.ndarray, 
        blank_id: int,
    ) -> List[DecoderOutput]:
        ...


class GreedyDecoder(Decoder):
    ...


class BeamSearchDecoderOptions:
    beam_size: int
    beam_size_token: int
    beam_threshold: float
    lm_weight: float

    def __init__(
        self,
        beam_size: int,
        beam_size_token: int,
        beam_threshold: float,
        lm_weight: float,
    ) -> None:
        ...


class BeamSearchDecoder(Decoder):
    def __init__(self, options: BeamSearchDecoderOptions) -> None:
        ...


class BeamSearchDecoderWithKenLM(Decoder):
    def __init__(
        self,
        options: BeamSearchDecoderOptions,
        model_path: str,
        vocab: List[str],
    ) -> None:
        ...


LMState = TypeVar("LMState")


class LMProtocol(Generic[LMState], Protocol):
    def start(self) -> LMState:
        ...

    def score(self, prev_state: LMState, token: int, n_vocab: int) -> Tuple[LMState, float]:
        ...

    def finish(self, prev_state: LMState) -> Tuple[LMState, float]:
        ...


class BeamSearchDecoderWithPyLM(Generic[LMState], Decoder):
    def __init__(
        self,
        options: BeamSearchDecoderOptions,
        lm: LMProtocol[LMState],
    ) -> None:
        ...