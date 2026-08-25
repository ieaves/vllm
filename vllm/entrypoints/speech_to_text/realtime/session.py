# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types a realtime model uses to drive a transcription turn.

A model implements ``SupportsRealtime.realtime_session`` as an async generator that
yields :class:`Command` values and receives a :class:`DecodeResult` back from each
:class:`Decode` it yields. The connection layer interprets the commands; it never
inspects model output or decides what the client sees.
"""

import asyncio
from dataclasses import dataclass, field

import numpy as np

from vllm.inputs import PromptType


@dataclass(frozen=True, slots=True)
class Decode:
    """A request to run the model once over ``prompt``."""

    prompt: PromptType
    max_tokens: int
    temperature: float = 0.0
    continue_session: bool = field(default=False)
    """Append to the turn's existing engine request instead of starting a fresh one.

    Fresh requests share no KV or prompt state, so a model whose processor accepts one
    audio item per request must leave this False. Models that interleave audio and text
    at a fixed rate set it True to keep the accumulated session.
    """

    logprobs: int | None = None
    """Number of prompt logprobs to return, for models that score their own prior output."""


@dataclass(frozen=True, slots=True)
class Emit:
    """Text to deliver to the client as a transcription delta.

    Deltas cannot be retracted, so only emit text the model will not revise.
    """

    text: str


Command = Decode | Emit


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """What one :class:`Decode` produced."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None
    prompt_tokens: int


class AudioStream:
    """The turn's audio, readable on the consumer's own cadence.

    Reads are satisfied from whatever has arrived, so a model paces its decodes by wall
    clock rather than by the granularity the client happens to send audio in.
    """

    def __init__(self, queue: "asyncio.Queue[np.ndarray | None]") -> None:
        self._queue = queue
        self._ended = False

    @property
    def ended(self) -> bool:
        """Whether the client has signalled the end of the turn."""
        return self._ended

    async def read(
        self,
        min_samples: int = 0,
        timeout: float | None = None,
    ) -> np.ndarray | None:
        """Collect audio until ``min_samples`` arrive, ``timeout`` elapses, or the turn ends.

        Args:
            min_samples: Stop collecting once this many samples are in hand.
            timeout: Seconds to wait before returning what has arrived; None waits
                until ``min_samples`` or the end of the turn.

        Returns:
            The samples collected, or None once the turn has ended and nothing is left.
        """
        if self._ended:
            return None

        chunks: list[np.ndarray] = []
        total = 0
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout

        while total < min_samples or not chunks:
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except (TimeoutError, asyncio.TimeoutError):
                break
            if chunk is None:
                self._ended = True
                break
            chunks.append(chunk)
            total += len(chunk)

        if not chunks:
            return None if self._ended else np.empty(0, dtype=np.float32)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
