# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

import numpy as np
import pybase64 as base64
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from .protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    TranscriptionDelta,
    TranscriptionDone,
)
from .serving import OpenAIServingRealtime
from .session import AudioStream, Decode, DecodeResult, Emit

logger = init_logger(__name__)


class RealtimeConnection:
    """Manages WebSocket lifecycle and state for realtime transcription.

    This class handles:
    - WebSocket connection lifecycle (accept, receive, send, close)
    - Event routing (session.update, append, commit)
    - Audio buffering via asyncio.Queue
    - Generation task management
    - Error handling and cleanup
    """

    def __init__(self, websocket: WebSocket, serving: OpenAIServingRealtime):
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None

        self._is_connected = False
        self._is_model_validated = False

        self._max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.connection_id)
        self._is_connected = True

        # Send session created event
        await self.send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error(sanitize_message(str(e)), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        if self.serving._is_model_supported(model):
            return None

        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model
        - input_audio_buffer.append: Add audio chunk to queue
        - input_audio_buffer.commit: Start transcription generation
        """
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            model = event.get("model")
            if model is None:
                await self.send_error("Missing required field: model", "invalid_event")
                return
            err = self._check_model(model)
            if err is not None:
                await self.send_error(err.error.message, "model_not_found")
                return
            self._is_model_validated = True
        elif event_type == "input_audio_buffer.append":
            append_event = InputAudioBufferAppend(**event)
            try:
                audio_bytes = base64.b64decode(append_event.audio)
                # Convert PCM16 bytes to float32 numpy array
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if len(audio_array) / 1024**2 > self._max_audio_filesize_mb:
                    raise VLLMValidationError(
                        "Maximum file size exceeded",
                        parameter="audio_filesize_mb",
                        value=len(audio_array) / 1024**2,
                    )
                if len(audio_array) == 0:
                    raise VLLMValidationError("Can't process empty audio.")

                # Put audio chunk in queue
                self.audio_queue.put_nowait(audio_array)

            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self.send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            if not self._is_model_validated:
                err_msg = (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )
                return

            commit_event = InputAudioBufferCommit(**event)
            # final signals that the audio is finished
            if commit_event.final:
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def start_generation(self):
        """Start the transcription turn."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        audio = AudioStream(self.audio_queue)
        session = self.serving.model_cls.realtime_session(
            audio, self.serving.model_config
        )
        self.generation_task = asyncio.create_task(self._run_session(session))

    async def _run_session(self, session: AsyncGenerator):
        """Interpret the model's commands for one turn.

        ``transcription.done`` carries exactly the concatenation of the deltas sent,
        because both come from the same ``Emit`` commands.
        """
        emitted: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        executor = _DecodeExecutor(self)

        try:
            result: DecodeResult | None = None
            while True:
                try:
                    command = await session.asend(result)
                except StopAsyncIteration:
                    break
                result = None

                if isinstance(command, Emit):
                    if command.text:
                        emitted.append(command.text)
                        await self.send(TranscriptionDelta(delta=command.text))
                elif isinstance(command, Decode):
                    result = await executor.run(command)
                    prompt_tokens += result.prompt_tokens
                    completion_tokens += len(result.token_ids)
                else:
                    raise TypeError(
                        f"unsupported realtime command: {type(command).__name__}"
                    )

                if not self._is_connected:
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            await self.send(TranscriptionDone(text="".join(emitted), usage=usage))

        except Exception as e:
            logger.exception("Error in realtime session: %s", e)
            await self.send_error(sanitize_message(str(e)), "processing_error")
        finally:
            await executor.aclose()
            await session.aclose()
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def send(
        self, event: SessionCreated | TranscriptionDelta | TranscriptionDone
    ):
        """Send event to client."""
        data = event.model_dump_json()
        await self.websocket.send_text(data)

    async def send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        error_event = ErrorEvent(error=message, code=code)
        await self.websocket.send_text(error_event.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        # Signal audio stream to stop
        self.audio_queue.put_nowait(None)

        # Cancel generation task if running
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

        logger.debug("Connection cleanup complete: %s", self.connection_id)


class _DecodeExecutor:
    """Runs the ``Decode`` commands of one turn.

    A decode with ``continue_session`` set joins the turn's streaming request so its
    KV persists; every other decode is an independent request that shares no state.
    """

    def __init__(self, connection: RealtimeConnection) -> None:
        self._connection = connection
        self._prompts: asyncio.Queue | None = None
        self._outputs: AsyncGenerator | None = None

    async def run(self, command: Decode) -> DecodeResult:
        """Execute one decode and return what it produced."""
        if command.continue_session:
            return await self._run_continued(command)
        return await self._run_fresh(command)

    async def _run_fresh(self, command: Decode) -> DecodeResult:
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        params = SamplingParams.from_optional(
            temperature=command.temperature,
            max_tokens=command.max_tokens,
            output_kind=RequestOutputKind.FINAL_ONLY,
            prompt_logprobs=command.logprobs,
        )
        request_id = f"rt-{self._connection.connection_id}-{uuid4()}"
        final = None
        async for output in self._connection.serving.engine_client.generate(
            prompt=command.prompt,
            sampling_params=params,
            request_id=request_id,
        ):
            final = output
        return _as_result(final)

    async def _run_continued(self, command: Decode) -> DecodeResult:
        from vllm.sampling_params import RequestOutputKind, SamplingParams

        if self._outputs is None:
            self._prompts = asyncio.Queue()
            params = SamplingParams.from_optional(
                temperature=command.temperature,
                max_tokens=command.max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )
            self._outputs = self._connection.serving.engine_client.generate(
                prompt=self._prompt_stream(),
                sampling_params=params,
                request_id=f"rt-{self._connection.connection_id}-{uuid4()}",
            ).__aiter__()

        assert self._prompts is not None
        await self._prompts.put(command.prompt)
        try:
            return _as_result(await self._outputs.__anext__())
        except StopAsyncIteration:
            return DecodeResult(text="", token_ids=(), finish_reason="stop", prompt_tokens=0)

    async def _prompt_stream(self) -> AsyncGenerator:
        from vllm.engine.protocol import StreamingInput

        assert self._prompts is not None
        while True:
            prompt = await self._prompts.get()
            if prompt is None:
                return
            rendered = await self._connection.serving.render_prompt(prompt)
            yield StreamingInput(prompt=rendered)

    async def aclose(self) -> None:
        """Release the streaming request, if one was opened."""
        if self._prompts is not None:
            self._prompts.put_nowait(None)
        if self._outputs is not None:
            await self._outputs.aclose()
            self._outputs = None


def _as_result(output) -> DecodeResult:
    """Project an engine output onto the model-facing result type."""
    if output is None or not output.outputs:
        return DecodeResult(text="", token_ids=(), finish_reason=None, prompt_tokens=0)
    completion = output.outputs[0]
    return DecodeResult(
        text=completion.text,
        token_ids=tuple(completion.token_ids),
        finish_reason=completion.finish_reason,
        prompt_tokens=len(output.prompt_token_ids or ()),
    )
