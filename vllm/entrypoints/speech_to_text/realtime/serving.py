# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cached_property
from typing import Literal, cast

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.renderers.inputs.preprocess import parse_model_prompt

logger = init_logger(__name__)


class OpenAIServingRealtime(GenerateBaseServing):
    """Realtime audio transcription service via WebSocket streaming.

    Provides streaming audio-to-text transcription by transforming audio chunks
    into StreamingInput objects that can be consumed by the engine.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )

        self.task_type: Literal["realtime"] = "realtime"

        logger.info("OpenAIServingRealtime initialized for task: %s", self.task_type)

    @cached_property
    def model_cls(self) -> type[SupportsRealtime]:
        """Get the model class that supports transcription."""
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsRealtime], model_cls)

    async def render_prompt(self, prompt: PromptType):
        """Render one prompt into the engine's input form.

        Only the continued-session path needs this; an independent decode passes its
        prompt to ``generate`` directly.

        Args:
            prompt: A prompt from a model's ``Decode`` command.
        """
        parsed_prompt = parse_model_prompt(self.model_config, prompt)
        (engine_input,) = await self.renderer.render_cmpl_async([parsed_prompt])
        return engine_input
