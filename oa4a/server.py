"""OA4A Server"""

import inspect
import logging
import os
from datetime import datetime

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from .amazon_bedrock_provider import AmazonBedrockProvider
from .mock_provider import MockProvider
from .model import (
    ChatCompletionStreamResponseDelta,
    Choice3,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    ListModelsResponse,
)
from .ollama_provider import OllamaProvider
from .provider import Provider


class InterceptHandler(logging.Handler):
    """
    Loguru interceptor for standard logging

    See https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
app = FastAPI(openapi_url="/v1/openapi.json")

PROVIDERS = {
    "mock": MockProvider,
    "ollama": OllamaProvider,
    "bedrock": AmazonBedrockProvider,
}

provider_name = os.environ.get("PROVIDER", None)
if not provider_name:
    raise ValueError("No provider specified using environment variable PROVIDER")

provider_type = PROVIDERS.get(provider_name, None)
if not provider_type:
    raise ValueError(f"Provider '{provider_name}' is not supported")

provider: Provider = provider_type()

logger.info(f"Using provider: {provider_name}")


@app.get("/v1/models", response_model_exclude_unset=True)
def list_models() -> ListModelsResponse:
    """
    Lists the currently available models,
    and provides basic information about each one such as the owner and availability.
    """
    response = provider.list_models()
    logger.debug(f"response: {response}")
    return response


@app.post("/v1/chat/completions", response_model_exclude_unset=True)
def create_chat_completion(
    request: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse | CreateChatCompletionStreamResponse:
    """Creates a model response for the given chat conversation."""

    logger.debug(f"request: {request}")

    response = provider.create_chat_completion(request)

    if isinstance(response, CreateChatCompletionResponse):
        logger.debug(f"response: {response}")
        return response

    def stream_response():
        first_response = next(response, None)
        if not first_response:
            raise ValueError("Provider didn't provide a response")

        response_id = first_response.id
        model = first_response.model
        yield first_response

        last_response = first_response
        for last_response in response:
            yield last_response

        if any(not choice.finish_reason for choice in last_response.choices):
            yield CreateChatCompletionStreamResponse(
                id=response_id,
                choices=[
                    Choice3(
                        delta=ChatCompletionStreamResponseDelta.model_validate({}),
                        finish_reason="stop",
                        index=0,
                    )
                ],
                created=int(datetime.now().timestamp()),
                model=model,
                object="chat.completion.chunk",
            )

        yield "[DONE]"

    def log_response(response_stream):
        for response in response_stream:
            if isinstance(response, BaseModel):
                logger.debug(f"stream response: {response}")
                yield response.model_dump_json(exclude_unset=True)
            else:
                yield response

    return EventSourceResponse(
        log_response(stream_response()),
        ping=3600,  # effectively disable ping since it violates the OpenAI API specs
    )
