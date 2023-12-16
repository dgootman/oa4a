"""OA4A Server"""

import inspect
import logging
import textwrap
import uuid
from datetime import datetime

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, SecretStr
from sse_starlette import EventSourceResponse

from .model import (
    ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice1,
    Choice3,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    ListModelsResponse,
    Model,
)


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


@app.get("/v1/models", response_model_exclude_unset=True)
def models() -> ListModelsResponse:
    """
    Lists the currently available models,
    and provides basic information about each one such as the owner and availability.
    """
    return ListModelsResponse(
        data=[
            Model(
                id="gpt-3.5-turbo",
                created=int(datetime.now().timestamp()),
                object="model",
                owned_by="dgootman",
            )
        ],
        object="list",
    )


@app.post("/v1/chat/completions", response_model_exclude_unset=True)
def create_chat_completion(
    request: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse | CreateChatCompletionStreamResponse:
    """Creates a model response for the given chat conversation."""

    logger.debug(f"request: {request}")

    message = "\n".join(
        [
            m.root.content.get_secret_value()
            for m in request.messages
            if isinstance(m.root, ChatCompletionRequestUserMessage)
            and isinstance(m.root.content, SecretStr)
        ]
    )
    response_content = SecretStr(
        f"Here's what I got:\n{textwrap.indent(message, '  ')}"
    )

    # pylint: disable-next=no-else-return
    if request.stream:

        def stream_response():
            response_id = f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}"

            response = CreateChatCompletionStreamResponse(
                id=response_id,
                choices=[
                    Choice3(
                        delta=ChatCompletionStreamResponseDelta(
                            content=response_content,
                            role="assistant",
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=int(datetime.now().timestamp()),
                model=request.model,
                system_fingerprint=None,
                object="chat.completion.chunk",
            )

            yield response

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
                model=request.model,
                system_fingerprint=None,
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

        return EventSourceResponse(log_response(stream_response()))
    else:
        response = CreateChatCompletionResponse(
            id=str(uuid.uuid4()),
            choices=[
                Choice1(
                    message=ChatCompletionResponseMessage(
                        content=response_content,
                        role="assistant",
                        function_call=None,
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=int(datetime.now().timestamp()),
            model=request.model,
            system_fingerprint=None,
            object="chat.completion",
        )

        logger.debug(f"response: {response}")
        return response
