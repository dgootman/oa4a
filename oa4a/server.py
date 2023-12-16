"""OA4A Server"""

import json
import textwrap
import uuid
from datetime import datetime

from fastapi import FastAPI
from pydantic import SecretStr
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

app = FastAPI(openapi_url="/v1/openapi.json")


@app.get("/v1/models")
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


@app.post("/v1/chat/completions")
def create_chat_completion(
    request: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse | CreateChatCompletionStreamResponse:
    """Creates a model response for the given chat conversation."""

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
            response = CreateChatCompletionStreamResponse(
                id=str(uuid.uuid4()),
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

            yield response.model_dump_json()
            yield json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            yield "[DONE]"

        return EventSourceResponse(stream_response())
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

        return response
