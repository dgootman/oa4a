"""Mock Provider"""

import textwrap
import uuid
from base64 import b64encode
from datetime import datetime
from pathlib import Path
from typing import Generator

from pydantic import SecretStr

from .model import (
    ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice1,
    Choice3,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateImageRequest,
    Image,
    ImagesResponse,
)
from .provider import Provider
from .provider_utils import store_image


# pylint: disable-next=too-few-public-methods
class MockProvider(Provider):
    """Mock Provider"""

    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
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
                    object="chat.completion.chunk",
                )

                yield response

            return stream_response()
        else:
            response = CreateChatCompletionResponse(
                id=str(uuid.uuid4()),
                choices=[
                    Choice1(
                        message=ChatCompletionResponseMessage(
                            content=response_content,
                            role="assistant",
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ],
                created=int(datetime.now().timestamp()),
                model=request.model,
                object="chat.completion",
            )

            return response

    def create_image(
        self,
        request: CreateImageRequest,
    ) -> ImagesResponse:
        """Creates an image given a prompt."""

        return ImagesResponse(
            data=[
                Image(url=store_image(b64encode(Path("ai.png").read_bytes()).decode()))
            ],
            created=int(datetime.now().timestamp()),
        )
