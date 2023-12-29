"""Ollama Provider"""

import json
import os
import uuid
from datetime import datetime
from typing import Generator

import requests
from loguru import logger
from pydantic import SecretStr

from .model import (
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice1,
    Choice3,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from .provider import Provider


# pylint: disable-next=too-few-public-methods
class OllamaProvider(Provider):
    """Ollama Provider"""

    def __init__(
        self,
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama2"),
    ) -> None:
        self.client = requests.Session()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    # pylint: disable-next=duplicate-code
    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
        """Creates a model response for the given chat conversation."""

        ollama_request = {
            "model": self.ollama_model,
            "messages": [
                {"role": m.root.role, "content": m.root.content.get_secret_value()}
                for m in request.messages
            ],
            "stream": request.stream,
        }

        logger.trace(f"ollama request: {json.dumps(ollama_request)}")

        response = self.client.post(
            f"{self.ollama_url}/api/chat",
            json=ollama_request,
            stream=True,
        )

        logger.debug(f"ollama response: {response}")
        if not response.ok:
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                if not response.text:
                    raise e
                raise RuntimeError(response.text) from e

        # pylint: disable=duplicate-code
        # pylint: disable-next=no-else-return
        if request.stream:

            def stream_response():
                response_id = f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}"

                for line in response.iter_lines(decode_unicode=True):
                    logger.trace(f"ollama response stream: {line}")

                    data = json.loads(line)

                    if data["done"]:
                        break

                    yield CreateChatCompletionStreamResponse(
                        id=response_id,
                        choices=[
                            Choice3(
                                delta=ChatCompletionStreamResponseDelta(
                                    content=SecretStr(data["message"]["content"]),
                                    role=data["message"]["role"],
                                ),
                                finish_reason=None,
                                index=0,
                            )
                        ],
                        created=int(datetime.now().timestamp()),
                        model=request.model,
                        object="chat.completion.chunk",
                    )

            return stream_response()
        else:
            logger.trace(f"ollama response text: {response.text}")

            data = response.json()

            response = CreateChatCompletionResponse(
                id=str(uuid.uuid4()),
                choices=[
                    Choice1(
                        message=ChatCompletionResponseMessage(
                            content=SecretStr(data["message"]["content"]),
                            role=data["message"]["role"],
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
