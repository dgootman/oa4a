"""Amazon Bedrock Provider"""

import json
import textwrap
import uuid
from datetime import datetime
from typing import Generator

import boto3
from loguru import logger
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import SecretStr

from .model import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage,
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
class AmazonBedrockProvider(Provider):
    """Amazon Bedrock Provider"""

    def __init__(self) -> None:
        self.client: BedrockRuntimeClient = boto3.client("bedrock-runtime")

    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
        """Creates a model response for the given chat conversation."""

        # https://docs.aws.amazon.com/bedrock/latest/userguide/api-methods-run-inference.html

        model_id = "meta.llama2-13b-chat-v1"

        def llama_prompt() -> str:
            """
            Generate a Llama 2 prompt following the algorithm here:
            https://github.com/facebookresearch/llama/blob/main/llama/generation.py
            """
            messages = request.messages
            if messages[0].root.role == "system":
                messages = (
                    [
                        ChatCompletionRequestMessage(
                            root=ChatCompletionRequestUserMessage(
                                content=SecretStr(
                                    textwrap.dedent(
                                        f"""
                                        <<SYS>>
                                        {messages[0].root.content.get_secret_value()}
                                        <</SYS>>
                                        {messages[1].root.content.get_secret_value()}
                                        """.strip()
                                    )
                                ),
                                role="user",
                            )
                        )
                    ]
                    + messages[2:]
                )

            return "\n\n".join(
                m.root.content.get_secret_value()
                if m.root.role == "assistant"
                else f"[INST]{m.root.content.get_secret_value()}[/INST]"
                for m in messages
            )

        prompt = llama_prompt()

        body = json.dumps(
            {"prompt": prompt}
            | (
                {"max_gen_len": request.max_tokens}
                if request.max_tokens and request.max_tokens != "inf"
                else {}
            )
            | ({"temperature": request.temperature} if request.temperature else {})
            | ({"top_p": request.top_p} if request.top_p else {})
        )

        logger.trace(f"bedrock request: {dict({'body': body, 'modelId': model_id})}")

        # pylint: disable-next=no-else-return
        if request.stream:
            response = self.client.invoke_model_with_response_stream(
                modelId=model_id, body=body
            )

            response_id = f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}"

            stream = response["body"]

            def stream_response():
                for event in stream:
                    chunk = event["chunk"]
                    line = chunk["bytes"].decode()
                    logger.trace(f"bedrock response stream: {line}")
                    data = json.loads(line)

                    yield CreateChatCompletionStreamResponse(
                        id=response_id,
                        choices=[
                            Choice3(
                                delta=ChatCompletionStreamResponseDelta(
                                    content=SecretStr(data["generation"]),
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

            return stream_response()
        else:
            accept = "application/json"
            content_type = "application/json"
            response = self.client.invoke_model(
                body=body, modelId=model_id, accept=accept, contentType=content_type
            )
            response_body = json.loads(response["body"].read())
            logger.trace(f"bedrock response: {response_body}")

            generation = response_body["generation"]

            response = CreateChatCompletionResponse(
                id=str(uuid.uuid4()),
                choices=[
                    Choice1(
                        message=ChatCompletionResponseMessage(
                            content=SecretStr(generation),
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
