"""Amazon Bedrock Provider"""

import json
import os
import textwrap
import uuid
from abc import ABC, abstractmethod
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

    class Engine(ABC):
        """Bedrock engine interface for supporting different models"""

        @staticmethod
        @abstractmethod
        def body(request: CreateChatCompletionRequest) -> dict:
            """Create Bedrock request body"""

        @staticmethod
        @abstractmethod
        def parse(data: dict) -> str:
            """Parse response content from Bedrock response"""

        @staticmethod
        @abstractmethod
        def stream_parse(data: dict) -> str:
            """Parse response content from Bedrock response stream"""

    class Llama2Engine(Engine):
        """Bedrock engine for Llama2 models"""

        @staticmethod
        def body(request: CreateChatCompletionRequest) -> dict:
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

            prompt = "\n\n".join(
                m.root.content.get_secret_value()
                if m.root.role == "assistant"
                else f"[INST]{m.root.content.get_secret_value()}[/INST]"
                for m in messages
            )

            return (
                {"prompt": prompt}
                | (
                    {"max_gen_len": request.max_tokens}
                    if request.max_tokens and request.max_tokens != "inf"
                    else {}
                )
                | ({"temperature": request.temperature} if request.temperature else {})
                | ({"top_p": request.top_p} if request.top_p else {})
            )

        @staticmethod
        def parse(data: dict) -> str:
            return data["generation"]

        @staticmethod
        def stream_parse(data: dict) -> str:
            return AmazonBedrockProvider.Llama2Engine.parse(data)

    class TitanEngine(Engine):
        """Bedrock engine for Titan models"""

        @staticmethod
        def body(request: CreateChatCompletionRequest) -> dict:
            """
            Generate an Amazon Titan prompt following the algorithm here:
            https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Text+Prompt+Engineering+Guidelines.pdf
            """
            role_prefixes = {"system": "", "user": "User: ", "assistant": "Bot: "}

            prompt = (
                "\n\n".join(
                    f"{role_prefixes[m.root.role]}{m.root.content.get_secret_value()}"
                    for m in request.messages
                )
                + "\n\nBot: "
            )

            return {
                "inputText": prompt,
                "textGenerationConfig": {}
                | (
                    {"maxTokenCount": request.max_tokens}
                    if request.max_tokens and request.max_tokens != "inf"
                    else {}
                )
                | ({"temperature": request.temperature} if request.temperature else {})
                | ({"topP": request.top_p} if request.top_p else {}),
            }

        @staticmethod
        def parse(data: dict) -> str:
            return "\n".join(result["outputText"] for result in data["results"])

        @staticmethod
        def stream_parse(data: dict) -> str:
            return data["outputText"]

    ENGINES: dict[str, Engine] = {
        "meta.llama2-13b-chat-v1": Llama2Engine(),
        "amazon.titan-text-express-v1": TitanEngine(),
    }

    def __init__(
        self,
        bedrock_model=os.environ.get("BEDROCK_MODEL", "amazon.titan-text-express-v1"),
    ) -> None:
        self.client: BedrockRuntimeClient = boto3.client("bedrock-runtime")
        self.bedrock_model = bedrock_model

    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
        """Creates a model response for the given chat conversation."""

        # https://docs.aws.amazon.com/bedrock/latest/userguide/api-methods-run-inference.html

        model_id = self.bedrock_model

        engine = AmazonBedrockProvider.ENGINES[model_id]

        body = json.dumps(engine.body(request))

        logger.trace(f"bedrock request: {dict({'body': body, 'modelId': model_id})}")

        # pylint: disable=duplicate-code
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
                    body = chunk["bytes"].decode()
                    logger.trace(f"bedrock response stream: {body}")
                    data = json.loads(body)
                    content = engine.stream_parse(data)

                    yield CreateChatCompletionStreamResponse(
                        id=response_id,
                        choices=[
                            Choice3(
                                delta=ChatCompletionStreamResponseDelta(
                                    content=SecretStr(content),
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
            body = response["body"].read().decode()
            logger.trace(f"bedrock response: {body}")
            data = json.loads(body)
            content = engine.parse(data)

            response = CreateChatCompletionResponse(
                id=str(uuid.uuid4()),
                choices=[
                    Choice1(
                        message=ChatCompletionResponseMessage(
                            content=SecretStr(content),
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
