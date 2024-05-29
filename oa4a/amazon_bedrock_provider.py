"""Amazon Bedrock Provider"""

import json
import os
import random
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
    CreateImageRequest,
    Image,
    ImagesResponse,
    ListModelsResponse,
    Model,
)
from .provider import Provider
from .provider_utils import store_image


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

    class Claude3Engine(Engine):
        """Bedrock engine for Claude 3 models"""

        @staticmethod
        def body(request: CreateChatCompletionRequest) -> dict:
            """
            Generate a Claude 3 prompt following the algorithm here:
            https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
            """
            messages = request.messages
            if messages[0].root.role == "system":
                system_prompt = messages[0]
                messages = messages[1:]
            else:
                system_prompt = None

            return (
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": (
                        request.max_tokens
                        if request.max_tokens and request.max_tokens != "inf"
                        else 4096
                    ),
                    "messages": [
                        {
                            "role": m.root.role,
                            "content": m.root.content.get_secret_value(),
                        }
                        for m in messages
                    ],
                }
                | ({"system": system_prompt} if system_prompt else {})
                | ({"temperature": request.temperature} if request.temperature else {})
                | ({"top_p": request.top_p} if request.top_p else {})
            )

        @staticmethod
        def parse(data: dict) -> str:
            data_type = data["type"]
            if data_type == "error":
                raise RuntimeError(data)

            if data_type == "message":
                return "".join(c["text"] for c in data["content"])

            if "content_block" in data:
                content = data["content_block"]
            elif "delta" in data:
                content = data["delta"]
            else:
                return ""

            if "text" not in content:
                if data_type != "message_delta":
                    logger.warning(f"Unexpected response without text: {data}")
                return ""

            return content["text"]

        @staticmethod
        def stream_parse(data: dict) -> str:
            return AmazonBedrockProvider.Claude3Engine.parse(data)

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
                (
                    m.root.content.get_secret_value()
                    if m.root.role == "assistant"
                    else f"[INST]{m.root.content.get_secret_value()}[/INST]"
                )
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
        "anthropic.claude-3-sonnet-20240229-v1:0": Claude3Engine(),
        "meta.llama2-13b-chat-v1": Llama2Engine(),
        "amazon.titan-text-express-v1": TitanEngine(),
    }

    def __init__(
        self,
        bedrock_model=os.environ.get("BEDROCK_MODEL", "amazon.titan-text-express-v1"),
    ) -> None:
        self.client: BedrockRuntimeClient = boto3.client("bedrock-runtime")
        self.bedrock_model = bedrock_model

    def list_models(self) -> ListModelsResponse:
        """
        Lists the currently available models,
        and provides basic information about each one such as the owner and availability.
        """
        return ListModelsResponse(
            data=[
                Model(
                    id=id,
                    created=int(Provider.START_TIME.timestamp()),
                    object="model",
                    owned_by="dgootman",
                )
                for id in AmazonBedrockProvider.ENGINES
            ],
            object="list",
        )

    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
        """Creates a model response for the given chat conversation."""

        # https://docs.aws.amazon.com/bedrock/latest/userguide/api-methods-run-inference.html

        model_id = request.model
        if model_id == Provider.DEFAULT_MODEL:
            model_id = self.bedrock_model

        engine = AmazonBedrockProvider.ENGINES[model_id]

        body = engine.body(request)

        # pylint: disable=duplicate-code
        # pylint: disable-next=no-else-return
        if request.stream:
            logger.trace(
                f"bedrock request: {dict({'body': json.dumps(body), 'modelId': model_id})}"
            )

            response = self.client.invoke_model_with_response_stream(
                modelId=model_id, body=json.dumps(body)
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
            data = self._invoke_model(model_id, body)
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

    def create_image(
        self,
        request: CreateImageRequest,
    ) -> ImagesResponse:
        """Creates an image given a prompt."""

        # https://docs.aws.amazon.com/bedrock/latest/userguide/api-methods-run-inference.html

        model_id = "amazon.titan-image-generator-v1"

        size = request.size or "256x256"
        width, height = [int(v) for v in size.split("x")]

        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": request.prompt.get_secret_value()},
            "imageGenerationConfig": {
                "numberOfImages": request.n,
                "quality": ("standard" if request.quality == "standard" else "premium"),
                "height": height,
                "width": width,
                "seed": random.randint(0, 2147483646),
            },
        }

        data = self._invoke_model(model_id, body)

        if data.get("error", None):
            raise RuntimeError(data["error"])

        def to_image(b64_image: str) -> Image:
            # pylint: disable-next=no-else-return
            if request.response_format == "b64_json":
                return Image(b64_json=SecretStr(b64_image))
            else:
                return Image(url=store_image(b64_image))

        response = ImagesResponse(
            data=list(map(to_image, data["images"])),
            created=int(datetime.now().timestamp()),
        )

        return response

    def _invoke_model(self, model_id: str, body: dict) -> dict:
        body_text = json.dumps(body)

        logger.trace(
            f"bedrock request: {dict({'body': body_text, 'modelId': model_id})}"
        )

        accept = "application/json"
        content_type = "application/json"
        response = self.client.invoke_model(
            body=body_text, modelId=model_id, accept=accept, contentType=content_type
        )
        body_text = response["body"].read().decode()

        logger.trace(f"bedrock response: {body_text}")

        return json.loads(body_text)
