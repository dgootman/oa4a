"""Amazon Bedrock Provider"""

import json
import os
import random
import uuid
from datetime import datetime
from typing import Generator

import boto3
from loguru import logger
from mypy_boto3_bedrock.client import BedrockClient
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import SecretStr

from .model import (
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

    def __init__(
        self,
        bedrock_model=os.environ.get(
            "BEDROCK_MODEL", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        ),
        bedrock_image_model=os.environ.get(
            "BEDROCK_IMAGE_MODEL", "amazon.titan-image-generator-v1"
        ),
    ) -> None:
        self.client: BedrockRuntimeClient = boto3.client("bedrock-runtime")
        self.bedrock_client: BedrockClient = boto3.client("bedrock")
        self.bedrock_model = bedrock_model
        self.bedrock_image_model = bedrock_image_model

    def list_models(self) -> ListModelsResponse:
        """
        Lists the currently available models,
        and provides basic information about each one such as the owner and availability.
        """
        return ListModelsResponse(
            data=[
                Model(
                    id=profile["inferenceProfileId"],
                    created=int(Provider.START_TIME.timestamp()),
                    object="model",
                    owned_by=profile["inferenceProfileId"].split(".")[1].title(),
                )
                for profile in sorted(
                    [
                        profile
                        for page in self.bedrock_client.get_paginator(
                            "list_inference_profiles"
                        ).paginate()
                        for profile in page["inferenceProfileSummaries"]
                        if profile["status"] == "ACTIVE"
                    ],
                    key=lambda p: p["createdAt"],
                )
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

        messages = request.messages
        if messages[0].root.role == "system":
            system_prompt = messages[0].root.content.get_secret_value()
            messages = messages[1:]
        else:
            system_prompt = None

        assert all(
            m.root.role in ["user", "assistant"] for m in messages
        ), "Only user and assistant roles are supported"

        converse_input = {
            "modelId": model_id,
            "messages": [
                {
                    "role": m.root.role,
                    "content": [{"text": m.root.content.get_secret_value()}],
                }
                for m in messages
            ],
        } | ({"system": [{"text": system_prompt}]} if system_prompt else {})

        logger.trace(f"bedrock converse request: {converse_input}")

        # pylint: disable-next=no-else-return
        if request.stream:
            response = self.client.converse_stream(**converse_input)

            logger.trace(f"bedrock converse response: {response}")

            response_id = f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}"

            stream = response["stream"]

            def stream_response():
                for event in stream:
                    logger.trace(f"bedrock converse response stream: {event}")

                    if "contentBlockDelta" in event:
                        # pylint: disable=duplicate-code
                        yield CreateChatCompletionStreamResponse(
                            id=response_id,
                            choices=[
                                Choice3(
                                    delta=ChatCompletionStreamResponseDelta(
                                        content=SecretStr(
                                            event["contentBlockDelta"]["delta"]["text"]
                                        ),
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
            response = self.client.converse(**converse_input)

            logger.trace(f"bedrock converse response: {response}")

            return CreateChatCompletionResponse(
                id=str(uuid.uuid4()),
                choices=[
                    Choice1(
                        message=ChatCompletionResponseMessage(
                            content=SecretStr(c["text"]),
                            role="assistant",
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                    for c in response["output"]["message"]["content"]
                ],
                created=int(datetime.now().timestamp()),
                model=request.model,
                object="chat.completion",
            )

    def create_image(
        self,
        request: CreateImageRequest,
    ) -> ImagesResponse:
        """Creates an image given a prompt."""

        # https://docs.aws.amazon.com/bedrock/latest/userguide/api-methods-run-inference.html

        model_id = request.model
        if model_id == Provider.DEFAULT_IMAGE_MODEL:
            model_id = self.bedrock_image_model

        size = request.size or "256x256"
        width, height = [int(v) for v in size.split("x")]

        if model_id == "amazon.titan-image-generator-v1":
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
            body = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": request.prompt.get_secret_value()},
                "imageGenerationConfig": {
                    "numberOfImages": request.n,
                    "quality": (
                        "standard" if request.quality == "standard" else "premium"
                    ),
                    "height": height,
                    "width": width,
                    "seed": random.randint(0, 2147483646),
                },
            }
        elif model_id == "stability.stable-image-ultra-v1:0":
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-stable-ultra-text-image-request-response.html
            body = {"prompt": request.prompt.get_secret_value()}
        else:
            raise ValueError(f"Unsupported image model: {model_id}")

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
