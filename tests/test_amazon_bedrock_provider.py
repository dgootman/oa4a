import os
from typing import Generator

import pytest
from loguru import logger
from pydantic import SecretStr

from oa4a.amazon_bedrock_provider import AmazonBedrockProvider
from oa4a.model import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateImageRequest,
)

chat_completion_test_data = [
    (model, system, message, reply, stream)
    for model in AmazonBedrockProvider.ENGINES.keys()
    for system, message, reply in [
        (None, "What's 1 + 2?", "3"),
        ('You are a dog. You can only reply with "woof"', "What's 1 + 2?", "woof"),
    ]
    for stream in [True, False]
]


@pytest.mark.parametrize("model,system,message,reply,stream", chat_completion_test_data)
def test_chat_completion(
    model: str, system: str, message: str, reply: str, stream: bool
):
    provider = AmazonBedrockProvider()
    response = provider.create_chat_completion(
        CreateChatCompletionRequest.model_validate(
            {
                "messages": ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": message}],
                "model": model,
                "stream": stream,
            }
        )
    )

    logger.debug(f"Response: {response}")

    if not stream:
        assert isinstance(response, CreateChatCompletionResponse)
        assert len(response.choices) == 1
        text = response.choices[0].message.content.get_secret_value()
    else:
        assert isinstance(response, Generator)
        contents = [r.choices[0].delta.content for r in response]
        assert all(isinstance(c, SecretStr) for c in contents)

        text = "".join(c.get_secret_value() for c in contents)
        logger.debug(f"Text: {text}")

    assert reply.lower() in text.lower()


def test_create_image():
    os.environ["BASE_URL"] = "https://localhost:8000"

    provider = AmazonBedrockProvider()
    response = provider.create_image(
        CreateImageRequest.model_validate({"prompt": "A cute baby sea otter"})
    )

    logger.debug(f"Response: {response}")

    assert response.data
    assert len(response.data) == 1
    assert response.data[0].url
    assert response.data[0].url.startswith("https://localhost:8000")
