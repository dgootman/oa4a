import os

import pytest
from loguru import logger

from oa4a.amazon_bedrock_provider import AmazonBedrockProvider
from oa4a.model import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateImageRequest,
)


@pytest.mark.parametrize("model", AmazonBedrockProvider.ENGINES.keys())
def test_chat_completion(model):
    provider = AmazonBedrockProvider()
    response = provider.create_chat_completion(
        CreateChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "What's 1 + 2?"}],
                "model": model,
            }
        )
    )

    logger.debug(f"Response: {response}")

    assert isinstance(response, CreateChatCompletionResponse)
    assert len(response.choices) == 1
    assert "3" in response.choices[0].message.content.get_secret_value()


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
