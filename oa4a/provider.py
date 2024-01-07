"""AI Provider interface"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generator

from .model import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    ListModelsResponse,
    Model,
)


class Provider(ABC):
    """AI Provider interface"""

    DEFAULT_MODEL = "gpt-3.5-turbo"
    START_TIME = datetime.now()

    def list_models(self) -> ListModelsResponse:
        """
        Lists the currently available models,
        and provides basic information about each one such as the owner and availability.
        """
        return ListModelsResponse(
            data=[
                Model(
                    id=Provider.DEFAULT_MODEL,
                    created=int(Provider.START_TIME.timestamp()),
                    object="model",
                    owned_by="dgootman",
                )
            ],
            object="list",
        )

    @abstractmethod
    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
        """Creates a model response for the given chat conversation."""
