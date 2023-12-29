"""AI Provider interface"""

from abc import ABC, abstractmethod
from typing import Generator

from .model import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


# pylint: disable-next=too-few-public-methods
class Provider(ABC):
    """AI Provider interface"""

    @abstractmethod
    def create_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> (
        CreateChatCompletionResponse
        | Generator[CreateChatCompletionStreamResponse, None, None]
    ):
        """Creates a model response for the given chat conversation."""
