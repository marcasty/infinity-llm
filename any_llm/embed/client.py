# very inspired by https://github.com/jxnl/instructor/blob/main/instructor/client.py

from __future__ import annotations

import openai
from openai.types.embedding import Embedding

from typing import Any, Callable, List, Union, overload
from typing_extensions import Self
from any_llm.utils import Provider, get_provider
from collections.abc import Awaitable

class AnyEmbedder:
    client: Any | None
    create_fn: Callable[..., Any]
    provider: Provider

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        provider: Provider,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.provider = provider
        self.kwargs = kwargs

    @overload
    def create(
        self: AsyncAnyEmbedder,
        input: Union[str, List[str]],
        **kwargs: Any,
    ) -> Awaitable[List[Embedding]]: ...

    @overload
    def create(
        self: Self,
        input: Union[str, List[str]],   
        **kwargs: Any,
    ) -> List[Embedding]: ...

    def create(
        self,
        input: Union[str, List[str]],
        **kwargs: Any
    ) -> List[Embedding]:
        kwargs = self.handle_kwargs(kwargs)
        
        return self.create_fn(
            input=input,
            **kwargs
        )

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs


class AsyncAnyEmbedder(AnyEmbedder):
    client: Any | None
    embed_fn: Callable[..., Any]
    provider: Provider

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        provider: Provider,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.provider = provider
        self.kwargs = kwargs

    async def create(
        self,
        input: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        kwargs = self.handle_kwargs(kwargs)

        return await self.embed_fn(
            input=input,
            **kwargs
        )
    
@overload
def embed_from_openai(
    client: openai.OpenAI,
    **kwargs: Any,
) -> AnyEmbedder:
    pass


@overload
def embed_from_openai(
    client: openai.AsyncOpenAI,
    **kwargs: Any,
) -> AsyncAnyEmbedder:
    pass


def embed_from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    **kwargs: Any,
) -> AnyEmbedder | AsyncAnyEmbedder:
    """
    accepts Provider.OPENAI, Provider.ANYSCALE, Provider.TOGETHER, Provider.DATABRICKS
    """
    if hasattr(client, "base_url"):
        provider = get_provider(str(client.base_url))
    else:
        provider = Provider.OPENAI

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        import warnings

        warnings.warn(
            "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. Unexpected behavior may occur with other client types.",
            stacklevel=2,
        )
    
    if isinstance(client, openai.OpenAI):
        return AnyEmbedder(
            client=client,
            create=client.embeddings.create,
            provider=provider,
            **kwargs,
        )

    if isinstance(client, openai.AsyncOpenAI):
        return AsyncAnyEmbedder(
            client=client,
            create=client.embeddings.create,
            provider=provider,
            **kwargs,
        )