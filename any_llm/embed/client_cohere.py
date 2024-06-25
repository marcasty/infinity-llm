# inspired by https://github.com/jxnl/instructor/blob/main/instructor/client_cohere.py
from __future__ import annotations

import cohere
from typing import Any, List, Optional, overload
from typing_extensions import Callable
import any_llm

def create_cohere_wrapper(embed_func: Callable):
    def wrapper(
        input: List[str],
        model: str,
        encoding_format: Optional[str] = None,
        **kwargs: Any
    ) -> List[List[float]]:
        assert len(input) <= 96, "Cohere can only embed up to 96 texts at a time"

        if encoding_format:
            kwargs['embedding_types'] = encoding_format

        return embed_func(texts=input, model=model, **kwargs)
    return wrapper

@overload
def embed_from_cohere(
    client: cohere.Client,
    **kwargs: Any,
) -> any_llm.AnyEmbedder: ...

@overload
def embed_from_cohere(
    client: cohere.AsyncClient,
    **kwargs: Any,
) -> any_llm.AsyncAnyEmbedder: ...


def embed_from_cohere(
    client: cohere.Client | cohere.AsyncClient,
    **kwargs: Any,
):
    assert isinstance(
        client, (cohere.Client, cohere.AsyncClient)
    ), "Client must be an instance of cohere.Cohere or cohere.AsyncCohere"

    wrapped_embed = create_cohere_wrapper(client.embed)

    if isinstance(client, cohere.Client):
        return any_llm.AnyEmbedder(
            client=client,
            create=wrapped_embed,
            provider=any_llm.Provider.COHERE,
            **kwargs,
        )

    async def async_wrapped_embed(*args, **kwargs):
        return await wrapped_embed(*args, **kwargs)


    return any_llm.AsyncAnyEmbedder(
        client=client,
        create=async_wrapped_embed,
        provider=any_llm.Provider.COHERE,
        **kwargs,
    )
