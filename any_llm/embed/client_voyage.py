# from https://github.com/jxnl/instructor/blob/main/instructor/client_mistral.py
# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

import voyageai
import any_llm
from typing import overload, Any, List

def create_voyage_wrapper(embed_func):
    def wrapper(
        input: List[str],
        model: str,
        **kwargs: Any
    ) -> List[List[float]]:
        
        return embed_func(texts=input, model=model,**kwargs)
    return wrapper

@overload
def from_voyage(
    client: voyageai.Client,
    **kwargs: Any,
) -> any_llm.AnyEmbedder: ...


@overload
def from_voyage(
    client: voyageai.AsyncClient,
    **kwargs: Any,
) -> any_llm.AsyncAnyEmbedder: ...

def from_voyage(
    client: voyageai.Client | voyageai.AsyncClient,
    **kwargs: Any,
) -> any_llm.AnyEmbedder | any_llm.AsyncAnyEmbedder:

    assert isinstance(
        client, (voyageai.Client, voyageai.AsyncClient)
    ), "Client must be an instance of voyageai.Client or voyageai.AsyncClient"

    wrapped_embed = create_voyage_wrapper(client.embed)

    if isinstance(client, voyageai.Client):
        return any_llm.AnyEmbedder(
            client=client,
            create=wrapped_embed,
            provider=any_llm.Provider.VOYAGE,
            **kwargs,
        )
    
    async def async_wrapped_embed(*args, **kwargs):
        return await wrapped_embed(*args, **kwargs)

    return any_llm.AsyncAnyEmbedder(
        client=client,
        create=async_wrapped_embed,
        provider=any_llm.Provider.VOYAGE,
        **kwargs,
    )