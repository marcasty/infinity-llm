# from https://github.com/jxnl/instructor/blob/main/instructor/client_mistral.py
# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

import mistralai.client
import mistralai.async_client as mistralaiasynccli
import any_llm
from typing import overload, Any

@overload
def embed_from_mistral(
    client: mistralai.client.MistralClient,
    **kwargs: Any,
) -> any_llm.AnyEmbedder: ...


@overload
def embed_from_mistral(
    client: mistralaiasynccli.MistralAsyncClient,
    **kwargs: Any,
) -> any_llm.AsyncAnyEmbedder: ...


def embed_from_mistral(
    client: mistralai.client.MistralClient | mistralaiasynccli.MistralAsyncClient,
    **kwargs: Any,
) -> any_llm.AnyEmbedder | any_llm.AsyncAnyEmbedder:

    assert isinstance(
        client, (mistralai.client.MistralClient, mistralaiasynccli.MistralAsyncClient)
    ), "Client must be an instance of mistralai.client.MistralClient or mistralai.async_cli.MistralAsyncClient"

    if isinstance(client, mistralai.client.MistralClient):
        return any_llm.AnyEmbedder(
            client=client,
            create=client.embeddings,
            provider=any_llm.Provider.MISTRAL,
            **kwargs,
        )

    else:
        return any_llm.AsyncAnyEmbedder(
            client=client,
            create=client.embeddings,
            provider=any_llm.Provider.MISTRAL,
            **kwargs,
        )