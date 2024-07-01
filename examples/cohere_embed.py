from any_llm import Provider
from any_llm import embed_from_any

# Create a Cohere embedding client
client = embed_from_any(Provider.COHERE)

# Example text to embed
text = "This is an example sentence to embed using Cohere."

# Get the embedding
embeddings, total_tokens = client.create(
    input=text, model="embed-english-v3.0", input_type="clustering"
)


print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Usage: {total_tokens}")
