from any_llm import Provider
from any_llm import embed_from_any

# Create a Mistral embedding client
client = embed_from_any(Provider.MISTRAL)

# Example text to embed
text = "This is an example sentence to embed using Mistral."

# Get the embedding
embeddings, total_tokens = client.create(input=text, model="mistral-embed")

# Assuming the embedding is a list of floats
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Usage: {total_tokens}")
