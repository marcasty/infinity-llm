from any_llm import Provider
from any_llm import embed_from_any

# Create a Voyage embedding client
client = embed_from_any(Provider.VOYAGE)

# Example text to embed
text = "This is an example sentence to embed using Voyage."

# Get the embedding
embedding = client.create(input=text, model="voyage-law-2")

print(f"Number of Embeddings: {len(embedding.embeddings)}")
print(f"Embedding dimension: {len(embedding.embeddings[0])}")
print(f"Usage: {embedding.total_tokens}")