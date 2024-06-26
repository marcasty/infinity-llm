from any_llm import Provider
from any_llm import embed_from_any

# Create an OpenAI embedding client
client = embed_from_any(Provider.OPENAI)

# Example text to embed
text = "This is an example sentence to embed using OpenAI."
# Get the embedding
embeddings, total_tokens = client.create(input=text, model="text-embedding-ada-002")

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Total tokens: {total_tokens}")