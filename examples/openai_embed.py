from any_llm import Provider
from any_llm import embed_from_any

# Create an OpenAI embedding client
client = embed_from_any(Provider.OPENAI)

# Example text to embed
text = "This is an example sentence to embed using OpenAI."
# Get the embedding
embedding = client.create(input=text, model="text-embedding-ada-002")

"""
CreateEmbeddingResponse(data=[Embedding(embedding=[-0.02307623252272606,...], index=0, object='embedding')], 
model='text-embedding-ada-002', 
object='list', 
usage=Usage(prompt_tokens=11, total_tokens=11))
"""

print(f"Number of embeddings: {len(embedding.data)}")
print(f"Embedding dimension: {len(embedding.data[0].embedding)}")

print(f"Total tokens: {embedding.usage.total_tokens}")