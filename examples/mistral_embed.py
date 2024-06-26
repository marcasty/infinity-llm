from any_llm import Provider
from any_llm import embed_from_any

# Create a Mistral embedding client
client = embed_from_any(Provider.MISTRAL)

# Example text to embed
text = "This is an example sentence to embed using Mistral."

# Get the embedding
embedding = client.create(input=text, model="mistral-embed")


"""
id='a8b6383ac1764642b624260bb2c678b0' 
object='list' 
data=[EmbeddingObject(object='embedding', embedding=[-0.00262451171875,...], index=0)], [EmbeddingObject(...)...] model='mistral-embed' 
usage=UsageInfo(prompt_tokens=13, total_tokens=13, completion_tokens=0)
"""

# Assuming the embedding is a list of floats
print(f"Number of embeddings: {len(embedding.data)}")
print(f"Embedding dimension: {len(embedding.data[0].embedding)}")
print(f"Usage: {embedding.usage.prompt_tokens}")