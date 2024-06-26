from any_llm import Provider
from any_llm import embed_from_any

# Create a Cohere embedding client
client = embed_from_any(Provider.COHERE)

# Example text to embed
text = "This is an example sentence to embed using Cohere."

# Get the embedding
embedding = client.create(input=text, model="embed-english-v3.0", input_type="clustering")

"""
> id='f0d20803-71b8-412c-8f89-17907ed2da51' 
embeddings=[[0.023345947, ...]] 
texts=['this is an example sentence to embed using cohere.'] 
meta=ApiMeta(api_version=ApiMetaApiVersion(version='1', is_deprecated=None, is_experimental=None), billed_units=ApiMetaBilledUnits(input_tokens=13.0, output_tokens=None, search_units=None, classifications=None), tokens=None, warnings=[]) 
response_type='embeddings_floats'
"""

print(f"Number of embeddings: {len(embedding.embeddings)}")
print(f"Embedding dimension: {len(embedding.embeddings[0])}")
print(f"Usage: {embedding.meta.billed_units.input_tokens}")