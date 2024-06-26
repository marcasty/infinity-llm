from any_llm import from_any, Provider

# initialize client
model_name = "claude-3-haiku-20240307"
client = from_any(provider=Provider.ANTHROPIC, model_name=model_name)
messages = [
    {"role": "user", "content": "Tell me about your day."}
    ]

# run chat completion
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    response_model=None,
    max_tokens=1024 # required
)
"""
Message(
    id='msg_01WF3k4w3XJuGwJYiNVZK7pu', 
    content=[TextBlock(text="I don't actually experience days or have personal experiences to share. I'm Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest.", type='text')], 
    model='claude-3-haiku-20240307', 
    role='assistant', 
    stop_reason='end_turn', 
    stop_sequence=None, 
    type='message', 
    usage=Usage(input_tokens=13, output_tokens=38)
)
"""

print(f"Query: {messages[0]['content']}")
print(f"Response: {response.content[0].text}")
print(f"Prompt Tokens: {response.usage.input_tokens}")
print(f"Completion Tokens: {response.usage.output_tokens}")