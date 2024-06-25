from any_llm import from_any, Provider

# initialize client
model_name = "gpt-3.5-turbo"
client = from_any(provider=Provider.OPENAI, model_name=model_name)
messages = [
    {"role": "system", "content": "You are a friend."},
    {"role": "user", "content": "Tell me about your day."}
    ]

# run chat completion
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    response_model=None
)

print(f"Query: {messages[1]['content']}")
print(f"Response: {response.choices[0].message.content}")
