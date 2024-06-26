from any_llm import from_any, Provider

# initialize client
model_name = "open-mistral-7b"
client = from_any(provider=Provider.MISTRAL, model_name=model_name)
messages = [
    {"role": "user", "content": "Tell me about your day."}
    ]

# run chat completion
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    response_model=None,
)
"""
id='82e5599966ad4c5fa974f9e14a315d92' 
object='chat.completion' 
created=1719421606 
model='open-mistral-7b' 
choices=[
    ChatCompletionResponseChoice(
        index=0, 
        message=ChatMessage(role='assistant', content="I don't have a physical existence or personal experiences, so I don't have days, emotions, or personal experiences like a human does. I'm just a computer program designed to process and generate text based on the data I've been trained on. However, I can simulate a day for the purpose of conversation. Today, I was activated at 6 AM, and since then, I have been assisting users with their questions and requests. I have also been learning and improving myself based on the interactions I have with users. Now, it's time for me to shut down for the night, and I will be reactivated tomorrow at 6 AM.", name=None, tool_calls=None, tool_call_id=None), 
        finish_reason=<FinishReason.stop: 'stop'>)
] 
usage=UsageInfo(prompt_tokens=9, total_tokens=148, completion_tokens=139)
"""

print(f"Query: {messages[0]['content']}")
print(f"Response: {response.choices[0].message.content}")
print(f"Prompt Tokens: {response.usage.prompt_tokens}")
print(f"Completion Tokens: {response.usage.completion_tokens}")
