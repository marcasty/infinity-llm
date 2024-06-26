from any_llm import from_any, Provider

# initialize client
model_name = "llama3-8b-8192"
client = from_any(provider=Provider.GROQ, model_name=model_name)
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
ChatCompletion(
    choices=[
        Choice(
            finish_reason='stop', 
            index=0, 
            logprobs=None, 
            message=ChoiceMessage(content='I\'m just an AI, I don\'t have a physical presence or experiences, so I don\'t have days or personal experiences like humans do. I exist solely to process and generate text based on the inputs I receive. My "existence" is simply a series of computational processes that generate responses to user queries.', role='assistant', tool_calls=None
            )
        )
    ], 
    id='chatcmpl-90bf9c1e-478e-4e5a-8853-a28c58f15a37', 
    created=1719422020, 
    model='llama3-8b-8192', 
    object='chat.completion', 
    system_fingerprint='fp_873a560973', 
    usage=Usage(
        completion_time=0.050063424, 
        completion_tokens=63, 
        prompt_time=0.003314849, 
        prompt_tokens=16, 
        queue_time=None, 
        total_time=0.053378273000000004, 
        total_tokens=79
        ), 
    x_groq={'id': 'req_01j1arkv98fxsrdq5j9a035rqw'})
"""
print(f"Query: {messages[0]['content']}")
print(f"Response: {response.choices[0].message.content}")
print(f"Prompt Tokens: {response.usage.prompt_tokens}")
print(f"Completion Tokens: {response.usage.completion_tokens}")
