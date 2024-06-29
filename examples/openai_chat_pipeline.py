import asyncio, json, os
from any_llm import Provider, Functionality, process_api_requests_from_file

def create_chat_prompts(num_prompts):
    prompts = []
    for i in range(num_prompts):
        prompt = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Tell me an interesting fact about the number {i+1}."}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "metadata": {"prompt_id": i}
        }
        prompts.append(prompt)
    return prompts

def save_prompts_to_jsonl(prompts, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')

async def main():
    # Create chat prompts
    num_prompts = 100
    prompts = create_chat_prompts(num_prompts)

    # Save prompts to JSONL file
    input_file = "tmp/chat_prompts.jsonl"
    save_prompts_to_jsonl(prompts, input_file)

    # Set up parameters for API call
    output_filepath = "tmp/chat_responses"
    provider = Provider.OPENAI
    functionality = Functionality.CHAT
    model_name = "gpt-3.5-turbo"

    # Process API requests
    await process_api_requests_from_file(
        requests_filepath=input_file,
        save_filepath=output_filepath,
        provider=provider,
        functionality=functionality,
        model_name=model_name,
        max_attempts=3,
    )

    print(f"Processing complete. Responses saved to {output_filepath}")

if __name__ == "__main__":
    asyncio.run(main())