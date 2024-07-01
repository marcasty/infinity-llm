import asyncio, json, os
from any_llm import Provider, Functionality, process_api_requests_from_file
from .pipeline_utils import save_prompts_to_jsonl


def create_chat_prompts(num_prompts, model):
    prompts = []
    for i in range(num_prompts):
        prompt = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Tell me an interesting fact about the number {i+1}.",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "metadata": {"prompt_id": i},
        }
        prompts.append(prompt)
    return prompts


async def main():
    # Create chat prompts
    num_prompts = 1000
    model = "open-mistral-7b"
    prompts = create_chat_prompts(num_prompts, model)

    # Save prompts to JSONL file
    input_file = "examples/tmp/chat_prompts.jsonl"
    save_prompts_to_jsonl(prompts, input_file)

    # Set up parameters for API call
    output_filepath = "examples/tmp/chat_responses"
    provider = Provider.MISTRAL
    functionality = Functionality.CHAT

    # Process API requests
    await process_api_requests_from_file(
        requests_filepath=input_file,
        save_filepath=output_filepath,
        provider=provider,
        functionality=functionality,
        model=model,
        max_attempts=3,
    )

    print(f"Processing complete. Responses saved to {output_filepath}")


if __name__ == "__main__":
    asyncio.run(main())
