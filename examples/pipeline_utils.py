import os, json

def save_prompts_to_jsonl(prompts, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
