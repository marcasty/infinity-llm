from any_llm import Provider, get_api_key, model_mapping
import cohere  
import tiktoken
from voyage import Client as voyage_client

def get_embed_input_tokens(request_json, provider: Provider) -> int:
    def get_text(inputs):
        if isinstance(inputs, str):
            return inputs
        elif isinstance(inputs, list):
            if all(isinstance(i, str) for i in inputs):
                return ''.join(inputs)
            else:
                raise TypeError('Expected a list of strings for "inputs" field in embedding request')
        else:
            raise TypeError('Expected a list of strings for "inputs" field in embedding request')
    
    if provider == Provider.OPENAI:
        encoding = tiktoken.encoding_for_model(request_json["model"])
        return encoding.encode(get_text(request_json["input"]))
    elif provider == Provider.ANYSCALE:
        return get_text(request_json['input'])
    elif provider == Provider.MISTRAL:
        return get_text(request_json['input'])
    elif provider == Provider.VOYAGE:
        client = voyage_client(api_key=get_api_key(provider))
        tokenized = client.tokenize(request_json['texts'], model=request_json["model"])
        return [token for sublist in tokenized for token in sublist]
    elif provider == Provider.COHERE:
        co = cohere.Client(api_key=get_api_key(provider))
        return co.tokenize(text=get_text(request_json['texts']), model=request_json["model"])




def get_chat_prompt_tokens(request_json, provider: Provider) -> int:
    def get_text(messages):
        return ''.join(value for message in messages for value in message.values())
    
    if provider == Provider.OPENAI:
        encoding = tiktoken.encoding_for_model(request_json["model"])
        return encoding.encode(get_text(request_json["messages"]))
    elif provider == Provider.ANTHROPIC:
        return get_text(request_json["messages"])
    elif provider in [Provider.GROQ, Provider.ANYSCALE]:
        return get_text(request_json["messages"])
    elif provider == Provider.MISTRAL:
        return get_text(request_json["messages"])
    elif provider == Provider.COHERE:
        co = cohere.Client(api_key=get_api_key(provider))
        return co.tokenize(text=get_text(request_json["messages"]), model=request_json["model"])