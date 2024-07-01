from any_llm import from_any, Provider
from pydantic import BaseModel, Field

# initialize client
model_name = "command-r"
client = from_any(provider=Provider.COHERE, model_name=model_name, max_tokens=1000)

task = """\
Given the following text, create a Group object for 'The Beatles' band

Text:
The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
"""
messages = [{"role": "user", "content": task}]


class Person(BaseModel):
    name: str = Field(description="name of the person")
    country_of_origin: str = Field(description="country of origin of the person")


class Group(BaseModel):
    group_name: str = Field(description="name of the group")
    members: list[Person] = Field(description="list of members in the group")


# run chat completion
response, completion = client.messages.create_with_completion(
    messages=messages, response_model=Group, temperature=0
)
"""
text='```json{   
    "group_name": "The Beatles",   
    "members": [
    {"name": "John Lennon", "country_of_origin": "England"},
    {"name": "Paul McCartney", "country_of_origin": "England"},
    {"name": "George Harrison", "country_of_origin": "England"},
    {"name": "Ringo Starr", "country_of_origin": "England"}
    ]}\```' 
generation_id='18017791-ff6a-4e81-87d7-e6cc37b5f8eb'
citations=None 
documents=None 
is_search_required=None 
search_queries=None 
search_results=None 
finish_reason='COMPLETE' 
tool_calls=None 
chat_history=[
    Message_User(message="Given the following text, create a Group object for 'The Beatles' band\n\nText:\nThe Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.\n", tool_calls=None, role='USER'), 
    Message_User(message="Extract a valid Group object based on the chat history and the json schema below.\n{'$defs': {'Person': {'properties': {'name': {'description': 'name of the person', 'title': 'Name', 'type': 'string'}, 'country_of_origin': {'description': 'country of origin of the person', 'title': 'Country Of Origin', 'type': 'string'}}, 'required': ['name', 'country_of_origin'], 'title': 'Person', 'type': 'object'}}, 'properties': {'group_name': {'description': 'name of the group', 'title': 'Group Name', 'type': 'string'}, 'members': {'description': 'list of members in the group', 'items': {'$ref': '#/$defs/Person'}, 'title': 'Members', 'type': 'array'}}, 'required': ['group_name', 'members'], 'title': 'Group', 'type': 'object'}\nThe JSON schema was obtained by running:\n```python\nschema = Group.model_json_schema()\n```\n\nThe output must be a valid JSON object that `Group.model_validate_json()` can successfully parse.\n", tool_calls=None, role='USER'), 
    Message_Chatbot(message='```json{ "group_name": "The Beatles", "members": [ { "name": "John Lennon", "country_of_origin": "England" }, {"name": "Paul McCartney", "country_of_origin": "England" }, { "name": "George Harrison","country_of_origin": "England"}, {"name": "Ringo Starr", "country_of_origin": "England"}]}\```', tool_calls=None, role='CHATBOT')]
    prompt=None 
    meta=ApiMeta(
        api_version=ApiMetaApiVersion(version='1', is_deprecated=None, is_experimental=None), 
        billed_units=ApiMetaBilledUnits(input_tokens=347, output_tokens=118, search_units=None, classifications=None), 
        tokens=ApiMetaTokens(input_tokens=415, output_tokens=118), 
        warnings=None
        ) 
        response_id='e6cf17ef-b421-45d3-88e4-97fc0fe87852'"""
print(completion)
print(f"Query: {messages[0]['content']}")
# print(f"Response: {response.choices[0].message.content}")
# print(f"Prompt Tokens: {response.usage.prompt_tokens}")
# print(f"Completion Tokens: {response.usage.completion_tokens}")
