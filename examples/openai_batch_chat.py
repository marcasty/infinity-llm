from typing import List
from pydantic import BaseModel
from any_llm.pipeline.batch import BatchJob
from openai import OpenAI
from openai.types.batch import Batch

# Define a sample response model
class PersonInfo(BaseModel):
    name: str
    age: int

# Sample messages for the batch
messages_batch = [
    [{"role": "user", "content": "Tell me about a person named Alice who is 30 years old."}],
    [{"role": "user", "content": "Describe a person named Bob who is 25 years old."}]
]

# Example usage
def create_batch_job():
    # Create the batch file
    BatchJob.create_from_messages(
        messages_batch=messages_batch,
        model="gpt-3.5-turbo",
        response_model=PersonInfo,
        url="/v1/chat/completions",
        file_path="examples/tmp/batch_requests.jsonl",
        max_tokens=100
    )

    print("Batch file created. Assume API calls have been made and responses saved to 'batch_responses.jsonl'")


def upload_batch_job(client: OpenAI) -> Batch:
    batch_file = client.files.create(
        file=open("examples/tmp/batch_requests.jsonl", "rb"),
        purpose="batch"
    )

    batch_job: Batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions", # /v1/embeddings
        completion_window="24h"
    )
    return batch_job


def check_status(client: OpenAI, batch_job_id: str):
    batch_job = client.batches.retrieve(batch_job_id)
    print(batch_job)


def retrieve_results(client: OpenAI, output_file_id: str, result_file_name: str):
    result = client.files.content(output_file_id).content

    with open(result_file_name, 'wb') as file:
        file.write(result)

def parse_results(file_path: str, response_model: BaseModel):
    # Parse the results with a response model
    results_with_model, errors_with_model = BatchJob.parse_from_file(
        file_path=file_path,
        response_model=response_model
    )

    print("\nResults with response model:")
    for person in results_with_model:
        print(f"Name: {person.name}, Age: {person.age}")

    print(f"\nErrors encountered: {len(errors_with_model)}")

if __name__ == "__main__":
    client = OpenAI()
    create_batch_job()
    batch_job = upload_batch_job(client)
    check_status(client, batch_job.id)
    retrieve_results(client, batch_job.output_file_id, "examples/tmp/batch_job_results.jsonl")
    parse_results("examples/tmp/batch_job_results.jsonl", PersonInfo)