# any-llm: Use any LLM API

any-llm is a collection of Python tools to make LLM APIs plug-and-play for fast, easy experimentation.
I use this in my own projects today, but it's very much a work in progress.

[![Twitter Follow](https://img.shields.io/twitter/follow/markycasty?style=social)](https://twitter.com/markycasty)

## Key Features

- **Chat Completion**: Mostly a thin wrapper around [jxnl/instructor](https://github.com/jxnl/instructor), supports a/sync chat completion and streaming for structured and unstructured chat completions.
- **Embeddings/Rerankers**: Easily use a slew of embedding and reranking models
- **Asynchronous Workloads**: Run all chat completion and embedding workloads in massively parallel fashion without worrying about rate limits. Nice for ETL pipelines.
- **OpenAI Batch Jobs**: Run large scale batch jobs with OpenAI's batch API.
