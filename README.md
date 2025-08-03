# RateCast Bot

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/jdmatv/jdmatv-ratecast)
[![Python Version](https://img.shields.io/badge/python-3.9+-informational)](https://www.python.org/)

RateCast is an AI-powered forecasting agent that automates research for complex questions. It uses a multi-stage pipeline to deconstruct a question, gather relevant information from Wikipedia, and synthesize it into a comprehensive background report using Large Language Models (LLMs).

The project is designed to produce a rich, evidence-based context that can be used to inform quantitative forecasts.

## Features

-   Automated question ingestion from Metaculus tournaments.
-   LLM-driven question decomposition into key drivers and search query generation.
-   Multi-stage, relevance-ranked information retrieval from Wikipedia.
-   LLM-based extraction and synthesis of factual content.
-   Flexible LLM service supporting OpenAI, Anthropic, OpenRouter, and local Ollama models.
-   Built-in API token usage and cost tracking.
-   Concurrent API calls with a configurable rate-limiter.
-   Modular prompt management using YAML and Jinja2.

## How It Works

The bot executes a research pipeline for each forecasting question:

1.  **Ingest**: Fetches an open question from a Metaculus tournament.
2.  **Decompose**: Uses an LLM to identify the question's key "drivers" and generate a list of Wikipedia search queries.
3.  **Search & Rank**: Executes the queries and uses an LLM to rank the search results by relevance, filtering out noise.
4.  **Extract & Synthesize**: For each relevant article, an LLM extracts factual snippets related to the drivers. These snippets are then synthesized into a coherent background report.
5.  **Expand**: The bot analyzes hyperlinks from the processed articles to find new, promising sources. It then repeats the extraction and synthesis steps to augment the initial report, adding more depth to the research.

## License

This project is licensed under the MIT License.
