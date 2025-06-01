from libs.service import CompletionsService
from pydantic import BaseModel, ValidationError
from typing import Optional
from apis.wikipedia import get_wiki_links
import random


def validate_json_with_retry(
    json_str: str,
    model: BaseModel,
    max_retries: int = 2
) -> dict:
    """
    Validate JSON string against a Pydantic model with retries.
    """
    for attempt in range(max_retries):
        try:
            return model.model_validate_json(json_str)
        except ValidationError as e:
            system_prompt = "Repair the JSON string. It should meet this schema: {schema}. Return a valid JSON string only."
            model_source = "{"+", ".join(f'"{k}": {v}' for k, v in model.__annotations__.items())+"}"
            service = CompletionsService()
            json_str = service.get_completion(
                messages=[
                    {"role": "system", "content": system_prompt.format(schema=model_source)},
                    {"role": "user", "content": json_str}
                ],
                model_name="qwen3:1.7b"
            )

    raise ValueError(f"Failed to validate JSON after {max_retries} attempts.")

def completions_with_retry(
    max_retries: int,
    validation_model: BaseModel,
    messages: list[dict],
    model_name: str,
    service: CompletionsService,
    temp: Optional[float]=None,
) -> BaseModel:
    for attempt in range(1, max_retries + 1):
        try:
            response = service.get_completion(messages=messages, model_name=model_name, temperature=temp)
            validated = validate_json_with_retry(response, validation_model)
            return validated

        except (ValidationError, ValueError) as e:
            print(f"Attempt {attempt} failed: {e}")

def batch_wiki_links(wiki_summaries: list[dict], batch_size) -> tuple[list[str], list[list[str]]]:
    existing_pages = [summary.get('page_name') for summary in wiki_summaries]
    all_links = [get_wiki_links(page) for page in existing_pages]
    all_links = list(set([link for sublist in all_links for link in sublist]) - set(existing_pages))
    
    # Shuffle the links to ensure randomness
    random.shuffle(all_links)
    
    # Create batches of links
    return existing_pages, [all_links[i:i + batch_size] for i in range(0, len(all_links), batch_size)]
    
    