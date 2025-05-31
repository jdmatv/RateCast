from libs.service import CompletionsService
from pydantic import BaseModel, ValidationError
import time
import inspect

def validate_json_with_retry(
        json_str: str,
        model: BaseModel,
        max_retries: int = 3
) -> dict:
    """
    Validate JSON string against a Pydantic model with retries.
    """
    for attempt in range(max_retries):
        try:
            return model.model_validate_json(json_str)
        except ValidationError as e:
            print(f"Validation error on attempt {attempt + 1}: {e}")

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
    