from libs.service import CompletionsService
from config import prompt_manager
from pydantic import BaseModel

def decompose_drivers(question_metadata: dict) -> list[str]:
    """
    Decompose the question metadata into a list of drivers.
    """

    messages = prompt_manager.render_prompt(
        prompt_name="pre_gen_decompose_drivers",
        question=question_metadata.get("question", ""),
        background=question_metadata.get("description", ""),
        resolution_criteria=question_metadata.get("resolution_criteria", "")
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="gpt-4.1-mini",
        temperature=0.6,
    )

    class DecomposedDriversResponse(BaseModel):
        summary: str
        factor_consideration: str
        drivers_list: list[str]

    return DecomposedDriversResponse.model_validate_json(response).drivers_list
    

