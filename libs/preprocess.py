from libs.service import CompletionsService
from config import prompt_manager
from pydantic import BaseModel, ValidationError
from apis.wikipedia import search_wiki, get_wiki_summary
from typing import Union

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
        model_name="qwen3:1.7b"
    )

    class DecomposedDriversResponse(BaseModel):
        summary: str
        factor_consideration: str
        drivers_list: list[str]

    try:
        return DecomposedDriversResponse.model_validate_json(response).drivers_list
    except ValidationError as e:
        print(f"Validation error: {e}")
        return eval(response).get("drivers_list", [])
    
def question_to_queries(question_metadata: dict) -> list[str]:
    """
    Convert question metadata to a list of queries.
    """

    messages = prompt_manager.render_prompt(
        prompt_name="pre_question_to_queries",
        question=question_metadata.get("question", ""),
        background=question_metadata.get("description", ""),
        resolution_criteria=question_metadata.get("resolution_criteria", "")
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="qwen3:1.7b",
    )

    class QueriesResponse(BaseModel):
        information_need_summary: str
        scratchpad_query_brainstrom: Union[str, list[str]]
        wikipedia_queries: list[str]

    try:
        return QueriesResponse.model_validate_json(response).wikipedia_queries
    except ValidationError as e:
        print(f"Validation error: {e}")
        return eval(response).get("wikipedia_queries", [])
    
def drivers_to_queries(
    question_metadata: dict, 
    drivers: list[str]
) -> list[str]:
    """
    Convert drivers to a list of queries.
    """

    messages = prompt_manager.render_prompt(
        prompt_name="pre_drivers_to_queries",
        question=question_metadata.get("question", ""),
        background=question_metadata.get("description", ""),
        resolution_criteria=question_metadata.get("resolution_criteria", ""),
        drivers=drivers
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="qwen3:1.7b"
    )

    class DriversQueriesResponse(BaseModel):
        driver_understanding: str
        scratchpad_query_brainstrom: Union[str, list[str]]
        wikipedia_queries: list[str]

    try:
        return DriversQueriesResponse.model_validate_json(response).wikipedia_queries
    except ValidationError as e:
        print(f"Validation error: {e}")
        return eval(response).get("wikipedia_queries", [])

def get_all_wiki_queries(
    question_metadata: dict
) -> list[str]:
    """
    Get all Wikipedia queries for a question.
    """

    print("Generating drivers and queries for question:", question_metadata.get("question", ""))
    drivers = decompose_drivers(question_metadata)
    queries = question_to_queries(question_metadata)

    if not drivers:
        return queries

    driver_queries = drivers_to_queries(question_metadata, drivers)

    # Combine all queries and remove duplicates
    all_queries = set(queries + driver_queries)
    
    return list(all_queries), drivers

def wiki_summary_relevance(
    question_metadata: dict,
    wiki_summary: str,
    drivers: list[str]
) -> float:
    """
    Calculate the relevance of a Wikipedia summary to the question metadata.
    """
    
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_select_pages",
        page_summary=wiki_summary,
        question=question_metadata.get("question", ""),
        drivers=drivers
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="qwen3:1.7b"
    )

    class RelevanceResponse(BaseModel):
        reason: str
        decision: str

    relevance = RelevanceResponse.model_validate_json(response).decision
    
    if "yes" in relevance.lower():
        return True
    else:
        return False

def search_wiki_rank(
    queries: list[str],
    drivers: list[str],
    question_metadata: dict,
    max_results: int = 3
) -> list[str]:
    """
    Search Wikipedia for the given queries and return the top results.
    """
    results = []
    for query in queries:
        search_results = search_wiki(query, max_results)
        results.extend(search_results)

    results = list(set(results))  # Remove duplicates

    # get summaries for the results
    result_summaries = [get_wiki_summary(title) for title in results]

    relevant_results = [result for result, summary in zip(results, result_summaries)\
                         if wiki_summary_relevance(question_metadata, summary, drivers)]

    return relevant_results
    
    

