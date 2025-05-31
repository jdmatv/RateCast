from libs.service import CompletionsService
from prompts.utils import validate_json_with_retry
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
        resolution_criteria=question_metadata.get("resolution_criteria", ""),
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="qwen3:0.6b"
    )

    class DecomposedDriversResponse(BaseModel):
        summary: str
        factor_consideration: str
        drivers_list: list[str]

    try:
        return validate_json_with_retry(response, DecomposedDriversResponse).drivers_list
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
        resolution_criteria=question_metadata.get("resolution_criteria", ""),
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="qwen3:0.6b",
    )

    class QueriesResponse(BaseModel):
        information_need_summary: str
        scratchpad_query_brainstorm: Union[str, list[str]]
        wikipedia_queries: list[str]

    try:
        return validate_json_with_retry(response, QueriesResponse).wikipedia_queries
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
        drivers=drivers,
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name="qwen3:1.7b"
    )

    class DriversQueriesResponse(BaseModel):
        driver_understanding: str
        scratchpad_query_brainstorm: Union[str, list[str]]
        wikipedia_queries: list[str]

    try:
        return validate_json_with_retry(response, DriversQueriesResponse).wikipedia_queries
    except ValidationError as e:
        print(f"Validation error: {e}")
        return eval(response).get("wikipedia_queries", [])

def get_all_wiki_queries(
    question_metadata: dict
) -> list[str]:
    """
    Get all Wikipedia queries for a question.
    """

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
    drivers: list[str],
    queries: list[str],
    model: str="qwen3:1.7b",
    out_type: str="binary"
) -> float:
    """
    Calculate the relevance of a Wikipedia summary to the question metadata.
    """
    
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_select_pages",
        page_summary=wiki_summary,
        question=question_metadata.get("question", ""),
        drivers=", ".join(drivers),
        queries=", ".join(queries),
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name=model
    )

    class RelevanceResponse(BaseModel):
        background: str
        page_summary: str
        reason: str
        decision: str
        score: int

    if out_type == "binary":
        relevance = validate_json_with_retry(response, RelevanceResponse).decision
        if any(answer in relevance.lower() for answer in ["yes", "maybe"]):
            return True
        else:
            return False
    elif out_type == "discrete":
        return validate_json_with_retry(response, RelevanceResponse).score

# write function to take list of wikipedia pages and return promsising ones batch and random them

def search_wiki_rank(
    queries: list[str],
    drivers: list[str],
    question_metadata: dict,
    max_results: int = 3,
    max_total_pages: int = 5,
    good_model: str = "qwen3:4b",
    bad_model: str = "qwen3:1.7b"
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

    relevant_results = [result for result, summary in zip(results, result_summaries) if\
                         wiki_summary_relevance(question_metadata, summary, drivers, queries, bad_model, "discrete")>=2]
    
    # relevant summaries
    if len(relevant_results) > max_total_pages:
        print(f"Identified {len(relevant_results)} prelimnary results: {'; '.join(relevant_results)}")
        relevant_summaries = [summary for summary, result in zip(result_summaries, results) if result in relevant_results]

        relevant_results = [result for result, summary in zip(relevant_results, relevant_summaries) if\
                            wiki_summary_relevance(question_metadata, summary, drivers, queries, good_model, "discrete")>=5]
        
    return relevant_results
    
    

