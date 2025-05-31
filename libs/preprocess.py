from libs.service import CompletionsService
from prompts.utils import validate_json_with_retry
from prompts.config import prompt_manager
from pydantic import BaseModel, ValidationError
from apis.wikipedia import search_wiki, get_wiki_summary, get_wiki_full_text_batched
from typing import Union

# this should be a good model e.g. o4-mini thinking
def decompose_drivers(question_metadata: dict, model="qwen3:14b") -> list[str]:
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
        model_name=model
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

# this can be a bad model (e.g., 4o-mini)
def question_to_queries(question_metadata: dict, model="qwen3:1.7b") -> list[str]:
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
        model_name=model
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

# this can be a bad model (e.g. 4o-mini)  
def drivers_to_queries(
    question_metadata: dict, 
    drivers: list[str],
    model="qwen3:1.7b"
) -> list[str]:
    """
    Convert drivers to a list of queries.
    """

    messages = prompt_manager.render_prompt(
        prompt_name="pre_drivers_to_queries",
        question=question_metadata.get("question", ""),
        background=question_metadata.get("description", ""),
        resolution_criteria=question_metadata.get("resolution_criteria", ""),
        drivers=", ".join(drivers),
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name=model
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
) -> tuple[list[str], list[str]]:
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

# this should be a small fast model (maybe qwen series)
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

def extract_wiki_sections(
    page_name: str,
    drivers: list[str],
    bad_model="qwen3:1.7b",
    good_model="qwen3:4b",
    filter_cycles=1
):
    """
    Extract relevant sections from a Wikipedia page based on the queries and drivers.
    """

    wiki_full_text = get_wiki_full_text_batched(page_name)
    
    extracted_summaries = []
    for section in wiki_full_text:
        messages = prompt_manager.render_prompt(
            article=section,
            prompt_name="wiki_pre_extract_score",
            drivers=", ".join(drivers),
            think="/think"
        )

        service = CompletionsService()
        response = service.get_completion(
            messages=messages,
            model_name=bad_model
        )

        class SectionExtractionResponse(BaseModel):
            paragraph_summary: str
            score: int
            extraction_reasoning: str
            extracted_gold: str

        try:
            extracted_summaries.append(validate_json_with_retry(response, SectionExtractionResponse).extracted_gold)
        except ValidationError as e:
            print(f"Validation error: {e}")
            extracted_summaries.append(eval(response).get("extracted_info", []))
    
    full_extraction = " ".join([i.strip() for i in extracted_summaries if i.strip()!=""])
    
    for _ in range(filter_cycles):
        full_extraction = filter_wikipedia_output(good_model, drivers, full_extraction)
    
    return {"page_name": page_name, "page_summary": full_extraction.strip()}

def filter_wikipedia_output(
    model: str,
    drivers: list[str],
    extraction: str
) -> str:
    
    if extraction.strip() == "":
        return extraction. strip()
    
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_remove_irrelevant_info",
        extracted_gold=extraction,
        drivers=", ".join(drivers),
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name=model
    )

    class FinalExtractionResponse(BaseModel):
        reasoning: str
        filtered_gold: str
    try:
        extraction = validate_json_with_retry(response, FinalExtractionResponse).filtered_gold
    except ValidationError as e:
        print(f"Validation error: {e}")
        extraction = eval(response).get("filtered_info", [])
    
    return extraction.strip()

def draft_wiki_background(
    question_metadata: dict,
    drivers: list[str],
    wiki_summaries: list[dict],
    mode: str = "qwen3:8b"
) -> str:
    
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_consolidate_summaries",
        question=question_metadata.get("question", ""),
        drivers=", ".join(drivers),
        wiki_summaries="\n".join([f"{summary['page_name']}: {summary['page_summary']}" for summary in wiki_summaries]),
        think="/no_think"
    )

    service = CompletionsService()
    response = service.get_completion(
        messages=messages,
        model_name=mode
    )
    class BackgroundResponse(BaseModel):
        reasoning: str
        consolidated_summary: str
    
    try:
        background = validate_json_with_retry(response, BackgroundResponse).consolidated_summary
    except ValidationError as e:
        print(f"Validation error: {e}")
        background = eval(response).get("consolidated_summary", "")
    
    return background.strip()

def gen_background_pipeline1(
    question_metadata: dict,
) -> tuple[str, list[str], list[dict]]:
    """
    Generate a background for the question using Wikipedia.
    """
    
    queries, drivers = get_all_wiki_queries(question_metadata)
    print(f"Generated {len(queries)} queries and {len(drivers)} drivers for the question.")

    relevant_pages = search_wiki_rank(
        queries=queries,
        drivers=drivers,
        question_metadata=question_metadata
    )

    raw_summaries = [extract_wiki_sections(page, drivers) for page in relevant_pages]
    print(f"Extracted summaries from {len(raw_summaries)} relevant pages.")

    background = draft_wiki_background(
        question_metadata=question_metadata,
        drivers=drivers,
        wiki_summaries=raw_summaries
    )
    print("Drafted background from the extracted summaries.")

    return background, drivers, raw_summaries





    


    

    
    

