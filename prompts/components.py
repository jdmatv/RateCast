from libs.service import CompletionsService
from prompts.config import prompt_manager
from prompts.utils import completions_with_retry, batch_wiki_links
from pydantic import BaseModel
from typing import Optional, Union
from apis.wikipedia import get_wiki_full_text_batched
from tqdm import tqdm
from libs.utils import logger, run_with_rate_limit_threaded

# this should be a good model e.g. o4-mini thinking
def decompose_drivers(
    question_metadata: dict, 
    model: str="qwen3:8b", 
    max_retries: int=3
) -> list[str]:
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

    class DecomposedDriversResponse(BaseModel):
        summary: str
        factor_consideration: str
        drivers_list: list[str]

    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=DecomposedDriversResponse,
        messages=messages,
        model_name=model,
        service=service
    )

    logger.info(f'{len(response.drivers_list)} drivers: {response.drivers_list}')

    return response.drivers_list

# this can be a bad model (e.g., 4o-mini)
def question_to_queries(
    question_metadata: dict, 
    model: str="qwen3:8b", 
    max_retries: int=3
) -> list[str]:
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

    class QueriesResponse(BaseModel):
        information_need_summary: str
        scratchpad_query_brainstorm: Union[str, list[str]]
        wikipedia_queries: list[str]

    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=QueriesResponse,
        messages=messages,
        model_name=model,
        service=service
    )

    return response.wikipedia_queries

# this can be a bad model (e.g. 4o-mini)  
def drivers_to_queries(
    question_metadata: dict, 
    drivers: list[str],
    model: str="qwen3:8b",
    max_retries: int=3
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

    class DriversQueriesResponse(BaseModel):
        driver_understanding: str
        scratchpad_query_brainstorm: Union[str, list[str]]
        wikipedia_queries: list[str]

    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=DriversQueriesResponse,
        messages=messages,
        model_name=model,
        service=service
    )

    return response.wikipedia_queries


# this should be a small fast model (maybe qwen series)
def wiki_summary_relevance(
    question_metadata: dict,
    wiki_summary: str,
    drivers: list[str],
    model: str="qwen3:8b",
    out_type: str="binary",
    max_retries: int=3
) -> float:
    """
    Calculate the relevance of a Wikipedia summary to the question metadata.
    """
    
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_select_pages",
        page_summary=wiki_summary,
        question=question_metadata.get("question", ""),
        drivers=", ".join(drivers),
        think="/no_think"
    )

    service = CompletionsService()

    class RelevanceResponse(BaseModel):
        background: str
        page_summary: str
        reason: str
        decision: str
        score: int

    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=RelevanceResponse,
        messages=messages,
        model_name=model,
        service=service
    )

    if out_type == "binary":
        relevance = response.decision
        if any(answer in relevance.lower() for answer in ["yes", "maybe"]):
            return True
        else:
            return False
    elif out_type == "discrete":
        return response.score

def extract_wiki_sections(
    page_name: str,
    drivers: list[str],
    bad_model: str="qwen3:8b",
    good_model: str="qwen3:8b",
    filter_cycles: int=1,
    max_retries: int=3,
    max_sections: Optional[int] = None
):
    """
    Extract relevant sections from a Wikipedia page based on the queries and drivers.
    """

    wiki_full_text = get_wiki_full_text_batched(page_name)

    if max_sections is not None:
        wiki_full_text = wiki_full_text[:max_sections]
    
    extracted_summaries = []
    for section in wiki_full_text:
        messages = prompt_manager.render_prompt(
            article=section,
            prompt_name="wiki_pre_extract_score",
            drivers=", ".join(drivers),
            think="/think"
        )

        service = CompletionsService()

        class SectionExtractionResponse(BaseModel):
            paragraph_summary: str
            score: int
            extraction_reasoning: str
            extracted_gold: str

        response = completions_with_retry(
            max_retries=max_retries, 
            validation_model=SectionExtractionResponse,
            messages=messages,
            model_name=bad_model,
            service=service
        )

        extracted_summaries.append(response.extracted_gold)
    
    full_extraction = " ".join([i.strip() for i in extracted_summaries if i.strip()!=""])
    
    for _ in range(filter_cycles):
        full_extraction = filter_wikipedia_output(good_model, drivers, full_extraction)
    
    return {"page_name": page_name, "page_summary": full_extraction.strip()}

def filter_wikipedia_output(
    model: str,
    drivers: list[str],
    extraction: str,
    max_retries: int = 3
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

    class FinalExtractionResponse(BaseModel):
        reasoning: str
        filtered_gold: str
    
    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=FinalExtractionResponse,
        messages=messages,
        model_name=model,
        service=service
    )
    
    return response.filtered_gold.strip()

def draft_wiki_background(
    question_metadata: dict,
    drivers: list[str],
    wiki_summaries: list[dict],
    model: str = "qwen3:8b",
    max_retries: int = 3
) -> str:
    
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_consolidate_summaries",
        question=question_metadata.get("question", ""),
        drivers=", ".join(drivers),
        wiki_summaries="\n".join([f"{summary['page_name']}: {summary['page_summary']}" for summary in wiki_summaries]),
        think="/no_think"
    )

    service = CompletionsService()

    class BackgroundResponse(BaseModel):
        reasoning: str
        consolidated_summary: str
    
    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=BackgroundResponse,
        messages=messages,
        model_name=model,
        service=service
    )
    
    return response.consolidated_summary.strip()

def review_wiki_pages(
    links: list[str],
    question_metadata: dict,
    drivers: list[str],
    background: str,
    existing_pages: list[str],
    model: str,
    max_retries: int
) -> list[str]:
    """
    Review a list of Wikipedia pages and return a list of relevant pages.
    """
    messages = prompt_manager.render_prompt(
        prompt_name="wiki_pre_select_new_pages",
        question=question_metadata.get("question", ""),
        existing_pages=", ".join(existing_pages),
        new_pages=', '.join([f"'{item}'" for item in links]),
        drivers=", ".join(drivers),
        background=background,
        think="/no_think"
    )

    service = CompletionsService()

    class ReviewResponse(BaseModel):
        reasoning: str
        pages_list: list[str]
    
    response = completions_with_retry(
        max_retries=max_retries, 
        validation_model=ReviewResponse,
        messages=messages,
        model_name=model,
        service=service
    )
    
    return response.pages_list

def review_wiki_pages_parallel(
    wiki_summaries: list[dict],
    question_metadata: dict,
    drivers: list[str],
    background: str,
    model: str = "qwen3:8b",
    link_batch_size: int = 40,
    max_retries: int = 3,
    max_batches: Optional[int]=None,
    max_workers: int = 10,
    rate_limit: int = 10
) -> list[str]:
    """
    Review list of Wikipedia pages and return a list of relevant pages in parallel.
    """
    existing_pages, all_links = batch_wiki_links(wiki_summaries, batch_size=link_batch_size)

    if max_batches is not None:
        all_links = all_links[:max_batches]

    candidate_links = run_with_rate_limit_threaded(
        func=review_wiki_pages,
        iterable=all_links,
        static_kwargs={
            "question_metadata": question_metadata,
            "drivers": drivers,
            "background": background,
            "existing_pages": existing_pages,
            "model": model,
            "max_retries": max_retries
        },
        max_workers=max_workers,
        tqdm_desc="Reviewing New Wikipedia Links",
        rate_limit_per_10_sec=rate_limit
    )

    # flatten the list of candidate links
    if isinstance(candidate_links, list) and all(isinstance(item, list) for item in candidate_links):
        candidate_links = [item for sublist in candidate_links for item in sublist]

    return review_wiki_pages(
        links=candidate_links,
        question_metadata=question_metadata,
        drivers=drivers,
        background=background,
        existing_pages=existing_pages,
        model=model,
        max_retries=max_retries
    )
