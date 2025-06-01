from prompts.components import (decompose_drivers, question_to_queries, drivers_to_queries, 
                                wiki_summary_relevance, extract_wiki_sections, draft_wiki_background,
                                extract_wiki_sections, review_wiki_pages)
from prompts.utils import search_wiki_queries
from apis.wikipedia import get_wiki_summary
from typing import Optional
from tqdm import tqdm

def get_all_wiki_queries(
    question_metadata: dict
) -> tuple[list[str], list[str]]:
    """
    Get all Wikipedia queries for a question using both driver decomposition and question metadata
    """

    drivers = decompose_drivers(question_metadata)
    queries = question_to_queries(question_metadata)

    if not drivers:
        return queries

    driver_queries = drivers_to_queries(question_metadata, drivers)

    # Combine all queries and remove duplicates
    all_queries = set(queries + driver_queries)
    
    return list(all_queries), drivers

def search_wiki_rank(
    results: list[str],
    drivers: list[str],
    question_metadata: dict,
    max_total_pages: int = 5,
    good_model: str = "qwen3:4b",
    bad_model: str = "qwen3:1.7b"
) -> list[str]:
    """
    Rank Wikipedia search results based on their relevance to the question metadata.
    """

    # get summaries for the results
    result_summaries = [get_wiki_summary(title) for title in results]

    relevant_results = [
        result for result, summary in tqdm(
            zip(results, result_summaries),
            total=len(results),
            desc="Coarse Filtering Wikipedia Results"
        )
        if wiki_summary_relevance(question_metadata, summary, drivers, bad_model, "discrete") >= 2
    ]
    
    # relevant summaries
    if len(relevant_results) > max_total_pages:
        relevant_summaries = [summary for summary, result in zip(result_summaries, results) if result in relevant_results]

        relevant_results = [
            result for result, summary in tqdm(
                zip(relevant_results, relevant_summaries),
                total=len(relevant_results),
                desc="Fine Filtering Wikipedia Results"
            )
            if wiki_summary_relevance(question_metadata, summary, drivers, good_model, "discrete") >= 5
        ]
        
    return relevant_results

def gen_background_pipeline1(
    question_metadata: dict,
    max_total_pages: int = 5,
    max_results_per_search: int = 3,
    max_sections_per_page: Optional[int] = None
) -> tuple[str, list[str], list[dict]]:
    """
    Generate a background for the question using Wikipedia.
    """
    
    queries, drivers = get_all_wiki_queries(question_metadata)

    search_results = search_wiki_queries(queries, max_results_per_search)
    relevant_pages = search_wiki_rank(
        results=search_results,
        drivers=drivers,
        question_metadata=question_metadata,
        max_total_pages=max_total_pages,
    )

    print(f"\nFirst Reading List: \n --{'\n --'.join(relevant_pages)}\n")

    raw_summaries = [
        extract_wiki_sections(page, drivers, max_sections=max_sections_per_page)
        for page in tqdm(relevant_pages, desc="Reading Wikipedia Pages")
    ]

    background = draft_wiki_background(
        question_metadata=question_metadata,
        drivers=drivers,
        wiki_summaries=raw_summaries
    )

    return background, drivers, raw_summaries

def gen_background_pipeline2(
    background: str,
    drivers: list[str],
    wiki_summaries: list[dict],
    question_metadata: dict,
    link_batch_size: int = 40,
    max_total_pages: int = 5,
    max_sections_per_page: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> tuple[str, list[str], list[dict]]:
    
    new_links = review_wiki_pages(
        wiki_summaries=wiki_summaries,
        question_metadata=question_metadata,
        drivers=drivers,
        background=background,
        link_batch_size=link_batch_size,
        max_batches=max_batches
    )

    relevant_pages = search_wiki_rank(
        results=new_links,
        drivers=drivers,
        question_metadata=question_metadata,
        max_total_pages=max_total_pages
    )

    if len(relevant_pages)>0:
        print(f"\nSecond Reading List: \n --{'\n --'.join(relevant_pages)}\n")

        raw_summaries = [
            extract_wiki_sections(page, drivers, max_sections=max_sections_per_page)
            for page in tqdm(relevant_pages, desc="Reading Wikipedia Pages")
        ]

        full_summaries = raw_summaries + wiki_summaries

        background = draft_wiki_background(
            question_metadata=question_metadata,
            drivers=drivers,
            wiki_summaries=full_summaries
        )

    return background, drivers, full_summaries

    
    





    


    

    
    

