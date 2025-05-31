from prompts.components import (decompose_drivers, question_to_queries, drivers_to_queries, 
                                wiki_summary_relevance, extract_wiki_sections, draft_wiki_background,
                                extract_wiki_sections)
from apis.wikipedia import search_wiki, get_wiki_summary, get_wiki_links
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

    relevant_results = [
        result for result, summary in tqdm(
            zip(results, result_summaries),
            total=len(results),
            desc="Coarse Filtering Wikipedia Results"
        )
        if wiki_summary_relevance(question_metadata, summary, drivers, queries, bad_model, "discrete") >= 2
    ]
    
    # relevant summaries
    if len(relevant_results) > max_total_pages:
        print(f"Identified {len(relevant_results)} prelimnary results: {'; '.join(relevant_results)}")
        relevant_summaries = [summary for summary, result in zip(result_summaries, results) if result in relevant_results]

        relevant_results = [
            result for result, summary in tqdm(
                zip(relevant_results, relevant_summaries),
                total=len(relevant_results),
                desc="Fine Filtering Wikipedia Results"
            )
            if wiki_summary_relevance(question_metadata, summary, drivers, queries, good_model, "discrete") >= 5
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

    relevant_pages = search_wiki_rank(
        queries=queries,
        drivers=drivers,
        question_metadata=question_metadata,
        max_results=max_results_per_search,
        max_total_pages=max_total_pages,
    )

    print(f"Reading list: \n{'\n --'.join(relevant_pages)}")

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
    max_total_pages: int = 5,
    max_results_per_search: int = 3,
    max_sections_per_page: Optional[int] = None
) -> tuple[str, list[str], list[dict]]:
    
    all_links = [get_wiki_links(summary.get('page_name')) for summary in wiki_summaries]
    all_links = [link for sublist in all_links for link in sublist]  # Flatten the list

    print(all_links)
    





    


    

    
    

