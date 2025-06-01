from prompts.components import (decompose_drivers, question_to_queries, drivers_to_queries, 
                                extract_wiki_sections_parallel, draft_wiki_background,
                                review_wiki_pages_parallel, check_relevance_with_filter)
from prompts.utils import search_wiki_queries
from apis.wikipedia import get_wiki_summary
from typing import Optional
from libs.utils import logger, run_with_rate_limit_threaded
import concurrent.futures

def get_all_wiki_queries(
    question_metadata: dict,
    model: str
) -> tuple[list[str], list[str]]:
    """
    Get all Wikipedia queries for a question using both driver decomposition and question metadata
    """

    # run first two calls in parallel
    drivers = None
    queries = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_drivers = executor.submit(decompose_drivers, question_metadata=question_metadata, model=model)
        future_queries = executor.submit(question_to_queries, question_metadata=question_metadata, model=model)
        try:
            drivers = future_drivers.result()
            queries = future_queries.result()
        except Exception as e:
            logger.error(f"An error occurred during threaded execution: {e}")

    if not drivers:
        return queries

    driver_queries = drivers_to_queries(question_metadata=question_metadata, 
                                        drivers=drivers,
                                        model=model)

    # Combine all queries and remove duplicates
    all_queries = set(queries + driver_queries)

    logger.info(f'Searching Wikipedia for: {all_queries}')
    
    return list(all_queries), drivers

def search_wiki_rank(
    results: list[str],
    drivers: list[str],
    question_metadata: dict,
    good_model: str,
    model: str,
    max_total_pages: int = 5,
    max_workers: int = 10,
    rate_limit: int = 10
) -> list[str]:
    """
    Rank Wikipedia search results based on their relevance to the question metadata.
    """

    # get summaries for the results
    result_summaries = [get_wiki_summary(title) for title in results]

    relevant_results = run_with_rate_limit_threaded(
        func=check_relevance_with_filter,
        iterable=list(zip(results, result_summaries)),
        static_kwargs={
            "question_metadata": question_metadata,
            "drivers": drivers,
            "model": model,
            "mode": "discrete",
            "threshold": 2
        },
        max_workers=max_workers,
        tqdm_desc="Coarse Filtering Wikipedia Results",
        rate_limit_per_10_sec=rate_limit
    )

    relevant_results = [result for result in relevant_results if result is not None]

    logger.info(f'Preliminary relevant results: {relevant_results}')
    
    # relevant summaries
    if len(relevant_results) > max_total_pages:
        relevant_summaries = [summary for summary, result in zip(result_summaries, results) if result in relevant_results]

        relevant_results = run_with_rate_limit_threaded(
            func=check_relevance_with_filter,
            iterable=list(zip(relevant_results, relevant_summaries)),
            static_kwargs={
                "question_metadata": question_metadata,
                "drivers": drivers,
                "model": good_model,
                "mode": "discrete",
                "threshold": 5
            },
            max_workers=max_workers,
            tqdm_desc="Fine Filtering Wikipedia Results",
            rate_limit_per_10_sec=rate_limit
        )

        relevant_results = [result for result in relevant_results if result is not None]
        
    return relevant_results

def gen_background_pipeline1(
    question_metadata: dict,
    model: str,
    good_model: str,
    max_total_pages: int = 5,
    max_results_per_search: int = 3,
    max_workers: int = 10,
    rate_limit: int = 10,
    max_sections_per_page: Optional[int] = None
) -> tuple[str, list[str], list[dict]]:
    """
    Generate a background for the question using Wikipedia.
    """
    
    queries, drivers = get_all_wiki_queries(question_metadata, model=model)

    search_results = search_wiki_queries(queries, max_results_per_search)
    
    relevant_pages = search_wiki_rank(
        results=search_results,
        drivers=drivers,
        question_metadata=question_metadata,
        max_total_pages=max_total_pages,
        max_workers=max_workers,
        rate_limit=rate_limit,
        model=model,
        good_model=good_model
    )

    if len(relevant_pages) == 0:
        logger.warning(f"\nNo relevant wikipedia pages identified.\n")
        return "", drivers, []

    print(f"\nFirst Reading List: \n --{'\n --'.join(relevant_pages)}\n")

    raw_summaries = [
        extract_wiki_sections_parallel(
            page,
            drivers=drivers,
            max_sections=max_sections_per_page,
            max_workers=max_workers,
            rate_limit=rate_limit,
            model=model,
            good_model=good_model
        ) 
        for page in relevant_pages
    ]

    background = draft_wiki_background(
        question_metadata=question_metadata,
        drivers=drivers,
        wiki_summaries=raw_summaries,
        model=good_model
    )

    return background, drivers, raw_summaries

def gen_background_pipeline2(
    background: str,
    drivers: list[str],
    wiki_summaries: list[dict],
    question_metadata: dict,
    model: str,
    good_model: str,
    link_batch_size: int = 40,
    max_total_pages: int = 5,
    max_workers: int = 10,
    rate_limit: int = 10,
    max_sections_per_page: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> tuple[str, list[str], list[dict]]:
    
    if len(wiki_summaries) == 0:
        return background, drivers, full_summaries
    
    new_links = review_wiki_pages_parallel(
        wiki_summaries=wiki_summaries,
        question_metadata=question_metadata,
        drivers=drivers,
        background=background,
        link_batch_size=link_batch_size,
        max_batches=max_batches,
        max_workers=max_workers,
        rate_limit=rate_limit,
        model=model
    )

    # remove links that are already in the summaries
    new_links = [link.lower() for link in new_links if link not in [summary.get('page_name').lower() for summary in wiki_summaries]]

    relevant_pages = search_wiki_rank(
        results=new_links,
        drivers=drivers,
        question_metadata=question_metadata,
        max_total_pages=max_total_pages,
        max_workers=max_workers,
        rate_limit=rate_limit,
        model=model,
        good_model=good_model
    )

    # remvove pages that are already in the summaries
    relevant_pages = [page for page in relevant_pages if page not in [summary.get('page_name') for summary in wiki_summaries]]

    if len(relevant_pages)>0:
        print(f"\nSecond Reading List: \n --{'\n --'.join(relevant_pages)}\n")

        raw_summaries = [
            extract_wiki_sections_parallel(
                page,
                drivers=drivers,
                max_sections=max_sections_per_page,
                max_workers=max_workers,
                rate_limit=rate_limit,
                model=model,
                good_model=good_model
            ) 
            for page in relevant_pages
        ]

        full_summaries = wiki_summaries + raw_summaries

        background = draft_wiki_background(
            question_metadata=question_metadata,
            drivers=drivers,
            wiki_summaries=full_summaries,
            model=good_model
        )
    
    else: full_summaries = wiki_summaries.copy()

    return background, drivers, full_summaries

    
    





    


    

    
    

