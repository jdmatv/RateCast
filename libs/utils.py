import json
import requests
import os
import logging
import sys

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
API_BASE_URL = "https://www.metaculus.com/api"
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}

def setup_logger(name: str = "ratecast_logger", level=logging.DEBUG) -> logging.Logger:
    """
    Set up a logger with console output and formatting.

    Args:
        name (str): Name of the logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler('logs/main.log', mode='a')
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def list_posts_from_tournament(
    tournament_id: int, 
    offset: int = 0, 
    count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data

def get_open_question_ids_from_tournament(
    tournament_id: int
) -> list[tuple[int, int]]:
    posts = list_posts_from_tournament(tournament_id)

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id

def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details

def get_question_metadata(post_id: int) -> str:
    """
    Get the question text from the post id.
    """
    post_details = get_post_details(post_id)
    question = post_details.get('question').get('title', '')
    description = post_details.get('question').get('description', '')
    resolution_criteria = post_details.get('question').get('resolution_criteria', '')
    fine_print = post_details.get('question').get('fine_print', '')

    return {"question": question,
            "description": description,
            "resolution_criteria": resolution_criteria,
            "fine_print": fine_print}

def get_open_questions_25q2():
    open_qs = get_open_question_ids_from_tournament(tournament_id = "32721")
    return [get_question_metadata(post_id) for _, post_id in open_qs]