import json
import requests
import os
import logging
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from time import time, sleep
from tqdm import tqdm

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
API_BASE_URL = "https://www.metaculus.com/api"
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
LOG_FILE = "logs/request_log.json"
LOG_LOCK = threading.Lock()

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

def count_message_tokens(
    messages: list[dict],
    model_name: str = "gpt-4.1-mini"
) -> int:
    """
    Count the number of tokens in a list of messages for a specific model.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = 0

    for message in messages:
        if 'content' in message:
            token_count += len(encoding.encode(message['content']))
        if 'role' in message:
            token_count += len(encoding.encode(message['role']))

    return token_count

def log_requests_and_enforce_rate(limit_per_window=10, window_sec=10, log_expiry_sec=60):
    """Check and update the request log to enforce rate limiting."""

    now = time()
    log_data = []

    # Load existing log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

    # Clean expired entries
    log_data = [ts for ts in log_data if now - ts < window_sec]

    # Reset if the log is stale
    if log_data and now - max(log_data) > log_expiry_sec:
        log_data = []

    if len(log_data) >= limit_per_window:
        sleep_time = window_sec - (now - min(log_data))
        if sleep_time > 0:
            logger.warning(f"Self-imposed rate limit exceeded. Sleeping for {sleep_time:.2f} seconds.")
            sleep(sleep_time)

    # Add current timestamp
    log_data.append(time())

    # Save log
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f)

def run_with_rate_limit_threaded(
    func,
    iterable,
    static_kwargs=None,
    max_workers=10,
    tqdm_desc="Processing",
    rate_limit_per_10_sec=10
):
    """
    Run a sync function with kwargs on a list of inputs using multithreading
    while enforcing a rate limit of N requests per 10 seconds.
    """
    static_kwargs = static_kwargs or {}
    results = []

    def wrapped(item):
        with LOG_LOCK:
            log_requests_and_enforce_rate(limit_per_window=rate_limit_per_10_sec, window_sec=10)

        return func(item, **static_kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapped, item): item for item in iterable}

        for future in tqdm(as_completed(futures), total=len(futures), desc=tqdm_desc):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = futures[future]
                print(f"Error processing {item}: {e}")

    return results