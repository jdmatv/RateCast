from libs.utils import *

def get_open_questions_25q2():
    return get_open_question_ids_from_tournament(
        api_base_url = "https://www.metaculus.com/api",
        tournament_id = "32721"
    )
