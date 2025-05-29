from libs.utils import *
from config import prompt_manager
import os
import pydantic

def decompose_drivers(question_metadata: dict) -> list[str]:
    """
    Decompose the question metadata into a list of drivers.
    """
    

