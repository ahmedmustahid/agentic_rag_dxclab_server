"""Get the parameters you need for the front end
Caution: Do not send confidential information to the front end
"""

import os

from dotenv import load_dotenv
from fastapi import APIRouter, Request

from ..schemas.app_schemas import Parameters

load_dotenv()

FRONT_MSG_LANG = os.getenv("FRONT_MSG_LANG")
router = APIRouter()


@router.post("/api/get_param")
def get_parameters(request: Request):
    """
    An API endpoint to get parameters required by the frontend.

    This endpoint provides environment variables and other parameters required by the frontend. However, sensitive information such as passwords should not be sent.
    Args:
      request (Request): The current request object.
    Returns:
      Parameters: A parameter object containing FRONT_MSG_LANG.
    """
    ret_param = Parameters(
        FRONT_MSG_LANG=FRONT_MSG_LANG,
    )
    return ret_param
