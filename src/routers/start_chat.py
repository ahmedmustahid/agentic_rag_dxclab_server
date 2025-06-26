"""Chat start process"""

import datetime
import logging

from dotenv import load_dotenv
from fastapi import APIRouter, Request

from ..schemas.app_schemas import ChatModel, StartChat
from .utils.csrf_utils import check_csrf
from .utils.log_dev import LogDev

load_dotenv()
log = LogDev()
PY_FILE_NAME = "[src/routers/start_chat.py]: "
router = APIRouter()


@router.post("/api/start_chat")
def start_chat(req: Request):
    """
    Chat start API endpoint.

    When starting a chat, it initializes the session.

    Args:
      req (Request): FastAPI request object. Used for session management.

    Returns:
      StartChat: Result object that includes the chat start date and time.

    Raises:
      Exception: If CSRF token verification fails.
    """
    try:
        check_csrf(req)
    except Exception as err:
        print(f"{PY_FILE_NAME}{err}")
        raise err
    try:
        # Get the current time in JST
        now_jst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            hours=9
        )
        chat_start_date = now_jst.isoformat()
        result = StartChat(chat_start_date=chat_start_date)
        return result
    except Exception as err:
        logging.error(f"Error: [/api/start_chat] {err}")
        raise err
