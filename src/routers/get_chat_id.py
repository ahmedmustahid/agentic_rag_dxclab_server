"""Generate chat_id and send it from backend to frontend"""

from uuid import uuid4

from dotenv import load_dotenv
from fastapi import APIRouter, Request

from ..schemas.app_schemas import ChatId

load_dotenv()

router = APIRouter()


@router.post("/api/get_chat_id")
def get_parameters(request: Request):
    """
    Generates a unique Chat ID and returns it as a ChatId schema object.

    This endpoint is called when the chat screen is opened and assigns a unique
    Chat ID. The generated Chat ID is used as a key to track and manage the chat conversation state.

    A unique Chat ID is assigned when the chat screen is opened. This ID is managed until the chat screen is closed, reloaded. This Chat ID is used as the key for the action log, so you can check the user's behavior by referring to the log. This is important information that will be the DB key, so it is generated in the backend.

    Parameters:
      request (Request): FastAPI Request object

    Returns:
      ChatId: ChatId schema object containing the newly generated Chat ID
    """
    chat_id = uuid4()
    ret = ChatId(
        chat_id=str(chat_id),
    )
    return ret
