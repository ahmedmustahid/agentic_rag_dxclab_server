"""Get or generate a CSRF token"""

import logging
import secrets

from dotenv import load_dotenv
from fastapi import APIRouter, Request

from ..schemas.app_schemas import CsrfToken

load_dotenv()

router = APIRouter()


@router.post("/api/get_csrf")
def get_csrf(request: Request):
    """
    Retrieves or generates a CSRF token and returns it.

    This endpoint retrieves a CSRF token from the session. If a CSRF token does not exist in the session,
    a new one is generated and stored in the session.

    Args:
      request (Request): The current request object.

    Returns:
      CsrfToken: A schema object containing the retrieved or generated CSRF token.

    Raises:
      Exception: If an error occurs while retrieving or generating the CSRF token.
    """
    try:
        csrf_token = request.session.get("csrf_token")
        if csrf_token == None:
            csrf_token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = csrf_token
        res_csrf = CsrfToken(csrf_token=csrf_token)
        return res_csrf
    except Exception as err:
        logging.error(f"Error: [/api/get_csrf] {err}")
        raise err
