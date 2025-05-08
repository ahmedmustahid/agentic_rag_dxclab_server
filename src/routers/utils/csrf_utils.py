"""Check CSRF Token"""

import logging

from fastapi import HTTPException, Request

from ..utils.constants import REST_API_401_ERROR, REST_API_500_ERROR


def check_csrf(request: Request):
    """CSRF Token Check Function

    Compares the CSRF Token sent from the front end with the CSRF Token created when the screen was opened and stored in the session.
    Returns an error if they do not match.

    Args:
      request (Request): Http request.

    Raises:
      HTTPException: Returns 401 if the CSRF Token does not exist in the session, and 403 if the CSRF Token does not match.
      Returns 500 otherwise.
    """

    try:
        csrf_ses = request.session.get("csrf_token")
        if csrf_ses == None:
            logging.error(f"[check_csrf] Session timed out.")
            request.session.clear()
            raise HTTPException(status_code=401, detail=REST_API_401_ERROR)
    except HTTPException as http_exc:
        logging.error(f"[check_csrf] {err} Internal server error.")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    except Exception as err:
        # Return 500 for all other exceptions
        logging.error(f"[check_csrf] {err} Internal server error.")
        raise HTTPException(status_code=500, detail=REST_API_500_ERROR)
    try:
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            csrf_token = request.headers.get("X-CSRF-Token")
            if not csrf_token or csrf_token != csrf_ses:
                request.session.clear()
                raise HTTPException(status_code=403, detail="CSRF token invalid")
    except HTTPException as http_exc:
        # For any status code other than 403, return 500.
        if http_exc.status_code == 403:
            raise http_exc
        else:
            logging.error(f"[check_csrf] {err} Internal server error.")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    except Exception as err:
        # Return 500 for all other exceptions
        logging.error(f"[check_csrf] {err} Internal server error.")
        raise HTTPException(status_code=500, detail=REST_API_500_ERROR)
