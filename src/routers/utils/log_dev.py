"""Output log for development"""

import os

from dotenv import load_dotenv

load_dotenv()

ENABLE_LOG_DEV = os.getenv("ENABLE_LOG_DEV")


class LogDev:
    """Log class for development

    Outputs development logs.

    Methods:
      print: Outputs logs.
    """

    def __init__(self):
        """constructor"""
        pass

    def print(self, msg):
        """Output log

        output log

        Args:
            msg (string): The message to write to the log

        Examples:
            log = LogDev()

            log.print("This is a test message.")

        Note:
            When the value of the environment variable ENABLE_LOG_DEV is True, the development log is output. When the value is False or other values, the log is not output.
        """
        if ENABLE_LOG_DEV.lower() == "true":
            print(msg)
