import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class MsgUtils:
    """
    Conversation Message Operation Utility Class

    Provides a set of functions for performing various operations related to conversations, such as extracting and formatting messages from conversation data, obtaining the latest user (HumanMessage) and assistant (AIMessage) messages, processing tool (ToolMessage) messages, and extracting conversation history.

    Note:
    The processing in this class assumes that classes such as HumanMessage, AIMessage, and ToolMessage have already been defined.
    """

    def __init__(self):
        pass

    def get_latest_human_msg(self, messages):
        """
        Gets the text content of the most recent HumanMessage

        Parameters:
          messages (list): List of message objects

        Returns:
          str or None: Text of the most recent HumanMessage, or None if not present
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content.strip("\n\r")
        return None

    def get_latest_ai_msg(self, messages):
        """
        Gets the text content of the most recent AIMessage

        Parameters:
          messages (list): List of message objects

        Returns:
          str or None: Text of the most recent AIMessage, None if not present
        """

        return next(
            (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )

    def get_latest_ai_tool_msg(self, messages):
        """
        Gets the text content of the most recent AIMessage or ToolMessage

        Parameters:
          messages (list): List of message objects

        Returns:
          str or None: Text of most recent AIMessage or ToolMessage, None if not present
        """
        return next(
            (
                msg.content
                for msg in reversed(messages)
                if isinstance(msg, (AIMessage, ToolMessage))
                and isinstance(getattr(msg, "content", msg), str)
            ),
            None,
        )

    def get_latest_tool_msg(self, messages):
        """
        Gets the text content of the most recent ToolMessage

        Parameters:
        messages (list): List of message objects

        Returns:
          str or None: Text of the most recent ToolMessage, None if not present
        """
        return next(
            (msg.content for msg in reversed(messages) if isinstance(msg, ToolMessage)),
            None,
        )

    def display_alternately(self, extracted_data):
        """
        Get the conversation between the user and the assistant

        Parameters:
          extracted_data (dict): A dictionary containing a list of messages under the "messages" key
        """
        for msg in extracted_data.get("messages", []):
            message = msg["message"]
            if message:
                print(f"{msg['type']}: {message}")

    def get_tool_names(self, data_obj):
        """
        Gets multiple tool names to call with function calling

        Parameters:
          data_obj: Object containing tool information (assuming passed as a string)

        Returns:
          list: List of tool names
          If the tool name is not found, returns False
        """
        data_str = str(data_obj)
        matches = re.findall(r"'name':\s*'([^']+)'", data_str)
        unique_names = list(dict.fromkeys(matches))
        if unique_names:
            return ", ".join(f"'{name}'" for name in unique_names)
        return False

    def get_history(self, messages):
        """
        Get conversation history

        (Trace backwards from the most recent message until "type": "start_turn" is found, and extract the ToolMessage and the AIMessage immediately preceding it (with "type" set to "plan_exec"))

        Parameters:
          messages (list): List of message objects

        Returns:
          str: Extracted conversation history, sorted by oldest
        """
        start_index = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if type(msg).__name__ == "AIMessage":
                try:
                    data = json.loads(msg.content) if msg.content else {}
                except json.JSONDecodeError:
                    data = {}
                if data.get("type") == "start_turn":
                    start_index = i
                    break
        if start_index is None:
            start_index = 0
        conv_history = messages[start_index:]
        result_lines = []
        for idx, msg in enumerate(conv_history):
            if type(msg).__name__ == "ToolMessage":
                plan_exec_value = ""
                for j in range(idx - 1, -1, -1):
                    prev_msg = conv_history[j]
                    if type(prev_msg).__name__ == "AIMessage":
                        try:
                            data = (
                                json.loads(prev_msg.content) if prev_msg.content else {}
                            )
                        except json.JSONDecodeError:
                            data = {}
                        if data.get("type") == "plan_exec":
                            plan_exec_value = data.get("plan_exec", "")
                            break
                result_lines.append("AIMessage: " + plan_exec_value)
                result_lines.append("ToolMessage: " + msg.content)
        return "\n".join(result_lines) + "\n"

    def _is_structured_json(self, text):
        """
        Determines whether a string is structured JSON (dictionary or list).
        If it is a simple string, returns False.

        Parameters:
          text (str): The string to be checked

        Returns:
          bool: True if it is structured JSON, False if not
        """
        try:
            parsed = json.loads(text)
            return isinstance(parsed, (dict, list))
        except ValueError:
            return False

    def get_pure_msg(self, messages):
        """
        Excludes messages with empty content and structured JSON (dict or list),
        adds a prefix for each message type, and returns it as text. Excludes ToolMessage.
        Includes the most recent HumanMessage, because excluding it would make LLM think it was investigating the previous question.

        Parameters:

          messages (list): List of message objects

        Returns:

          str: Text with prefixed messages on each line
        """
        return "\n".join(
            (
                "AIMessage: " + msg.content
                if isinstance(msg, AIMessage)
                else (
                    "HumanMessage: " + msg.content
                    if isinstance(msg, HumanMessage)
                    else msg.content
                )
            )
            for msg in messages
            if msg.content.strip() and not self._is_structured_json(msg.content.strip())
        )

    def extract_messages(self, messages):
        """
        Extract only user and assistant messages from the conversation information

        Parameters:
          messages (list): List of message objects

        Returns:
          dict: Dictionary with list of extracted messages stored under "messages" key
        """
        output_messages = []
        for msg in messages:
            if hasattr(msg, "__class__") and hasattr(msg, "content"):
                msg_type = msg.__class__.__name__
                res_msg = msg.content.strip()
                output_messages.append({"type": msg_type, "message": res_msg})
            else:
                output_messages.append({"type": "Unknown", "message": str(msg)})
        return {"messages": output_messages}


__all__ = ["MsgUtils"]
