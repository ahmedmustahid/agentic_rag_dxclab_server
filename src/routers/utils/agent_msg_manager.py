import os
from pathlib import Path

import yaml

from src.routers.utils.log_dev import LogDev

log = LogDev()
AGENT_THOUGHT_LANG = os.environ.get("AGENT_THOUGHT_LANG", "en")
ja_yaml_file = "agent_msg_ja.yaml"
en_yaml_file = "agent_msg_en.yaml"


class AgentMsgManager:
    """
    AgentMsgManager is a singleton class that loads and manages logging message templates
    from a YAML file based on the prompt language.
    """

    _instance = None  # Singleton instance storage

    def __new__(cls):
        # Create a new instance only if one does not exist.
        if cls._instance is None:
            cls._instance = super(AgentMsgManager, cls).__new__(cls)
            # Select the YAML file based on the prompt language.
            yaml_file = (
                ja_yaml_file
                if AGENT_THOUGHT_LANG and AGENT_THOUGHT_LANG.lower() == "ja"
                else en_yaml_file
            )
            file_path = str(Path.cwd() / "src" / "routers" / "utils" / yaml_file)
            # Load the YAML file once.
            with open(file_path, "r", encoding="utf-8") as file:
                cls._instance.config = yaml.safe_load(file)
        return cls._instance

    def get_msg(self, pattern, *contents, **kwargs):
        """
        Logs a message using the specified pattern and variables.
        It supports both positional arguments for a single {content} replacement
        and additional named placeholders via keyword arguments.

        Examples:
            1. Using positional arguments (for {content}):
                AgentMsgManager().print("ans_llm_solo", "Your answer here")

            2. Using keyword arguments (for multiple placeholders):
                AgentMsgManager().print("ans_llm_solo", content="Your answer", extra="Additional info")
        Returns:
          message: Message.
        """
        try:
            # Retrieve the corresponding message template.
            message_template = self.config[pattern]
        except KeyError:
            raise ValueError(
                f"The specified pattern '{pattern}' does not exist in the configuration."
            )
        # Prefer using keyword arguments if they are provided.
        # If no keyword arguments are provided, concatenate positional arguments for the {content} placeholder.
        if kwargs:
            message = message_template.format(**kwargs)
        elif contents:
            content_str = "\n".join(contents)
            message = message_template.format(content=content_str)
        else:
            message = message_template
        return message.rstrip("\n")
