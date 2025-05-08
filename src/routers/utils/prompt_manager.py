import os
from pathlib import Path

import yaml

PROMPT_LANG = os.environ.get("PROMPT_LANG")
prompts_ja_file = "prompts_ja.yaml"
prompts_en_file = "prompts_en.yaml"


class PromptManager:
    """
    PromptManager is a singleton class that manages prompt templates loaded from a YAML file.

    This class ensures that the YAML file is read only once, and provides methods to load and retrieve
    prompt templates by their names. If the file path is not explicitly provided, it determines the file
    based on the PROMPT_LANG environment variable (using 'prompts_ja.yaml' if PROMPT_LANG is 'JA', otherwise
    'prompts_en.yaml').
    """

    _instance = None  # Class variable for the singleton instance
    _initialized = (
        False  # Flag to indicate whether initialization has already been done
    )

    def __new__(cls):
        """
        Create and return the singleton instance of PromptManager.

        If an instance already exists, the existing instance is returned without creating a new one.
        environment variable.

        Returns:
            PromptManager: The singleton instance of PromptManager.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the PromptManager instance by loading prompt templates from the specified YAML file.
        The YAML file is loaded only once, even if __init__ is called multiple times, thanks to the
        _initialized flag.
        """
        if not self._initialized:
            prompt_file = prompts_ja_file if PROMPT_LANG == "JA" else prompts_en_file
            file_path = str(Path.cwd() / "src" / "routers" / "utils" / prompt_file)
            self.prompts = self.load_prompts_from_yaml(file_path)
            self._initialized = True

    def load_prompts_from_yaml(self, file_path: str) -> dict:
        """
        Load prompt templates from a YAML file and return them as a dictionary.

        Args:
            file_path (str): The path to the YAML file to load.

        Returns:
            dict: A dictionary containing the prompt templates.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_prompt(self, template_name: str) -> str:
        """
        Return the prompt corresponding to the specified template name.

        Args:
            template_name (str): The name of the prompt to retrieve (e.g., "create_plan").

        Returns:
            str: The prompt template corresponding to the specified name.

        Raises:
            ValueError: If the specified template name does not exist.
        """
        if template_name in self.prompts:
            return self.prompts[template_name]
        else:
            available = ", ".join(self.prompts.keys())
            raise ValueError(
                f"The prompt '{template_name}' does not exist. Available templates are: {available}"
            )
