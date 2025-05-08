import os


class PromptService:
    """Service for managing and loading prompts."""

    def __init__(self, prompt_dir=None):
        """
        Initialize the PromptService.

        Args:
            prompt_dir: Optional directory path for prompts. If None, defaults to
            ../prompts relative to this file.
        """
        if prompt_dir is None:
            self.prompt_dir = os.path.join(os.path.dirname(__file__), "../prompts")
        else:
            self.prompt_dir = prompt_dir

        self._prompt_cache = {}

    def get_prompt(self, file_name: str) -> str:
        """
        Load a prompt from a text file with caching.

        Args:
            file_name: The name of the prompt file to load.

        Returns:
            The content of the prompt file.

        Raises:
            FileNotFoundError: If the prompt file doesn't exist.
        """
        file_path = os.path.join(self.prompt_dir, file_name)

        # Check if the prompt is already in the cache
        if file_path not in self._prompt_cache:
            try:
                with open(file_path, "r") as f:
                    self._prompt_cache[file_path] = f.read()
            except FileNotFoundError:
                print(f"Error: Prompt file '{file_path}' not found.")
                raise FileNotFoundError(f"Prompt file '{file_path}' not found.")

        return self._prompt_cache[file_path]
