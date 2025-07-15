

class LLMBase:
    def __init__(self):
        pass

    def prompt(self, prompt: dict) -> str:
        """
        Generate a response from the LLM based on the provided prompt.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")