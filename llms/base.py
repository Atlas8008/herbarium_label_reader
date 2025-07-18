

class LLMBase:
    def __init__(self, model_name: str, rate_limit_wait: bool = False, retries_on_error: int = 10):

        self.model_name = model_name
        self.rate_limit_wait = rate_limit_wait
        self.retries_on_error = retries_on_error

    def prompt(self, prompt: dict) -> str:
        """
        Generate a response from the LLM based on the provided prompt.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")