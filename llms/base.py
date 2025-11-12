import time
from abc import ABC, abstractmethod


class LLMBase(ABC):
    CLIENT_ERROR = None # Class for API client errors, to be defined in subclasses
    SERVER_ERROR = None # Class for API server errors, to be defined in subclasses

    def __init__(self, model_name: str, rate_limit_wait: bool = False, retries_on_error: int = 10, temperature: float = None):
        self.model_name = model_name
        self.rate_limit_wait = rate_limit_wait
        self.retries_on_error = retries_on_error
        self.temperature = temperature

    def prompt(self, prompt: list, on_error_fn=None) -> str:
        prepared_prompt = self._prepare_prompt(prompt)
        n_retries = self.retries_on_error
        backoff = 60

        while True:
            try:
                response = self._get_response(prepared_prompt)
                break
            except self.CLIENT_ERROR as e:
                if self.rate_limit_wait and self.get_api_error_status_code(e) == 429:
                    print(f"Rate limit exceeded. Waiting for {backoff} seconds before retrying...")
                    if on_error_fn:
                        on_error_fn(e, f"Rate limit exceeded. Waiting for {backoff} seconds before retrying...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 3600)  # Exponential backoff
                    continue
                else:
                    raise e
            except self.SERVER_ERROR as e:
                if n_retries > 0:
                    print(f"Internal server error: {e}. Retrying in {backoff} seconds... ({n_retries - 1} left)")
                    if on_error_fn:
                        on_error_fn(e, f"Internal server error: {e}. Retrying in {backoff} seconds... ({n_retries - 1} left)")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 3600)  # Exponential backoff

                    n_retries -= 1
                    continue
                else:
                    raise e

        return response

    def _prepare_prompt(self, prompt: list) -> dict:
        """
        Prepare the prompt for the LLM.
        This method can be overridden by subclasses if needed.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_response(self, prompt_kwargs: dict) -> str:
        """
        Get the response from the LLM based on the prepared prompt.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_api_error_status_code(self, error: Exception) -> int:
        """
        Get the API error status code from the exception.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")