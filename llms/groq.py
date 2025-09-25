import io
import base64
from PIL import Image
from groq import Groq, APIStatusError, InternalServerError

from .base import LLMBase


def encode_image(image: Image.Image, format="jpeg") -> str:
    with io.BytesIO() as output:
        image.save(output, format=format)
        return f"data:image/{format};base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"


def wrap_image(image: Image.Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": encode_image(image)},
    }


def wrap_text(text: str) -> dict:
    return {
        "type": "text",
        "text": text,
    }


def wrap_prompt(prompt: list) -> list:
    return [
        {"role": "user", "content": prompt},
    ]


PREFIXES = {
    "llama-4-scout-17b-16e-instruct": "meta-llama/",
    "llama-4-maverick-17b-128e-instruct": "meta-llama/",
}


class GroqModel(LLMBase):
    CLIENT_ERROR = APIStatusError
    SERVER_ERROR = InternalServerError

    def __init__(self, model_name, *args, **kwargs):
        if model_name in PREFIXES:
            model_name = PREFIXES[model_name] + model_name
        super().__init__(model_name, *args, **kwargs)
        self.client = Groq()

    def _prepare_prompt(self, prompt: list) -> dict:
        return {
            "messages": wrap_prompt([
                wrap_image(d) if isinstance(d, Image.Image) else wrap_text(d)
                for d in prompt
            ])
        }

    def _get_response(self, prepared_prompt: dict) -> str:
        return self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            **prepared_prompt,
        ).choices[0].message.content

    def get_api_error_status_code(self, error: Exception) -> int:
        return error.status_code