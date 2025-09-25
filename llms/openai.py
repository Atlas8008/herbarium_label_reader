import io
import base64
from PIL import Image
from openai import OpenAI, APIStatusError, InternalServerError

from .base import LLMBase


def encode_image(image: Image.Image, format="jpeg") -> str:
    with io.BytesIO() as output:
        image.save(output, format=format)
        return f"data:image/{format};base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"


def wrap_image(image: Image.Image) -> dict:
    return {
        "type": "input_image",
        "image_url": encode_image(image),
    }


def wrap_text(text: str) -> dict:
    return {
        "type": "input_text",
        "text": text,
    }


def wrap_prompt(prompt: list) -> list:
    return [
        {"role": "user", "content": prompt},
    ]


class OpenAIModel(LLMBase):
    CLIENT_ERROR = APIStatusError
    SERVER_ERROR = InternalServerError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI()

    def _prepare_prompt(self, prompt: list) -> dict:
        return {
            "input": wrap_prompt([
                wrap_image(d) if isinstance(d, Image.Image) else wrap_text(d)
                for d in prompt
            ])
        }

    def _get_response(self, prepared_prompt: dict) -> str:
        return self.client.responses.create(
            model=self.model_name,
            temperature=self.temperature,
            **prepared_prompt,
        ).output_text

    def get_api_error_status_code(self, error: Exception) -> int:
        return error.status_code