import io
from PIL import Image
from google import genai

from .base import LLMBase


def img_to_bytes(image: Image.Image, format="jpeg") -> bytes:
    """
    Convert a PIL Image to bytes.

    Args:
        image (Image.Image): The image to convert.

    Returns:
        bytes: The image in byte format.
    """
    with io.BytesIO() as output:
        image.save(output, format=format)
        return output.getvalue()


class GeminiModel(LLMBase):
    CLIENT_ERROR = genai.errors.ClientError
    SERVER_ERROR = genai.errors.ServerError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = genai.Client()

    def _prepare_prompt(self, prompt: list) -> dict:
        return {
            "contents": [
                genai.types.Part.from_bytes(
                    data=img_to_bytes(d),
                    mime_type="image/jpeg",
                ) if isinstance(d, Image.Image) else d
                for d in prompt
            ]
        }

    def _get_response(self, prepared_prompt: dict) -> str:
        return self.client.models.generate_content(
            model=self.model_name,
            config={
                "temperature": self.temperature,
            },
            **prepared_prompt,
        ).text

    def get_api_error_status_code(self, error: Exception) -> int:
        # genai errors have .code attribute
        return error.code