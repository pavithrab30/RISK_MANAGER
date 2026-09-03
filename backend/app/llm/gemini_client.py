"""
Gemini client - used in two places, both deliberately NOT the primary
answer generator:

  1. As the eval-harness LLM-judge for faithfulness/groundedness. Using a
     different model family than the generator (Groq/Llama) avoids the
     self-preference bias a model has when grading its own output.
  2. As an optional multimodal fallback for questions about a figure/chart
     that Docling could not turn into structured text (e.g. a plot with no
     data table) - we send the cropped page image directly since Gemini's
     free tier supports vision input.

Uses the current `google-genai` SDK (the old `google-generativeai` package
is deprecated/unmaintained as of 2025). Also free-tier, no credit card
required: https://aistudio.google.com/apikey
"""
from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import LLMProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is not set - see .env.example")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system: str, user: str, temperature: float) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature
            ),
        )
        return response.text or ""

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        try:
            return self._call(system, user, temperature)
        except Exception as exc:
            logger.error("gemini_call_failed", error=str(exc))
            raise LLMProviderError(f"Gemini generation failed: {exc}") from exc

    def complete_with_image(self, system: str, user: str, image_path: str) -> str:
        try:
            from google.genai import types

            with open(image_path, "rb") as f:
                image_bytes = f.read()
            mime = "image/png" if image_path.lower().endswith("png") else "image/jpeg"
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    user,
                ],
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.0),
            )
            return response.text or ""
        except Exception as exc:
            logger.error("gemini_vision_call_failed", error=str(exc))
            raise LLMProviderError(f"Gemini vision call failed: {exc}") from exc
