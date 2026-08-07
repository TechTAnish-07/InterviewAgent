import os
from fastapi import HTTPException
from litellm import acompletion
from litellm.exceptions import APIError
from app.config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_NAME
from app.prompts.resume_prompt import RESUME_NORMALIZATION_SYSTEM_PROMPT


class ResumeService:
    def strip_code_fences(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```markdown"):
            cleaned = cleaned[len("```markdown"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned.strip()

    async def normalize_resume(self, raw_text: str) -> str:
        if not raw_text or len(raw_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="rawText must be at least 50 characters long"
            )

        model_name = os.getenv("MODEL_NAME", MODEL_NAME)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=502,
                detail="LLM normalization failed"
            )

        try:
            response = await acompletion(
                model=model_name,
                messages=[
                    {"role": "system", "content": RESUME_NORMALIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.0,
                api_key=api_key,
            )
            content = response.choices[0].message.content or ""
            cleaned = self.strip_code_fences(content)
            print(f"[ResumeService LLM Response Output]:\n{cleaned}")
            return cleaned
        except Exception as e:
            print(f"[ResumeService LLM Error] {e}")
            raise HTTPException(
                status_code=502,
                detail="LLM normalization failed"
            )


resume_service = ResumeService()
