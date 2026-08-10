import os
from fastapi import HTTPException
from litellm import acompletion
from litellm.exceptions import APIError
from app.config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_NAME
from app.prompts.resume_prompt import (
    RESUME_NORMALIZATION_SYSTEM_PROMPT,
    RESUME_RELEVANCE_CHECK_PROMPT,
)


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

    async def check_relevance(self, resume_text: str, job_title: str) -> dict:
        if not resume_text or not job_title:
            return {
                "relevant": True,
                "reason": "Resume assessment completed."
            }

        model_name = os.getenv("MODEL_NAME", MODEL_NAME)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            print("[ResumeService LLM Error] Missing API key for check_relevance, failing open")
            return {
                "relevant": True,
                "reason": "Resume assessment completed."
            }

        prompt = RESUME_RELEVANCE_CHECK_PROMPT.format(
            job_title=job_title,
            resume_text=resume_text[:4000]
        )

        try:
            response = await acompletion(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                api_key=api_key,
            )
            content = response.choices[0].message.content or ""
            print(f"[ResumeService LLM Relevance Response]:\n{content}")

            relevant = True
            reason = "Resume assessment completed."

            lines = content.strip().splitlines()
            for line in lines:
                line_str = line.strip()
                if line_str.upper().startswith("RELEVANT:"):
                    val = line_str.split(":", 1)[1].strip().upper()
                    if "NO" in val:
                        relevant = False
                    elif "YES" in val:
                        relevant = True
                elif line_str.upper().startswith("REASON:"):
                    reason_val = line_str.split(":", 1)[1].strip()
                    if reason_val:
                        reason = reason_val

            return {"relevant": relevant, "reason": reason}

        except Exception as e:
            print(f"[ResumeService LLM Relevance Error] {e}")
            return {
                "relevant": True,
                "reason": "Resume assessment completed."
            }


resume_service = ResumeService()

