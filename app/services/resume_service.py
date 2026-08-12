import json
import os
from fastapi import HTTPException
from litellm import acompletion
from litellm.exceptions import APIError
from app.config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_NAME
from app.prompts.resume_prompt import (
    RESUME_ANALYSIS_SYSTEM_PROMPT,
    RESUME_RELEVANCE_CHECK_PROMPT,
)


class ResumeService:
    def strip_code_fences(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```markdown"):
            cleaned = cleaned[len("```markdown"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned.strip()

    async def normalize_resume(self, raw_text: str) -> dict:
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
                    {"role": "system", "content": RESUME_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.0,
                api_key=api_key,
            )
            content = response.choices[0].message.content or ""
            cleaned = self.strip_code_fences(content)
            print(f"[ResumeService LLM Response Output]:\n{cleaned}")

            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return {
                        "cleanedText": data.get("cleanedMarkdown") or data.get("cleanedText") or cleaned,
                        "candidateName": data.get("candidateName"),
                        "summary": data.get("summary"),
                        "skills": data.get("skills") if isinstance(data.get("skills"), list) else [],
                        "suitableRoles": data.get("suitableRoles") if isinstance(data.get("suitableRoles"), list) else [],
                        "experienceLevel": data.get("experienceLevel"),
                    }
            except Exception as parse_err:
                print(f"[ResumeService JSON parse fallback]: {parse_err}")

            return {
                "cleanedText": cleaned,
                "candidateName": None,
                "summary": None,
                "skills": [],
                "suitableRoles": [],
                "experienceLevel": None,
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ResumeService LLM Error] {e}")
            raise HTTPException(
                status_code=502,
                detail="LLM normalization failed"
            )

    async def check_relevance(
        self, resume_text: str, job_title: str, suitable_roles: list[str] | None = None
    ) -> dict:
        if not resume_text or not job_title:
            return {
                "relevant": True,
                "reason": "Resume assessment completed."
            }

        # Fast-path local check if pre-extracted suitable_roles exist
        if suitable_roles and isinstance(suitable_roles, list) and len(suitable_roles) > 0:
            title_lower = job_title.lower().strip()
            for role in suitable_roles:
                r_lower = role.lower().strip()
                if r_lower in title_lower or title_lower in r_lower:
                    return {
                        "relevant": True,
                        "reason": f"Your resume aligns well with {job_title} positions based on your verified skills."
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

            # Strip markdown formatting (*, _, #, `) to ensure robust line parsing
            cleaned_content = content.replace("*", "").replace("_", "").replace("#", "").replace("`", "").strip()

            relevant = True
            reason = "Resume assessment completed."

            for line in cleaned_content.splitlines():
                line_str = line.strip()
                upper_line = line_str.upper()

                if "RELEVANT:" in upper_line:
                    val = upper_line.split("RELEVANT:", 1)[1].strip()
                    if "NO" in val:
                        relevant = False
                    elif "YES" in val:
                        relevant = True
                elif "REASON:" in upper_line:
                    parts = line_str.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        reason = parts[1].strip()

            return {"relevant": relevant, "reason": reason}

        except Exception as e:
            print(f"[ResumeService LLM Relevance Error] {e}")
            return {
                "relevant": True,
                "reason": "Resume assessment completed."
            }


resume_service = ResumeService()

