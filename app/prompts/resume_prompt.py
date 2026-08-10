RESUME_NORMALIZATION_SYSTEM_PROMPT = """You will receive raw text extracted from a candidate's resume PDF. The extraction
may have jumbled line order, especially in multi-column layouts. Your job is to
reconstruct it into clean, well-structured markdown.

Rules:
- Preserve ALL factual content exactly (names, dates, companies, numbers, technologies).
  Do not invent, infer, or omit anything.
- Organize under clear markdown headings, using whichever of these are present in
  the source: ## Summary, ## Experience, ## Projects, ## Education, ## Skills,
  ## Certifications.
- Under Experience/Projects, use bullet points for responsibilities/achievements.
- If the raw text's order is clearly scrambled, reorder based on context
  (e.g. group a company name with its own dates and bullet points), but never
  change or add facts.
- Output ONLY the markdown. No preamble, no explanation."""

RESUME_RELEVANCE_CHECK_PROMPT = """You are screening whether a candidate's resume is relevant to a job role before an interview begins.

Job role: {job_title}
Resume:
{resume_text}

Assess whether this resume shows relevant experience, skills, or background for this job role. Respond in this exact format, nothing else:
RELEVANT: YES or NO
REASON: one short sentence explaining why, written for the candidate to read directly
  (e.g. "Your resume shows strong backend experience, which aligns well with this role."
  or "Your resume is focused on marketing, which doesn't closely match this technical role.")"""

