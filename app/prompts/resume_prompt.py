RESUME_ANALYSIS_SYSTEM_PROMPT = """You will receive raw text extracted from a candidate's resume PDF.
Analyze the text and extract structured metadata along with normalized markdown.

Output MUST be a valid JSON object matching this exact schema:
{
  "candidateName": "Full name of candidate or null if missing",
  "summary": "A 2-sentence executive summary of the candidate's technical profile and years of experience.",
  "skills": ["List", "of", "core", "technical", "skills", "tools", "frameworks"],
  "suitableRoles": ["List", "of", "3-6", "job", "titles", "this", "candidate", "is", "qualified", "for"],
  "experienceLevel": "Entry / Mid-Level / Senior / Lead",
  "cleanedMarkdown": "Clean, well-structured Markdown reconstruction of the resume under ## Summary, ## Experience, ## Projects, ## Education, ## Skills headings."
}

Rules:
- Preserve ALL factual content exactly. Do not invent facts.
- suitableRoles MUST contain realistic target job roles matching the candidate's actual skills and experience (e.g. ['Software Engineer', 'Backend Engineer', 'Python Developer'] for a Python dev, or ['Frontend Engineer', 'React Developer'] for a React dev).
- Output ONLY raw valid JSON. No markdown code fences (no ```json), no preamble, no explanation."""


RESUME_RELEVANCE_CHECK_PROMPT = """You are screening whether a candidate's resume is relevant to a job role before an interview begins.

Job role: {job_title}
Resume:
{resume_text}

Assess whether this resume shows relevant experience, skills, or background for this job role. Do not use markdown bolding, asterisks, or bullet points. Respond in this exact format, nothing else:
RELEVANT: YES or NO
REASON: one short sentence explaining why, written for the candidate to read directly
  (e.g. "Your resume shows strong backend experience, which aligns well with this role."
  or "Your resume is focused on marketing, which doesn't closely match this technical role.")"""

