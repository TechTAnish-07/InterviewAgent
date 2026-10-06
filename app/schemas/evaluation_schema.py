from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TopicAssessment(BaseModel):
    topic: str = Field(..., description="Standardized topic name (e.g. 'Dynamic Programming', 'System Design: Caching', 'Concurrency')")
    category: str = Field(default="Technical", description="Category: 'Data Structures & Algorithms', 'System Design', 'Language Internals', 'Problem Solving', 'Communication', etc.")
    score: float = Field(..., ge=1.0, le=10.0, description="Proficiency score from 1.0 to 10.0")
    status: str = Field(..., description="Status: 'strong' (score >= 7.5), 'weak' (score <= 5.5), or 'moderate'")
    evidence: str = Field(..., description="Specific evidence or quote from the candidate's responses")
    recommendation: Optional[str] = Field(default=None, description="Actionable suggestion for improvement if weak/moderate")


class CompetencyScores(BaseModel):
    technical_depth: float = Field(default=7.0, ge=1.0, le=10.0, description="Technical understanding and depth (1-10)")
    problem_solving: float = Field(default=7.0, ge=1.0, le=10.0, description="Problem solving and analytical thought process (1-10)")
    communication: float = Field(default=7.0, ge=1.0, le=10.0, description="Clarity, structure, and verbal delivery (1-10)")
    code_quality: Optional[float] = Field(default=None, ge=1.0, le=10.0, description="Code correctness, syntax, and efficiency if coding occurred")


class InterviewEvaluationResult(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Unique session identifier")
    user_id: Optional[str] = Field(default=None, description="Unique candidate/user identifier")
    candidate_name: Optional[str] = Field(default=None, description="Candidate name")
    job_role: Optional[str] = Field(default=None, description="Job title / target role")
    overall_score: float = Field(..., ge=1.0, le=10.0, description="Overall weighted interview score from 1.0 to 10.0")
    strong_topics: List[TopicAssessment] = Field(default_factory=list, description="List of areas where candidate demonstrated strong mastery")
    weak_topics: List[TopicAssessment] = Field(default_factory=list, description="List of areas where candidate had notable gaps or struggles")
    all_topics: List[TopicAssessment] = Field(default_factory=list, description="Complete breakdown of all topics assessed")
    competencies: CompetencyScores = Field(default_factory=CompetencyScores, description="Competency sub-scores")
    overview: str = Field(default="", description="2-3 sentence overview of candidate performance")
    markdown_feedback: str = Field(default="", description="Full human-readable Markdown feedback report")
    created_at: Optional[str] = Field(default=None, description="Timestamp of the evaluation")


class UserTopicMastery(BaseModel):
    user_id: str
    topic: str
    category: str = "Technical"
    attempts_count: int = 1
    average_score: float = 5.0
    latest_score: float = 5.0
    mastery_level: str = "DEVELOPING"  # NEEDS_WORK, DEVELOPING, PROFICIENT, MASTERY
    latest_status: str = "moderate"     # weak, moderate, strong
    latest_recommendation: Optional[str] = None
    last_tested_at: str


class UserAnalyticsSummary(BaseModel):
    user_id: str
    candidate_name: Optional[str] = None
    total_interviews: int = 0
    average_overall_score: float = 0.0
    competency_averages: Dict[str, float] = Field(default_factory=dict)
    top_strengths: List[UserTopicMastery] = Field(default_factory=list)
    top_weaknesses: List[UserTopicMastery] = Field(default_factory=list)
    all_topic_mastery: List[UserTopicMastery] = Field(default_factory=list)
    recent_evaluations: List[Dict[str, Any]] = Field(default_factory=list)
