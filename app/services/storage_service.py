import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.evaluation_schema import (
    CompetencyScores,
    InterviewEvaluationResult,
    TopicAssessment,
    UserAnalyticsSummary,
    UserTopicMastery,
)

logger = logging.getLogger("interview-agent.storage")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "interviews.db"


class StorageService:
    """
    SQLite-backed persistence layer for storing:
    1. Per-session interview evaluation records (scores, strong & weak topics, markdown feedback).
    2. Longitudinal candidate topic mastery and performance metrics over time.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create database directory %s: %s", self.db_path.parent, e)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize required database tables and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table: interview_evaluations (per-session evaluation records)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interview_evaluations (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    candidate_name TEXT,
                    job_role TEXT,
                    overall_score REAL NOT NULL,
                    technical_depth_score REAL,
                    problem_solving_score REAL,
                    communication_score REAL,
                    code_quality_score REAL,
                    strong_topics_json TEXT,
                    weak_topics_json TEXT,
                    all_topics_json TEXT,
                    overview TEXT,
                    markdown_feedback TEXT,
                    raw_json TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Table: user_topic_mastery (longitudinal performance per topic per user)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_topic_mastery (
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    category TEXT NOT NULL,
                    attempts_count INTEGER DEFAULT 1,
                    average_score REAL NOT NULL,
                    latest_score REAL NOT NULL,
                    mastery_level TEXT NOT NULL,
                    latest_status TEXT NOT NULL,
                    latest_recommendation TEXT,
                    last_tested_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, topic)
                )
            """)

            # Indexes for efficient lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_user_id ON interview_evaluations(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_created_at ON interview_evaluations(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_user ON user_topic_mastery(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_score ON user_topic_mastery(user_id, average_score)")
            conn.commit()

    @staticmethod
    def _calculate_mastery_level(avg_score: float, attempts: int) -> str:
        """
        Derive mastery level category from historical average and attempts:
        - MASTERY: avg >= 8.5 and attempts >= 2
        - PROFICIENT: avg >= 7.0
        - DEVELOPING: 5.0 <= avg < 7.0
        - NEEDS_WORK: avg < 5.0
        """
        if avg_score >= 8.5 and attempts >= 2:
            return "MASTERY"
        if avg_score >= 7.0:
            return "PROFICIENT"
        if avg_score >= 5.0:
            return "DEVELOPING"
        return "NEEDS_WORK"

    def save_evaluation(self, eval_result: InterviewEvaluationResult) -> Dict[str, Any]:
        """
        Persist an interview evaluation and update candidate's longitudinal topic progress.
        """
        session_id = eval_result.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        user_id = eval_result.user_id or "anonymous"
        created_at = eval_result.created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        strong_topics_json = json.dumps([t.model_dump() for t in eval_result.strong_topics], ensure_ascii=False)
        weak_topics_json = json.dumps([t.model_dump() for t in eval_result.weak_topics], ensure_ascii=False)
        all_topics_json = json.dumps([t.model_dump() for t in eval_result.all_topics], ensure_ascii=False)
        raw_json = json.dumps(eval_result.model_dump(), ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Upsert into interview_evaluations
            cursor.execute("""
                INSERT INTO interview_evaluations (
                    session_id, user_id, candidate_name, job_role,
                    overall_score, technical_depth_score, problem_solving_score,
                    communication_score, code_quality_score,
                    strong_topics_json, weak_topics_json, all_topics_json,
                    overview, markdown_feedback, raw_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    candidate_name=excluded.candidate_name,
                    job_role=excluded.job_role,
                    overall_score=excluded.overall_score,
                    technical_depth_score=excluded.technical_depth_score,
                    problem_solving_score=excluded.problem_solving_score,
                    communication_score=excluded.communication_score,
                    code_quality_score=excluded.code_quality_score,
                    strong_topics_json=excluded.strong_topics_json,
                    weak_topics_json=excluded.weak_topics_json,
                    all_topics_json=excluded.all_topics_json,
                    overview=excluded.overview,
                    markdown_feedback=excluded.markdown_feedback,
                    raw_json=excluded.raw_json,
                    created_at=excluded.created_at
            """, (
                session_id,
                user_id,
                eval_result.candidate_name,
                eval_result.job_role,
                eval_result.overall_score,
                eval_result.competencies.technical_depth,
                eval_result.competencies.problem_solving,
                eval_result.competencies.communication,
                eval_result.competencies.code_quality,
                strong_topics_json,
                weak_topics_json,
                all_topics_json,
                eval_result.overview,
                eval_result.markdown_feedback,
                raw_json,
                created_at,
            ))

            # Update longitudinal topic mastery for all evaluated topics
            assessed_topics = eval_result.all_topics or (eval_result.strong_topics + eval_result.weak_topics)
            seen_topics = set()

            for t in assessed_topics:
                norm_topic = t.topic.strip()
                if not norm_topic or norm_topic in seen_topics:
                    continue
                seen_topics.add(norm_topic)

                cursor.execute("""
                    SELECT attempts_count, average_score
                    FROM user_topic_mastery
                    WHERE user_id = ? AND topic = ?
                """, (user_id, norm_topic))
                existing = cursor.fetchone()

                if existing:
                    prev_attempts = existing["attempts_count"]
                    prev_avg = existing["average_score"]
                    new_attempts = prev_attempts + 1
                    # Exponential moving average / weighted running average
                    new_avg = round((prev_avg * prev_attempts + t.score) / new_attempts, 2)
                else:
                    new_attempts = 1
                    new_avg = round(t.score, 2)

                mastery_level = self._calculate_mastery_level(new_avg, new_attempts)

                cursor.execute("""
                    INSERT INTO user_topic_mastery (
                        user_id, topic, category, attempts_count,
                        average_score, latest_score, mastery_level,
                        latest_status, latest_recommendation, last_tested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, topic) DO UPDATE SET
                        category=excluded.category,
                        attempts_count=excluded.attempts_count,
                        average_score=excluded.average_score,
                        latest_score=excluded.latest_score,
                        mastery_level=excluded.mastery_level,
                        latest_status=excluded.latest_status,
                        latest_recommendation=excluded.latest_recommendation,
                        last_tested_at=excluded.last_tested_at
                """, (
                    user_id,
                    norm_topic,
                    t.category,
                    new_attempts,
                    new_avg,
                    t.score,
                    mastery_level,
                    t.status,
                    t.recommendation,
                    created_at,
                ))

            conn.commit()
            logger.info("Saved evaluation for session %s (user: %s). Processed %d topics.", session_id, user_id, len(seen_topics))

        return {"status": "success", "session_id": session_id, "user_id": user_id}

    def get_evaluation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full evaluation for a specific session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interview_evaluations WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._format_eval_row(row)

    def get_user_evaluations(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve interview evaluation history for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM interview_evaluations
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            return [self._format_eval_row(r) for r in rows]

    def get_user_analytics(self, user_id: str) -> UserAnalyticsSummary:
        """
        Generate aggregated candidate analytics:
        - Overall interview average and total sessions
        - Top strengths (best scoring topics)
        - Top weaknesses (weakest scoring topics needing work)
        - Full topic mastery matrix
        - Competency breakdown
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Evaluations stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_interviews,
                    AVG(overall_score) as avg_score,
                    AVG(technical_depth_score) as avg_tech,
                    AVG(problem_solving_score) as avg_ps,
                    AVG(communication_score) as avg_comm,
                    AVG(code_quality_score) as avg_code,
                    candidate_name
                FROM interview_evaluations
                WHERE user_id = ?
            """, (user_id,))
            eval_stat = cursor.fetchone()

            total_interviews = eval_stat["total_interviews"] if eval_stat else 0
            candidate_name = eval_stat["candidate_name"] if eval_stat and eval_stat["candidate_name"] else None
            avg_overall = round(eval_stat["avg_score"] or 0.0, 2) if eval_stat else 0.0

            competency_averages = {
                "technical_depth": round(eval_stat["avg_tech"] or 0.0, 2) if eval_stat else 0.0,
                "problem_solving": round(eval_stat["avg_ps"] or 0.0, 2) if eval_stat else 0.0,
                "communication": round(eval_stat["avg_comm"] or 0.0, 2) if eval_stat else 0.0,
            }
            if eval_stat and eval_stat["avg_code"] is not None:
                competency_averages["code_quality"] = round(eval_stat["avg_code"], 2)

            # All topic masteries
            cursor.execute("""
                SELECT * FROM user_topic_mastery
                WHERE user_id = ?
                ORDER BY average_score DESC
            """, (user_id,))
            topic_rows = cursor.fetchall()

            all_mastery = [self._format_topic_row(r) for r in topic_rows]

            # Top strengths (highest average score, score >= 7.0)
            top_strengths = [t for t in all_mastery if t.average_score >= 7.0][:5]

            # Top weaknesses (lowest average score, score < 6.5)
            # Sort ascending by average_score
            weak_candidates = [t for t in all_mastery if t.average_score < 6.5 or t.latest_status == "weak"]
            weak_candidates.sort(key=lambda x: x.average_score)
            top_weaknesses = weak_candidates[:5]

            # Recent evaluations preview
            cursor.execute("""
                SELECT session_id, job_role, overall_score, created_at
                FROM interview_evaluations
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 5
            """, (user_id,))
            recent = [dict(r) for r in cursor.fetchall()]

            return UserAnalyticsSummary(
                user_id=user_id,
                candidate_name=candidate_name,
                total_interviews=total_interviews,
                average_overall_score=avg_overall,
                competency_averages=competency_averages,
                top_strengths=top_strengths,
                top_weaknesses=top_weaknesses,
                all_topic_mastery=all_mastery,
                recent_evaluations=recent,
            )

    def get_user_weak_topics_for_interview(self, user_id: str, limit: int = 3) -> List[str]:
        """
        Fetch top weak topic names for a candidate to personalize the next interview.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT topic FROM user_topic_mastery
                WHERE user_id = ? AND (average_score < 6.5 OR latest_status = 'weak' OR mastery_level = 'NEEDS_WORK')
                ORDER BY average_score ASC, last_tested_at DESC
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            return [r["topic"] for r in rows]

    @staticmethod
    def _format_eval_row(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if data.get("strong_topics_json"):
            data["strong_topics"] = json.loads(data["strong_topics_json"])
        if data.get("weak_topics_json"):
            data["weak_topics"] = json.loads(data["weak_topics_json"])
        if data.get("all_topics_json"):
            data["all_topics"] = json.loads(data["all_topics_json"])
        if data.get("raw_json"):
            try:
                data["raw_evaluation"] = json.loads(data["raw_json"])
            except Exception:
                pass
        return data

    @staticmethod
    def _format_topic_row(row: sqlite3.Row) -> UserTopicMastery:
        return UserTopicMastery(
            user_id=row["user_id"],
            topic=row["topic"],
            category=row["category"],
            attempts_count=row["attempts_count"],
            average_score=row["average_score"],
            latest_score=row["latest_score"],
            mastery_level=row["mastery_level"],
            latest_status=row["latest_status"],
            latest_recommendation=row["latest_recommendation"],
            last_tested_at=row["last_tested_at"],
        )


# Global singleton instance
storage_service = StorageService()
