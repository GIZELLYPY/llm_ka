"""Extraction Contract Schema."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Difficulty = Literal["beginner", "intermediate", "advanced", "unknown"]
Intent = Literal["how-to", "debugging", "conceptual", "opinion", "unknown"]

class ExtractedQuestion(BaseModel):
    question_id: int = Field(..., description="StackOverflow question id")
    title: str
    primary_technology: Optional[str] = Field(None, description="Main tech, e.g. python, java, sql, pyspark")
    secondary_technologies: List[str] = Field(default_factory=list)
    problem_type: Optional[str] = Field(None, description="short label: e.g. dataframe manipulation, authenticfation, performance, parsing")
    intent: Intent = "unknown"
    difficulty: Difficulty = "unknown"
    key_entities: List[str] = Field(default_factory=list, description="key libraries/tools/terms mentioned")
    summary: str = Field(..., description="One-sentence summary of the question")
    confidence: float = Field(..., ge=0.0, le=1.0, description="model confidence 0..1")
    

