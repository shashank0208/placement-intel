# schemas.py
# Pydantic data models for structured interview records.
# Used by stage 3 (structuring) and downstream stages.

from typing import Optional
from pydantic import BaseModel, Field


class Round(BaseModel):
    """One interview round with its questions and metadata."""
    round_name: str = Field(description="Name of the round, e.g. 'Technical Round 1', 'HR Round'")
    topics: list[str] = Field(default_factory=list, description="Topics covered, e.g. ['LLM', 'SQL', 'Case Study']")
    questions: list[str] = Field(default_factory=list, description="Specific questions asked in this round")
    difficulty: Optional[str] = Field(default=None, description="Easy / Medium / Hard")
    tips: Optional[str] = Field(default=None, description="Candidate's advice for this round")


class InterviewRecord(BaseModel):
    """A single candidate's full interview experience at a company."""
    company: str = Field(description="Company name, e.g. 'ZS Associates'")
    role: str = Field(description="Role applied for, e.g. 'Data Science Associate'")
    year: Optional[int] = Field(default=None, description="Year of the interview")
    source: str = Field(description="Where this data came from: 'form', 'web', 'reddit', 'synthetic'")
    overall_experience: Optional[str] = Field(default=None, description="Positive / Neutral / Negative")
    rounds: list[Round] = Field(default_factory=list, description="List of interview rounds")
    raw_text: Optional[str] = Field(default=None, description="Original unstructured text before parsing")
    offer_received: Optional[bool] = Field(default=None, description="Whether the candidate got an offer")


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the FAISS index with its score."""
    text: str = Field(description="The chunk text content")
    score: float = Field(description="Cosine similarity score from retrieval")
    source: str = Field(default="unknown", description="Origin of this chunk")
    metadata: dict = Field(default_factory=dict, description="Any extra metadata")


class RAGResult(BaseModel):
    """Final output of the RAG pipeline for one query."""
    query: str = Field(description="The user's original question")
    answer: str = Field(description="Generated answer grounded in retrieved context")
    chunks_used: list[RetrievedChunk] = Field(default_factory=list, description="Chunks that contributed to the answer")
    groundedness_score: float = Field(default=0.0, description="How well the answer is supported by chunks")
    decision: str = Field(default="answered", description="'answered', 'abstained', or 'low_confidence'")
