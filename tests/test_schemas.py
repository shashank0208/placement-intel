# tests/test_schemas.py
# Tests for Pydantic schemas: InterviewRecord, Round, RetrievedChunk, RAGResult.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import InterviewRecord, Round, RetrievedChunk, RAGResult


def test_round_creation():
    """A Round should accept all expected fields."""
    r = Round(
        round_name="Technical Round 1",
        topics=["LLM", "SQL"],
        questions=["What is tokenization?"],
        difficulty="Medium",
        tips="Study embeddings deeply",
    )
    assert r.round_name == "Technical Round 1"
    assert len(r.topics) == 2
    print("  âœ“ Round creation works")


def test_interview_record_creation():
    """An InterviewRecord should hold rounds and metadata."""
    record = InterviewRecord(
        company="ZS Associates",
        role="Data Science Associate",
        year=2024,
        source="form",
        overall_experience="Positive",
        rounds=[
            Round(round_name="R1", topics=["ML"], questions=["Explain gradient descent"]),
        ],
        offer_received=True,
    )
    assert record.company == "ZS Associates"
    assert len(record.rounds) == 1
    print("  âœ“ InterviewRecord creation works")


def test_interview_record_optional_fields():
    """Optional fields should default to None."""
    record = InterviewRecord(company="Test", role="Test", source="test")
    assert record.year is None
    assert record.offer_received is None
    assert record.rounds == []
    print("  âœ“ Optional fields default correctly")


def test_rag_result_creation():
    """RAGResult should hold query, answer, and metadata."""
    result = RAGResult(
        query="Test question?",
        answer="Test answer.",
        groundedness_score=0.85,
        decision="answered",
    )
    assert result.decision == "answered"
    print("  âœ“ RAGResult creation works")


if __name__ == "__main__":
    print("\n  Running schema tests...")
    test_round_creation()
    test_interview_record_creation()
    test_interview_record_optional_fields()
    test_rag_result_creation()
    print("\n  All schema tests passed! âœ“\n")
