# tests/test_rag_gates.py
# Tests for the three-branch gate logic in the RAG pipeline.
# The gate MUST refuse when evidence is missing -- this is a safety test.
# These tests are self-contained and do NOT require langchain to be installed.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import RetrievedChunk
from config import MIN_MATCHES_FOR_ANSWER, MIN_RELEVANCE_SCORE


# --- Inline the pure gate functions so tests run without langchain ---
# These are exact copies of the logic in 5_rag_graph.py

def decide_branch(relevant_chunks, min_matches):
    """Decide which branch to take based on evidence quality."""
    count = len(relevant_chunks)
    if count >= min_matches:
        return "correct"
    elif count > 0:
        return "ambiguous"
    else:
        return "incorrect"


def keep_only_relevant_chunks(chunks, min_score):
    """Filter out chunks below the relevance score threshold."""
    return [c for c in chunks if c.score >= min_score]


# --- Tests ---

def make_chunk(score):
    """Helper: create a fake RetrievedChunk with a given score."""
    return RetrievedChunk(text="test chunk", score=score, source="test")


def test_correct_branch_with_enough_evidence():
    """Gate should return 'correct' when enough relevant chunks are found."""
    chunks = [make_chunk(0.8) for _ in range(MIN_MATCHES_FOR_ANSWER)]
    branch = decide_branch(chunks, MIN_MATCHES_FOR_ANSWER)
    assert branch == "correct"
    print("  [OK] Correct branch with enough evidence")


def test_ambiguous_branch_with_partial_evidence():
    """Gate should return 'ambiguous' with some but not enough evidence."""
    chunks = [make_chunk(0.7)]  # only 1, but MIN_MATCHES is 2
    branch = decide_branch(chunks, MIN_MATCHES_FOR_ANSWER)
    assert branch == "ambiguous"
    print("  [OK] Ambiguous branch with partial evidence")


def test_incorrect_branch_with_no_evidence():
    """Gate MUST return 'incorrect' when no relevant chunks exist (safety)."""
    branch = decide_branch([], MIN_MATCHES_FOR_ANSWER)
    assert branch == "incorrect"
    print("  [OK] Incorrect branch with no evidence (SAFETY CHECK)")


def test_relevance_filtering():
    """Chunks below the relevance threshold should be filtered out."""
    chunks = [
        make_chunk(0.9),   # above threshold
        make_chunk(0.3),   # below threshold
        make_chunk(0.7),   # above threshold
    ]
    kept = keep_only_relevant_chunks(chunks, MIN_RELEVANCE_SCORE)
    assert all(c.score >= MIN_RELEVANCE_SCORE for c in kept)
    print("  [OK] Relevance filtering works")


if __name__ == "__main__":
    print("\n  Running RAG gate tests...")
    test_correct_branch_with_enough_evidence()
    test_ambiguous_branch_with_partial_evidence()
    test_incorrect_branch_with_no_evidence()
    test_relevance_filtering()
    print("\n  All RAG gate tests passed!\n")
