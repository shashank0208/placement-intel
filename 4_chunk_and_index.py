# 4_chunk_and_index.py
# Stage 4: Split structured records into per-question chunks, embed, and save a FAISS index.
# This creates the retrieval layer that the RAG pipeline (stage 5) queries.

import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import (
    EMBEDDING_MODEL_NAME, OPENAI_API_KEY, CLEAN_DATA_DIR,
    FAISS_INDEX_PATH, CHUNK_SIZE, SUMMARY_SEPARATOR
)


def load_structured_records(input_path):
    """Load structured interview records from stage 3 output."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_chunk(record, round_data, question):
    """Build a rich text chunk around a single interview question."""
    parts = [
        f"Company: {record.get('company', 'Unknown')}",
        f"Role: {record.get('role', 'Unknown')}",
        f"Round: {round_data.get('round_name', 'Unknown')}",
        f"Topics: {', '.join(round_data.get('topics', []))}",
        f"Question: {question}",
    ]
    # Supplementary round context improves retriever semantic alignment
    for field in ("tips", "difficulty"):
        if round_data.get(field):
            parts.append(f"{field.capitalize()}: {round_data[field]}")
    return "\n".join(parts)


def build_overview_chunk(record):
    """Build a summary chunk for the overall interview experience."""
    rounds = [r.get("round_name", "?") for r in record.get("rounds", [])]
    round_str = f"Rounds: {', '.join(rounds)}" if rounds else ""
    parts = [
        f"Company: {record.get('company', 'Unknown')}",
        f"Role: {record.get('role', 'Unknown')}",
        f"Year: {record.get('year', 'Unknown')}",
        f"Overall Experience: {record.get('overall_experience', 'Unknown')}",
        f"Offer Received: {record.get('offer_received', 'Unknown')}",
        f"Number of Rounds: {len(rounds)}",
        round_str,
    ]
    # Retain non-empty overview lines to keep index content compact
    return "\n".join([p for p in parts if p])


def create_question_doc(record, round_data, question):
    """Create a LangChain document for an individual interview question."""
    return Document(
        page_content=build_question_chunk(record, round_data, question),
        metadata={
            "company": record.get("company", "Unknown"),
            "role": record.get("role", "Unknown"),
            "round": round_data.get("round_name", "Unknown"),
            "source": record.get("source", "unknown"),
            "type": "question",
        },
    )


def create_overview_doc(record):
    """Create a LangChain document for the full interview overview."""
    return Document(
        page_content=build_overview_chunk(record),
        metadata={
            "company": record.get("company", "Unknown"),
            "role": record.get("role", "Unknown"),
            "source": record.get("source", "unknown"),
            "type": "overview",
        },
    )


def records_to_documents(records):
    """Convert structured records into LangChain Documents (one per question + overviews)."""
    documents = []
    # Both granular questions and holistic summaries are indexed for hybrid retrieval
    for record in records:
        for round_data in record.get("rounds", []):
            for question in round_data.get("questions", []):
                documents.append(create_question_doc(record, round_data, question))
        documents.append(create_overview_doc(record))
    return documents


def build_and_save_index(documents):
    """Embed all documents and save the FAISS index to disk."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=OPENAI_API_KEY,
    )
    # Batch embedding is handled automatically by FAISS integration
    index = FAISS.from_documents(documents, embeddings)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    index.save_local(FAISS_INDEX_PATH)
    return index


def load_existing_index():
    """Load a previously saved FAISS index from disk."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=OPENAI_API_KEY,
    )
    # Pipeline creates local index so dangerous deserialization is safe
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


def print_chunk_stats(documents):
    """Print chunk distribution counts by document type."""
    q_chunks = [d for d in documents if d.metadata.get("type") == "question"]
    o_chunks = [d for d in documents if d.metadata.get("type") == "overview"]
    print(f"  Created {len(documents)} chunks")
    print(f"    {len(q_chunks)} question chunks")
    print(f"    {len(o_chunks)} overview chunks")


def run_chunk_and_index():
    """Run the full chunking and indexing stage."""
    print(f"{SUMMARY_SEPARATOR}\nSTAGE 4: CHUNK & INDEX\n{SUMMARY_SEPARATOR}")
    input_path = os.path.join(CLEAN_DATA_DIR, "structured_records.json")
    records = load_structured_records(input_path)
    print(f"  Loaded {len(records)} structured records")
    documents = records_to_documents(records)
    print_chunk_stats(documents)
    # Save embeddings to disk so stage 5 can perform fast similarity lookups
    index = build_and_save_index(documents)
    print(f"\n  FAISS index saved to: {FAISS_INDEX_PATH}\n{SUMMARY_SEPARATOR}")
    return index


if __name__ == "__main__":
    run_chunk_and_index()
