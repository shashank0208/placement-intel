# 3_structure.py
# Stage 3: Convert cleaned free-text interview experiences into structured records.
# Uses an LLM with low temperature for deterministic extraction into Pydantic models.

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import (
    LLM_MODEL_NAME, OPENAI_API_KEY, EXTRACTION_TEMPERATURE,
    CLEAN_DATA_DIR, SUMMARY_SEPARATOR
)
from schemas import InterviewRecord


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a data extraction assistant. Extract structured interview information "
        "from the text below. Return valid JSON matching the schema exactly. "
        "If a field is not mentioned, use null. Do not invent information."
    )),
    ("human", (
        "Extract the interview record from this text:\n\n"
        "---\n{text}\n---\n\n"
        "Return a JSON object with these fields:\n"
        "- company (string)\n"
        "- role (string)\n"
        "- year (integer or null)\n"
        "- source (string: keep as '{source}')\n"
        "- overall_experience (string: 'Positive', 'Neutral', or 'Negative', or null)\n"
        "- offer_received (boolean or null)\n"
        "- rounds (list of objects, each with: round_name, topics, questions, difficulty, tips)\n"
        "\nReturn ONLY the JSON, no markdown fences."
    ))
])


def build_extraction_chain():
    """Create the LLM chain for structured extraction."""
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=EXTRACTION_TEMPERATURE,
    )
    return EXTRACTION_PROMPT | llm


def parse_llm_json(llm_output_text):
    """Parse the LLM text response into a Python dictionary."""
    text = llm_output_text.strip()
    # Strip code block wrappers because some LLMs wrap JSON output in markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def extract_single_record(chain, record):
    """Run the LLM to extract structure from one cleaned text record."""
    try:
        result = chain.invoke({
            "text": record["text"],
            "source": record.get("source", "unknown"),
        })
        parsed = parse_llm_json(result.content)
        return InterviewRecord(**parsed)
    except Exception as e:
        # Surface errors to assist in identifying malformed records during pipeline runs
        print(f"  [error] Failed to extract record: {e}")
        return None


def extract_all_records(records):
    """Extract structured data from all cleaned records using the LLM."""
    chain = build_extraction_chain()
    structured = []
    failed = 0
    for i, record in enumerate(records):
        print(f"  Extracting {i + 1}/{len(records)}...", end="\r")
        result = extract_single_record(chain, record)
        if result:
            structured.append(result)
        else:
            failed += 1
    return structured, failed


def save_structured_records(records, output_path):
    """Save structured InterviewRecords to disk as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = [r.model_dump() for r in records]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_clean_records(input_path):
    """Load cleaned records from stage 2 output file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_stage_summary(structured, failed, output_path):
    """Print metrics summarizing extracted interview records and questions."""
    total_q = sum(len(rd.questions) for r in structured for rd in r.rounds)
    print(f"\n  Structured {len(structured)} records ({failed} failed)")
    print(f"  Total questions extracted: {total_q}")
    print(f"  Saved to: {output_path}")
    print(SUMMARY_SEPARATOR)


def run_structure():
    """Run the full structuring stage and print a summary."""
    print(f"{SUMMARY_SEPARATOR}\nSTAGE 3: STRUCTURE\n{SUMMARY_SEPARATOR}")
    input_path = os.path.join(CLEAN_DATA_DIR, "cleaned_records.json")
    records = load_clean_records(input_path)
    print(f"  Loaded {len(records)} cleaned records")
    structured, failed = extract_all_records(records)
    output_path = os.path.join(CLEAN_DATA_DIR, "structured_records.json")
    save_structured_records(structured, output_path)
    print_stage_summary(structured, failed, output_path)
    return structured


if __name__ == "__main__":
    run_structure()
