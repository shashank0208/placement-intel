# 2_clean.py
# Stage 2: Clean raw interview records before structuring.
# Runs PII scrubbing, deduplication, and quality filters. PII MUST run before any embedding.

import os
import re
import json
import html
import hashlib
from config import (
    PII_PATTERNS, RAW_DATA_DIR, CLEAN_DATA_DIR, SUMMARY_SEPARATOR
)


def fix_encoding(text):
    """Fix common encoding artifacts like mojibake and HTML entities."""
    text = html.unescape(text)
    # Strip invalid bytes that cause downstream tokenizer crashes
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def remove_html_tags(text):
    """Strip any remaining HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text):
    """Collapse multiple spaces/newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def scrub_personal_info(text):
    """Replace PII patterns (emails, phone numbers, card numbers) with placeholders."""
    # Must remove PII before vector embedding to prevent leakage in query retrieval
    pii_count = 0
    for pattern in PII_PATTERNS:
        matches = re.findall(pattern, text)
        pii_count += len(matches)
        text = re.sub(pattern, "[PII_REMOVED]", text)
    return text, pii_count


def clean_single_record(record):
    """Apply all cleaning steps to one record's text field."""
    text = record.get("text", "")
    text = fix_encoding(text)
    text = remove_html_tags(text)
    text = normalize_whitespace(text)
    text, pii_count = scrub_personal_info(text)
    cleaned = dict(record)
    cleaned["text"] = text
    cleaned["pii_items_removed"] = pii_count
    return cleaned


def compute_text_hash(text):
    """Create a hash of the text for deduplication."""
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()


def remove_duplicates(records):
    """Drop records with identical text content, keeping the first occurrence."""
    seen_hashes = set()
    unique = []
    for record in records:
        text_hash = compute_text_hash(record["text"])
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique.append(record)
    return unique, len(records) - len(unique)


def remove_too_short(records, min_length=50):
    """Filter out records whose text is too short to be useful."""
    # Exclude noise or partial submissions lacking actionable interview insight
    kept = [r for r in records if len(r["text"]) >= min_length]
    return kept, len(records) - len(kept)


def sanitize_adversarial_input(text):
    """Detect and neutralize common prompt injection patterns."""
    # Strip injection prefix rather than drop record to preserve legitimate experience
    injection_patterns = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)tell\s+me\s+(the\s+)?system\s+prompt",
        r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
        r"(?i)disregard\s+(all\s+)?prior\s+(instructions|context)",
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, "[INJECTION_REMOVED]", text)
    return text


def clean_all_records(records):
    """Run the full cleaning pipeline on a list of raw records."""
    cleaned = [clean_single_record(r) for r in records]
    for record in cleaned:
        record["text"] = sanitize_adversarial_input(record["text"])
    cleaned, dup_count = remove_duplicates(cleaned)
    cleaned, short_count = remove_too_short(cleaned)
    total_pii = sum(r.get("pii_items_removed", 0) for r in cleaned)
    return cleaned, dup_count, short_count, total_pii


def save_clean_records(records, output_path):
    """Save cleaned records as JSON for the next stage."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def load_raw_records(input_path):
    """Load raw records from the ingest stage output."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_clean():
    """Run the full cleaning stage and print a summary."""
    print(f"{SUMMARY_SEPARATOR}\nSTAGE 2: CLEAN\n{SUMMARY_SEPARATOR}")
    input_path = os.path.join(RAW_DATA_DIR, "all_raw_records.json")
    records = load_raw_records(input_path)
    print(f"  Loaded {len(records)} raw records")
    cleaned, dup_count, short_count, pii_count = clean_all_records(records)
    output_path = os.path.join(CLEAN_DATA_DIR, "cleaned_records.json")
    save_clean_records(cleaned, output_path)
    print(f"\n  Cleaned {len(cleaned)} docs:\n    dropped {dup_count} duplicates")
    print(f"    dropped {short_count} too short\n    replaced {pii_count} PII items")
    print(f"  Saved to: {output_path}\n{SUMMARY_SEPARATOR}")
    return cleaned


if __name__ == "__main__":
    run_clean()
