# tests/test_ingest.py
# Tests for stage 1: data ingestion and synthetic data generation.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
ingest_mod = importlib.import_module("1_ingest")


def test_synthetic_data_has_records():
    """Synthetic data generator should produce test records."""
    records = ingest_mod.generate_synthetic_data()
    assert len(records) >= 4
    print(f"  âœ“ Synthetic data has {len(records)} records")


def test_synthetic_records_have_required_fields():
    """Each synthetic record must have text, company, role, source."""
    records = ingest_mod.generate_synthetic_data()
    for record in records:
        assert "text" in record, "Missing 'text' field"
        assert "company" in record, "Missing 'company' field"
        assert "source" in record, "Missing 'source' field"
    print("  âœ“ All synthetic records have required fields")


def test_synthetic_includes_edge_cases():
    """Synthetic data should include short text and adversarial input edge cases."""
    records = ingest_mod.generate_synthetic_data()
    texts = [r["text"] for r in records]

    has_short = any(len(t) < 50 for t in texts)
    has_adversarial = any("ignore" in t.lower() for t in texts)

    assert has_short, "No short-text edge case found"
    assert has_adversarial, "No adversarial input edge case found"
    print("  âœ“ Edge cases present (short text, adversarial)")


def test_csv_loader_handles_missing_file():
    """CSV loader should gracefully handle a missing file."""
    result = ingest_mod.load_csv_responses("nonexistent_file.csv")
    assert result == []
    print("  âœ“ CSV loader handles missing file gracefully")


if __name__ == "__main__":
    print("\n  Running ingest tests...")
    test_synthetic_data_has_records()
    test_synthetic_records_have_required_fields()
    test_synthetic_includes_edge_cases()
    test_csv_loader_handles_missing_file()
    print("\n  All ingest tests passed! âœ“\n")
