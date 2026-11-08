# run_all.py
# Runs the full data pipeline: ingest → clean → structure → chunk & index.
# Stages 5 and 6 (RAG and evaluation) are run separately since they need an API key.

import importlib
import time


def import_stage(module_name):
    """Import a numbered stage module using importlib (since names start with digits)."""
    # Dynamic import allows numbered module names forbidden in standard import syntax
    return importlib.import_module(module_name)


def execute_stages():
    """Execute the sequential stages from ingestion to indexing."""
    # Each stage must complete in order before search indexing is viable
    import_stage("1_ingest").run_ingest()
    import_stage("2_clean").run_clean()
    import_stage("3_structure").run_structure()
    import_stage("4_chunk_and_index").run_chunk_and_index()


def run_all():
    """Execute stages 1 through 4 in sequence with timing."""
    start = time.time()
    print("\n" + "=" * 60 + "\n  PLACEMENT INTEL — FULL PIPELINE RUN\n" + "=" * 60 + "\n")
    execute_stages()
    elapsed = time.time() - start
    print(f"\n  Pipeline complete in {elapsed:.1f} seconds\n" + "=" * 60)


if __name__ == "__main__":
    run_all()
