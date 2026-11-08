# Decisions Log

## Architecture Decisions

### 1. Numbered file names
Files are numbered (1_ingest.py, 2_clean.py, ...) to make the pipeline order obvious at a glance. Python can't import these directly, so we use `importlib.import_module()` where needed.

### 2. Plain functions over classes
Following the simplicity rules: no inheritance chains, no abstract base classes. Classes are used only where Pydantic requires them (data models).

### 3. Per-question chunking
Each interview question becomes its own chunk (with context: company, role, round). This gives fine-grained retrieval — a query about "LLM tokenization questions" matches the specific question chunk, not a whole interview blob.

### 4. Three-branch gate (safety-critical)
The RAG pipeline has a mandatory gate:
- **Correct**: ≥2 relevant chunks → generate answer
- **Ambiguous**: 1 chunk → generate with low-confidence flag
- **Incorrect**: 0 chunks → refuse to answer

This prevents hallucination when the system has no evidence.

### 5. PII scrubbing before embedding
PII must be removed BEFORE text enters the vector store. This is enforced by the pipeline order (clean → structure → embed) and tested explicitly.

### 6. Synthetic test data
The project includes synthetic interview records with edge cases (short text, PII, prompt injection) so the pipeline can be tested without real user data.

### 7. Temperature settings
- Extraction (stage 3): Temperature 0.0 for deterministic, faithful extraction
- Generation (stage 5): Temperature 0.3 for slightly varied but grounded answers
- Groundedness scoring: Temperature 0.0 for consistent evaluation
