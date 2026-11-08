# 🎯 Placement Intel

**Interview preparation assistant powered by RAG (Retrieval-Augmented Generation).**

Built specifically for ZS Associates Data Science / AI Associate interview preparation, with a focus on LLM primitives: tokens, embeddings, temperature, and RAG system design.

## Pipeline Overview

| Stage | File | What it does |
|-------|------|-------------|
| 1 | `1_ingest.py` | Load data from CSV forms, web pages, and synthetic test data |
| 2 | `2_clean.py` | Fix encoding, scrub PII, deduplicate, filter short texts |
| 3 | `3_structure.py` | LLM extracts structured InterviewRecords from free text |
| 4 | `4_chunk_and_index.py` | Split into per-question chunks, embed, save FAISS index |
| 5 | `5_rag_graph.py` | Retrieve → grade → gate → generate → groundedness check |
| 6 | `6_evaluate.py` | Hit-rate, MRR, groundedness, abstention accuracy |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env with your OpenAI API key

# 3. Run the data pipeline (stages 1-4)
python run_all.py

# 4. Ask questions (CLI mode)
python app.py

# 5. Or use the Streamlit interface
streamlit run app.py
```

## Safety Features

- **PII scrubbing** runs before any text is embedded or sent to an LLM
- **Three-branch gate** refuses to answer when evidence is insufficient
- **Groundedness check** verifies answers are supported by retrieved context
- **Adversarial input sanitization** neutralizes prompt injection attempts

## Testing

```bash
python tests/test_ingest.py
python tests/test_clean.py
python tests/test_schemas.py
python tests/test_rag_gates.py
```

## Key Configuration

All settings live in `config.py`:
- `EXTRACTION_TEMPERATURE = 0.0` — greedy decoding for structured extraction
- `GENERATION_TEMPERATURE = 0.3` — slightly creative for answer generation
- `MIN_MATCHES_FOR_ANSWER = 2` — minimum chunks needed to attempt an answer
- `GROUNDEDNESS_THRESHOLD = 0.7` — minimum score to accept a generated answer
