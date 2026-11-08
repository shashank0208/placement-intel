# config.py
# Central configuration for the placement_intel pipeline.
# Every tunable setting lives here so nothing is scattered across files.

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Settings ---
LLM_MODEL_NAME = "gpt-4o-mini"              # model used for structuring and RAG generation
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # model used for vector embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Sampling Parameters ---
EXTRACTION_TEMPERATURE = 0.0    # greedy decoding for structured extraction (deterministic)
GENERATION_TEMPERATURE = 0.3    # slightly creative for RAG answer generation
GENERATION_TOP_P = 0.9          # nucleus sampling cutoff for generation

# --- Chunking ---
CHUNK_SIZE = 1500               # max characters per chunk
CHUNK_OVERLAP = 200             # overlap between adjacent chunks to preserve context

# --- Retrieval ---
TOP_K_RETRIEVAL = 5             # number of chunks to retrieve from FAISS
MIN_RELEVANCE_SCORE = 0.65      # minimum cosine similarity to consider a chunk relevant

# --- Grading & Gates ---
MIN_MATCHES_FOR_ANSWER = 2      # minimum relevant chunks needed to attempt an answer
GROUNDEDNESS_THRESHOLD = 0.7    # minimum groundedness score to accept a generated answer
MAX_RETRIES = 2                 # max times to retry generation if groundedness fails

# --- PII Scrubbing ---
PII_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
    r'\b\d{10}\b',                                                # 10-digit phone numbers
    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',                           # US phone format
    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',             # credit card-like numbers
]

# --- File Paths ---
RAW_DATA_DIR = "data/raw"
CLEAN_DATA_DIR = "data/clean"
INDEX_DIR = "data/index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")

# --- Data Sources ---
FORM_CSV_PATH = os.path.join(RAW_DATA_DIR, "interview_responses.csv")
ALLOWED_URLS = [
    # Add URLs of allowed web pages to scrape for interview experiences
]

# --- Stage Summary Format ---
SUMMARY_SEPARATOR = "=" * 60   # visual separator for stage summaries
