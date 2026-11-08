# 5_rag_graph.py
# Stage 5: RAG pipeline with retrieval, grading, three-branch gating, and groundedness check.
# The gate MUST refuse to answer when evidence is insufficient — this is a safety requirement.

import json
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from config import (
    LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, OPENAI_API_KEY,
    GENERATION_TEMPERATURE, GENERATION_TOP_P, TOP_K_RETRIEVAL,
    MIN_RELEVANCE_SCORE, MIN_MATCHES_FOR_ANSWER, GROUNDEDNESS_THRESHOLD,
    MAX_RETRIES, FAISS_INDEX_PATH, SUMMARY_SEPARATOR
)
from schemas import RetrievedChunk, RAGResult

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful interview preparation assistant. Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say so honestly. Do not make up information."),
    ("human", "Context:\n{context}\n\nQuestion: {query}\n\nProvide a helpful, specific answer based on the context above.")
])

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 'You are a fact-checking assistant. Score how well the answer is supported by the provided context. Return ONLY a JSON object: {"score": <float 0-1>, "explanation": "<brief reason>"}'),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}\n\nHow well is this answer grounded in the context? Score from 0.0 to 1.0.")
])

def load_existing_index():
    """Load a previously saved FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def retrieve_chunks(index, query, top_k):
    """Fetch the top-k most similar chunks from the FAISS index."""
    results = index.similarity_search_with_score(query, k=top_k)
    chunks = []
    for doc, score in results:
        # FAISS returns L2 distance; convert to 0-1 similarity for easier thresholding
        similarity = 1.0 / (1.0 + score)
        chunks.append(RetrievedChunk(
            text=doc.page_content, score=similarity,
            source=doc.metadata.get("source", "unknown"), metadata=doc.metadata
        ))
    return chunks

def keep_only_relevant_chunks(chunks, min_score):
    """Filter out chunks below the relevance score threshold."""
    return [c for c in chunks if c.score >= min_score]

# This gate is a safety requirement to prevent hallucination without supporting evidence.
def decide_branch(relevant_chunks, min_matches):
    """Decide which branch to take based on evidence quality."""
    count = len(relevant_chunks)
    if count >= min_matches:
        return "correct"
    elif count > 0:
        return "ambiguous"
    return "incorrect"

def format_context(chunks):
    """Join retrieved chunk texts into a single context string."""
    return "\n\n---\n\n".join(c.text for c in chunks)

def generate_answer(query, chunks):
    """Use the LLM to generate an answer grounded in the retrieved chunks."""
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY,
        temperature=GENERATION_TEMPERATURE, top_p=GENERATION_TOP_P
    )
    chain = ANSWER_PROMPT | llm
    return chain.invoke({"context": format_context(chunks), "query": query}).content

def parse_groundedness_score(raw_text):
    """Extract numeric groundedness score from JSON LLM output."""
    cleaned = raw_text.strip().strip("`").removeprefix("json").strip()
    return float(json.loads(cleaned).get("score", 0.0))

def check_groundedness(answer, chunks):
    """Score how well the generated answer is supported by the retrieved context."""
    llm = ChatOpenAI(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.0)
    chain = GROUNDEDNESS_PROMPT | llm
    try:
        res = chain.invoke({"context": format_context(chunks), "answer": answer})
        return parse_groundedness_score(res.content)
    except Exception as e:
        print(f"  [warning] Groundedness check failed: {e}")
        return 0.0

def generate_and_verify(query, relevant):
    """Generate answer and verify groundedness with retries if ungrounded."""
    answer = generate_answer(query, relevant)
    groundedness = check_groundedness(answer, relevant)
    for retry in range(MAX_RETRIES):
        if groundedness >= GROUNDEDNESS_THRESHOLD:
            break
        print(f"  [retry {retry + 1}] Groundedness {groundedness:.2f} below threshold, regenerating...")
        answer = generate_answer(query, relevant)
        groundedness = check_groundedness(answer, relevant)
    return answer, groundedness

def run_rag_query(index, query):
    """Execute the full RAG pipeline for a single query."""
    all_chunks = retrieve_chunks(index, query, TOP_K_RETRIEVAL)
    relevant = keep_only_relevant_chunks(all_chunks, MIN_RELEVANCE_SCORE)
    branch = decide_branch(relevant, MIN_MATCHES_FOR_ANSWER)
    if branch == "incorrect":
        return RAGResult(
            query=query, answer="I don't have enough information to answer this question reliably.",
            chunks_used=[], groundedness_score=0.0, decision="abstained"
        )
    answer, groundedness = generate_and_verify(query, relevant)
    decision = "answered" if (branch == "correct" and groundedness >= GROUNDEDNESS_THRESHOLD) else "low_confidence"
    return RAGResult(query=query, answer=answer, chunks_used=relevant, groundedness_score=groundedness, decision=decision)

def print_query_result(query, result):
    """Print the formatted evaluation details of a single query result."""
    print(f"\n  Q: {query}")
    print(f"  Decision: {result.decision}")
    print(f"  Groundedness: {result.groundedness_score:.2f}")
    print(f"  Chunks used: {len(result.chunks_used)}")
    print(f"  A: {result.answer[:200]}...")

def run_demo_queries():
    """Run a few demo queries to test the RAG pipeline."""
    print(f"{SUMMARY_SEPARATOR}\nSTAGE 5: RAG DEMO\n{SUMMARY_SEPARATOR}")
    index = load_existing_index()
    demo_queries = [
        "What kind of questions does ZS Associates ask about LLMs?",
        "What is the interview process at ZS Associates?",
        "Does ZS ask about RAG systems?",
        "What are the salary details at ZS Associates?",
    ]
    for query in demo_queries:
        print_query_result(query, run_rag_query(index, query))
    print(SUMMARY_SEPARATOR)

if __name__ == "__main__":
    run_demo_queries()
