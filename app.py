# app.py
# User-facing interface: CLI mode and a minimal Streamlit page.
# Run with `python app.py` for CLI or `streamlit run app.py` for web.

import sys
import importlib
from config import SUMMARY_SEPARATOR


def load_rag_components():
    """Load the RAG query function and FAISS index dynamically."""
    # Module name starts with a digit so importlib is required
    rag_mod = importlib.import_module("5_rag_graph")
    return rag_mod.load_existing_index(), rag_mod.run_rag_query


def print_cli_answer(result):
    """Format and display the query answer and groundedness score."""
    print(f"\n  Answer ({result.decision}, groundedness: {result.groundedness_score:.2f}):")
    print(f"  {result.answer}")
    print(f"  [{len(result.chunks_used)} chunks used]")


def cli_loop(index, run_rag_query):
    """Run the interactive query-answer prompt loop."""
    while True:
        query = input("\n  Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if query:
            result = run_rag_query(index, query)
            print_cli_answer(result)


def run_cli():
    """Interactive CLI for asking interview prep questions."""
    print(f"{SUMMARY_SEPARATOR}\n  PLACEMENT INTEL — Interview Prep Assistant\n  Type 'quit' to exit\n{SUMMARY_SEPARATOR}")
    try:
        index, run_rag_query = load_rag_components()
    except Exception as e:
        print(f"  [error] Could not load index: {e}\n  Run `python run_all.py` first to build the pipeline.")
        return
    cli_loop(index, run_rag_query)


def init_streamlit_session(st):
    """Configure page layout and initialize message history."""
    st.set_page_config(page_title="Placement Intel", page_icon="🎯")
    st.title("🎯 Placement Intel")
    st.caption("Interview preparation assistant powered by RAG")
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history(st):
    """Render prior chat messages stored in session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_chunk_details(st, result):
    """Display retrieved source chunks inside an expander for transparency."""
    with st.expander("Details"):
        st.write(f"**Decision:** {result.decision}")
        st.write(f"**Groundedness:** {result.groundedness_score:.2f}")
        st.write(f"**Chunks used:** {len(result.chunks_used)}")
        for i, chunk in enumerate(result.chunks_used):
            st.text(f"--- Chunk {i+1} (score: {chunk.score:.3f}) ---")
            st.text(chunk.text[:300])


def handle_user_prompt(st, prompt, index, run_rag_query):
    """Process a user prompt through the RAG pipeline and render response."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = run_rag_query(index, prompt)
        st.markdown(result.answer)
        render_chunk_details(st, result)
    st.session_state.messages.append({"role": "assistant", "content": result.answer})


def run_streamlit():
    """Minimal Streamlit UI for the interview prep assistant."""
    import streamlit as st
    init_streamlit_session(st)
    get_index = st.cache_resource(load_rag_components)
    try:
        index, run_rag_query = get_index()
    except Exception as e:
        st.error(f"Could not load index: {e}. Run `python run_all.py` first.")
        return
    render_chat_history(st)
    if prompt := st.chat_input("Ask about interview experiences..."):
        handle_user_prompt(st, prompt, index, run_rag_query)


# Detect whether we're running under Streamlit or as a script
if "streamlit" in sys.modules and hasattr(sys.modules["streamlit"], "runtime"):
    run_streamlit()
elif __name__ == "__main__":
    run_cli()
