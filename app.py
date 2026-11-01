from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from learnml_core import (
    build_index,
    embed_texts,
    load_index,
    retrieve_chunks,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
STORAGE_DIR = ROOT / "storage"
DOCS_DIR = ROOT / "knowledge_base" / "documents"
DEFAULT_GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

st.set_page_config(
    page_title="LearnML | AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1100px; padding-top: 1.5rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.25);
             border-radius: 16px; background: linear-gradient(120deg, rgba(45,125,210,.12), rgba(22,180,150,.08));}
      .subtle {color: #7b8491; font-size: .92rem;}
      .source-card {padding: .8rem 1rem; border: 1px solid rgba(128,128,128,.25);
                    border-radius: 12px; margin: .45rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def create_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def generate_teacher_answer(
    client: Any,
    model: str,
    question: str,
    mode: str,
    context_chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> str:
    if not context_chunks:
        return (
            "I couldn't find relevant material in the indexed LearnML library. "
            "Try rephrasing the question or ask the project maintainer to add a relevant source document."
        )

    sources_text = []
    for i, item in enumerate(context_chunks, start=1):
        page = item.get("page")
        page_text = f", page {page}" if page else ""
        sources_text.append(
            f"[S{i}] Source: {item.get('source', 'Unknown document')}{page_text}\n"
            f"Subject: {item.get('subject', 'General')}\n"
            f"Passage: {item.get('text', '')}"
        )

    previous = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
    )
    style = (
        "Teach a beginner. Build intuition first, define terms simply, use an analogy, "
        "then give a small example. Avoid assuming prior knowledge."
        if mode == "Beginner"
        else "Teach at an advanced undergraduate level. Include relevant equations, "
        "assumptions, derivations or algorithmic detail, and a practical Python example when useful."
    )

    prompt = f"""You are LearnML, a patient and accurate Data Science and Machine Learning teacher.

TEACHING STYLE:
{style}

IMPORTANT SOURCE RULES:
- Use the retrieved passages as the primary evidence for claims about the indexed books.
- Cite source-supported statements using [S1], [S2], etc. Do not invent citations, page numbers, chapters, or quotations.
- If the retrieved passages do not adequately answer the question, say what is missing. You may add general background knowledge, but clearly label it as 'Additional explanation' and do not attribute it to a source.
- Never claim that a book says something unless the retrieved passages support it.
- Use clear headings and explain symbols whenever you show a formula.
- Answer the question directly and teach, rather than merely summarising the passages.

RECENT CONVERSATION:
{previous or 'No previous conversation.'}

RETRIEVED KNOWLEDGE:
{chr(10).join(sources_text)}

USER QUESTION:
{question}

Write a helpful, structured teacher-style answer. Add a short 'Sources used' line at the end using only the source IDs that you actually cited."""

    try:
        response = client.interactions.create(model=model, input=prompt)
        output = getattr(response, "output_text", None)
        if output:
            return output.strip()
    except Exception:
        # Compatibility fallback for accounts or SDK versions that do not expose Interactions yet.
        pass
    fallback = client.models.generate_content(model=model, contents=prompt)
    fallback_text = getattr(fallback, "text", None)
    if fallback_text:
        return str(fallback_text).strip()
    raise RuntimeError("The Gemini response did not contain text. Try another generation model.")


@st.cache_data(show_spinner=False)
def cached_index_paths() -> tuple[bool, str, str]:
    chunks_path = STORAGE_DIR / "chunks.json"
    vectors_path = STORAGE_DIR / "embeddings.npy"
    return chunks_path.exists() and vectors_path.exists(), str(chunks_path), str(vectors_path)


def current_api_key() -> str:
    # Secrets are read from Streamlit Secrets or the local .env file.
    # They are intentionally never requested or displayed in the app UI.
    return get_secret("GEMINI_API_KEY").strip()


with st.sidebar:
    st.title("🎓 LearnML")
    st.caption("Your Data Science & ML study companion")
    page = st.radio("Navigate", ["AI Tutor", "Knowledge Library", "Setup & Help"], index=0)
    st.divider()
    st.caption("Ask questions and learn from the indexed library.")

# Model settings are intentionally kept out of the interface.
# Configure them through .env locally or Streamlit Secrets when deployed.
generation_model = get_secret("GEMINI_MODEL", DEFAULT_GENERATION_MODEL)
embedding_model = get_secret("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

index_exists, _, _ = cached_index_paths()

if page == "AI Tutor":
    st.markdown(
        "<div class='hero'><h1 style='margin:0'>LearnML AI Tutor</h1>"
        "<p style='margin:.45rem 0 0 0'>Ask questions about statistics, data science, machine learning, deep learning and NLP. Learn with explanations grounded in the indexed library.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    if not index_exists:
        st.warning("The knowledge index has not been built yet. Open **Knowledge Library** and build it first.")
        st.stop()

    index = load_index(STORAGE_DIR)
    if not index["chunks"]:
        st.warning("The indexed library is empty. Add a document and rebuild the index.")
        st.stop()

    left, right = st.columns([3, 1])
    with left:
        mode = st.segmented_control("Teaching style", options=["Beginner", "Advanced"], default="Beginner")
    with right:
        top_k = st.selectbox("Sources to retrieve", [3, 5, 7], index=1)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    top_bar_left, top_bar_right = st.columns([5, 1])
    with top_bar_right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View retrieved sources"):
                    for source in msg["sources"]:
                        page_text = f" · page {source.get('page')}" if source.get("page") else ""
                        st.markdown(f"**{source.get('source')}**{page_text}")
                        if source.get("source_url"):
                            st.caption(source.get("source_url"))
                        st.caption(source.get("text", "")[:900])

    question = st.chat_input("Ask a Data Science or ML question...")
    if question:
        if not current_api_key():
            st.error("Gemini API key is not configured. Add GEMINI_API_KEY to your local .env file or Streamlit app secrets.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the library and preparing your lesson..."):
                    client = create_client(current_api_key())
                    query_vector = embed_texts(
                        client, [question], embedding_model, task_type="RETRIEVAL_QUERY"
                    )[0]
                    retrieved = retrieve_chunks(index["chunks"], index["embeddings"], query_vector, top_k=top_k)
                    history = st.session_state.messages[:-1]
                    answer = generate_teacher_answer(
                        client, generation_model, question, mode, retrieved, history
                    )
                    st.markdown(answer)
                    if retrieved:
                        with st.expander("View retrieved sources"):
                            for i, source in enumerate(retrieved, start=1):
                                page_text = f" · page {source.get('page')}" if source.get("page") else ""
                                st.markdown(f"**[S{i}] {source.get('source')}**{page_text} · similarity {source.get('score', 0):.3f}")
                                if source.get("source_url"):
                                    st.caption(source.get("source_url"))
                                st.caption(source.get("text", "")[:900])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": retrieved}
            )
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.info("Check that your Gemini API key is valid and the selected model is available to your account.")

elif page == "Knowledge Library":
    st.title("Knowledge Library")
    st.write("Build a searchable library from the included starter notes and books added by the project maintainer.")

    index = load_index(STORAGE_DIR) if index_exists else {"chunks": [], "embeddings": np.empty((0, 0))}
    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed chunks", len(index["chunks"]))
    sources = sorted({c.get("source", "Unknown") for c in index["chunks"]})
    col2.metric("Indexed documents", len(sources))
    col3.metric("Index ready", "Yes" if index_exists else "No")

    st.subheader("Documents in the project")
    available_files = sorted(
        p.name for p in DOCS_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    if available_files:
        for name in available_files:
            st.write(f"📘 {name}")
    else:
        st.info("No book files found yet. The original LearnML starter notes are included in the documents folder.")

    st.caption("To add another book, place its PDF, TXT or Markdown file in knowledge_base/documents/ locally, then rebuild the index. Only add documents you have permission to use.")

    st.subheader("Build / rebuild the search index")
    st.caption("This uses Gemini embeddings to turn document chunks into vectors. The index is saved locally in the storage folder.")
    if st.button("Build knowledge index", type="primary", use_container_width=True):
        if not current_api_key():
            st.error("Gemini API key is not configured. Add GEMINI_API_KEY to your local .env file or Streamlit app secrets.")
        else:
            progress = st.progress(0, text="Preparing documents...")
            status = st.empty()
            try:
                client = create_client(current_api_key())
                def report(percent: int, message: str) -> None:
                    progress.progress(max(0, min(100, percent)), text=message)
                    status.write(message)
                result = build_index(
                    client=client,
                    documents_dir=DOCS_DIR,
                    storage_dir=STORAGE_DIR,
                    embedding_model=embedding_model,
                    progress_callback=report,
                )
                st.cache_data.clear()
                st.success(f"Index built successfully: {result['chunks']} chunks from {result['documents']} documents.")
                st.rerun()
            except Exception as exc:
                st.error(f"Index building failed: {exc}")
                st.info("Try again after checking your API key, network connection and Gemini API quota.")

    if index_exists:
        st.subheader("Indexed source list")
        for name in sources:
            st.write(f"- {name}")
        with st.expander("Index storage details"):
            st.code("storage/chunks.json\nstorage/embeddings.npy")
            st.caption("These two files can be committed to GitHub after you build the index. Do not upload your API key.")

elif page == "Setup & Help":
    st.title("Setup & Help")
    st.markdown("""
    ### 1. Get a Gemini API key
    Create one in [Google AI Studio](https://aistudio.google.com/app/apikey).
    For local use, add it to your `.env` file as `GEMINI_API_KEY=...`.
    For Streamlit Community Cloud, add `GEMINI_API_KEY` under the app's **Settings → Secrets**.
    The key is never entered into or displayed in the LearnML interface.

    ### 2. Add learning material
    - The project includes original starter notes.
    - Run `python download_books.py` to download the openly licensed *Dive into Deep Learning* PDF.
    - Download *OpenIntro Statistics* from its [official book page](https://www.openintro.org/book/os/) (the PDF is available at a $0 contribution) and place it in `knowledge_base/documents/`.
    - To add another permitted document, place it in `knowledge_base/documents/` and rebuild the index.

    ### 3. Build your index
    Open **Knowledge Library** and click **Build knowledge index**. This creates `storage/chunks.json` and `storage/embeddings.npy`.

    ### 4. Start learning
    Open **AI Tutor**, select Beginner or Advanced mode, and ask a question.

    ### 5. Deploy
    Commit the app, requirements, knowledge-base text files and the two generated `storage/` index files to GitHub. Add `GEMINI_API_KEY` under Streamlit Community Cloud app secrets. Do not commit `.env` or API keys. Model names can be configured through `.env` or Streamlit Secrets; they are not shown in the interface.
    """)
    st.info("LearnML v1 intentionally keeps the core simple: document loading, chunking, Gemini embeddings, cosine-similarity retrieval and Gemini answer generation. Quizzes and progress tracking can be added after this core flow is stable.")

st.sidebar.divider()
st.sidebar.caption("LearnML · simple RAG tutor · v1")
