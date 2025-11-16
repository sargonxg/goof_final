import os
import tempfile
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

initialize_vertex_ai = None
create_or_get_corpus = None
upload_pdf_to_corpus = None
list_corpus_files = None

try:
    from rag.shared_libraries.prepare_corpus_and_data import (
        initialize_vertex_ai as _init_va,
        create_or_get_corpus as _create_corpus,
        upload_pdf_to_corpus as _upload_pdf,
        list_corpus_files as _list_files,
    )
    initialize_vertex_ai = _init_va
    create_or_get_corpus = _create_corpus
    upload_pdf_to_corpus = _upload_pdf
    list_corpus_files = _list_files
    init_error = None
except Exception as e:
    init_error = e

agent = None
agent_import_error = None
try:
    from rag.main import agent as _agent
    agent = _agent
except Exception as e:
    agent_import_error = e

REMOTE_RAG_ENDPOINT_URL = os.getenv("REMOTE_RAG_ENDPOINT_URL", "").strip()

st.set_page_config(
    page_title="Vertex RAG – Multi‑PDF Uploader & Query",
    layout="wide",
)

if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "corpus_files" not in st.session_state:
    st.session_state["corpus_files"] = []


def log(msg: str) -> None:
    st.session_state["logs"].append(msg)


def generate_display_name_from_pdf(temp_path: str, original_name: str) -> str:
    """Open the PDF, read the first page, and derive a short title."""
    try:
        reader = PdfReader(temp_path)
        if not reader.pages:
            return original_name
        text = reader.pages[0].extract_text() or ""
        text = " ".join(text.split())
        if not text:
            return original_name
        snippet = text[:120]
        snippet = snippet.replace("/", "-").replace("\\", "-").replace("\n", " ")
        return snippet + ".pdf"
    except Exception as e:
        log(f"[rename] Failed to read PDF for {original_name}: {e}")
        return original_name


def get_corpus() -> Any:
    if initialize_vertex_ai is None or create_or_get_corpus is None:
        return None
    initialize_vertex_ai()
    corpus = create_or_get_corpus()
    return corpus


def refresh_corpus_files(corpus_name: str) -> None:
    files = []
    if list_corpus_files is not None:
        try:
            files = list_corpus_files(corpus_name=corpus_name)
        except TypeError:
            files = []
        except Exception as e:
            log(f"[corpus] Error listing corpus files: {e}")
            files = []
    st.session_state["corpus_files"] = files


def display_corpus_stats(files: Any) -> None:
    st.markdown("### 📊 Corpus statistics")
    if not files:
        st.info("No file metadata available from list_corpus_files.")
        return
    if isinstance(files, list):
        st.write(f"**Total documents:** {len(files)}")
        try:
            st.dataframe(files)
        except Exception:
            st.json(files)
    else:
        st.json(files)


def query_local_agent(question: str) -> str:
    if agent is None:
        raise RuntimeError("Local ADK agent is not available.")
    result = agent.run(question)
    return str(result)


def query_remote_agent(question: str) -> str:
    import requests
    if not REMOTE_RAG_ENDPOINT_URL:
        raise RuntimeError("REMOTE_RAG_ENDPOINT_URL is not set.")
    url = REMOTE_RAG_ENDPOINT_URL.rstrip("/") + "/chat"
    resp = requests.post(url, json={"query": question}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", data)


st.title("📚 Vertex RAG – Multi‑PDF Uploader & Deep Query")

if init_error:
    st.error(
        "Could not import RAG helpers from rag.shared_libraries.prepare_corpus_and_data.py.\n\n"
        f"Error: {init_error}"
    )

tabs = st.tabs(
    [
        "📂 Upload & Corpus",
        "💬 Ask Questions",
        "🛠 Backend Activity",
        "🏗 Architecture & Models",
    ]
)

with tabs[0]:
    st.header("Upload PDFs to Vertex RAG Engine")
    if initialize_vertex_ai is None or create_or_get_corpus is None or upload_pdf_to_corpus is None:
        st.warning("RAG helpers not available. Ensure ADK RAG sample code is present.")
    else:
        corpus = get_corpus()
        if corpus is None:
            st.error("Failed to initialize or get corpus.")
        else:
            st.success(f"Active corpus: {corpus.name}")
            st.caption(
                "Files uploaded here will be added to this corpus. Display names are auto‑generated "
                "based on the first page of each PDF."
            )
            uploaded_files = st.file_uploader(
                "Upload one or more PDF documents",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if uploaded_files and st.button("Upload to RAG corpus"):
                total = len(uploaded_files)
                progress = st.progress(0)
                for idx, f in enumerate(uploaded_files, start=1):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    display_name = generate_display_name_from_pdf(tmp_path, f.name)
                    log(f"[upload] {f.name} -> {display_name} into corpus {corpus.name}")
                    try:
                        upload_pdf_to_corpus(
                            corpus_name=corpus.name,
                            pdf_path=tmp_path,
                            display_name=display_name,
                            description="Uploaded via Streamlit UI",
                        )
                    except Exception as e:
                        log(f"[upload] Error uploading {f.name}: {e}")
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    progress.progress(idx / total)
                st.success(f"Uploaded {total} files.")
                refresh_corpus_files(corpus.name)

            st.divider()
            st.subheader("Current corpus contents")
            files = st.session_state.get("corpus_files", [])
            if st.button("🔄 Refresh corpus file list"):
                refresh_corpus_files(corpus.name)
                files = st.session_state.get("corpus_files", [])
            display_corpus_stats(files)

with tabs[1]:
    st.header("Ask deep, complex questions about your documents")
    st.write(
        "Enter long, detailed queries. The agent will use retrieval‑augmented generation "
        "over your RAG corpus (via the ADK agent or a remote RAG endpoint)."
    )
    if REMOTE_RAG_ENDPOINT_URL:
        engine_choice = st.radio(
            "Which backend to use?",
            ["Local ADK agent", "Remote RAG endpoint"],
        )
    else:
        engine_choice = "Local ADK agent"
        st.info("REMOTE_RAG_ENDPOINT_URL is not set; using local ADK agent if available.")
    question = st.text_area(
        "Your question",
        height=150,
        placeholder="Ask anything that requires deep reasoning over your uploaded PDFs...",
    )
    if st.button("Run query"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Querying RAG backend…"):
                    if engine_choice == "Remote RAG endpoint" and REMOTE_RAG_ENDPOINT_URL:
                        answer = query_remote_agent(question)
                        log(f"[query/remote] {question}")
                    else:
                        if agent is None:
                            raise RuntimeError(
                                "Local ADK agent is not available and no remote endpoint is configured."
                            )
                        answer = query_local_agent(question)
                        log(f"[query/local] {question}")
                st.markdown("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error during query: {e}")
                log(f"[query/error] {e}")

with tabs[2]:
    st.header("Backend activity & logs")
    st.write(
        "This panel shows a simple in‑memory log of key events: initialization, uploads, "
        "corpus operations, and queries."
    )
    if st.button("Clear logs"):
        st.session_state["logs"] = []
    logs = st.session_state.get("logs", [])
    if logs:
        st.text_area("Log output", value="\n".join(logs), height=400)
    else:
        st.info("No log entries yet.")

with tabs[3]:
    st.header("Architecture & models in use")
    st.markdown(
        """
This app sits on top of your existing Vertex AI RAG / ADK agent infrastructure.

**Components:**
1. Streamlit UI (this app)
2. RAG helpers from `rag.shared_libraries.prepare_corpus_and_data`
3. Vertex AI RAG Engine corpus
4. ADK agent (`rag.main.agent`) using Gemini/Vertex models and RAG tools
5. Optional remote RAG endpoint (`REMOTE_RAG_ENDPOINT_URL`)

You can customize this panel to describe your exact models, corpora, and tools.
"""
    )
