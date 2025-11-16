import os
import tempfile
import time
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except Exception as import_error:  # pragma: no cover - surfaced in UI
    genai = None
    types = None
    GENAI_IMPORT_ERROR = import_error
else:
    GENAI_IMPORT_ERROR = None

MAX_PDFS = 10
DEFAULT_MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title="Gemini File Search RAG",
    layout="wide",
)

if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "store_name" not in st.session_state:
    st.session_state["store_name"] = ""
if "store_display_name" not in st.session_state:
    st.session_state["store_display_name"] = ""
if "upload_history" not in st.session_state:
    st.session_state["upload_history"] = []
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []


def log(message: str) -> None:
    """Persist a message into the in-memory log and surface in the UI."""
    st.session_state["logs"].append(message)


def get_client(api_key: str) -> "genai.Client":
    if not genai:
        raise RuntimeError(
            "google-genai is not installed. Add it to requirements.txt and pip install the package."
        )
    if not api_key:
        raise ValueError("Provide a Gemini API key to continue.")
    cached_key = st.session_state.get("_genai_cached_key")
    if cached_key == api_key and st.session_state.get("_genai_client") is not None:
        return st.session_state["_genai_client"]
    client = genai.Client(api_key=api_key)
    st.session_state["_genai_cached_key"] = api_key
    st.session_state["_genai_client"] = client
    log("[client] Initialized Gemini client")
    return client


def wait_for_operation(client: "genai.Client", operation: any, poll_seconds: float = 2.0) -> any:
    """Poll the long-running operation until it reports completion."""
    current_op = operation
    while getattr(current_op, "done", True) is False:
        time.sleep(poll_seconds)
        if hasattr(current_op, "name"):
            current_op = client.operations.get(name=current_op.name)
        else:
            current_op = client.operations.get(current_op)
    return current_op


def create_file_search_store(client: "genai.Client", display_name: str) -> str:
    config: Dict[str, str] = {}
    if display_name:
        config["display_name"] = display_name
    store = client.file_search_stores.create(config=config or None)
    st.session_state["store_name"] = store.name
    st.session_state["store_display_name"] = getattr(store, "display_name", display_name)
    log(f"[store] Created file search store {store.name}")
    return store.name


def upload_pdfs(
    client: "genai.Client",
    store_name: str,
    uploaded_files: List["st.runtime.uploaded_file_manager.UploadedFile"],
) -> None:
    if not store_name:
        raise ValueError("Select or create a File Search store first.")
    if len(uploaded_files) > MAX_PDFS:
        raise ValueError(f"You can upload at most {MAX_PDFS} PDFs per batch.")

    progress = st.progress(0.0)
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        log(f"[upload] Sending {uploaded_file.name} to {store_name}")
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name,
                file=tmp_path,
                config={"display_name": uploaded_file.name},
            )
            wait_for_operation(client, operation)
            st.session_state["upload_history"].insert(
                0,
                {
                    "file": uploaded_file.name,
                    "store": store_name,
                    "size_kb": round(uploaded_file.size / 1024, 2),
                },
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        progress.progress(index / len(uploaded_files))
    log(f"[upload] Uploaded {len(uploaded_files)} files")


def run_file_search_query(
    client: "genai.Client", store_name: str, question: str, model_name: str
) -> str:
    if types is None:
        raise RuntimeError("google-genai types unavailable – reinstall google-genai.")
    if not store_name:
        raise ValueError("No File Search store selected.")
    if not question.strip():
        raise ValueError("Enter a question to run File Search.")
    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store_name],
                )
            )
        ]
    )
    response = client.models.generate_content(
        model=model_name or DEFAULT_MODEL,
        contents=question,
        config=config,
    )
    grounding = []
    if response.candidates:
        meta = response.candidates[0].grounding_metadata
        if meta and getattr(meta, "sources", None):
            for source in meta.sources:
                doc = source.get("title") or source.get("uri") or "document"
                grounding.append(doc)
    answer_text = response.text
    st.session_state["conversation"].insert(
        0,
        {
            "question": question,
            "answer": answer_text,
            "citations": grounding,
        },
    )
    return answer_text


st.title("📚 Gemini File Search – Multi PDF RAG")
st.caption(
    "Upload up to 10 PDFs, let Gemini File Search index them, and instantly run retrieval‑augmented questions."
)

if GENAI_IMPORT_ERROR:
    st.error(
        "google-genai could not be imported. Install it and restart the app.\n\n"
        f"Details: {GENAI_IMPORT_ERROR}"
    )

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        value=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        type="password",
        help="Create one at https://aistudio.google.com/app/apikey",
    )
    st.session_state["store_name"] = st.text_input(
        "Active File Search store name",
        value=st.session_state.get("store_name", ""),
        help="Paste an existing store name or create a new one below.",
    )
    with st.form("create_store_form"):
        st.write("Create a new File Search store")
        new_display_name = st.text_input(
            "Store display name",
            value=st.session_state.get("store_display_name", ""),
        )
        submitted = st.form_submit_button("Create store", use_container_width=True)
        if submitted:
            try:
                client = get_client(api_key)
                store_name = create_file_search_store(client, new_display_name)
                st.success(f"Created {store_name}")
            except Exception as exc:
                st.error(f"Unable to create store: {exc}")

upload_tab, query_tab, activity_tab = st.tabs([
    "1️⃣ Upload PDFs",
    "2️⃣ Ask questions",
    "3️⃣ Activity & citations",
])

with upload_tab:
    st.subheader("Upload and index PDFs")
    st.info(
        "Gemini File Search stores chunks, embeds, and indexes each document. Limit: 100 MB per PDF and 10 PDFs per upload batch."
    )
    uploaded_files = st.file_uploader(
        "Drop your PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select up to 10 PDFs at a time.",
    )
    if uploaded_files:
        if len(uploaded_files) > MAX_PDFS:
            st.error(f"You selected {len(uploaded_files)} files – limit is {MAX_PDFS} per batch.")
        if st.button("Upload to File Search", use_container_width=True):
            try:
                client = get_client(api_key)
                with st.spinner("Uploading and indexing PDFs…"):
                    upload_pdfs(client, st.session_state.get("store_name", ""), uploaded_files)
                st.success("Upload complete. You can now query these documents.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
                log(f"[error] upload failed {exc}")

    if st.session_state.get("upload_history"):
        st.markdown("### Recent uploads")
        st.dataframe(st.session_state["upload_history"][:20])

with query_tab:
    st.subheader("Ask Gemini with File Search context")
    model_choice = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Both models support File Search."
    )
    question = st.text_area(
        "Your question",
        height=180,
        placeholder="Ex: Summarize the risk sections across these PDFs…",
    )
    if st.button("Run File Search query", use_container_width=True):
        try:
            client = get_client(api_key)
            with st.spinner("Grounding response with your PDFs…"):
                answer = run_file_search_query(
                    client,
                    st.session_state.get("store_name", ""),
                    question,
                    model_choice,
                )
            st.markdown("### Answer")
            st.write(answer)
            if st.session_state["conversation"][0]["citations"]:
                st.caption(
                    "Citations: " + ", ".join(st.session_state["conversation"][0]["citations"])
                )
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            log(f"[error] query failed {exc}")

    if st.session_state.get("conversation"):
        st.markdown("### Conversation history")
        for turn in st.session_state["conversation"][:5]:
            with st.expander(turn["question"][:80] + ("…" if len(turn["question"]) > 80 else "")):
                st.markdown(f"**Question**: {turn['question']}")
                st.markdown(f"**Answer**: {turn['answer']}")
                if turn["citations"]:
                    st.caption("Citations: " + ", ".join(turn["citations"]))

with activity_tab:
    st.subheader("Activity log")
    st.caption("All significant actions recorded for easy debugging.")
    if st.button("Clear logs"):
        st.session_state["logs"] = []
    if st.session_state["logs"]:
        st.text_area("Logs", value="\n".join(st.session_state["logs"]), height=300)
    else:
        st.info("No logs yet – upload a PDF or run a query to get started.")

    if st.session_state.get("conversation"):
        st.markdown("### Latest citations")
        latest = st.session_state["conversation"][0]
        if latest["citations"]:
            st.write(latest["citations"])
        else:
            st.write("No citations returned yet.")
