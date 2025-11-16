"""Local fallback implementation for RAG helper utilities.

This module mirrors the interface provided by
``rag.shared_libraries.prepare_corpus_and_data`` from the Vertex AI
reference samples so that the Streamlit UI remains functional in
environments where the full ADK sample code is not available.

The implementation is intentionally simple: uploaded PDFs are converted to
plain text and stored on disk, while metadata is recorded in a JSON file.
Queries perform a lightweight keyword match and return an extract from the
most relevant document.  This keeps the demo usable without requiring
access to Vertex AI resources or the ADK agent runtime.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader

DATA_DIR = Path(os.getenv("LOCAL_RAG_STORAGE", "./.local_rag_store")).resolve()
DEFAULT_CORPUS_NAME = os.getenv("LOCAL_CORPUS_NAME", "local-corpus")
MAX_SNIPPET_CHARS = 600


@dataclass
class LocalCorpus:
    """Tiny stand-in for the Vertex AI Corpus object."""

    name: str


def initialize_vertex_ai() -> None:
    """Create the data directory used by the local corpus implementation."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _corpus_dir(corpus_name: str) -> Path:
    return DATA_DIR / corpus_name


def _metadata_path(corpus_name: str) -> Path:
    return _corpus_dir(corpus_name) / "metadata.json"


def _load_metadata(corpus_name: str) -> List[Dict[str, Any]]:
    meta_path = _metadata_path(corpus_name)
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_metadata(corpus_name: str, entries: List[Dict[str, Any]]) -> None:
    meta_path = _metadata_path(corpus_name)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def create_or_get_corpus() -> LocalCorpus:
    corpus_name = DEFAULT_CORPUS_NAME
    _corpus_dir(corpus_name).mkdir(parents=True, exist_ok=True)
    return LocalCorpus(name=corpus_name)


def _extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    contents: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        contents.append(text)
    raw_text = "\n".join(contents).strip()
    return raw_text


def upload_pdf_to_corpus(
    *,
    corpus_name: str,
    pdf_path: str,
    display_name: str,
    description: str = "",
) -> Dict[str, Any]:
    """Persist extracted text and metadata for a PDF."""
    corpus_directory = _corpus_dir(corpus_name)
    corpus_directory.mkdir(parents=True, exist_ok=True)

    text = _extract_pdf_text(pdf_path)
    safe_name = Path(display_name).with_suffix(".txt").name
    text_path = corpus_directory / safe_name
    text_path.write_text(text, encoding="utf-8")

    metadata = _load_metadata(corpus_name)
    entry = {
        "display_name": display_name,
        "description": description,
        "text_path": safe_name,
        "characters": len(text),
    }
    metadata.append(entry)
    _save_metadata(corpus_name, metadata)
    return entry


def list_corpus_files(*, corpus_name: str) -> List[Dict[str, Any]]:
    return _load_metadata(corpus_name)


def _load_corpus_texts(corpus_name: str) -> List[Dict[str, Any]]:
    metadata = _load_metadata(corpus_name)
    corpus_directory = _corpus_dir(corpus_name)
    documents: List[Dict[str, Any]] = []
    for entry in metadata:
        text_path = corpus_directory / entry.get("text_path", "")
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        documents.append({"text": text, **entry})
    return documents


class LocalRagAgent:
    """Minimal RAG-like agent that performs keyword matching over the corpus."""

    def __init__(self, corpus_name: str | None = None) -> None:
        self.corpus_name = corpus_name or DEFAULT_CORPUS_NAME

    def run(self, query: str) -> str:
        documents = _load_corpus_texts(self.corpus_name)
        if not documents:
            return (
                "No documents are available in the local corpus. Upload PDFs first "
                "and try again."
            )
        if not query.strip():
            return "Please provide a non-empty question."

        query_terms = {token for token in query.lower().split() if len(token) > 2}

        def _score(doc: Dict[str, Any]) -> int:
            text = doc["text"].lower()
            return sum(text.count(term) for term in query_terms)

        best_doc = max(documents, key=_score)
        snippet = best_doc["text"][:MAX_SNIPPET_CHARS]
        metadata_str = best_doc.get("display_name", "Unnamed document")
        return (
            f"Top match: {metadata_str}\n\n"
            f"Snippet:\n{snippet}\n\n"
            "(Local keyword search – add the official ADK agent for full RAG capabilities.)"
        )


def build_local_agent() -> LocalRagAgent:
    initialize_vertex_ai()
    corpus = create_or_get_corpus()
    return LocalRagAgent(corpus_name=corpus.name)
