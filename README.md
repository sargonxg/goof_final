# RAG Streamlit UI – Multi-PDF Upload & Query (Flat Layout)

This repo provides a Streamlit application to:

- Upload multiple PDFs into a Vertex AI RAG Engine corpus
- Rename files based on their content
- View basic statistics about the RAG corpus
- Ask deep, complex queries using an ADK RAG agent
- See backend activity and a description of the architecture/models in use

All files are in the repository root; there are no subfolders.
Place this alongside the Google ADK RAG sample so that:

- `rag/shared_libraries/prepare_corpus_and_data.py` exists
- `rag/main.py` exists and exports `agent`.
