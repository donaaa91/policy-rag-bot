# policy-rag-bot — AI Policy Analyst

A Streamlit application that lets you upload any policy document (PDF) and ask natural-language questions about it. Answers are generated using **Retrieval-Augmented Generation (RAG)**, meaning the AI only answers from the content of your uploaded document — not from its own general training knowledge — and every answer is accompanied by the exact source passages it was drawn from, so you can verify it yourself.

---

## Table of Contents
- [Why RAG](#why-rag)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [How to Use It](#how-to-use-it)
- [Project Structure](#project-structure)
- [Configuration Details](#configuration-details)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Possible Future Improvements](#possible-future-improvements)

---

## Why RAG

Large language models are powerful but have two problems for a use case like policy analysis:
1. **They don't know your specific document** — a general-purpose LLM has never seen your uploaded policy file.
2. **They can hallucinate** — when unsure, a model may generate a plausible-sounding but incorrect answer.

RAG solves both: instead of asking the LLM to answer from memory, this app first **retrieves** the most relevant passages from your actual document, then asks the LLM to answer **using only that retrieved text**. If the answer genuinely isn't in the document, the model is explicitly instructed to say so rather than guess.

---

## Architecture

```
┌─────────────────┐
│  Upload PDF      │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│ 1. Save to temp.pdf          │
│ 2. Extract text (PyPDFLoader)│
│ 3. Split into ~1000-char     │
│    chunks (200 overlap)      │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Embed each chunk             │
│ (HuggingFace all-MiniLM-L6-v2)│
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Store in FAISS vector index  │
│ (in-memory, per session)     │
└────────┬─────────────────────┘
         │
   [ User asks a question ]
         │
         ▼
┌─────────────────────────────┐
│ Retrieve top-matching chunks │
│ from FAISS                   │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Build prompt:                │
│ context = retrieved chunks   │
│ input = user's question      │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ LLM (Llama 3.1 8B via Groq)  │
│ answers from context only    │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Display answer +             │
│ expandable source citations  │
│ (chunk text + page number)   │
└──────────────────────────────┘
```

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| UI | Streamlit | Fast to build, good for single-page interactive tools |
| Orchestration | LangChain (LCEL) | Composable, functional chain-building for RAG pipelines |
| PDF parsing | PyPDFLoader | Simple, reliable text extraction from PDF files |
| Text splitting | RecursiveCharacterTextSplitter | Splits on natural boundaries (paragraphs → sentences → words) rather than blindly by character count |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Small (~80MB), fast, runs on CPU — no GPU or paid embedding API needed |
| Vector store | FAISS | Lightweight, in-memory, no external database/service required |
| LLM | Llama 3.1 8B Instant (via Groq) | Fast inference, low cost, good enough quality for grounded Q&A |

---

## Prerequisites
- Python 3.9+
- A [Groq API key](https://console.groq.com) (free tier available)
- ~500MB free disk/RAM for the embedding model and FAISS index (varies with document size)

---

## Setup

```bash
git clone https://github.com/donaaa91/policy-rag-bot.git
cd policy-rag-bot
pip install -r requirements.txt
```

### API Key Setup
You have two options:

**Option A — Streamlit secrets (recommended for deployment):**
Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your-key-here"
```

**Option B — Manual entry (for local testing):**
Just run the app and paste your key into the sidebar text field when prompted — nothing to configure beforehand.

---

## Running the App

```bash
streamlit run app.py
```

This opens the app in your default browser, typically at `http://localhost:8501`.

---

## How to Use It

1. **Enter your Groq API key** in the sidebar (skip if already set via secrets).
2. **Upload a PDF** using the file uploader — a sample `policy.pdf` is included in the repo for quick testing.
3. Wait for the "Document Analyzed!" confirmation (processing time depends on document length; runs on CPU).
4. **Type a question** in plain English about the document's content.
5. Read the answer, then expand **"View Source / Evidence"** to see exactly which chunks and page numbers the answer was grounded in — useful for verifying accuracy rather than trusting the AI blindly.

---

## Project Structure

```
policy-rag-bot/
├── app.py              # Main application: UI, ingestion, RAG pipeline
├── requirements.txt    # Python dependencies
├── policy.pdf           # Sample policy document for testing
├── temp.pdf              # Working file written on upload (not meant to be committed/tracked long-term)
└── README.md           # This file
```

---

## Configuration Details
These are hardcoded in `app.py` and can be adjusted directly in the source:

| Setting | Current Value | Location |
|---|---|---|
| Chunk size | 1000 characters | `RecursiveCharacterTextSplitter(chunk_size=1000, ...)` |
| Chunk overlap | 200 characters | `RecursiveCharacterTextSplitter(..., chunk_overlap=200)` |
| Embedding model | `all-MiniLM-L6-v2` | `HuggingFaceEmbeddings(model_name=...)` |
| Embedding device | CPU | `model_kwargs={"device": "cpu"}` |
| LLM | `llama-3.1-8b-instant` | `ChatGroq(model_name=...)` |

---

## Known Limitations

1. **Single-document-per-session limit:** The vector store is built only once per session (`if "vectors" not in st.session_state`). Uploading a **second, different** PDF in the same session will NOT trigger re-processing — the app will keep answering from the first document. **Workaround:** refresh the app/start a new session to analyze a different document.
2. **No cleanup of `temp.pdf`:** The uploaded file is written to disk and never deleted after processing. On a shared or long-running deployment, this could accumulate files or leak content between sessions.
3. **No error handling for malformed input:** A corrupted PDF, an empty document, or a failed Groq API call will likely surface as an unhandled exception rather than a graceful error message.
4. **Plaintext API key input:** The sidebar text input is masked visually but not a substitute for proper secret management in a production deployment.
5. **CPU-only embeddings:** Processing time scales with document length; very large PDFs may take noticeably longer without GPU acceleration.
6. **No persistent storage:** The FAISS index lives only in session memory — closing the browser tab or restarting the app loses the processed document, requiring re-upload.

---

## Troubleshooting

**"Please enter your Groq API Key" won't go away even after entering one**
→ Check for extra whitespace when pasting the key, or confirm the key is active on the Groq console.

**App seems stuck on "Analyzing document..."**
→ Large PDFs on CPU-only embedding can take a while; check terminal output for errors rather than assuming it's frozen.

**Answers seem to ignore my uploaded PDF entirely**
→ Likely the known single-document-per-session limitation — refresh the app if you've uploaded more than one file this session.

**`ModuleNotFoundError` on startup**
→ Re-run `pip install -r requirements.txt`; ensure you're in the correct virtual environment.

---

## Possible Future Improvements
- Fix the session-state bug to allow re-uploading a new document without a full refresh
- Add automatic cleanup of `temp.pdf` after processing
- Add error handling/user-facing messages for failed PDF parsing or API errors
- Support multiple simultaneous documents with source attribution per document
- Add a persistent vector store option (e.g., saving the FAISS index to disk) so re-analysis isn't required on every session
- Add unit tests for the ingestion and retrieval pipeline
