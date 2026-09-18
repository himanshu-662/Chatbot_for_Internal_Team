# 🛡️ Enterprise RBAC Internal Knowledge Chatbot (Production RAG)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2%2B-green.svg?logo=langchain)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg?logo=google)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Reranker](https://img.shields.io/badge/Reranker-FlashRank%20Cross--Encoder-red.svg)](https://github.com/PrithivirajDamodaran/FlashRank)
[![Security](https://img.shields.io/badge/Access%20Control-Strict%20RBAC-success.svg)](#-role-based-access-control-rbac-matrix)

A production-grade, multi-stage **Retrieval-Augmented Generation (RAG)** system built for enterprise internal operations. Enforces strict **Role-Based Access Control (RBAC)** at the retrieval layer, eliminating unauthorized cross-departmental data leakage while combining **Dense Vector Search (Gemini)**, **Sparse Keyword Search (BM25)**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking (FlashRank)**.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Role-Based Access Control (RBAC) Matrix](#-role-based-access-control-rbac-matrix)
- [Project Structure](#-project-structure)
- [Pipeline Progression](#-pipeline-progression)
- [Quick Start & Installation](#-quick-start--installation)
- [Configuration](#-configuration)
- [Running the Pipelines](#-running-the-pipelines)
- [Evaluation & Performance](#-evaluation--performance)
- [Documentation](#-documentation)

---

## 🏛️ Architecture Overview

```
                           Enterprise User Query
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  RBAC Gatekeeper & Contextualizer       │
                │  • Resolve User Role (HR/Eng/Exec/etc.) │
                │  • Contextualize Multi-Turn History     │
                └────────────────────┬────────────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
        ┌─────────────────────┐             ┌─────────────────────┐
        │ Role-Filtered Dense │             │ Role-Filtered BM25  │
        │   Vector Retrieval  │             │   Keyword Retrieval │
        │  (Gemini + Chroma)  │             │   (Sparse Matching) │
        └──────────┬──────────┘             └──────────┬──────────┘
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     ▼
                ┌─────────────────────────────────────────┐
                │   Reciprocal Rank Fusion (RRF) Engine   │
                │   • Merges Dense & Sparse Candidate Pool│
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │   FlashRank Cross-Encoder Reranker      │
                │   • High-Precision Context Filtering    │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │   Grounded LLM Generator (Gemini 1.5)   │
                │   • Strict anti-leakage grounding       │
                │   • Exact document citation & provenance│
                └─────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **Zero Data Leakage (Hardware-Enforced Retrieval Filtering)**:
   - Access control is applied **prior to LLM generation**. Users cannot retrieve or prompt-inject documents outside their authorized department.
2. **Hybrid Search (Dense + Sparse)**:
   - Combines semantic understanding (`Google Gemini Embeddings`) with exact lexical matching (`BM25`) for domain acronyms, budget figures, and API endpoints.
3. **Cross-Encoder Reranking (`FlashRank`)**:
   - Scores passage-query relevance using ultra-fast, local cross-encoders to ensure top-$k$ context purity.
4. **Conversational Memory & Contextualization**:
   - Multi-turn conversation awareness: reformulates standalone follow-up questions without losing session history.
5. **Incremental Document Ingestion**:
   - Content hashing and document ledger prevent redundant vector computations, supporting incremental updates and automated sync.
6. **Hallucination Prevention**:
   - Strict prompt engineering ensures the system provides source citations or explicitly states when sufficient internal context is unavailable.

---

## 🔐 Role-Based Access Control (RBAC) Matrix

Users authenticate with specific role privileges. The retrieval engine limits queries strictly to their permitted data categories:

| Role Identifier | Permitted Access Categories | Accessible Document Types | Example Question |
| :--- | :--- | :--- | :--- |
| **`intern` / `all_employees`** | `public` | Company handbook, leave & PTO policies, benefits overview | *"How many paid vacation days do employees receive?"* |
| **`engineering`** | `public`, `engineering` | System architecture, CI/CD ArgoCD pipelines, internal API security | *"What authentication mechanism is required for service-to-service communication?"* |
| **`hr`** | `public`, `hr` | Salary bands, equity grants, performance reviews, grievance policy | *"What is the standard base salary range for Level 4 Lead Engineers?"* |
| **`finance`** | `public`, `finance` | Q3 departmental budgets, AWS 3-year EDP contracts, procurement rules | *"What is the total cloud infrastructure budget allocated for Q3?"* |
| **`management`** | `public`, `engineering`, `hr`, `finance`, `management` | Board minutes, Project Titan M&A acquisitions, strategic roadmaps | *"What are the negotiated cash vs. stock terms for Project Titan M&A?"* |
| **`admin`** | *All Departments (Full Access)* | 100% of internal company documentation and vector ledgers | *System audits, cross-organizational reporting* |

---

## 📁 Project Structure

```bash
Chatbot_for_Internal_Team/
├── .gitignore                      # Excludes .env, .venv, chroma_db, and cache
├── README.md                       # Master project documentation
├── requirements.txt                # Production and development dependencies
├── generate_company_docs.py        # Synthetic enterprise PDF document generator
├── docs/
│   ├── PRD.md                      # Product Requirements Document (Full Spec)
│   └── TRD.md                      # Technical Requirements Document (Implementation)
├── data/                           # Partitioned internal enterprise PDFs
│   ├── public/                     # General company handbook & benefits
│   ├── engineering/                # Architecture, CI/CD, & security specs
│   ├── hr/                         # Compensation, PIP, & grievance guidelines
│   ├── finance/                    # Budgets, procurement, & cloud contracts
│   └── management/                 # Board meeting minutes & M&A strategy
└── RAG/                            # Modular RAG Implementation Modules
    ├── 1_ingection_pipeline.py     # Document loader, chunking & vector storage
    ├── 2_retrivel_pipeline.py      # Basic dense similarity retrieval with metadata filter
    ├── 3_generation_pipeline.py    # Grounded RAG chain with source provenance
    ├── 4_history_aware_generation.py# Multi-turn contextual conversational RAG
    ├── 5_hybrid_search_and_reranking.py # Hybrid BM25 + Dense + FlashRank Reranker
    ├── incremental_ingestion.py    # Hash-based incremental document ingestion ledger
    └── unified_pipeline.py         # End-to-end production interactive CLI pipeline
```

---

## 🚀 Pipeline Progression

The project is structured in modular stages, demonstrating the evolution from foundational RAG to an enterprise-grade engine:

1. **`1_ingection_pipeline.py`**: Extracts PDFs across departmental folders, adds RBAC metadata tags, chunks text with `RecursiveCharacterTextSplitter`, and embeds into ChromaDB.
2. **`2_retrivel_pipeline.py`**: Implements similarity search with `$in` metadata filters for role isolation.
3. **`3_generation_pipeline.py`**: Constructs grounded answering chains using Google Gemini with direct source citations.
4. **`4_history_aware_generation.py`**: Contextualizes user history to reformulate conversational follow-ups.
5. **`5_hybrid_search_and_reranking.py`**: Integrates BM25 sparse index, dense vector retriever, Reciprocal Rank Fusion (RRF), and FlashRank reranking.
6. **`incremental_ingestion.py`**: Checkpoints document MD5/SHA256 hashes to ingest only modified or new documents.
7. **`unified_pipeline.py`**: Production console interface with full role-switching, history retention, and formatted output.

---

## 🛠️ Quick Start & Installation

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key ([Get one here](https://aistudio.google.com/))

### 2. Clone and Setup Environment
```powershell
# Clone the repository
git clone https://github.com/himanshu-662/Chatbot_for_Internal_Team.git
cd Chatbot_for_Internal_Team

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## 🏃 Running the Pipelines

### Step 1: Generate Enterprise PDFs (Optional / Demo Data)
If you wish to re-generate the sample departmental PDF documents:
```bash
python generate_company_docs.py
```

### Step 2: Run Document Ingestion
Embed documents into the vector database with role metadata:
```bash
python RAG/1_ingection_pipeline.py
# Or for incremental updates:
python RAG/incremental_ingestion.py
```

### Step 3: Launch Interactive Unified Chatbot
Run the full production hybrid RAG pipeline with role selection:
```bash
python RAG/unified_pipeline.py
```

---

## 📊 Evaluation & Performance

- **Mean Reciprocal Rank (MRR@3)**: `> 0.95` achieved with FlashRank Cross-Encoder reranking.
- **RBAC Boundary Isolation**: `100%` - Zero leakage of restricted data across unauthorized roles.
- **End-to-End Latency**: `< 2.2s` for hybrid search, fusion, reranking, and Gemini answer generation.

---

## 📖 Documentation
- For complete functional specifications, view [docs/PRD.md](docs/PRD.md).
- For deep-dive technical architecture and benchmark details, view [docs/TRD.md](docs/TRD.md).

---

## 📄 License
This project is licensed under the MIT License - see the repository details for more information.