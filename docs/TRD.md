# Technical Requirements Document (TRD)

## Project: Enterprise Role-Based Access Control (RBAC) Internal Knowledge Chatbot
**Document Version:** 1.0.0  
**Status:** Implemented / Active  
**System Architecture:** Hybrid Dense-Sparse RAG with Cross-Encoder Reranking & RBAC Enforcement  
**Core Frameworks:** LangChain, ChromaDB, Google Gemini API, FlashRank, BM25  

---

## 1. System Architecture Overview

The system implements a production-grade, 5-stage modular Retrieval-Augmented Generation (RAG) pipeline designed for corporate knowledge bases. It features pre-retrieval role-based access filtering, dense vector embeddings, sparse keyword indexing, reciprocal rank fusion (RRF), cross-encoder re-ranking, and context-grounded conversational LLM generation.

```
                                  +---------------------------------------+
                                  |     Source Corporate Documents        |
                                  |  (data/{public,eng,hr,fin,mgmt}/*.pdf)|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       Document Loader & Ingestion     |
                                  |   - DirectoryLoader (PDF, DOCX, TXT)  |
                                  |   - Auto-extracts 'access_level'      |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   RecursiveCharacterTextSplitter      |
                                  |   (chunk_size=500, chunk_overlap=50)  |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     Google Gemini Embedding Model     |
                                  |      (models/gemini-embedding-2)      |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |        Persistent ChromaDB Store      |
                                  |       (HNSW Cosine Space Index)       |
                                  +-------------------+-------------------+
                                                      |
======================================================|======================================================
                                     RUNTIME QUERY EXECUTION FLOW
======================================================|======================================================
                                                      |
             +--------------------+                   |
             |  User Query (CLI)  |                   |
             +---------+----------+                   |
                       |                              |
                       v                              |
    +------------------------------------+            |
    | Query Contextualizer (Gemini Flash)|            |
    | - Resolves pronoun & chat history  |            |
    +------------------+-----------------+            |
                       |                              |
                       v                              |
    +------------------------------------+            |
    | RBAC Gatekeeper & Role Filter Gen  |            |
    | e.g. {"access_level": {"$in": ...}}|            |
    +------------------+-----------------+            |
                       |                              |
         +-------------+-------------+                |
         |                           |                |
         v                           v                |
+------------------+        +------------------+      |
| Role-Filtered    |        | Role-Filtered    |      |
| BM25 Retriever   |        | Chroma Retriever |<-----+
| (Keyword Match)  |        | (Dense Vectors)  |
+--------+---------+        +--------+---------+
         |                           |
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         |   Reciprocal Rank Fusion  |
         |         (RRF k=60)        |
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         |    FlashRank Re-ranker    |
         | (ms-marco-TinyBERT-L-2-v2)|
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         | Context Formatting Engine |
         |   [DEPT - FILENAME] tags  |
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         | Gemini Generative LLM     |
         | (gemini-flash-lite-latest)|
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         | Context-Grounded Response |
         +---------------------------+
```

---

## 2. Technology Stack & Dependencies

| Layer / Component | Technology / Library | Version / Model Specification | Purpose |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | `>= 3.10` | Core application logic and async orchestration. |
| **LLM Orchestration** | `langchain`, `langchain-community`, `langchain-core` | `^0.2.x` / `^0.3.x` | Pipeline abstractions, message history, document loaders. |
| **Generative LLM** | Google Gemini (`ChatGoogleGenerativeAI`) | `gemini-flash-lite-latest` | Fast, low-latency, context-grounded response synthesis. |
| **Vector Embeddings** | Google Gemini (`GoogleGenerativeAIEmbeddings`) | `models/gemini-embedding-2` | 768-dimensional dense semantic text representations. |
| **Vector Database** | `langchain-chroma` (`ChromaDB`) | Embedded local storage | On-disk persistent vector store with HNSW index and metadata filtering. |
| **Keyword Retriever** | `rank_bm25` / `BM25Retriever` | In-memory token indexer | Exact keyword and technical acronym matching. |
| **Cross-Encoder Reranker**| `flashrank` | `ms-marco-TinyBERT-L-2-v2` | Fast CPU-based neural passage reranking. |
| **Document Parsers** | `pypdf`, `docx2txt` | Latest | Binary parsing for enterprise PDF and Word documents. |
| **Document Generator** | `reportlab` | Latest | PDF generation utility for mocking corporate documentation. |
| **Config Management** | `python-dotenv` | Latest | Environment variable resolution (`GEMINI_API_KEY`). |

---

## 3. Data Ingestion & Preprocessing Specification (`1_ingection_pipeline.py`)

### 3.1 Directory-Based Classification Mapping
The ingestion pipeline dynamically derives document sensitivity and departmental boundaries from the source directory layout inside `data/`:

```
data/
├── public/          -> access_level: "public"
├── engineering/     -> access_level: "engineering"
├── hr/              -> access_level: "hr"
├── finance/         -> access_level: "finance"
└── management/      -> access_level: "management"
```

### 3.2 Metadata Schema
Every chunk indexed into the vector store contains the following strongly typed metadata attributes:

```json
{
  "source": "d:\\PROJECTS\\RAG-Projects\\Chatbot_for_Internal_Team\\data\\hr\\salary_bands_and_compensation.pdf",
  "filename": "salary_bands_and_compensation.pdf",
  "department": "hr",
  "access_level": "hr",
  "page": 0
}
```

### 3.3 Text Splitting Configuration
- **Splitter Class**: `RecursiveCharacterTextSplitter`
- **Chunk Size**: `500` characters
- **Chunk Overlap**: `50` characters (10% sliding window)
- **Separators**: `["\n\n", "\n", " ", ""]` (preserves paragraph and sentence boundaries)
- **Empty Chunk Filtering**: Chunks with `chunk.page_content.strip() == ""` are dropped before embedding.

### 3.4 Rate Limiting & Batching Strategy
To prevent Google Gemini API HTTP 429 quota exceptions during bulk embedding:
- **Batch Size**: `5` chunks per API request.
- **Throttling Pause**: `10` seconds sleep between successive batches.
- **Index Metric**: `collection_metadata={"hnsw:space": "cosine"}`.

---

## 4. Security & Role-Based Access Control (RBAC) Architecture

### 4.1 Permission Mapping Matrix

```python
ROLE_PERMISSIONS = {
    "all_employees": ["public"],
    "intern": ["public"],
    "engineering": ["public", "engineering"],
    "hr": ["public", "hr"],
    "finance": ["public", "finance"],
    "management": ["public", "engineering", "hr", "finance", "management"],
    "admin": ["public", "engineering", "hr", "finance", "management"]
}
```

### 4.2 Database Pre-Filtering Mechanics
To guarantee absolute security isolation, document filtering occurs at the database retrieval level:
1. **Single Category Filter**:
   ```python
   search_filter = {"access_level": allowed_levels[0]}
   ```
2. **Multi-Category Filter**:
   ```python
   search_filter = {"access_level": {"$in": allowed_levels}}
   ```
3. **Admin Bypass**:
   Users with role `admin`, `superuser`, or `root` bypass metadata filtering and search across 100% of indexed vectors.

---

## 5. Mathematical Formulations & Search Algorithms (`5_hybrid_search_and_reranking.py`)

### 5.1 Dense Semantic Vector Search
Calculates the Cosine Similarity between the embedded query vector $\mathbf{q}$ and document chunk vector $\mathbf{d}$:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

### 5.2 Sparse Keyword Search (Okapi BM25)
Calculates term frequency-inverse document frequency relevance:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^{N} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$ is term frequency in document $D$.
- $|D|$ and $\text{avgdl}$ are document length and average document length.
- Default parameters: $k_1 = 1.5, b = 0.75$.

### 5.3 Reciprocal Rank Fusion (RRF)
Combines dense and sparse ranked lists without requiring score normalization:

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
- $M = \{\text{Dense Retriever}, \text{Sparse BM25 Retriever}\}$.
- $r_m(d)$ is the rank position of document $d$ in retriever $m$ (1-indexed).
- $k = 60$ (smoothing constant preventing high-rank domination).
- Top $N = 8$ fused candidate chunks are passed to the reranker.

### 5.4 Cross-Encoder Neural Reranking
Passes pairs of `(Query, Passage_i)` through the `ms-marco-TinyBERT-L-2-v2` cross-encoder network to calculate cross-attention scores:

$$s_i = \text{CrossEncoder}(\text{Query}, \text{Passage}_i)$$

The final top $k=3$ passages with highest $s_i$ are injected into the LLM context.

---

## 6. Pipeline Code Modules & Specifications

### 6.1 `RAG/1_ingection_pipeline.py`
- **Responsibility**: Ingests files, parses PDFs/Docs, attaches RBAC metadata, splits text, computes Gemini embeddings, and persists ChromaDB.
- **Key Functions**:
  - `load_documents(docs_path)`: Scans filesystem and loads multi-format documents.
  - `split_documents(documents, chunk_size, chunk_overlap)`: Recursive character chunker.
  - `create_vector_store(chunks, persist_directory, reset_db)`: Batch-embeds and writes to ChromaDB.

### 6.2 `RAG/2_retrivel_pipeline.py`
- **Responsibility**: Implements standalone RBAC vector retrieval and automated test suite.
- **Key Functions**:
  - `get_allowed_access_levels(role)`: Maps role string to permitted category array.
  - `build_retriever_for_role(role, k)`: Creates Chroma retriever with `$in` filter.
  - `run_rbac_test_suite()`: Executes 7 role-permission boundary tests.

### 6.3 `RAG/3_generation_pipeline.py`
- **Responsibility**: Single-turn context-grounded Q&A with anti-hallucination prompt guardrails.
- **Key Functions**:
  - `generate_answer(query, role, k)`: Retrieves authorized chunks, formats context, invokes `gemini-flash-lite-latest`.
  - `run_rbac_generation_tests()`: Automated verification of generation boundaries.

### 6.4 `RAG/4_history_aware_generation.py`
- **Responsibility**: Multi-turn conversational session with history-aware question reformulation and CLI session manager.
- **Key Functions**:
  - `contextualize_question(query, chat_history)`: Reformulates ambiguous follow-up questions into standalone queries.
  - `generate_answer(query, relevant_documents, chat_history, role)`: Synthesizes responses using full conversation context.
  - `main()`: Interactive REPL supporting `/role`, `/whoami`, `/clear`, `/help`, and `exit`.

### 6.5 `RAG/5_hybrid_search_and_reranking.py`
- **Responsibility**: Standalone reference pipeline combining BM25 keyword search, Chroma dense vector search, RRF fusion, and FlashRank cross-encoder reranking.
- **Key Functions**:
  - `build_bm25_retriever(role, k)`: In-memory sparse retriever filtered by role.
  - `build_semantic_retriever(role, k)`: Dense Chroma vector retriever filtered by role.
  - `reciprocal_rank_fusion(dense_docs, sparse_docs, rrf_k, top_n)`: Merges multi-retriever candidate lists.
  - `rerank_documents(query, candidate_docs, top_k)`: FlashRank TinyBERT neural scoring.
  - `generate_hybrid_answer(query, role, final_k, verbose)`: End-to-end hybrid Q&A pipeline.

### 6.6 `RAG/incremental_ingestion.py` (Production Ingestion Engine)
- **Responsibility**: Event-driven incremental sync comparing filesystem SHA-256 hashes against `chroma_db/ingestion_manifest.json`.
- **Key Features**:
  - Unchanged files are skipped (zero redundant embedding API calls).
  - New and modified files are chunked and batch-embedded (batch size 5 with rate-limit pacing).
  - Stale chunk IDs for deleted or updated documents are pruned from ChromaDB.
  - Supports `--force-reset` for full clean rebuilds.

### 6.7 `RAG/unified_pipeline.py` (Production Multi-Turn Hybrid Orchestrator)
- **Responsibility**: Unified enterprise pipeline uniting conversational query contextualization, RBAC filtering, BM25 + Dense retrieval, RRF ($k=60$), FlashRank Cross-Encoder reranking, and citation-grounded LLM synthesis.
- **Key Classes & Methods**:
  - `RBACManager`: Centralized role resolution and `$in` metadata filter generator.
  - `QueryContextualizer`: Reformulates multi-turn conversational queries using Gemini Flash.
  - `HybridSearchEngine`: Dual-retriever engine with RRF candidate ranking.
  - `NeuralReranker`: Cross-encoder passage scoring with `ms-marco-TinyBERT-L-2-v2`.
  - `GroundedGenerator`: Strict anti-leakage prompt and context formatter.
  - `UnifiedRAGPipeline.query(...)`: End-to-end orchestrator with per-stage latency stopwatch.
  - `start_interactive_cli(...)`: Production CLI supporting `/role`, `/whoami`, `/verbose`, `/history`, `/clear`, `/benchmark`, and `/help`.

---

## 7. Prompt Engineering & Guardrail Specifications

### 7.1 Standalone Query Contextualization Prompt
```
System: Given a chat history and the latest user question which might reference context 
in the chat history, formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, just reformulate it if needed 
and otherwise return it exactly as is.
```

### 7.2 Grounded RBAC Generation System Prompt
```
System: You are an enterprise AI assistant for DMS Solutions with strict Role-Based Access Control (RBAC).
The current user's role is: '{ROLE}'.
Their permitted document categories are: {ALLOWED_CATEGORIES}.

Strict Guidelines:
1. Answer the user's question using ONLY the provided authorized documents context.
2. Provide a direct, factual, and concise answer.
3. If the provided context does NOT contain enough information to answer the question, reply exactly with: 
   'I don't have access to this information or it is not available based on your role permissions.'
4. Never hallucinate or disclose information that is not present in the provided context documents.
```

---

## 8. Error Handling, Resilience & Operational Constraints

1. **Windows Platform Compatibility**:
   - Explicit `sys.stdout.reconfigure(encoding='utf-8')` is invoked in all entry points to eliminate `charmap` Unicode print crashes on Windows terminal hosts.
2. **Dynamic Path Calculation**:
   - All file references use `os.path.dirname(os.path.abspath(__file__))` and `os.path.join(parent_dir, ...)` to ensure reliable cross-directory script execution regardless of working directory (`cwd`).
3. **Empty Retrieval Handling**:
   - When vector filtering returns 0 chunks, the system immediately returns the deterministic refusal message without invoking the LLM, conserving API tokens and latency.

---

## 9. Future Technical Roadmap & Enhancements

```
+---------------------------------------------------------------------------------------------+
| 1. Distributed Vector Database Migration                                                    |
|    - Transition from local ChromaDB to Qdrant or Milvus cluster for multi-node scalability  |
|    - Payload indexing on 'access_level' for sub-millisecond filtering at million-vector scale|
+---------------------------------------------------------------------------------------------+
| 2. API & Real-Time Streaming Gateway                                                        |
|    - FastAPI REST & WebSocket endpoints with Server-Sent Events (SSE) token streaming        |
|    - Pydantic v2 request/response validation contracts                                      |
+---------------------------------------------------------------------------------------------+
| 3. Enterprise Identity & SSO Integration                                                    |
|    - JWT / OIDC token decoding in API Gateway                                               |
|    - Automated role extraction from Active Directory / Okta SAML claims                      |
+---------------------------------------------------------------------------------------------+
| 4. Telemetry, Observability & Evaluation                                                    |
|    - LangSmith / OpenTelemetry tracing for latency and retrieval metrics                    |
|    - Automated RAG Triad evaluation (Context Relevance, Groundedness, Answer Relevance)     |
+---------------------------------------------------------------------------------------------+
```
