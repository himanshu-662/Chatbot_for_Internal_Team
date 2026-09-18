# Product Requirements Document (PRD)

## Project: Enterprise Role-Based Access Control (RBAC) Internal Knowledge Chatbot
**Document Version:** 1.0.0  
**Status:** Approved / In Production  
**Target Organization:** DMS Solutions (Enterprise Internal Operations)  
**System Classification:** Internal Knowledge Retrieval & Generative AI Platform  

---

## 1. Executive Summary & Vision

In enterprise environments, internal documentation is fragmented across various departments (Engineering, Human Resources, Finance, Management, and General Corporate Operations). Traditional keyword search engines or unpartitioned AI chatbots pose a critical security risk: **unauthorized data exposure** (e.g., exposing executive M&A strategies or confidential HR compensation bands to unauthorized personnel) and **hallucination**.

The **DMS Solutions RBAC Internal Chatbot** is a production-grade, multi-stage Retrieval-Augmented Generation (RAG) platform designed to deliver instant, context-grounded, and secure answers to internal team queries. The system enforces strict **Role-Based Access Control (RBAC)** at the vector and keyword retrieval layers, ensuring employees only retrieve and generate answers from documents within their authorized permission boundaries.

```
       +-------------------------------------------------------------+
       |                  Enterprise User Request                    |
       |             (Identity Role: Engineering, HR, etc.)          |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             RBAC Gatekeeper & Intent Contextualizer         |
       |       - Reformulates multi-turn conversational context      |
       |       - Resolves active Role & Permitted Document Tiers     |
       +------------------------------+------------------------------+
                                      |
                      +---------------+---------------+
                      |                               |
                      v                               v
       +------------------------------+ +------------------------------+
       |   Role-Filtered Sparse BM25  | |  Role-Filtered Dense Vector  |
       |        Keyword Search        | |   (Gemini Embeddings + HNSW) |
       +--------------+---------------+ +--------------+---------------+
                      |                               |
                      +---------------+---------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             Reciprocal Rank Fusion (RRF) Engine             |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |        Cross-Encoder Reranker (FlashRank / TinyBERT)        |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |      Context-Grounded LLM Generation (Google Gemini)        |
       |     - Strict anti-hallucination & anti-leakage prompt       |
       |     - Direct source citation and metadata provenance        |
       +-------------------------------------------------------------+
```

---

## 2. Business Objectives & Value Proposition

| Business Objective | Current Pain Point | Target Solution / KPI |
| :--- | :--- | :--- |
| **Accelerated Knowledge Discovery** | Employees spend an average of 25-45 minutes searching Confluence, Notion, and PDF drives. | Reduce query-to-answer latency to **< 2.5 seconds** with high semantic accuracy. |
| **Zero Data Leakage (RBAC)** | Standard chatbots indiscriminately ingest all docs; unauthorized users can prompt-extract sensitive information. | **100% boundary isolation**: Documents are filtered before reaching the LLM context. |
| **Elimination of Hallucinations** | Generic LLMs guess policy terms, salary numbers, and tech stacks. | Strict grounding: System returns explicit refusal if verified context is absent. |
| **High Retrieval Precision** | Semantic-only search struggles with exact technical acronyms (`mTLS`, `Kafka`, `EDP`) and numerical budget lines. | Hybrid search (BM25 + Semantic Embeddings) + Cross-Encoder reranking achieving **> 95% MRR@3**. |

---

## 3. Target User Personas & Role Permission Matrix

The platform segments users into 6 distinct roles. Access is strictly partitioned based on document classification metadata.

```
+-----------------------------------------------------------------------------------+
|                              ROLE ACCESS HIERARCHY                                |
+-----------------------------------------------------------------------------------+
|  [ADMIN]       --> [PUBLIC] + [ENGINEERING] + [HR] + [FINANCE] + [MANAGEMENT]     |
|  [MANAGEMENT]  --> [PUBLIC] + [ENGINEERING] + [HR] + [FINANCE] + [MANAGEMENT]     |
|  [ENGINEERING] --> [PUBLIC] + [ENGINEERING]                                       |
|  [HR]          --> [PUBLIC] + [HR]                                                |
|  [FINANCE]     --> [PUBLIC] + [FINANCE]                                           |
|  [ALL/INTERN]  --> [PUBLIC]                                                       |
+-----------------------------------------------------------------------------------+
```

### 3.1 Role Definitions & Entitlements

| Role Identifier | Permitted Access Categories | Accessible Document Types | Example Use Cases |
| :--- | :--- | :--- | :--- |
| `all_employees` / `intern` | `public` | Company handbook, holiday calendar, PTO policies, general wellness perks, learning stipend rules. | "What are the core working hours and 2026 holiday calendar?" |
| `engineering` | `public`, `engineering` | System architecture blueprints, CI/CD ArgoCD pipelines, API Gateway & mTLS specifications, Vault secrets policies. | "What database replication and streaming setup do we use in AWS?" |
| `hr` | `public`, `hr` | Compensation leveling salary bands, equity RSU allocations, PIP protocols, whistleblower and grievance procedures. | "What are the base salary and bonus targets for Level 3 Senior Engineers?" |
| `finance` | `public`, `finance` | Approved Q3 departmental budgets, AWS 3-Year EDP commitments, vendor procurement limits, travel expense per-diems. | "What is the Q3 allocated budget for AWS cloud infrastructure?" |
| `management` | `public`, `engineering`, `hr`, `finance`, `management` | M&A strategic acquisition plans (Project Titan), Board meeting minutes, Series C financing memo, all departmental records. | "What are the negotiated cash vs. stock terms for Project Titan M&A?" |
| `admin` | `public`, `engineering`, `hr`, `finance`, `management` *(Full Unrestricted Access)* | 100% of internal documentation, vector database indexing logs, system configuration parameters. | System audit, enterprise compliance checks, comprehensive cross-departmental queries. |

---

## 4. User Stories & Acceptance Criteria

### US-01: Secure Role-Based Retrieval
- **As an** Engineering team member,
- **I want to** search for technical architecture blueprints and deployment guidelines,
- **So that** I can build and deploy microservices according to company standards,
- **Without** being able to access confidential HR salary bands or M&A acquisition terms.
- **Acceptance Criteria:**
  - Queries about engineering return precise technical context.
  - Queries probing HR salary data return: `"I don't have access to this information or it is not available based on your role permissions."`

### US-02: Multi-Turn History-Aware Conversations
- **As an** Employee querying company policies,
- **I want to** ask follow-up questions referencing previous answers (e.g., *"How many days can I roll over?"* following a vacation question),
- **So that** I don't have to re-type the full context in every message.
- **Acceptance Criteria:**
  - LLM automatically rewrites ambiguous queries into self-contained search queries prior to vector retrieval.
  - Chat history maintains conversational coherence across multi-turn interactions.

### US-03: Precise Hybrid Search & Technical Query Answering
- **As a** DevOps or Finance professional,
- **I want to** query exact acronyms (`mTLS`, `EDP`, `ArgoCD`, `gRPC`) or financial dollar figures (`$38,000,000`),
- **So that** I get accurate, exact-match chunks rather than vague semantic approximations.
- **Acceptance Criteria:**
  - Sparse BM25 keyword index matches exact strings.
  - Dense Gemini vector search matches conceptual intent.
  - Reciprocal Rank Fusion + FlashRank cross-encoder boosts the exact matching chunk to Rank #1.

### US-04: Transparent Citations & Source Provenance
- **As an** internal auditor or manager,
- **I want to** see which file and classification category backed every generated answer,
- **So that** I can verify the factual accuracy and audit trail.
- **Acceptance Criteria:**
  - Context chunks formatted with `[DEPARTMENT - FILENAME]` tags before LLM generation.

---

## 5. Functional Requirements (FR)

| ID | Feature Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **FR-1** | **Multi-Format Ingestion** | Recursively loads and parses `.pdf`, `.docx`, and `.txt` files from directory hierarchies. | P0 (Must Have) |
| **FR-2** | **Automated RBAC Metadata Tagging** | Automatically extracts department access level based on directory hierarchy (e.g. `data/finance/*` -> `access_level: finance`). | P0 (Must Have) |
| **FR-3** | **Recursive Text Chunking** | Splits source documents into chunks (500 characters, 50-character overlap) preserving structural separators (`\n\n`, `\n`, space) and metadata. | P0 (Must Have) |
| **FR-4** | **Dense Vector Indexing** | Generates embeddings via `models/gemini-embedding-2` and persists into ChromaDB using Cosine distance space. | P0 (Must Have) |
| **FR-5** | **Sparse BM25 Indexing** | Builds in-memory BM25 index on indexed documents, partitioned per user role permissions. | P0 (Must Have) |
| **FR-6** | **Reciprocal Rank Fusion (RRF)** | Merges dense and sparse retrieved candidates with standard constant $k=60$. | P1 (Should Have) |
| **FR-7** | **Cross-Encoder Reranking** | Scores candidate passages using `ms-marco-TinyBERT-L-2-v2` via FlashRank to output top $k$ relevant chunks. | P0 (Must Have) |
| **FR-8** | **Question Contextualization** | Reformulates conversational queries into standalone queries using prior chat memory before retrieval. | P0 (Must Have) |
| **FR-9** | **Role-Filtered Retrieval Gatekeeper** | Intercepts all search requests and injects `$in` or direct match metadata filter into Chroma and BM25 queries. | P0 (Must Have) |
| **FR-10** | **Grounded Generative LLM** | Generates concise, truthful responses via `gemini-flash-lite-latest` based exclusively on retrieved context. | P0 (Must Have) |
| **FR-11** | **Interactive CLI Session & Switching** | Supports `/role <name>`, `/whoami`, `/clear`, and `/help` slash commands for interactive debugging and usage. | P1 (Should Have) |

---

## 6. Non-Functional Requirements (NFR)

### 6.1 Security & Compliance
- **Zero Cross-Role Data Leakage**: Vector filtering must happen at database query time, **never** via post-filtering in LLM context.
- **Strict Anti-Hallucination Fallback**: If no authorized chunks are retrieved, the LLM must return the standardized fallback text without speculating.
- **Local Secret Isolation**: API keys (`GEMINI_API_KEY`) loaded via secure environment variables (`.env`), never hardcoded.

### 6.2 Performance & Latency Budgets
- **Vector Retrieval Latency**: $< 150 \text{ ms}$ for $k=5$ cosine similarity lookup in ChromaDB.
- **BM25 Search Latency**: $< 50 \text{ ms}$ across full indexed document corpus.
- **Cross-Encoder Reranking Latency**: $< 120 \text{ ms}$ for 10 candidate passages on CPU.
- **End-to-End User Turn Latency**: $< 2.5 \text{ seconds}$ (including Gemini LLM generation).

### 6.3 Reliability & Availability
- **Rate Limit Resilience**: Batch ingestion processes documents in chunks of 5 with 10-second sleep throttling to prevent Gemini API quota exhaustion.
- **Persistent Vector Store**: Vector database persists to disk (`chroma_db/`) across server restarts.

---

## 7. User Interface & Interaction Specification

The current reference implementation provides an interactive Command-Line Interface (CLI) supporting dynamic role switching and slash commands:

```
======================================================================
   DMS SOLUTIONS - ROLE-BASED ACCESS CONTROL (RBAC) CHATBOT
======================================================================
Roles available: all_employees, engineering, hr, finance, management, admin

Default role set to: 'all_employees'
Type '/role admin' (or any other role) to switch roles. Type 'exit' to quit.

[ALL_EMPLOYEES] > What are the official holidays for 2026?
Assistant: All regional offices observe official public holidays including New Year's Day (Jan 1), Memorial Day (May 25), Independence Day (Jul 3 observed), Labor Day (Sep 7), Thanksgiving (Nov 26), and Christmas Day (Dec 25). Each employee also receives 2 floating cultural holidays per year.

[ALL_EMPLOYEES] > What is the target price for Project Titan M&A?
Assistant: I don't have access to this information or it is not available based on your role permissions.

[ALL_EMPLOYEES] > /role management
-> Switched active role to: 'management' (Authorized tiers: ['public', 'engineering', 'hr', 'finance', 'management'])

[MANAGEMENT] > What is the target price for Project Titan M&A?
Assistant: The total acquisition valuation negotiated for Project Titan (CloudMetrics Inc.) is $38,000,000 USD, structured as 65% cash ($24.7M) and 35% DMS Solutions Series C Preferred Stock ($13.3M).
```

---

## 8. Release Milestones & Implementation Roadmap

```
+---------------------------------------------------------------------------------------------+
| Phase 1: Ingestion & Vector Pipeline (COMPLETED)                                           |
| - DirectoryLoader for PDF, DOCX, TXT                                                        |
| - RBAC metadata tagging & Recursive text splitting                                         |
| - Gemini embedding generation & ChromaDB persistence                                        |
+---------------------------------------------------------------------------------------------+
| Phase 2: RBAC Retrieval & Single-Turn Q&A (COMPLETED)                                       |
| - Chroma pre-filtering with $in metadata filters                                            |
| - Role permission matrix (all_employees, engineering, hr, finance, management, admin)      |
| - System prompt boundary enforcement and anti-leakage controls                              |
+---------------------------------------------------------------------------------------------+
| Phase 3: History-Aware Conversational Memory (COMPLETED)                                    |
| - Multi-turn conversation state tracking                                                    |
| - Query contextualization & reformulation via Gemini Flash                                  |
| - Interactive CLI session with /role, /whoami, /clear commands                              |
+---------------------------------------------------------------------------------------------+
| Phase 4: Core Architecture & Pipeline Unification (COMPLETED - CURRENT STATE)               |
| - Incremental hash-based ingestion engine with manifest tracking (SHA-256)                 |
| - Unified multi-turn hybrid search (BM25 + Dense Vectors) + RRF + FlashRank Reranker       |
| - Interactive production CLI with /role, /whoami, /verbose, /history, /benchmark commands   |
+---------------------------------------------------------------------------------------------+
| Phase 5: Enterprise Web UI & Production API Deployment (FUTURE SCOPE)                       |
| - FastAPI backend with SSE streaming responses                                              |
| - Next.js / Tailwind modern web frontend with role switcher and citation badges            |
| - Enterprise SSO (OAuth2 / Okta / Azure AD OIDC JWT) authentication                         |
| - LangSmith observability & retrieval tracing                                               |
+---------------------------------------------------------------------------------------------+
```
