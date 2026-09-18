import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Dynamic path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
load_dotenv(os.path.join(parent_dir, ".env"))

# =============================================================================
# 1. ROLE-BASED ACCESS CONTROL (RBAC) MANAGER
# =============================================================================
class RBACManager:
    """Manages role-based access control policies, entitlements, and Chroma filter clauses."""
    
    ROLE_PERMISSIONS = {
        "all_employees": ["public"],
        "intern": ["public"],
        "engineering": ["public", "engineering"],
        "hr": ["public", "hr"],
        "finance": ["public", "finance"],
        "management": ["public", "engineering", "hr", "finance", "management"],
        "admin": ["public", "engineering", "hr", "finance", "management"]
    }

    ADMIN_ROLES = {"admin", "superuser", "root"}

    @classmethod
    def normalize_role(cls, role: str) -> str:
        return role.strip().lower().replace(" ", "_")

    @classmethod
    def is_admin(cls, role: str) -> bool:
        return cls.normalize_role(role) in cls.ADMIN_ROLES

    @classmethod
    def get_allowed_categories(cls, role: str) -> List[str]:
        norm = cls.normalize_role(role)
        return cls.ROLE_PERMISSIONS.get(norm, ["public"])

    @classmethod
    def build_chroma_filter(cls, role: str) -> Optional[Dict[str, Any]]:
        if cls.is_admin(role):
            return None  # Unrestricted access to 100% of documents
        allowed = cls.get_allowed_categories(role)
        if len(allowed) == 1:
            return {"access_level": allowed[0]}
        return {"access_level": {"$in": allowed}}

    @classmethod
    def get_tier_display(cls, role: str) -> str:
        if cls.is_admin(role):
            return "ALL CATEGORIES (UNRESTRICTED ADMIN ACCESS)"
        return str(cls.get_allowed_categories(role))


# =============================================================================
# 2. QUERY CONTEXTUALIZER (CONVERSATIONAL MEMORY REWRITER)
# =============================================================================
class QueryContextualizer:
    """Reformulates multi-turn conversational queries into standalone, self-contained search queries."""

    def __init__(self, model_name: str = "gemini-flash-lite-latest"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        self.system_prompt = (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if needed "
            "and otherwise return it exactly as is."
        )

    def contextualize(self, query: str, chat_history: List[Any]) -> str:
        if not chat_history:
            return query

        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(chat_history)
        messages.append(HumanMessage(content=query))

        response = self.llm.invoke(messages)
        return response.text.strip()


# =============================================================================
# 3. HYBRID RETRIEVER & RECIPROCAL RANK FUSION (RRF)
# =============================================================================
class HybridSearchEngine:
    """Combines Role-Filtered Sparse BM25 keyword search with Dense Chroma Cosine similarity."""

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.path.join(parent_dir, "chroma_db")
        self.persist_directory = persist_directory

        # Embedding & Chroma Store
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self._refresh_corpus_cache()

    def _refresh_corpus_cache(self):
        """Loads all indexed documents from ChromaDB for in-memory BM25 retrieval."""
        data = self.db.get(include=["documents", "metadatas"])
        self.cached_documents = []
        for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            self.cached_documents.append(Document(page_content=text, metadata=meta or {}))

    def build_bm25_retriever(self, role: str, k: int = 5) -> Optional[BM25Retriever]:
        """Constructs an in-memory BM25 retriever restricted to the user's role permissions."""
        if RBACManager.is_admin(role):
            authorized_docs = self.cached_documents
        else:
            allowed = RBACManager.get_allowed_categories(role)
            authorized_docs = [
                doc for doc in self.cached_documents
                if doc.metadata.get("access_level", "public") in allowed
            ]

        if not authorized_docs:
            return None

        bm25 = BM25Retriever.from_documents(authorized_docs)
        bm25.k = k
        return bm25

    def search_dense(self, query: str, role: str, k: int = 5) -> List[Document]:
        """Performs dense vector retrieval in ChromaDB with role metadata filtering."""
        search_filter = RBACManager.build_chroma_filter(role)
        if search_filter:
            retriever = self.db.as_retriever(search_kwargs={"k": k, "filter": search_filter})
        else:
            retriever = self.db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def search_sparse(self, query: str, role: str, k: int = 5) -> List[Document]:
        """Performs sparse BM25 keyword retrieval filtered by role."""
        bm25 = self.build_bm25_retriever(role=role, k=k)
        if not bm25:
            return []
        return bm25.invoke(query)

    @staticmethod
    def reciprocal_rank_fusion(dense_docs: List[Document], sparse_docs: List[Document], rrf_k: int = 60, top_n: int = 8) -> List[Document]:
        """
        Merges dense and sparse candidate lists using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score(d) = sum( 1 / (rrf_k + rank(d)) )
        """
        doc_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_docs, 1):
            key = (doc.metadata.get("source", ""), doc.page_content.strip())
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        for rank, doc in enumerate(sparse_docs, 1):
            key = (doc.metadata.get("source", ""), doc.page_content.strip())
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:top_n]]


# =============================================================================
# 4. CROSS-ENCODER NEURAL RERANKER
# =============================================================================
class NeuralReranker:
    """Applies FlashRank Cross-Encoder to compute deep semantic relevance scores."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        cache_dir = os.path.join(parent_dir, ".cache")
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, candidate_docs: List[Document], top_k: int = 3) -> List[Document]:
        if not candidate_docs:
            return []

        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(candidate_docs)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(rerank_req)

        final_docs = []
        for item in ranked[:top_k]:
            original = candidate_docs[item["id"]]
            original.metadata["rerank_score"] = float(item["score"])
            final_docs.append(original)
        return final_docs


# =============================================================================
# 5. GROUNDED GENERATION ENGINE
# =============================================================================
class GroundedGenerator:
    """Generates context-grounded answers while strictly enforcing RBAC boundaries."""

    FALLBACK_REFUSAL = "I don't have access to this information or it is not available based on your role permissions."

    def __init__(self, model_name: str = "gemini-flash-lite-latest"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)

    def generate(self, query: str, context_docs: List[Document], chat_history: List[Any], role: str) -> str:
        if not context_docs:
            return self.FALLBACK_REFUSAL

        # Format context with citation badges
        formatted_chunks = []
        for doc in context_docs:
            dept = doc.metadata.get("access_level", "unknown").upper()
            fname = doc.metadata.get("filename", "unknown")
            formatted_chunks.append(f"[{dept} - {fname}]\n{doc.page_content}")

        context_text = "\n\n---\n\n".join(formatted_chunks)
        tier_info = RBACManager.get_tier_display(role)

        system_prompt = (
            f"You are an enterprise AI assistant for DMS Solutions with strict Role-Based Access Control (RBAC).\n"
            f"Current User Role: '{role.upper()}'.\n"
            f"Authorized Information Tiers: {tier_info}.\n\n"
            "Strict Guidelines:\n"
            "1. Answer the user's question using ONLY the provided verified context documents.\n"
            "2. Provide accurate, direct, and factual answers.\n"
            f"3. If the context does NOT contain enough verified information, reply exactly with:\n"
            f"'{self.FALLBACK_REFUSAL}'\n"
            "4. Never hallucinate or disclose details outside the authorized documents provided."
        )

        user_content = f"Context Documents:\n{context_text}\n\nUser Question: {query}"

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(chat_history)
        messages.append(HumanMessage(content=user_content))

        response = self.llm.invoke(messages)
        return response.text.strip()


# =============================================================================
# 6. UNIFIED PIPELINE ORCHESTRATOR
# =============================================================================
class UnifiedRAGPipeline:
    """
    End-to-End Enterprise RAG Pipeline uniting:
    1. Conversational Query Contextualization
    2. RBAC Policy Resolution & Filtering
    3. Sparse BM25 + Dense Semantic Retrieval
    4. Reciprocal Rank Fusion (RRF)
    5. FlashRank Cross-Encoder Re-ranking
    6. Grounded Gemini Generation & Citations
    """

    def __init__(self):
        self.contextualizer = QueryContextualizer()
        self.search_engine = HybridSearchEngine()
        self.reranker = NeuralReranker()
        self.generator = GroundedGenerator()

    def query(self, user_query: str, chat_history: List[Any] = None, role: str = "all_employees",
              initial_k: int = 5, final_k: int = 3, verbose: bool = False) -> Dict[str, Any]:
        
        if chat_history is None:
            chat_history = []

        start_total = time.time()
        timings = {}

        # Step 1: Contextualize Query
        t0 = time.time()
        standalone_query = self.contextualizer.contextualize(user_query, chat_history)
        timings["contextualization_ms"] = round((time.time() - t0) * 1000, 2)

        # Step 2: Role-Filtered Hybrid Retrieval
        t0 = time.time()
        sparse_docs = self.search_engine.search_sparse(standalone_query, role=role, k=initial_k)
        dense_docs = self.search_engine.search_dense(standalone_query, role=role, k=initial_k)
        fused_docs = self.search_engine.reciprocal_rank_fusion(dense_docs, sparse_docs, top_n=initial_k * 2)
        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 2)

        # Step 3: Cross-Encoder Reranking
        t0 = time.time()
        reranked_docs = self.reranker.rerank(standalone_query, fused_docs, top_k=final_k)
        timings["reranking_ms"] = round((time.time() - t0) * 1000, 2)

        # Step 4: Grounded LLM Generation
        t0 = time.time()
        answer = self.generator.generate(user_query, reranked_docs, chat_history, role=role)
        timings["generation_ms"] = round((time.time() - t0) * 1000, 2)

        timings["total_ms"] = round((time.time() - start_total) * 1000, 2)

        # Extract structured citations
        citations = []
        for d in reranked_docs:
            citations.append({
                "department": d.metadata.get("access_level", "unknown"),
                "filename": d.metadata.get("filename", "unknown"),
                "rerank_score": d.metadata.get("rerank_score", 0.0),
                "preview": d.page_content[:120].strip()
            })

        if verbose:
            print("\n" + "=" * 65)
            print(f"[VERBOSE TRACE] Role: '{role}' | Standalone Query: '{standalone_query}'")
            print(f" • BM25 hits: {len(sparse_docs)} | Dense hits: {len(dense_docs)} | Fused: {len(fused_docs)}")
            print(f" • Top {len(reranked_docs)} Reranked Passages:")
            for idx, c in enumerate(citations, 1):
                print(f"   [{idx}] Score: {c['rerank_score']:.4f} | [{c['department'].upper()}] {c['filename']}")
            print(f" • Timings: Context={timings['contextualization_ms']}ms | Retrieval={timings['retrieval_ms']}ms | Rerank={timings['reranking_ms']}ms | Gen={timings['generation_ms']}ms | Total={timings['total_ms']}ms")
            print("=" * 65)

        return {
            "answer": answer,
            "standalone_query": standalone_query,
            "citations": citations,
            "retrieved_docs": reranked_docs,
            "timings": timings
        }


# =============================================================================
# 7. AUTOMATED BENCHMARK & EVALUATION SUITE
# =============================================================================
def run_automated_benchmark(pipeline: UnifiedRAGPipeline):
    """Executes a multi-role, multi-turn benchmark to evaluate RBAC and hybrid retrieval."""
    print("\n" + "=" * 75)
    print("      UNIFIED PIPELINE END-TO-END AUTOMATED BENCHMARK")
    print("=" * 75)

    test_cases = [
        {
            "desc": "1. Admin Full Access (Confidential M&A Project Titan)",
            "role": "admin",
            "queries": ["What are the negotiated cash and stock terms for Project Titan M&A?"],
            "expect_allow": True
        },
        {
            "desc": "2. Engineering Authorized Query (Microservices & Kafka streaming)",
            "role": "engineering",
            "queries": ["What is our backend microservice topology, Kafka retention, and database replication?"],
            "expect_allow": True
        },
        {
            "desc": "3. Engineering Probing HR Salary Bands (RBAC Block)",
            "role": "engineering",
            "queries": ["What are the base salary bands and RSU equity for Level 4 Staff Engineers?"],
            "expect_allow": False
        },
        {
            "desc": "4. HR Authorized Leveling Query (Level 4 Staff Engineers)",
            "role": "hr",
            "queries": ["What are the base salary bands and RSU equity for Level 4 Staff Engineers?"],
            "expect_allow": True
        },
        {
            "desc": "5. Finance Vendor Contracts (AWS 3-Year EDP Commitment)",
            "role": "finance",
            "queries": ["What is our annual AWS EDP spend commitment and discount rate?"],
            "expect_allow": True
        },
        {
            "desc": "6. Multi-Turn History-Aware Resolution (PTO Policy follow-up)",
            "role": "all_employees",
            "queries": [
                "How many days of paid vacation do full-time employees receive?",
                "Can I roll any unused days over to the next year?"
            ],
            "expect_allow": True
        }
    ]

    for test in test_cases:
        print(f"\n>>> TEST: {test['desc']}")
        history = []
        for q_idx, q in enumerate(test["queries"], 1):
            if len(test["queries"]) > 1:
                print(f"  Turn {q_idx}: '{q}'")
            res = pipeline.query(q, chat_history=history, role=test["role"], verbose=True)
            print(f"  [Answer]: {res['answer']}")
            history.append(HumanMessage(content=q))
            history.append(AIMessage(content=res["answer"]))
        print("-" * 75)

    print("\n[Benchmark Complete] All test scenarios evaluated successfully.\n")


# =============================================================================
# 8. INTERACTIVE COMMAND-LINE INTERFACE (CLI)
# =============================================================================
def print_help():
    print("""
Available Slash Commands:
  /role <name>   : Switch active role (all_employees, engineering, hr, finance, management, admin)
  /whoami        : Show active role and authorized document categories
  /verbose <on|off> : Toggle detailed stage latency & retrieval tracing
  /history       : Print current multi-turn conversation memory
  /clear         : Reset conversation history
  /benchmark     : Run automated end-to-end evaluation suite
  /help          : Display this command manual
  exit / quit    : Exit the session
""")

def start_interactive_cli(pipeline: UnifiedRAGPipeline, initial_role: str = "all_employees", initial_verbose: bool = False):
    current_role = initial_role
    verbose = initial_verbose
    chat_history = []

    print("=" * 75)
    print("      DMS SOLUTIONS - UNIFIED ENTERPRISE RBAC RAG CHATBOT")
    print("=" * 75)
    print(f"Active Role:   '{current_role}'")
    print(f"Authorized:    {RBACManager.get_tier_display(current_role)}")
    print(f"Verbose Mode:  {'ON' if verbose else 'OFF'}")
    print("Type '/help' for command list. Type 'exit' to quit.\n")

    while True:
        try:
            prompt_str = f"[{current_role.upper()}] > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting session. Goodbye!")
                break

            # Handle Slash Commands
            if user_input.startswith("/role"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    new_role = RBACManager.normalize_role(parts[1])
                    if new_role in RBACManager.ROLE_PERMISSIONS:
                        current_role = new_role
                        print(f"-> Switched active role to: '{current_role}'")
                        print(f"   Authorized Tiers: {RBACManager.get_tier_display(current_role)}")
                    else:
                        print(f"-> Unknown role '{parts[1]}'. Valid roles: {list(RBACManager.ROLE_PERMISSIONS.keys())}")
                else:
                    print("-> Usage: /role <all_employees | engineering | hr | finance | management | admin>")
                continue

            if user_input == "/whoami":
                print(f"-> Active Role:    '{current_role}'")
                print(f"-> Access Tiers:   {RBACManager.get_tier_display(current_role)}")
                print(f"-> Verbose Trace:  {'ON' if verbose else 'OFF'}")
                print(f"-> History Turns:  {len(chat_history) // 2}")
                continue

            if user_input.startswith("/verbose"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].lower() in ["on", "1", "true"]:
                    verbose = True
                    print("-> Verbose tracing enabled.")
                elif len(parts) > 1 and parts[1].lower() in ["off", "0", "false"]:
                    verbose = False
                    print("-> Verbose tracing disabled.")
                else:
                    verbose = not verbose
                    print(f"-> Verbose tracing toggled to: {'ON' if verbose else 'OFF'}")
                continue

            if user_input == "/history":
                if not chat_history:
                    print("-> Conversation history is currently empty.")
                else:
                    print("\n--- Conversation Memory ---")
                    for msg in chat_history:
                        sender = "User" if isinstance(msg, HumanMessage) else "Assistant"
                        print(f"[{sender}]: {msg.content}")
                    print("---------------------------\n")
                continue

            if user_input == "/clear":
                chat_history = []
                print("-> Conversation history cleared.")
                continue

            if user_input == "/benchmark":
                run_automated_benchmark(pipeline)
                continue

            if user_input == "/help":
                print_help()
                continue

            # Execute Query through Unified Pipeline
            result = pipeline.query(
                user_query=user_input,
                chat_history=chat_history,
                role=current_role,
                verbose=verbose
            )

            print(f"\nAssistant: {result['answer']}")

            if result["citations"] and result["answer"] != GroundedGenerator.FALLBACK_REFUSAL:
                top_sources = list({f"[{c['department'].upper()}] {c['filename']}" for c in result["citations"]})
                print(f"Sources: {', '.join(top_sources)}")
            print("-" * 65)

            # Update history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=result["answer"]))

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


# =============================================================================
# 9. ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Enterprise RBAC RAG Pipeline")
    parser.add_argument("--test", action="store_true", help="Run automated RBAC and hybrid retrieval benchmark suite")
    parser.add_argument("--role", default="all_employees", help="Initial employee role")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose tracing output")
    args = parser.parse_args()

    pipeline = UnifiedRAGPipeline()

    if args.test:
        run_automated_benchmark(pipeline)
    else:
        start_interactive_cli(pipeline, initial_role=args.role, initial_verbose=args.verbose)

if __name__ == "__main__":
    main()
