import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

# Configure standard output to use UTF-8 (prevents crashes when printing Unicode on Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Calculate dynamic paths relative to the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load the environment variables from .env at the project root
load_dotenv(os.path.join(parent_dir, ".env"))

persistent_directory = os.path.join(parent_dir, "chroma_db")

# Initialize Embedding Model & Chroma Vector Store
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

# Initialize Cross-Encoder FlashRank Reranker
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=os.path.join(parent_dir, ".cache"))

# Role-Based Access Control (RBAC) Permission Matrix
ROLE_PERMISSIONS = {
    "all_employees": ["public"],
    "intern": ["public"],
    "engineering": ["public", "engineering"],
    "hr": ["public", "hr"],
    "finance": ["public", "finance"],
    "management": ["public", "engineering", "hr", "finance", "management"],
    "admin": ["public", "engineering", "hr", "finance", "management"]
}

def get_allowed_access_levels(role: str) -> list:
    """Returns the list of allowed document categories for a given role."""
    normalized_role = role.strip().lower().replace(" ", "_")
    return ROLE_PERMISSIONS.get(normalized_role, ["public"])

def get_all_indexed_documents():
    """Extracts all indexed documents and metadata from ChromaDB for BM25 indexing."""
    data = db.get(include=["documents", "metadatas"])
    documents = []
    for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
        documents.append(Document(page_content=text, metadata=meta or {}))
    return documents

# Cache loaded documents for BM25
ALL_DOCUMENTS = get_all_indexed_documents()

def build_bm25_retriever(role: str = "all_employees", k: int = 5):
    """Constructs a BM25 keyword retriever restricted to the documents authorized for the user's role."""
    normalized_role = role.strip().lower().replace(" ", "_")
    
    if normalized_role in ["admin", "superuser", "root"]:
        authorized_docs = ALL_DOCUMENTS
    else:
        allowed_levels = get_allowed_access_levels(role)
        authorized_docs = [
            doc for doc in ALL_DOCUMENTS 
            if doc.metadata.get("access_level", "public") in allowed_levels
        ]
        
    if not authorized_docs:
        return None
        
    bm25 = BM25Retriever.from_documents(authorized_docs)
    bm25.k = k
    return bm25

def build_semantic_retriever(role: str = "all_employees", k: int = 5):
    """Constructs a Chroma semantic dense vector retriever filtered by role."""
    normalized_role = role.strip().lower().replace(" ", "_")
    
    if normalized_role in ["admin", "superuser", "root"]:
        return db.as_retriever(search_kwargs={"k": k})
        
    allowed_levels = get_allowed_access_levels(role)
    if len(allowed_levels) == 1:
        search_filter = {"access_level": allowed_levels[0]}
    else:
        search_filter = {"access_level": {"$in": allowed_levels}}
        
    return db.as_retriever(search_kwargs={"k": k, "filter": search_filter})

def reciprocal_rank_fusion(dense_docs, sparse_docs, rrf_k: int = 60, top_n: int = 8):
    """
    Combines dense (semantic) and sparse (BM25) search results using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score(d) = sum( 1 / (rrf_k + rank(d)) )
    """
    doc_scores = {}
    doc_map = {}

    # Score Semantic / Dense Results
    for rank, doc in enumerate(dense_docs, 1):
        # Create a unique key based on content snippet and source
        key = (doc.metadata.get("source", ""), doc.page_content.strip())
        doc_map[key] = doc
        doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    # Score BM25 / Sparse Results
    for rank, doc in enumerate(sparse_docs, 1):
        key = (doc.metadata.get("source", ""), doc.page_content.strip())
        doc_map[key] = doc
        doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    # Sort candidates by combined RRF score
    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
    fused_docs = [doc_map[k] for k in sorted_keys[:top_n]]
    return fused_docs

def rerank_documents(query: str, candidate_docs: list, top_k: int = 3):
    """
    Applies FlashRank Cross-Encoder reranking to score semantic relevance between query and text chunks.
    """
    if not candidate_docs:
        return []
        
    passages = []
    for idx, doc in enumerate(candidate_docs):
        passages.append({
            "id": idx,
            "text": doc.page_content,
            "meta": doc.metadata
        })
        
    rerank_request = RerankRequest(query=query, passages=passages)
    ranked_results = ranker.rerank(rerank_request)
    
    final_docs = []
    for res in ranked_results[:top_k]:
        original_doc = candidate_docs[res["id"]]
        # Attach the reranker relevance score to metadata
        original_doc.metadata["rerank_score"] = float(res["score"])
        final_docs.append(original_doc)
        
    return final_docs

def hybrid_search_and_rerank(query: str, role: str = "all_employees", initial_k: int = 5, final_k: int = 3, verbose: bool = True):
    """
    Executes full Hybrid Search (BM25 + Semantic Vector) + RRF Fusion + Cross-Encoder Reranking.
    """
    allowed_levels = get_allowed_access_levels(role)
    normalized_role = role.strip().lower().replace(" ", "_")
    tier_info = "ALL DOCUMENTS (UNRESTRICTED)" if normalized_role in ["admin", "superuser"] else allowed_levels

    if verbose:
        print(f"\n{'='*70}")
        print(f"HYBRID RETRIEVAL & RERANKING")
        print(f"Role: '{role}' | Authorized Categories: {tier_info}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")

    # 1. Sparse BM25 Search
    bm25_retriever = build_bm25_retriever(role=role, k=initial_k)
    sparse_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    
    # 2. Dense Semantic Search
    semantic_retriever = build_semantic_retriever(role=role, k=initial_k)
    dense_docs = semantic_retriever.invoke(query)

    if verbose:
        print(f"[1] BM25 Keyword Search retrieved: {len(sparse_docs)} candidate chunk(s)")
        print(f"[2] Dense Semantic Vector Search retrieved: {len(dense_docs)} candidate chunk(s)")

    # 3. Reciprocal Rank Fusion (RRF)
    fused_docs = reciprocal_rank_fusion(dense_docs, sparse_docs, top_n=initial_k * 2)
    if verbose:
        print(f"[3] RRF Fusion merged candidate pool: {len(fused_docs)} unique chunk(s)")

    # 4. Cross-Encoder Reranking
    reranked_docs = rerank_documents(query, fused_docs, top_k=final_k)
    if verbose:
        print(f"[4] Cross-Encoder Reranker selected top {len(reranked_docs)} high-confidence chunk(s)")
        for idx, doc in enumerate(reranked_docs, 1):
            score = doc.metadata.get("rerank_score", 0.0)
            dept = doc.metadata.get("access_level", "unknown").upper()
            fname = doc.metadata.get("filename", "unknown")
            print(f"    Rank {idx} [Score: {score:.4f}] [{dept}] ({fname})")
            
    return reranked_docs

def generate_hybrid_answer(query: str, role: str = "all_employees", final_k: int = 3, verbose: bool = True):
    """
    End-to-end question answering pipeline using Hybrid Search (BM25 + Semantic) + Reranking.
    """
    relevant_documents = hybrid_search_and_rerank(query, role=role, initial_k=5, final_k=final_k, verbose=verbose)
    
    if not relevant_documents:
        return "I don't have access to this information or it is not available based on your role permissions."

    # Format context with citations
    formatted_docs = []
    for doc in relevant_documents:
        dept = doc.metadata.get("access_level", "unknown").upper()
        fname = doc.metadata.get("filename", "unknown")
        formatted_docs.append(f"[{dept} - {fname}]\n{doc.page_content}")
        
    context_text = "\n\n---\n\n".join(formatted_docs)

    system_prompt = (
        f"You are an enterprise AI assistant for DMS Solutions with strict Role-Based Access Control (RBAC).\n"
        f"Current User Role: '{role.upper()}'.\n\n"
        "Strict Guidelines:\n"
        "1. Answer the user's question using ONLY the provided verified context documents.\n"
        "2. Provide accurate, direct, and factual answers.\n"
        "3. If the context does not contain the answer, reply exactly with: "
        "'I don't have access to this information or it is not available based on your role permissions.'\n"
        "4. Do NOT disclose confidential information outside the user's role."
    )

    user_prompt = f"Context Documents:\n{context_text}\n\nQuestion: {query}"

    model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = model.invoke(messages)
    return response.text.strip()

def run_hybrid_comparison_tests():
    """Demonstrates how Hybrid Search + Reranking excels at keyword acronyms, specific numbers, and semantic meaning."""
    print("=" * 75)
    print("   RUNNING HYBRID SEARCH (BM25 + SEMANTIC) & RERANKING TEST BENCHMARK")
    print("=" * 75)

    test_cases = [
        # Test 1: Exact technical acronym search ("mTLS", "Kong", "Vault")
        {
            "role": "engineering",
            "query": "What is the token auth, mTLS, and Vault secret rotation policy?",
            "desc": "Technical Acronym & Keyword Match (mTLS, Vault)"
        },
        # Test 2: Specific numeric / financial data ("$38,000,000", "Project Titan")
        {
            "role": "admin",
            "query": "What are the exact acquisition terms and cash vs stock breakdown for Project Titan?",
            "desc": "Precise Numerical & Financial Terms Extraction"
        },
        # Test 3: Specific HR level compensation ("Level 4", "RSUs")
        {
            "role": "hr",
            "query": "What is the base salary band and bonus percentage for Level 4 Staff Engineers?",
            "desc": "Structured Level & Range Query"
        },
        # Test 4: General semantic question (Office working policy)
        {
            "role": "all_employees",
            "query": "Can I work from home on Wednesdays and what are the core hours?",
            "desc": "Semantic / Natural Language Understanding"
        },
        # Test 5: RBAC boundary enforcement (Finance asking about engineering secrets)
        {
            "role": "finance",
            "query": "What are the internal API secret rotation keys and Vault passwords?",
            "desc": "RBAC Security Enforcement Check"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n>>> TEST BENCHMARK {i}: {test['desc']}")
        ans = generate_hybrid_answer(test["query"], role=test["role"], verbose=True)
        print(f"\n[Generated Answer]:\n{ans}\n")
        print("-" * 75)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Search (BM25 + Semantic) and Reranking Pipeline")
    parser.add_argument("--test", action="store_true", help="Run automated Hybrid Search benchmark tests")
    parser.add_argument("--role", default=None, help="User role (all_employees, engineering, hr, finance, management, admin)")
    args = parser.parse_args()

    if args.test:
        run_hybrid_comparison_tests()
    else:
        role = args.role
        if not role:
            print("=" * 65)
            print("   DMS Solutions - Hybrid Search (BM25 + Vector) & Reranking")
            print("=" * 65)
            print("Select an Employee Role:")
            print("  1. all_employees (Public handbook & perks)")
            print("  2. engineering   (Architecture, CI/CD, API Security)")
            print("  3. hr            (Salary Bands, PIP guidelines)")
            print("  4. finance       (Budgets, Vendor Contracts, Travel)")
            print("  5. management    (M&A Strategy, Board Minutes)")
            print("  6. admin         (FULL UNRESTRICTED ACCESS)")
            print("=" * 65)
            choice = input("Enter choice (1-6) or role name [default: all_employees]: ").strip()
            
            role_map = {
                "1": "all_employees",
                "2": "engineering",
                "3": "hr",
                "4": "finance",
                "5": "management",
                "6": "admin"
            }
            role = role_map.get(choice, choice if choice else "all_employees")

        print(f"\nActive Role: '{role}'")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                user_query = input(f"[HYBRID - {role.upper()}] Enter query: ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ["exit", "quit"]:
                    break
                
                answer = generate_hybrid_answer(user_query, role=role, verbose=True)
                print(f"\nAnswer:\n{answer}\n")
                print("-" * 65)
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}\n")
