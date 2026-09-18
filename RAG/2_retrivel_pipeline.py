import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure standard output to use UTF-8 (prevents crashes when printing Unicode on Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Calculate dynamic paths relative to the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load the environment variables from .env at the project root
load_dotenv(os.path.join(parent_dir, ".env"))

persistent_directory = os.path.join(parent_dir, "chroma_db")

# Load the embedding model and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

# Role-Based Access Control (RBAC) Permission Matrix
# 'admin' has complete, unrestricted access to ALL documents across all categories.
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
    """Returns the list of allowed document categories/access levels for a given role."""
    normalized_role = role.strip().lower().replace(" ", "_")
    return ROLE_PERMISSIONS.get(normalized_role, ["public"])

def build_retriever_for_role(role: str = "all_employees", k: int = 3):
    """Constructs a Chroma retriever pre-filtered by the employee's role permissions."""
    normalized_role = role.strip().lower().replace(" ", "_")
    
    # Superuser/Admin has unrestricted access across 100% of all documents
    if normalized_role in ["admin", "superuser", "root"]:
        return db.as_retriever(search_kwargs={"k": k})
        
    allowed_levels = get_allowed_access_levels(role)
    
    if len(allowed_levels) == 1:
        search_filter = {"access_level": allowed_levels[0]}
    else:
        search_filter = {"access_level": {"$in": allowed_levels}}
        
    return db.as_retriever(search_kwargs={"k": k, "filter": search_filter})

def retrieve_documents(query: str, role: str = "all_employees", k: int = 3):
    """
    Searches and retrieves documents matching the query, strictly restricted to 
    the permissions associated with the provided role.
    """
    allowed_levels = get_allowed_access_levels(role)
    normalized_role = role.strip().lower().replace(" ", "_")
    tier_info = "ALL DOCUMENTS (UNRESTRICTED)" if normalized_role in ["admin", "superuser"] else allowed_levels
    print(f"\n[RBAC] Query: '{query}' | Role: '{role}' | Allowed Categories: {tier_info}")
    
    retriever = build_retriever_for_role(role=role, k=k)
    relevant_documents = retriever.invoke(query)
    
    print(f"[RBAC] Retrieved {len(relevant_documents)} document chunk(s) authorized for role '{role}'.")
    return relevant_documents

def run_rbac_test_suite():
    """Demonstrates RBAC retrieval across various employee roles including admin."""
    test_cases = [
        # 1. Admin querying confidential salary bands (Allowed - Full Access)
        {
            "role": "admin",
            "query": "What are the salary bands and equity for Level 3 Senior Engineers?",
            "expected_access": "ALLOWED (admin - full access)"
        },
        # 2. Admin querying confidential M&A strategy (Allowed - Full Access)
        {
            "role": "admin",
            "query": "What is Project Titan and the target company acquisition valuation?",
            "expected_access": "ALLOWED (admin - full access)"
        },
        # 3. Engineering querying technical info (Allowed)
        {
            "role": "engineering",
            "query": "What is our microservices infrastructure, Kafka streaming, and database setup?",
            "expected_access": "ALLOWED (engineering)"
        },
        # 4. Engineering querying confidential HR salary bands (BLOCKED by RBAC)
        {
            "role": "engineering",
            "query": "What are the salary bands and equity for Level 3 Senior Engineers?",
            "expected_access": "BLOCKED (HR/Admin only)"
        },
        # 5. HR querying salary bands (Allowed)
        {
            "role": "hr",
            "query": "What are the salary bands and equity for Level 3 Senior Engineers?",
            "expected_access": "ALLOWED (hr)"
        },
        # 6. Finance querying department budget allocations (Allowed)
        {
            "role": "finance",
            "query": "What is the Q3 approved budget for the Engineering Division?",
            "expected_access": "ALLOWED (finance)"
        },
        # 7. General Employee querying public holidays (Allowed)
        {
            "role": "all_employees",
            "query": "What are the official company holidays and PTO benefits for 2026?",
            "expected_access": "ALLOWED (public)"
        }
    ]

    print("=" * 70)
    print("      ROLE-BASED ACCESS CONTROL (RBAC) RETRIEVAL TEST SUITE")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n>>> TEST CASE {i} [{test['expected_access']}]")
        docs = retrieve_documents(test["query"], role=test["role"])
        
        if docs:
            for j, doc in enumerate(docs, 1):
                dept = doc.metadata.get('access_level', 'unknown')
                fname = doc.metadata.get('filename', 'unknown')
                preview = doc.page_content[:130].replace("\n", " ")
                print(f"   [Doc {j}] [{dept.upper()}] ({fname}): {preview}...")
        else:
            print("   -> [ACCESS DENIED / NO RELEVANT AUTHORIZED DOCUMENTS FOUND]")
            
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Run the comprehensive RBAC demonstration test suite
    run_rbac_test_suite()