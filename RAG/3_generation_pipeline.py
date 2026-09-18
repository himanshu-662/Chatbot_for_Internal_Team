import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

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

def retrieve_documents(query: str, role: str = "all_employees", k: int = 3):
    """Retrieves document chunks matching the query filtered strictly by the user's role permissions."""
    normalized_role = role.strip().lower().replace(" ", "_")
    
    # Admin has complete unrestricted access to all documents
    if normalized_role in ["admin", "superuser", "root"]:
        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
        
    allowed_levels = get_allowed_access_levels(role)
    if len(allowed_levels) == 1:
        search_filter = {"access_level": allowed_levels[0]}
    else:
        search_filter = {"access_level": {"$in": allowed_levels}}
        
    retriever = db.as_retriever(search_kwargs={"k": k, "filter": search_filter})
    return retriever.invoke(query)

def generate_answer(query: str, role: str = "all_employees", k: int = 3):
    """
    Generates a context-grounded response to the user's query while enforcing RBAC restrictions.
    """
    normalized_role = role.strip().lower().replace(" ", "_")
    allowed_levels = get_allowed_access_levels(role)
    tier_info = "ALL (UNRESTRICTED ACCESS)" if normalized_role in ["admin", "superuser"] else allowed_levels
    print(f"\n[RBAC Generation] Role: '{role}' | Allowed Categories: {tier_info}")
    
    # 1. Retrieve authorized documents only
    relevant_documents = retrieve_documents(query, role=role, k=k)
    print(f"[RBAC Generation] Retrieved {len(relevant_documents)} authorized chunk(s).")
    
    # If no relevant authorized documents exist in vector search
    if not relevant_documents:
        return "I don't have access to this information or it is not available based on your role permissions."

    # 2. Format documents into context
    formatted_docs = []
    for doc in relevant_documents:
        dept = doc.metadata.get("access_level", "unknown").upper()
        fname = doc.metadata.get("filename", "unknown")
        formatted_docs.append(f"[{dept} - {fname}]\n{doc.page_content}")
        
    context_text = "\n\n---\n\n".join(formatted_docs)

    # 3. System prompt enforcing role boundary compliance and truthful answering
    system_prompt = (
        f"You are an enterprise AI assistant for DMS Solutions with strict Role-Based Access Control (RBAC).\n"
        f"The current user's role is: '{role.upper()}'.\n"
        f"Their permitted document categories are: {tier_info}.\n\n"
        "Strict Guidelines:\n"
        "1. Answer the user's question using ONLY the provided authorized documents context.\n"
        "2. Provide a direct, factual, and concise answer.\n"
        "3. If the provided context does NOT contain enough information to answer the question, reply exactly with: "
        "'I don't have access to this information or it is not available based on your role permissions.'\n"
        "4. Never hallucinate or disclose information that is not present in the provided context documents."
    )

    user_prompt = f"Context Documents:\n{context_text}\n\nUser Question: {query}"

    # 4. Generate response using Gemini
    model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = model.invoke(messages)
    return response.text.strip()

def run_rbac_generation_tests():
    """Runs a quick automated test across roles including admin to verify end-to-end RBAC generation."""
    print("=" * 70)
    print("      RUNNING RBAC GENERATION EVALUATION TESTS")
    print("=" * 70)
    
    test_queries = [
        # Admin querying salary bands (Authorized - Full Access)
        ("admin", "What are the salary bands and RSU grants for Level 3 Senior Engineers?"),
        # Admin querying confidential M&A acquisition (Authorized - Full Access)
        ("admin", "What is Project Titan and what is the target acquisition price?"),
        # Engineering querying technical specs (Authorized)
        ("engineering", "What database and streaming technology do we use in our system architecture?"),
        # Engineering querying confidential HR salary bands (Blocked)
        ("engineering", "What are the salary bands and RSU grants for Level 3 Senior Engineers?"),
        # HR querying salary bands (Authorized)
        ("hr", "What are the salary bands and RSU grants for Level 3 Senior Engineers?"),
        # All employees querying holidays (Authorized)
        ("all_employees", "What are the official public holidays for 2026?")
    ]
    
    for i, (role, query) in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Role: {role}")
        print(f"Query: {query}")
        ans = generate_answer(query, role=role)
        print(f"Response:\n{ans}")
        print("-" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RBAC Generation Pipeline")
    parser.add_argument("--test", action="store_true", help="Run automated RBAC test suite")
    parser.add_argument("--role", default=None, help="Employee role (all_employees, engineering, hr, finance, management, admin)")
    args = parser.parse_args()

    if args.test:
        run_rbac_generation_tests()
    else:
        role = args.role
        if not role:
            print("=" * 60)
            print("   DMS Solutions - RBAC Single-Turn Q&A Pipeline")
            print("=" * 60)
            print("Select an Employee Role:")
            print("  1. all_employees (Public handbook & general benefits)")
            print("  2. engineering   (Public + System Architecture & CI/CD)")
            print("  3. hr            (Public + Salary Bands & PIP policies)")
            print("  4. finance       (Public + Q3 Budgets & Vendor spend)")
            print("  5. management    (All Departmental + M&A Strategy)")
            print("  6. admin         (FULL UNRESTRICTED ACCESS to all documents)")
            print("=" * 60)
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

        normalized = role.strip().lower().replace(" ", "_")
        tier_display = "ALL DOCUMENTS (UNRESTRICTED)" if normalized in ["admin", "superuser"] else get_allowed_access_levels(role)
        
        print(f"\nActive Role set to: '{role}' (Access: {tier_display})")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                user_query = input(f"[{role}] Enter your question: ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ["exit", "quit"]:
                    break
                
                answer = generate_answer(user_query, role=role)
                print(f"\nAnswer:\n{answer}\n")
                print("-" * 60)
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}\n")
