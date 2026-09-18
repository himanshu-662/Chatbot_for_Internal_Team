import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

# Step 1: Contextualize the user question based on chat history
def contextualize_question(query, chat_history):
    """Reformulates the latest question to be standalone if it references past conversation context."""
    if not chat_history:
        return query  # No history, so no need to contextualize

    contextualize_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed "
        "and otherwise return it exactly as is."
    )

    messages = [SystemMessage(content=contextualize_system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=query))

    model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    response = model.invoke(messages)
    standalone_query = response.text.strip()
    
    return standalone_query

# Step 2: Retrieve relevant documents filtered strictly by role
def retrieve_documents(query, role="all_employees", k=3):
    """Searches for and returns relevant documents based on the query and role permissions."""
    normalized_role = role.strip().lower().replace(" ", "_")
    
    # Admin has complete unrestricted access across all documents
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

# Step 3: Generate the final answer using retrieved context, history, and role constraints
def generate_answer(query, relevant_documents, chat_history, role="all_employees"):
    """Generates an answer using the retrieved documents context and full chat history within role constraints."""
    normalized_role = role.strip().lower().replace(" ", "_")
    allowed_levels = get_allowed_access_levels(role)
    tier_info = "ALL (UNRESTRICTED ACCESS)" if normalized_role in ["admin", "superuser"] else allowed_levels
    
    if not relevant_documents:
        return "I don't have access to this information or it is not available based on your role permissions."

    # Format retrieved document chunks with metadata tags
    formatted_docs = []
    for doc in relevant_documents:
        dept = doc.metadata.get("access_level", "unknown").upper()
        fname = doc.metadata.get("filename", "unknown")
        formatted_docs.append(f"[{dept} - {fname}]\n{doc.page_content}")
        
    context_text = "\n\n---\n\n".join(formatted_docs)

    system_prompt = (
        f"You are an enterprise AI assistant for DMS Solutions with strict Role-Based Access Control (RBAC).\n"
        f"Current User Role: '{role.upper()}'.\n"
        f"Authorized Information Tiers: {tier_info}.\n\n"
        "Strict Guidelines:\n"
        "1. Answer the user's question using ONLY the provided context documents.\n"
        "2. Provide direct, factual, and concise responses.\n"
        "3. If the context does not contain the answer, reply exactly with: "
        "'I don't have access to this information or it is not available based on your role permissions.'\n"
        "4. Under no circumstances should you disclose or guess details outside the authorized documents provided."
    )

    user_content = f"Context Documents:\n{context_text}\n\nUser Question: {query}"

    # Build the full conversation thread for the LLM
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_content))

    model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    response = model.invoke(messages)
    return response.text.strip()

def print_help():
    print("""
Available Commands:
  /role <name>   : Switch active role (e.g. /role admin, /role engineering, /role hr, /role finance, /role management, /role all_employees)
  /whoami        : Show current active role and permitted document tiers
  /clear         : Clear conversation history
  /help          : Show this help message
  exit / quit    : Exit the chatbot
""")

# Main loop to interact with RAG using chat history and dynamic role switching
def main():
    print("=" * 70)
    print("   DMS SOLUTIONS - ROLE-BASED ACCESS CONTROL (RBAC) CHATBOT")
    print("=" * 70)
    print("Roles available: all_employees, engineering, hr, finance, management, admin\n")
    
    current_role = "all_employees"
    print(f"Default role set to: '{current_role}'")
    print("Type '/role admin' (or any other role) to switch roles. Type 'exit' to quit.\n")
    
    chat_history = []  # Initialize conversation history

    while True:
        try:
            normalized = current_role.strip().lower().replace(" ", "_")
            allowed = "ALL (UNRESTRICTED)" if normalized in ["admin", "superuser"] else get_allowed_access_levels(current_role)
            
            user_input = input(f"[{current_role.upper()}] > ").strip()
            
            if not user_input:
                continue
            
            # Handle exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting session. Goodbye!")
                break
                
            # Handle slash commands
            if user_input.startswith("/role"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    new_role = parts[1].strip().lower()
                    if new_role in ROLE_PERMISSIONS:
                        current_role = new_role
                        norm_new = new_role.replace(" ", "_")
                        tiers = "ALL DOCUMENTS (UNRESTRICTED)" if norm_new in ["admin", "superuser"] else get_allowed_access_levels(current_role)
                        print(f"-> Switched active role to: '{current_role}' (Authorized tiers: {tiers})")
                    else:
                        print(f"-> Unknown role '{new_role}'. Valid roles: {list(ROLE_PERMISSIONS.keys())}")
                else:
                    print("-> Usage: /role <all_employees | engineering | hr | finance | management | admin>")
                continue
                
            if user_input == "/whoami":
                print(f"-> Current Role: '{current_role}'")
                print(f"-> Authorized Categories: {allowed}")
                continue
                
            if user_input == "/clear":
                chat_history = []
                print("-> Chat history cleared.")
                continue
                
            if user_input == "/help":
                print_help()
                continue
            
            # 1. Contextualize query with history
            standalone_query = contextualize_question(user_input, chat_history)
            
            # 2. Retrieve documents filtered by current role
            relevant_docs = retrieve_documents(standalone_query, role=current_role)
            
            # 3. Generate answer
            answer = generate_answer(user_input, relevant_docs, chat_history, role=current_role)
            
            print(f"\nAssistant: {answer}\n")
            print("-" * 60)
            
            # 4. Update Chat History
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            
        except KeyboardInterrupt:
            print("\nExiting session. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()
