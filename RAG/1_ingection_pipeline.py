import os
import sys
import time
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Configure standard output to use UTF-8 (prevents crashes when printing Unicode on Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Calculate dynamic paths relative to the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load the environment variables from .env at the project root
load_dotenv(os.path.join(parent_dir, ".env"))

# Loading the files and attaching Role-Based Access Control (RBAC) metadata
def load_documents(docs_path="./data"):
    """Loads documents from the specified directory and tags each with department/access_level metadata."""
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified path {docs_path} does not exist.")
    
    loaders = {
        "**/*.txt": TextLoader,
        "**/*.pdf": PyPDFLoader,
        "**/*.docx": Docx2txtLoader,
    }
    
    documents = []
    for glob, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(docs_path, glob=glob, loader_cls=loader_cls)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"Loaded {len(loaded_docs)} documents matching '{glob}'.")
        except Exception as e:
            print(f"Error loading '{glob}' files: {e}")

    if not documents:
        raise ValueError(f"No valid text, PDF, or Word files found in the directory {docs_path}.")
    
    abs_docs_path = os.path.abspath(docs_path)
    
    # Tag each document with RBAC metadata based on folder structure
    dept_counts = {}
    for doc in documents:
        source_path = doc.metadata.get('source', '')
        if source_path:
            norm_source = os.path.normpath(source_path)
            rel_source = os.path.relpath(norm_source, abs_docs_path)
            parts = rel_source.split(os.sep)
            
            # The top-level subdirectory inside data/ defines the department / access category
            # e.g., 'public', 'engineering', 'hr', 'finance', 'management'
            department = parts[0].lower() if len(parts) > 1 else "public"
            filename = os.path.basename(norm_source)
            
            # Attach structured RBAC metadata
            doc.metadata["department"] = department
            doc.metadata["access_level"] = department
            doc.metadata["filename"] = filename
            
            dept_counts[department] = dept_counts.get(department, 0) + 1

    print(f"\nTotal documents loaded: {len(documents)}")
    print(f"Documents by department / access_level: {dept_counts}\n")
    
    for i, doc in enumerate(documents[:3]):
        print(f"--- Document {i+1} Sample ---")
        print(f" Source: {doc.metadata.get('source')}")
        print(f" Department / Access Level: {doc.metadata.get('access_level')}")
        print(f" Filename: {doc.metadata.get('filename')}")
        print(f" Content Preview: {doc.page_content[:120].strip()}...\n")

    return documents

# Chunking the files
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks while preserving RBAC metadata."""
    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    chunk_dept_counts = {}
    for chunk in chunks:
        dept = chunk.metadata.get("access_level", "unknown")
        chunk_dept_counts[dept] = chunk_dept_counts.get(dept, 0) + 1

    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunks by department / access_level: {chunk_dept_counts}\n")

    if chunks:
        print("--- Sample Chunk with RBAC Metadata ---")
        sample = chunks[0]
        print(f" Source: {sample.metadata.get('source')}")
        print(f" Access Level: {sample.metadata.get('access_level')}")
        print(f" Department: {sample.metadata.get('department')}")
        print(f" Content Preview: {sample.page_content[:150].strip()}...")
        print("-" * 50 + "\n")

    return chunks

# Embedding and storing in a vector database
def create_vector_store(chunks, persist_directory=None, reset_db=True):
    """Create a persistent vector store using Chroma and Google Generative AI embeddings."""
    if persist_directory is None:
        persist_directory = os.path.join(parent_dir, "chroma_db")
        
    if reset_db and os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory} for fresh RBAC indexing...")
        shutil.rmtree(persist_directory, ignore_errors=True)
        time.sleep(1)

    print(f"Creating embeddings and storing in ChromaDB at {persist_directory}...")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Create vector store with ChromaDB
    print(f"--- Initializing vector store with ChromaDB ---")
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Batch processing to respect rate limits
    batch_size = 5
    total_chunks = len(chunks)
    print(f"Total chunks to process: {total_chunks}")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num} of {total_batches}...")
        vector_store.add_documents(batch)

        # Sleep to respect rate limits
        if i + batch_size < total_chunks:
            time.sleep(10)

    print(f"\nVector store created and persisted at {persist_directory}")
    print(f"--- Vector Store created successfully with RBAC metadata! ---")
    return vector_store

# Main function
def main():
    print("=== Starting the RBAC Ingestion Pipeline ===")

    docs_path = os.path.join(parent_dir, "data")
    persist_db_path = os.path.join(parent_dir, "chroma_db")

    # 1. Loading the files with RBAC metadata
    documents = load_documents(docs_path=docs_path)  

    # 2. Chunking the files (inherits RBAC metadata)
    chunks = split_documents(documents)

    # 3. Embedding and storing in ChromaDB with metadata
    vector_store = create_vector_store(chunks, persist_directory=persist_db_path, reset_db=True)

if __name__ == "__main__":
    main()