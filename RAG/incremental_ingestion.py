import os
import sys
import time
import json
import hashlib
import shutil
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 (prevents crashes when printing Unicode on Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Calculate paths relative to current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load environment variables
load_dotenv(os.path.join(parent_dir, ".env"))

MANIFEST_FILENAME = "ingestion_manifest.json"

def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file to detect content modifications."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def sanitize_id_prefix(rel_path: str) -> str:
    """Generates a clean, deterministic ID prefix for document chunks."""
    normalized = rel_path.replace("\\", "/").strip("/")
    clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in normalized)
    return clean

def load_manifest(manifest_path: str) -> dict:
    """Loads existing ingestion manifest or returns a fresh structure."""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Manifest] Warning: Failed to read existing manifest ({e}). Creating new.")
    return {
        "version": "1.0",
        "last_sync": None,
        "files": {}
    }

def save_manifest(manifest_path: str, manifest: dict):
    """Saves the updated manifest to disk."""
    manifest["last_sync"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def load_single_document(filepath: str, department: str, rel_path: str):
    """Loads a single document and attaches RBAC metadata."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".docx":
        loader = Docx2txtLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        return []

    docs = loader.load()
    filename = os.path.basename(filepath)
    for doc in docs:
        doc.metadata["department"] = department
        doc.metadata["access_level"] = department
        doc.metadata["filename"] = filename
        doc.metadata["rel_path"] = rel_path
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Splits loaded documents into smaller overlapping chunks while retaining metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if c.page_content.strip()]

def sync_incremental_ingestion(docs_path=None, persist_directory=None, force_reset=False):
    """
    Performs incremental hash-based synchronization between data/ and ChromaDB.
    - Unchanged files: Skipped (saves embedding API calls and time).
    - New / Modified files: Embedded and upserted with deterministic chunk IDs.
    - Deleted files: Obsolete vector chunk IDs pruned from ChromaDB.
    """
    if docs_path is None:
        docs_path = os.path.join(parent_dir, "data")
    if persist_directory is None:
        persist_directory = os.path.join(parent_dir, "chroma_db")

    manifest_path = os.path.join(persist_directory, MANIFEST_FILENAME)

    print("=" * 75)
    print("      INCREMENTAL DOCUMENT INGESTION & RBAC SYNC ENGINE")
    print("=" * 75)
    print(f"Data Directory:    {docs_path}")
    print(f"Chroma Directory:  {persist_directory}")
    print(f"Manifest File:     {manifest_path}")

    # Handle force-reset
    if force_reset:
        print("\n[Force-Reset] Wiping existing vector store and manifest for full re-index...")
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory, ignore_errors=True)
            time.sleep(1)
        manifest = {"version": "1.0", "last_sync": None, "files": {}}
    else:
        manifest = load_manifest(manifest_path)

    # Initialize Embedding Model & Chroma Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 1. Scan current filesystem
    supported_exts = {".pdf", ".docx", ".txt"}
    current_files = {}

    abs_docs_path = os.path.abspath(docs_path)
    for root, _, files in os.walk(docs_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_exts:
                full_path = os.path.abspath(os.path.join(root, file))
                rel_path = os.path.relpath(full_path, abs_docs_path).replace("\\", "/")
                parts = rel_path.split("/")
                department = parts[0].lower() if len(parts) > 1 else "public"
                sha256 = compute_file_hash(full_path)
                mtime = os.path.getmtime(full_path)
                size_bytes = os.path.getsize(full_path)

                current_files[rel_path] = {
                    "full_path": full_path,
                    "rel_path": rel_path,
                    "department": department,
                    "sha256": sha256,
                    "mtime": mtime,
                    "size_bytes": size_bytes
                }

    stored_files = manifest.get("files", {})

    # 2. Identify Added, Modified, Unchanged, and Deleted files
    added_files = []
    modified_files = []
    unchanged_files = []
    deleted_files = []

    for rel_path, info in current_files.items():
        if rel_path not in stored_files:
            added_files.append(info)
        elif stored_files[rel_path].get("sha256") != info["sha256"]:
            modified_files.append(info)
        else:
            unchanged_files.append(info)

    for rel_path in stored_files.keys():
        if rel_path not in current_files:
            deleted_files.append(rel_path)

    print(f"\nScan Summary:")
    print(f"  • Unchanged (Skipped):  {len(unchanged_files)}")
    print(f"  • Newly Added:          {len(added_files)}")
    print(f"  • Modified (Updated):   {len(modified_files)}")
    print(f"  • Deleted (Pruned):     {len(deleted_files)}")
    print("-" * 75)

    # 3. Handle Deleted Files (Prune from ChromaDB)
    if deleted_files:
        print("\n[Pruning] Removing deleted files from vector index...")
        for rel_path in deleted_files:
            old_chunk_ids = stored_files[rel_path].get("chunk_ids", [])
            if old_chunk_ids:
                try:
                    vector_store.delete(ids=old_chunk_ids)
                    print(f"  [-] Pruned {len(old_chunk_ids)} chunk(s) for deleted file: {rel_path}")
                except Exception as e:
                    print(f"  [!] Warning pruning chunks for {rel_path}: {e}")
            del stored_files[rel_path]

    # 4. Process Modified Files (Delete old chunks first)
    files_to_index = added_files + modified_files
    if modified_files:
        print("\n[Updating] Purging stale vector chunks for modified files...")
        for info in modified_files:
            rel_path = info["rel_path"]
            old_chunk_ids = stored_files[rel_path].get("chunk_ids", [])
            if old_chunk_ids:
                try:
                    vector_store.delete(ids=old_chunk_ids)
                    print(f"  [-] Removed {len(old_chunk_ids)} previous chunk(s) for: {rel_path}")
                except Exception as e:
                    print(f"  [!] Warning removing old chunks for {rel_path}: {e}")

    # 5. Ingest & Embed New and Modified Files
    if not files_to_index:
        print("\n[Up to Date] All documents match manifest hashes. No embedding calls needed!")
        save_manifest(manifest_path, manifest)
        print("=" * 75 + "\n")
        return vector_store

    print(f"\n[Embedding] Indexing {len(files_to_index)} new/modified document(s)...")
    
    total_new_chunks = 0
    batch_size = 5

    for idx, info in enumerate(files_to_index, 1):
        rel_path = info["rel_path"]
        full_path = info["full_path"]
        dept = info["department"]

        print(f"\n({idx}/{len(files_to_index)}) Processing: {rel_path} [{dept.upper()}]")
        docs = load_single_document(full_path, department=dept, rel_path=rel_path)
        if not docs:
            print(f"  [!] No text extracted from {rel_path}")
            continue

        chunks = split_documents(docs)
        if not chunks:
            print(f"  [!] No valid chunks generated for {rel_path}")
            continue

        id_prefix = sanitize_id_prefix(rel_path)
        chunk_ids = [f"{id_prefix}_chk_{c_idx}" for c_idx in range(len(chunks))]

        print(f"  -> Generated {len(chunks)} chunk(s). Uploading in batches to ChromaDB...")

        # Batch embed into Chroma
        for b_start in range(0, len(chunks), batch_size):
            b_chunks = chunks[b_start : b_start + batch_size]
            b_ids = chunk_ids[b_start : b_start + batch_size]
            vector_store.add_documents(documents=b_chunks, ids=b_ids)
            
            # Pacing to avoid Gemini API rate limits
            if b_start + batch_size < len(chunks):
                time.sleep(6)

        # Update manifest record
        stored_files[rel_path] = {
            "sha256": info["sha256"],
            "mtime": info["mtime"],
            "size_bytes": info["size_bytes"],
            "department": dept,
            "access_level": dept,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids
        }
        total_new_chunks += len(chunks)

    manifest["files"] = stored_files
    save_manifest(manifest_path, manifest)

    print(f"\n[Success] Incremental sync complete! Indexed {total_new_chunks} chunk(s) across {len(files_to_index)} file(s).")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 75 + "\n")
    return vector_store

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Incremental Document Ingestion Engine")
    parser.add_argument("--force-reset", action="store_true", help="Force complete rebuild of ChromaDB and manifest")
    args = parser.parse_args()

    sync_incremental_ingestion(force_reset=args.force_reset)
