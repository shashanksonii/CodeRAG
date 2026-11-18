import os
import json
import hashlib
import pathspec
from typing import Dict, List, Any, Callable, Tuple
from langchain_core.documents import Document
from vectorstore.chroma_client import get_chroma_vectorstore, delete_file_from_vectorstore
from indexer.ast_chunker import chunk_python_file
from indexer.generic_chunker import chunk_generic_file

TEXT_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.md', '.json', 
    '.yaml', '.yml', '.txt', '.sh', '.ini', '.cfg', '.toml', '.go', '.java', 
    '.c', '.cpp', '.h', '.rs', '.sql', '.properties', '.xml', '.gradle', '.bat'
}

def load_hashes(persist_directory: str) -> Dict[str, str]:
    hash_file = os.path.join(persist_directory, "file_hashes.json")
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load file_hashes.json: {e}")
    return {}

def save_hashes(persist_directory: str, hashes: Dict[str, str]):
    os.makedirs(persist_directory, exist_ok=True)
    hash_file = os.path.join(persist_directory, "file_hashes.json")
    try:
        with open(hash_file, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save file_hashes.json: {e}")

def get_gitignore_spec(root_dir: str) -> pathspec.PathSpec:
    gitignore_path = os.path.join(root_dir, ".gitignore")
    lines = []
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Failed to read .gitignore at {gitignore_path}: {e}")
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)

def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}")
        return ""

class CodebaseIndexer:
    def __init__(self, codebase_root: str, persist_directory: str = "./chroma_db", collection_name: str = "codebase_rag"):
        self.codebase_root = os.path.abspath(codebase_root)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = get_chroma_vectorstore(persist_directory, collection_name)
        self.hashes = load_hashes(persist_directory)

    def walk_codebase(self) -> List[Tuple[str, str]]:
        """
        Walks the codebase recursively, filtering files by gitignore and common junk directories.
        Returns a list of tuples: (absolute_path, relative_path)
        """
        spec = get_gitignore_spec(self.codebase_root)
        junk_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"}
        files_to_index = []
        
        for dirpath, dirnames, filenames in os.walk(self.codebase_root):
            # Exclude junk directories in-place to stop os.walk traversing them
            dirnames[:] = [d for d in dirnames if d not in junk_dirs]
            
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, self.codebase_root)
                
                # Check gitignore
                if spec.match_file(rel_path):
                    continue
                    
                # Filter out files without text extensions or non-text files
                _, ext = os.path.splitext(filename)
                if ext.lower() not in TEXT_EXTENSIONS:
                    continue
                    
                files_to_index.append((full_path, rel_path))
                
        return files_to_index

    def index_codebase(self, progress_callback: Callable[[Dict[str, Any]], None] = None) -> Dict[str, Any]:
        """
        Indexes the codebase incrementally. 
        - progress_callback: a callable receiving status updates (dict)
        Returns a stats dict.
        """
        stats = {
            "scanned": 0,
            "indexed": 0,
            "deleted": 0,
            "skipped": 0,
            "total_chunks": 0
        }
        
        if progress_callback:
            progress_callback({"status": "scanning", "message": "Scanning codebase files..."})
            
        files_to_index = self.walk_codebase()
        stats["scanned"] = len(files_to_index)
        
        scanned_rel_paths = set()
        to_process = []
        
        # Determine which files are new/modified
        for full_path, rel_path in files_to_index:
            scanned_rel_paths.add(rel_path)
            file_hash = compute_file_hash(full_path)
            
            if rel_path not in self.hashes or self.hashes[rel_path] != file_hash:
                to_process.append((full_path, rel_path, file_hash))
            else:
                stats["skipped"] += 1
                
        # Clean up deleted files from the vector store
        deleted_files = []
        for rel_path in list(self.hashes.keys()):
            if rel_path not in scanned_rel_paths:
                deleted_files.append(rel_path)
                
        if deleted_files:
            if progress_callback:
                progress_callback({"status": "deleting", "message": f"Cleaning up {len(deleted_files)} deleted files from index..."})
            for rel_path in deleted_files:
                delete_file_from_vectorstore(self.vectorstore, rel_path)
                del self.hashes[rel_path]
                stats["deleted"] += 1
            save_hashes(self.persist_directory, self.hashes)

        total_to_process = len(to_process)
        if total_to_process == 0:
            if progress_callback:
                progress_callback({"status": "complete", "message": "Indexing complete. No files changed.", "stats": stats})
            return stats

        # Process new and modified files
        for idx, (full_path, rel_path, file_hash) in enumerate(to_process):
            if progress_callback:
                progress_callback({
                    "status": "indexing", 
                    "message": f"Indexing file {idx+1}/{total_to_process}: {rel_path}",
                    "current": idx + 1,
                    "total": total_to_process,
                    "file": rel_path
                })
                
            # 1. Chunk file
            _, ext = os.path.splitext(rel_path)
            chunks = []
            try:
                if ext.lower() == '.py':
                    chunks = chunk_python_file(full_path)
                else:
                    chunks = chunk_generic_file(full_path)
            except Exception as e:
                print(f"Error chunking file {rel_path}: {e}")
                continue
                
            # 2. Delete any old chunks for this file
            delete_file_from_vectorstore(self.vectorstore, rel_path)
            
            # 3. Add to Chroma
            if chunks:
                documents = []
                ids = []
                for c in chunks:
                    # Stable chunk_id
                    chunk_id = f"{rel_path}:{c['start_line']}"
                    
                    doc = Document(
                        page_content=c["content"],
                        metadata={
                            "file_path": rel_path,
                            "start_line": c["start_line"],
                            "end_line": c["end_line"],
                            "chunk_type": c["chunk_type"],
                            "parent_class": c["parent_class"],
                            "name": c["name"]
                        }
                    )
                    documents.append(doc)
                    ids.append(chunk_id)
                
                try:
                    self.vectorstore.add_documents(documents=documents, ids=ids)
                    stats["total_chunks"] += len(chunks)
                except Exception as e:
                    print(f"Error adding documents for {rel_path} to vector store: {e}")
                    # If it raises a RuntimeError about Ollama connection, we let it bubble up
                    # so that we can show instructions in Streamlit.
                    raise e
                    
            # 4. Save hash
            self.hashes[rel_path] = file_hash
            save_hashes(self.persist_directory, self.hashes)
            stats["indexed"] += 1
            
        if progress_callback:
            progress_callback({"status": "complete", "message": "Indexing complete.", "stats": stats})
            
        return stats
