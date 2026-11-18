import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_generic_file(
    file_path: str, 
    chunk_size_tokens: int = 800, 
    chunk_overlap_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits a generic non-Python file (js, ts, md, json, etc.) using LangChain's 
    RecursiveCharacterTextSplitter. Tracks approximate line ranges by searching 
    for chunk content in the original source code.
    
    Returns a list of chunk dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    if not content.strip():
        return []

    # Map tokens to approximate character sizes (1 token ~= 4 characters)
    char_size = chunk_size_tokens * 4
    char_overlap = chunk_overlap_tokens * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        keep_separator=True
    )
    
    splits = splitter.split_text(content)
    
    chunks = []
    last_start_idx = 0
    
    for split in splits:
        if not split.strip():
            continue
            
        # Locate where this split starts in the original content.
        # Since splits are sequential and can overlap, we start searching from last_start_idx.
        idx = content.find(split, last_start_idx)
        
        # Fallback in case spacing details differ slightly:
        if idx == -1:
            # Let's search the whole document or search for a prefix
            prefix = split[:min(100, len(split))].strip()
            idx = content.find(prefix, last_start_idx)
            if idx == -1:
                idx = content.find(prefix) # absolute fallback
                
        if idx == -1:
            # Absolute fallback if still not found
            idx = last_start_idx

        # Calculate line ranges
        start_line = content[:idx].count("\n") + 1
        end_line = content[:idx + len(split)].count("\n") + 1
        
        # Update last_start_idx for next chunk. 
        # Since there is overlap, the next chunk might start slightly before the end of the current chunk,
        # but it will definitely start at or after the start of this chunk.
        last_start_idx = idx
        
        chunks.append({
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_type": "generic",
            "parent_class": "",
            "name": "",
            "content": split
        })
        
    return chunks
