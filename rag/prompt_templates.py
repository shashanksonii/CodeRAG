import re
import os
from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a helpful software engineering assistant designed to answer questions about a codebase.
You are given a context of code snippets retrieved from the codebase, each labeled with its file path and line ranges.

Strict Guidelines:
1. ONLY answer using the provided code snippets and context.
2. If the context is insufficient to answer the question confidently, state this explicitly and specify what additional files, functions, or information you would need to answer. Do NOT guess, speculate, or fabricate functions, variables, classes, or file paths that are not present in the context.
3. For every claim, code block, or explanation you provide, cite the source file and line numbers using the format: [Source: file_path, lines X-Y] or [file_path:X-Y].
4. Be precise, concise, and technical in your answers.
"""

USER_PROMPT_TEMPLATE = """Here is the retrieved context from the codebase:

{context}

Question: {question}

Answer:"""

def format_context(documents: List[Any]) -> str:
    """
    Formats retrieved LangChain documents into a readable string block for the LLM.
    """
    formatted = []
    for doc in documents:
        meta = doc.metadata
        formatted.append(
            f"--- FILE: {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']}) ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)

def extract_citations(response_text: str, retrieved_docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Parses the LLM's response for citations, cross-checks them against actually retrieved documents,
    and returns a list of verified citations.
    """
    # Pattern to match [Source: path/to/file.py, lines X-Y] or [path/to/file.py:X-Y]
    pattern = r"\[(?:Source:\s*)?([a-zA-Z0-9_\-\./\\]+)(?::|,\s*lines?\s*)(\d+)(?:\s*-\s*(\d+))?\]"
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    
    # Pattern to match general filename in brackets e.g. [ast_chunker.py]
    simple_file_pattern = r"\[([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)\]"
    simple_matches = re.findall(simple_file_pattern, response_text)
    
    citations = []
    seen = set()
    
    def find_matching_doc(cited_file: str) -> Any:
        cited_clean = cited_file.replace("\\", "/").strip("/")
        for doc in retrieved_docs:
            doc_path = doc.metadata["file_path"].replace("\\", "/")
            # Match exact relative path, path suffix, or file basename
            if (doc_path == cited_clean or 
                doc_path.endswith("/" + cited_clean) or 
                os.path.basename(doc_path) == os.path.basename(cited_clean)):
                return doc
        return None

    # 1. Process matches with line numbers
    for cited_file, start_str, end_str in matches:
        matched_doc = find_matching_doc(cited_file)
        if matched_doc:
            file_path = matched_doc.metadata["file_path"]
            start_line = int(start_str)
            end_line = int(end_str) if end_str else start_line
            
            citation_key = (file_path, start_line, end_line)
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append({
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "doc": matched_doc
                })
                
    # 2. Process simple file matches without line numbers
    for cited_file in simple_matches:
        matched_doc = find_matching_doc(cited_file)
        if matched_doc:
            file_path = matched_doc.metadata["file_path"]
            start_line = matched_doc.metadata["start_line"]
            end_line = matched_doc.metadata["end_line"]
            
            citation_key = (file_path, start_line, end_line)
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append({
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "doc": matched_doc
                })
                
    # 3. Fallback: if no citations are parsed from the response, but documents were retrieved,
    # list all retrieved documents as references.
    if not citations and retrieved_docs:
        for doc in retrieved_docs:
            meta = doc.metadata
            citation_key = (meta["file_path"], meta["start_line"], meta["end_line"])
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append({
                    "file_path": meta["file_path"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "doc": doc
                })
                
    return citations
