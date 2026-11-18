import ast
import os
from typing import List, Dict, Any

def add_module_level_chunks(file_path: str, start: int, end: int, lines: List[str], chunks: List[Dict[str, Any]], max_lines: int = 60):
    """
    Splits a range of module-level lines into chunks of at most max_lines lines.
    This prevents single massive module-level chunks from exceeding LLM/embedding token limits.
    """
    total_lines = len(lines)
    current_start = start
    while current_start <= end:
        current_end = min(current_start + max_lines - 1, end)
        s_idx = max(0, current_start - 1)
        e_idx = min(total_lines, current_end)
        block_content = "\n".join(lines[s_idx:e_idx])
        
        if block_content.strip():
            chunks.append({
                "file_path": file_path,
                "start_line": current_start,
                "end_line": current_end,
                "chunk_type": "module_level",
                "parent_class": "",
                "name": "",
                "content": block_content
            })
        current_start = current_end + 1

def chunk_python_file(file_path: str, threshold: int = 40, max_module_lines: int = 60) -> List[Dict[str, Any]]:
    """
    Parses a Python file using AST and splits it into logical chunks:
    - Top-level functions -> function chunks
    - Classes -> class chunks (sub-chunked if total class body > threshold)
    - Module-level code -> grouped into module-level chunks (split if > max_module_lines lines)
    
    Each chunk is represented as a dictionary:
    {
        "file_path": str,
        "start_line": int,
        "end_line": int,
        "chunk_type": str,  # "function" | "class" | "method" | "module_level"
        "parent_class": str,  # Name of parent class or ""
        "name": str,  # Name of the function/class/method or ""
        "content": str
    }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    lines = content.splitlines()
    total_lines = len(lines)
    
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as se:
        # If there's a syntax error, we cannot parse the AST.
        # Fallback: split the file into chunks of max_module_lines lines.
        print(f"Syntax error parsing AST for {file_path}: {se}. Falling back to line-based splitting.")
        fallback_chunks = []
        add_module_level_chunks(file_path, 1, total_lines, lines, fallback_chunks, max_lines=max_module_lines)
        return fallback_chunks

    chunks = []
    
    # We will keep track of which lines are covered by functions/classes
    covered_lines = [False] * (total_lines + 1)
    
    def mark_covered(start: int, end: int):
        for i in range(max(1, start), min(total_lines + 1, end + 1)):
            covered_lines[i] = True

    # Helper to slice lines (1-indexed bounds)
    def get_lines_content(start: int, end: int) -> str:
        s_idx = max(0, start - 1)
        e_idx = min(total_lines, end)
        return "\n".join(lines[s_idx:e_idx])

    # Iterate over top-level statements
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level function
            start = node.lineno
            end = node.end_lineno
            name = node.name
            
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "chunk_type": "function",
                "parent_class": "",
                "name": name,
                "content": get_lines_content(start, end)
            })
            mark_covered(start, end)
            
        elif isinstance(node, ast.ClassDef):
            # Class definition
            class_start = node.lineno
            class_end = node.end_lineno
            class_name = node.name
            
            class_height = class_end - class_start + 1
            
            # Find direct methods in this class
            methods = [
                child for child in node.body 
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            
            if class_height <= threshold or not methods:
                # Small class or class with no methods, keep as single chunk
                chunks.append({
                    "file_path": file_path,
                    "start_line": class_start,
                    "end_line": class_end,
                    "chunk_type": "class",
                    "parent_class": "",
                    "name": class_name,
                    "content": get_lines_content(class_start, class_end)
                })
                mark_covered(class_start, class_end)
            else:
                # Large class, sub-chunk it
                # 1. Class Header chunk: from class start line to the line before the first method
                first_method_start = methods[0].lineno
                header_end = max(class_start, first_method_start - 1)
                
                chunks.append({
                    "file_path": file_path,
                    "start_line": class_start,
                    "end_line": header_end,
                    "chunk_type": "class",
                    "parent_class": "",
                    "name": class_name,
                    "content": get_lines_content(class_start, header_end)
                })
                mark_covered(class_start, header_end)
                
                # 2. Method chunks: each method is its own chunk
                for m in methods:
                    m_start = m.lineno
                    m_end = m.end_lineno
                    
                    chunks.append({
                        "file_path": file_path,
                        "start_line": m_start,
                        "end_line": m_end,
                        "chunk_type": "method",
                        "parent_class": class_name,
                        "name": m.name,
                        "content": get_lines_content(m_start, m_end)
                    })
                    mark_covered(m_start, m_end)

    # Identify and group module-level code
    # Find contiguous ranges of uncovered lines
    start_uncovered = None
    
    for line_num in range(1, total_lines + 1):
        if not covered_lines[line_num]:
            if start_uncovered is None:
                start_uncovered = line_num
        else:
            if start_uncovered is not None:
                # End of a contiguous uncovered block
                end_uncovered = line_num - 1
                add_module_level_chunks(
                    file_path=file_path, 
                    start=start_uncovered, 
                    end=end_uncovered, 
                    lines=lines, 
                    chunks=chunks, 
                    max_lines=max_module_lines
                )
                start_uncovered = None
                
    # Handle block at the end of the file
    if start_uncovered is not None:
        add_module_level_chunks(
            file_path=file_path, 
            start=start_uncovered, 
            end=total_lines, 
            lines=lines, 
            chunks=chunks, 
            max_lines=max_module_lines
        )
            
    # Sort chunks by start line to keep them ordered
    chunks.sort(key=lambda x: x["start_line"])
    return chunks
