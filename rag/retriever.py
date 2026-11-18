from typing import List, Tuple, Any
from langchain_community.vectorstores import Chroma

def retrieve_relevant_chunks(
    vectorstore: Chroma, 
    query: str, 
    k: int = 6, 
    threshold: float = 0.4
) -> List[Tuple[Any, float]]:
    """
    Given a user question, retrieves the top-k most relevant chunks from Chroma DB.
    Includes similarity score calculation.
    If the highest similarity score is below `threshold`, returns an empty list, 
    indicating that no sufficiently relevant context was found.
    """
    try:
        # LangChain's similarity_search_with_relevance_scores returns tuples of (Document, float_score)
        # where float_score is normalized to [0, 1] (higher score = more relevant)
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        # In case the collection is empty or has not been initialized
        return []
        
    if not results:
        return []
        
    # Check if the highest score is below the threshold
    top_doc, top_score = results[0]
    if top_score < threshold:
        print(f"Top chunk relevance score {top_score:.3f} is below threshold {threshold:.3f}")
        return []
        
    return results
