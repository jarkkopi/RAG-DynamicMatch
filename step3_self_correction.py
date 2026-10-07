import json
from step2_hybrid_search import CANDIDATES, dense_search, bm25_search, reciprocal_rank_fusion, build_vocab

# --- Step 3: Self-Correction Gate & Evaluator ---

class RelevanceEvaluator:
    """Evaluates whether retrieved documents are actually relevant to the query."""
    
    @staticmethod
    def evaluate_relevance(query, doc_text):
        """
        Heuristic / Rule-based evaluator for testing self-correction gates.
        In production, this is called via an LLM prompt (e.g. Gemini 3.6 Flash / GPT-4o-mini).
        Returns: (is_relevant: bool, reason: str)
        """
        query_terms = set(query.lower().split())
        doc_terms = set(doc_text.lower().split())
        
        # Check matching key terms
        matched_terms = query_terms.intersection(doc_terms)
        
        # Specific domain rules (e.g., if query asks for Rust but doc only has PyTorch/Go)
        if "rust" in query.lower() and "rust" not in doc_text.lower():
            return False, "Query explicitly requests Rust, but candidate lacks Rust experience."
            
        if "gpu" in query.lower() and not any(term in doc_text.lower() for term in ["gpu", "cuda", "distributed"]):
            return False, "Query requests GPU experience, but candidate lacks GPU/CUDA skills."

        if len(matched_terms) >= 1:
            return True, f"Relevant match found with key terms: {matched_terms}"
            
        return False, "Insufficient keyword or semantic overlap found."

def rag_pipeline_with_self_correction(query):
    print("=" * 70)
    print(f"RAG PIPELINE EXECUTION FOR QUERY: '{query}'")
    print("=" * 70)
    
    vocab = build_vocab(CANDIDATES)
    
    # 1. Hybrid Retrieval Phase
    dense_res = dense_search(query, CANDIDATES, vocab)
    bm25_res = bm25_search(query, CANDIDATES)
    hybrid_res = reciprocal_rank_fusion(dense_res, bm25_res)
    
    top_doc_id, top_name, score = hybrid_res[0]
    top_doc = next(d for d in CANDIDATES if d["id"] == top_doc_id)
    
    print(f"\n[Phase 1 - Hybrid Retrieval]: Candidate Selected -> {top_name} (RRF Score: {score:.6f})")
    print(f"   Retrieved Text: \"{top_doc['profile']}\"")
    
    # 2. Self-Correction Evaluation Gate
    print("\n[Phase 2 - Self-Correction Evaluation Gate]: Checking relevance...")
    is_relevant, reason = RelevanceEvaluator.evaluate_relevance(query, top_doc["profile"])
    
    if is_relevant:
        print(f"   [RESULT]: APPROVED (PASSED GATE)")
        print(f"   Reason: {reason}")
        print("\n[Phase 3 - Generation Context Augmentation]:")
        final_prompt = (
            f"Context: {top_doc['profile']}\n"
            f"User Query: {query}\n"
            f"Task: Generate candidate match report based ONLY on the context."
        )
        print(f"   Status: Context appended successfully for LLM generation.")
        return True, top_doc
    else:
        print(f"   [RESULT]: REJECTED (FAILED GATE)")
        print(f"   Reason: {reason}")
        print("\n[Phase 3 - Fallback Strategy]:")
        print("   Status: Triggering fallback mechanism (Returning 'No matching candidates found' to prevent hallucination).")
        return False, None

def main():
    print("HOUR 3 DEMO: RAG SELF-CORRECTION & RELEVANCE GATE\n")
    
    # Test 1: Valid query with existing candidate skills
    rag_pipeline_with_self_correction("Looking for a backend engineer with PostgreSQL experience")
    print("\n" + "-"*70 + "\n")
    
    # Test 2: Impossible query (No Rust developer in database) -> Triggers Self-Correction Fallback!
    rag_pipeline_with_self_correction("Looking for a Rust developer with GPU experience")

if __name__ == "__main__":
    main()
