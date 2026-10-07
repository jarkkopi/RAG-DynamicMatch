import math
import re
from collections import Counter

# --- Step 2: Compare Naive Vector Search vs. BM25 Keyword Search vs. Hybrid Search ---

CANDIDATES = [
    {
        "id": "doc1",
        "name": "Alice",
        "profile": "5 years experience in PyTorch, distributed training, high-performance computing, and CUDA optimization."
    },
    {
        "id": "doc2",
        "name": "Bob",
        "profile": "Senior Backend Engineer specializing in PostgreSQL database optimization, Go microservices, and Kubernetes cluster management."
    },
    {
        "id": "doc3",
        "name": "Charlie",
        "profile": "ML Engineer focused on RAG pipelines, ChromaDB vector indexing, LLM fine-tuning, and LangChain."
    },
    {
        "id": "doc4",
        "name": "David",
        "profile": "Full-stack cloud developer with expertise in React, Node.js, AWS Lambda, and PostgreSQL data modeling."
    }
]

# --- 1. Dense Semantic Vector Search (Cosine Similarity on Word Embeddings) ---

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_vocab(docs):
    vocab = set()
    for d in docs:
        vocab.update(tokenize(d["profile"]))
    return sorted(list(vocab))

def text_to_vec(text, vocab):
    tokens = tokenize(text)
    counts = Counter(tokens)
    return [counts.get(w, 0) for w in vocab]

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def dense_search(query, docs, vocab):
    q_vec = text_to_vec(query, vocab)
    scores = []
    for doc in docs:
        d_vec = text_to_vec(doc["profile"], vocab)
        sim = cosine_similarity(q_vec, d_vec)
        scores.append((doc["id"], doc["name"], sim))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

# --- 2. BM25 / Exact Keyword Search ---

def bm25_score(query, doc_text, avg_doc_len, N, df_dict, k1=1.5, b=0.75):
    """Calculates BM25 score for a document given a query."""
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc_text)
    d_len = len(d_tokens)
    d_counts = Counter(d_tokens)
    
    score = 0.0
    for term in q_tokens:
        if term not in d_counts:
            continue
        tf = d_counts[term]
        df = df_dict.get(term, 0)
        # IDF calculation
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        # BM25 TF weight formula
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (d_len / avg_doc_len))
        score += idf * (numerator / denominator)
    return score

def bm25_search(query, docs):
    N = len(docs)
    doc_tokens = [tokenize(d["profile"]) for d in docs]
    avg_doc_len = sum(len(t) for t in doc_tokens) / N
    
    # Document frequency dictionary
    df_dict = Counter()
    for tokens in doc_tokens:
        for unique_term in set(tokens):
            df_dict[unique_term] += 1
            
    scores = []
    for doc in docs:
        score = bm25_score(query, doc["profile"], avg_doc_len, N, df_dict)
        scores.append((doc["id"], doc["name"], score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

# --- 3. Reciprocal Rank Fusion (RRF) for Hybrid Search ---

def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """Combines Dense & BM25 rankings using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    doc_map = {}
    
    # Process dense rankings
    for rank, (doc_id, name, score) in enumerate(dense_results):
        doc_map[doc_id] = name
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    # Process BM25 rankings
    for rank, (doc_id, name, score) in enumerate(bm25_results):
        doc_map[doc_id] = name
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, doc_map[doc_id], score) for doc_id, score in sorted_rrf]

# --- Experiment Execution ---

def run_experiment(query):
    print("=" * 70)
    print(f"QUERY: '{query}'")
    print("=" * 70)
    
    vocab = build_vocab(CANDIDATES)
    
    dense_res = dense_search(query, CANDIDATES, vocab)
    bm25_res = bm25_search(query, CANDIDATES)
    hybrid_res = reciprocal_rank_fusion(dense_res, bm25_res)
    
    print(f"\n1. Dense Similarity Search Top 2:")
    for rank, (doc_id, name, score) in enumerate(dense_res[:2]):
        print(f"   Rank {rank+1}: {name:<8} (Score: {score:.4f})")
        
    print(f"\n2. BM25 Keyword Search Top 2:")
    for rank, (doc_id, name, score) in enumerate(bm25_res[:2]):
        print(f"   Rank {rank+1}: {name:<8} (BM25 Score: {score:.4f})")
        
    print(f"\n3. Hybrid Search (RRF Fusion) Top 2:")
    for rank, (doc_id, name, score) in enumerate(hybrid_res[:2]):
        print(f"   Rank {rank+1}: {name:<8} (RRF Score: {score:.6f})")
    print("\n")

def main():
    print("HOUR 2 EXPERIMENT: DENSE VS. BM25 VS. HYBRID SEARCH\n")
    
    # Test 1: Exact Technical Keyword / Acronym Query
    run_experiment("PostgreSQL")
    
    # Test 2: Conceptual / Semantic Query
    run_experiment("Who handles heavy server infrastructure and container clusters?")
    
    # Test 3: Mixed Query (Concept + Exact Acronym)
    run_experiment("RAG pipeline developer with CUDA optimization background")

if __name__ == "__main__":
    main()
