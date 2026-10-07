import time
from step2_hybrid_search import dense_search, bm25_search, reciprocal_rank_fusion
from step3_self_correction import RelevanceEvaluator

# --- Step 4: Multi-Agent Benchmark & Evaluation Suite ---

BENCHMARK_DATASET = [
    {
        "id": "c1",
        "name": "Alice Vance",
        "bio": "Senior ML Engineer with 6 years experience specializing in PyTorch, distributed training on GPU clusters, and CUDA kernel optimization for LLM inference."
    },
    {
        "id": "c2",
        "name": "Bob Miller",
        "bio": "Principal Infrastructure Architect specializing in PostgreSQL performance tuning, Go microservices, Kubernetes, and Terraform cloud deployment."
    },
    {
        "id": "c3",
        "name": "Charlie Zhang",
        "bio": "AI R&D Specialist expert in RAG pipelines, ChromaDB vector search, hybrid retrieval fusion, LangChain, and prompt engineering."
    },
    {
        "id": "c4",
        "name": "Diana Ross",
        "bio": "Senior Fullstack Engineer skilled in TypeScript, React, GraphQL APIs, PostgreSQL, and responsive web interface design."
    }
]

BENCHMARK_QUERIES = [
    {
        "query": "Who is specialized in CUDA GPU optimization for ML?",
        "expected_id": "c1"
    },
    {
        "query": "PostgreSQL database administrator and Kubernetes expert",
        "expected_id": "c2"
    },
    {
        "query": "Looking for RAG vector search and hybrid retrieval developer",
        "expected_id": "c3"
    },
    {
        "query": "React TypeScript UI developer",
        "expected_id": "c4"
    },
    {
        "query": "Rust language backend developer for high performance network programming",
        "expected_id": None  # Impossible query -> tests Self-Correction rejection
    }
]

def chunk_text(text, chunk_size=100, overlap=20):
    """Splits text into fixed-size chunks with overlapping windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def build_chunked_dataset(candidates, chunk_size, overlap):
    chunked_docs = []
    for c in candidates:
        sub_chunks = chunk_text(c["bio"], chunk_size, overlap)
        for i, chunk in enumerate(sub_chunks):
            chunked_docs.append({
                "id": f"{c['id']}_chunk_{i}",
                "candidate_id": c["id"],
                "name": c["name"],
                "profile": chunk
            })
    return chunked_docs

def evaluate_retrieval(chunked_docs, queries, method="hybrid"):
    """Evaluates top-1 retrieval accuracy across the benchmark queries."""
    from step2_hybrid_search import build_vocab
    vocab = build_vocab(chunked_docs)
    
    correct = 0
    total = len(queries)
    
    start_time = time.time()
    for item in queries:
        q = item["query"]
        expected = item["expected_id"]
        
        dense_res = dense_search(q, chunked_docs, vocab)
        bm25_res = bm25_search(q, chunked_docs)
        
        if method == "dense":
            top_match = dense_res[0]
            candidate_id = next(d["candidate_id"] for d in chunked_docs if d["id"] == top_match[0])
        elif method == "bm25":
            top_match = bm25_res[0]
            candidate_id = next(d["candidate_id"] for d in chunked_docs if d["id"] == top_match[0])
        elif method == "hybrid":
            rrf_res = reciprocal_rank_fusion(dense_res, bm25_res)
            top_match = rrf_res[0]
            candidate_id = next(d["candidate_id"] for d in chunked_docs if d["id"] == top_match[0])
        elif method == "self_corrected":
            rrf_res = reciprocal_rank_fusion(dense_res, bm25_res)
            top_doc = next(d for d in chunked_docs if d["id"] == rrf_res[0][0])
            is_relevant, _ = RelevanceEvaluator.evaluate_relevance(q, top_doc["profile"])
            # If self-correction rejects, candidate_id is None
            candidate_id = top_doc["candidate_id"] if is_relevant else None
            
        if candidate_id == expected:
            correct += 1
            
    latency_ms = (time.time() - start_time) * 1000 / total
    accuracy = (correct / total) * 100
    return accuracy, latency_ms

def main():
    print("======================================================================")
    print("   STEP 4: RAG BENCHMARK & MULTI-AGENT EVALUATION MATRIX")
    print("======================================================================\n")
    
    chunk_configs = [
        {"size": 100, "overlap": 20},
        {"size": 250, "overlap": 50},
        {"size": 500, "overlap": 100}
    ]
    
    methods = ["dense", "bm25", "hybrid", "self_corrected"]
    
    print(f"{'Chunk Size':<12} | {'Method':<15} | {'Accuracy (%)':<14} | {'Avg Latency (ms)':<16}")
    print("-" * 65)
    
    for config in chunk_configs:
        dataset = build_chunked_dataset(BENCHMARK_DATASET, config["size"], config["overlap"])
        for m in methods:
            acc, lat = evaluate_retrieval(dataset, BENCHMARK_QUERIES, method=m)
            print(f"{config['size']:<12} | {m:<15} | {acc:<14.1f} | {lat:<16.3f}")

if __name__ == "__main__":
    main()
