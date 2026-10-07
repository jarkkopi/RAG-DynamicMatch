# Dynamic Resume & Profile Matcher (RAG Application)

A step-by-step Retrieval-Augmented Generation (RAG) CLI application demonstrating vector database ingestion, hybrid retrieval (Dense + BM25), agentic self-correction gates, and automated multi-agent evaluation benchmarking.

---

## 🛠️ Architecture & Learning Steps

```
[Candidate Profiles] ──> [1. Chunking & ChromaDB Ingestion]
                                    │
                                    ▼
[User Query] ──────────> [2. Hybrid Search (Dense + BM25 RRF)]
                                    │
                                    ▼
                         [3. Self-Correction Gate]
                         ┌──────────┴──────────┐
                  [Relevant]              [Irrelevant]
                      │                        │
                      ▼                        ▼
           [Prompt Augmentation]      [Graceful Fallback]
```

1. **Step 1: Vector Storage & Ingestion (`step1_ingest.py` / `step1_concept.py`)**
   * Stores candidate profiles in a local **ChromaDB** vector database using `all-MiniLM-L6-v2` dense embeddings.
   * Demonstrates vector cosine similarity math in pure Python.

2. **Step 2: Hybrid Search (`step2_hybrid_search.py`)**
   * Combines **Dense Vector Search** (semantic context) and **BM25 Keyword Search** (exact term precision).
   * Fuses ranked results using **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF Score}(d) = \sum_{m} \frac{1}{60 + \text{rank}_m(d)}$$

3. **Step 3: Self-Correction Relevance Gate (`step3_self_correction.py`)**
   * Evaluates retrieved documents *before* LLM prompt augmentation.
   * Rejects out-of-domain queries (e.g. searching for missing skills) to prevent AI hallucinations.

4. **Step 4: RAG Benchmarking & Evaluation Matrix (`step4_benchmark.py`)**
   * Automates parameter testing across chunk sizes (100, 250, 500 chars) and measures **Accuracy (%)** and **Latency (ms)**.

---

## 🚀 Quick Start & Environment Setup

### Prerequisites
* Conda / Miniconda (`AIENG` environment)
* Python 3.11+

## 💻 Running the Application Steps

Run each module sequentially:

```powershell
# Step 1A: ChromaDB Neural Vector Store & Search
python step1_ingest.py

# Step 1B: Pure Math Cosine Similarity Demo (Zero-Dependency)
python step1_concept.py

# Step 2: Dense vs. BM25 vs. Hybrid RRF Search
python step2_hybrid_search.py

# Step 3: Self-Correction Gate & Fallback Evaluation
python step3_self_correction.py

# Step 4: RAG Parameter Benchmark Matrix
python step4_benchmark.py
```

---
