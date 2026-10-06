import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import chromadb
from chromadb.utils import embedding_functions

def main():
    print("--- Step 1: Vector DB Setup & Ingestion ---")
    
    # 1. Initialize local ChromaDB client (ephemeral in-memory for testing, or persistent)
    client = chromadb.Client()
    
    # Using default embedding function (All-MiniLM-L6-v2 by default in ChromaDB)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # Create or get collection
    collection = client.create_collection(
        name="resumes", 
        embedding_function=default_ef
    )
    
    # 2. Sample candidate profiles/resumes
    documents = [
        "Alice: 5 years experience in PyTorch, distributed training, and CUDA optimization.",
        "Bob: Senior Backend Engineer specializing in PostgreSQL, Go, and Kubernetes cluster management.",
        "Charlie: ML Engineer focused on RAG pipelines, ChromaDB vector indexing, and LangChain.",
        "Diana: Frontend Developer with expertise in React, TypeScript, and UI/UX design.",
        "Evan: DevOps & Cloud Architect with AWS, Terraform, and Docker experience."
    ]
    
    ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    metadatas = [
        {"name": "Alice", "role": "ML/CUDA Engineer"},
        {"name": "Bob", "role": "Backend Engineer"},
        {"name": "Charlie", "role": "RAG Engineer"},
        {"name": "Diana", "role": "Frontend Developer"},
        {"name": "Evan", "role": "DevOps Architect"}
    ]
    
    # 3. Ingest documents into Vector DB
    print(f"Ingesting {len(documents)} candidate profiles into ChromaDB...")
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print("Ingestion complete! Total items in collection:", collection.count())
    
    # 4. Perform a simple test query
    query = "Who has experience with GPU optimization and neural networks?"
    print(f"\nQuerying: '{query}'")
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    
    print("\n--- Search Results ---")
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        doc_text = results['documents'][0][i]
        distance = results['distances'][0][i] if 'distances' in results and results['distances'] else N/A
        metadata = results['metadatas'][0][i]
        print(f"Rank {i+1}: [ID: {doc_id}] (Distance: {distance:.4f})")
        print(f"  Candidate: {metadata['name']} - {metadata['role']}")
        print(f"  Profile: {doc_text}\n")

if __name__ == "__main__":
    main()
