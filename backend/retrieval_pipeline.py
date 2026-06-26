import numpy as np
import faiss
import time
from sentence_transformers import CrossEncoder

class MultiStageRetrieval:
    def __init__(self, embedding_dim=256, nlist=100, m=8, bits_per_subquantizer=8):
        """
        Initializes the Multi-Stage Retrieval pipeline.
        Stage 1: FAISS IVF-PQ
        Stage 2: Cross-Encoder Reranking
        """
        self.embedding_dim = embedding_dim
        
        # FAISS IVF-PQ setup for fast approximate nearest neighbor search
        quantizer = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, bits_per_subquantizer)
        
        # Cross-encoder for reranking
        # Note: In production, load this from a local cache or specific device
        print("Loading Cross-Encoder model...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        
        self.candidate_metadata = {} # Mapping from FAISS ID to metadata (e.g., text)

    def train_and_add_embeddings(self, embeddings, metadata_list):
        """Mock method to populate FAISS with 100k+ embeddings."""
        print(f"Training FAISS index on {len(embeddings)} vectors...")
        assert embeddings.shape[1] == self.embedding_dim
        
        # IVF-PQ requires training
        self.index.train(embeddings)
        self.index.add(embeddings)
        
        for i, meta in enumerate(metadata_list):
            self.candidate_metadata[i] = meta
            
        print("FAISS index populated successfully.")

    def retrieve_and_rerank(self, query_text, query_embedding, karmgraph_candidates, top_k_stage1=1000, top_k_stage2=100):
        """
        Executes the multi-stage pipeline.
        1. FAISS retrieval
        2. KarmGraph merging
        3. Cross-Encoder reranking
        """
        start_time = time.time()
        
        # Stage 1: FAISS IVF-PQ Retrieval
        # Reshape query embedding for FAISS
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Perform search
        distances, indices = self.index.search(query_embedding_np, top_k_stage1)
        
        faiss_candidates = indices[0].tolist()
        
        stage1_time = time.time() - start_time
        print(f"Stage 1 (FAISS) completed in {stage1_time*1000:.2f}ms. Retrieved {len(faiss_candidates)} candidates.")

        # Stage 1.5: Merge with KarmGraph expanded candidates (second-degree connections)
        # Merge lists, remove duplicates, keeping FAISS candidates prioritized or just pooled
        merged_candidate_ids = list(set(faiss_candidates + karmgraph_candidates))
        print(f"Merged with {len(karmgraph_candidates)} KarmGraph candidates. Total pool: {len(merged_candidate_ids)}")

        # Stage 2: Cross-Encoder Reranking
        # Prepare pairs for cross-encoder: (query, candidate_document)
        rerank_pairs = []
        valid_candidate_ids = []
        for cid in merged_candidate_ids:
            if cid in self.candidate_metadata:
                doc_text = self.candidate_metadata[cid]
                rerank_pairs.append([query_text, doc_text])
                valid_candidate_ids.append(cid)
                
        if not rerank_pairs:
            return []

        # Score pairs
        print("Stage 2 (Cross-Encoder) reranking...")
        scores = self.cross_encoder.predict(rerank_pairs)
        
        # Sort by score descending
        ranked_results = sorted(zip(valid_candidate_ids, scores), key=lambda x: x[1], reverse=True)
        
        # Return Top N
        top_results = ranked_results[:top_k_stage2]
        
        total_time = time.time() - start_time
        print(f"Total Retrieval Pipeline Time: {total_time*1000:.2f}ms")
        
        return top_results

# Mock Execution
if __name__ == "__main__":
    print("Initializing Multi-Stage Retrieval Pipeline (Dry Run)...")
    pipeline = MultiStageRetrieval(embedding_dim=256)
    
    # Generate 10,000 mock embeddings for testing (instead of 100k to save memory in dry run)
    mock_size = 10000
    mock_embeddings = np.random.random((mock_size, 256)).astype('float32')
    mock_metadata = [f"Candidate Resume {i}" for i in range(mock_size)]
    
    pipeline.train_and_add_embeddings(mock_embeddings, mock_metadata)
    
    query_text = "Looking for a Senior Python Developer with machine learning experience."
    query_emb = np.random.random(256).astype('float32')
    
    # Mock KarmGraph candidates (e.g., IDs 5, 100005)
    mock_karmgraph_cids = [5, 12, 45, 9999]
    
    results = pipeline.retrieve_and_rerank(
        query_text=query_text,
        query_embedding=query_emb,
        karmgraph_candidates=mock_karmgraph_cids,
        top_k_stage1=1000,
        top_k_stage2=10
    )
    
    print("Top 10 Reranked Candidates:")
    for cid, score in results:
        print(f"ID: {cid}, Score: {score:.4f}")
