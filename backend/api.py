# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
# pyrefly: ignore [missing-import]
from transformers import pipeline
import asyncio
import sys
import os

# Ensure backend and data_science modules are in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.retrieval_pipeline import MultiStageRetrieval

app = FastAPI(title="VaaniMatch Ranking API", version="1.0.0")

# --- Global Components (Loaded on startup in production) ---
print("Loading Zero-Shot Classifier and Retrieval Pipeline...")
# We use a smaller model for demonstration/speed
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
retrieval_pipeline = MultiStageRetrieval()

# Define dynamic weight sets
# Weights: [α=Semantic, β=Trajectory, γ=Behavior, δ=Graph, ε=Verification]
WEIGHT_MAP = {
    "Technical": [0.40, 0.25, 0.15, 0.15, 0.05],
    "Leadership": [0.25, 0.35, 0.20, 0.15, 0.05],
    "Gig": [0.20, 0.10, 0.25, 0.15, 0.30]
}

# --- Pydantic Models ---
class RankRequest(BaseModel):
    jd_text: str
    industry_type: str

class CandidateRawScores(BaseModel):
    candidate_id: str
    semantic_score: float # 0-1
    trajectory_score: float # 0-100 (needs scaling)
    behavior_score: float # integer counts (needs scaling)
    graph_score: float # 0-1
    verification_score: float # 0 or 1

class RankResponse(BaseModel):
    jd_classification: str
    applied_weights: List[float]
    ranked_candidates: List[Dict]

# Mock database retrieval of raw component scores for candidates
async def fetch_mock_candidate_scores() -> List[CandidateRawScores]:
    # In reality, this queries Postgres/Redis for candidates returned by FAISS
    return [
        CandidateRawScores(candidate_id="c_1", semantic_score=0.85, trajectory_score=80, behavior_score=45, graph_score=0.9, verification_score=1.0),
        CandidateRawScores(candidate_id="c_2", semantic_score=0.92, trajectory_score=40, behavior_score=12, graph_score=0.4, verification_score=0.0),
        CandidateRawScores(candidate_id="c_3", semantic_score=0.60, trajectory_score=95, behavior_score=120, graph_score=0.8, verification_score=1.0)
    ]

# --- Endpoints ---

@app.post("/rank", response_model=RankResponse)
async def rank_candidates(request: RankRequest):
    """
    Asynchronous endpoint to classify a JD and dynamically rank candidates.
    """
    if not request.jd_text:
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")

    # 1. Zero-Shot Classification
    candidate_labels = ["Technical", "Leadership", "Gig"]
    # Run synchronously in a threadpool to not block asyncio event loop
    classification_result = await asyncio.to_thread(
        classifier, request.jd_text, candidate_labels
    )
    
    # Get top predicted label
    jd_class = classification_result['labels'][0]
    weights = WEIGHT_MAP.get(jd_class, WEIGHT_MAP["Technical"]) # Default fallback

    # 1.5 Real Retrieval via MultiStage Pipeline
    # Mocking query embedding and karmgraph candidates for orchestration
    import numpy as np
    mock_query_emb = np.random.random(256).astype('float32')
    mock_karmgraph = [1, 2, 3]
    
    retrieved_top_k = await asyncio.to_thread(
        retrieval_pipeline.retrieve_and_rerank,
        request.jd_text, mock_query_emb, mock_karmgraph, 100, 10
    )
    
    # 2. Fetch Raw Candidates (Mocked database call using retrieved IDs)
    candidates = await fetch_mock_candidate_scores()
    
    # If the retrieval pipeline returned valid candidates, we can map them
    if retrieved_top_k:
        # Override mock IDs with actual retrieved IDs
        for i, (cid, score) in enumerate(retrieved_top_k):
            if i < len(candidates):
                candidates[i].candidate_id = str(cid)
                candidates[i].semantic_score = float(score) # Use cross-encoder score

    # 3. Dynamic Scoring & Normalization
    ranked_list = []
    
    # Determine max values for normalization (usually tracked globally, mocked here)
    max_trajectory = 100.0
    max_behavior = 150.0 # Mock max engagement count
    
    alpha, beta, gamma, delta, epsilon = weights

    for c in candidates:
        # Normalize features to 0-1 range
        norm_semantic = c.semantic_score
        norm_trajectory = c.trajectory_score / max_trajectory
        norm_behavior = min(c.behavior_score / max_behavior, 1.0)
        norm_graph = c.graph_score
        norm_verification = c.verification_score

        # Calculate weighted total score
        total_score = (
            (alpha * norm_semantic) +
            (beta * norm_trajectory) +
            (gamma * norm_behavior) +
            (delta * norm_graph) +
            (epsilon * norm_verification)
        )
        
        ranked_list.append({
            "candidate_id": c.candidate_id,
            "total_score": round(total_score, 4),
            "normalized_components": {
                "semantic": round(norm_semantic, 2),
                "trajectory": round(norm_trajectory, 2),
                "behavior": round(norm_behavior, 2),
                "graph": round(norm_graph, 2),
                "verification": round(norm_verification, 2)
            }
        })

    # Sort by total score descending
    ranked_list.sort(key=lambda x: x["total_score"], reverse=True)

    return RankResponse(
        jd_classification=jd_class,
        applied_weights=weights,
        ranked_candidates=ranked_list
    )

# Mock execution block for testing
if __name__ == "__main__":
    import uvicorn
    # In a real environment, you run this via: uvicorn api:app --reload
    print("Run this API using: uvicorn api:app --host 0.0.0.0 --port 8000")
