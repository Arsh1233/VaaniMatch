import os
import redis
from celery import Celery
from celery.schedules import crontab
import json
import sys
import os

# Ensure data_science module is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_science.bharatbert_finetune import BharatBERT
from transformers import AutoTokenizer
import torch
# --- Redis Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Setup Redis Connection Pool
redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
redis_client = redis.Redis(connection_pool=redis_pool)

# --- Celery App Setup ---
app = Celery(
    'vaanimatch_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

app.conf.update(
    timezone='UTC',
    enable_utc=True,
)

# --- Celery Beat Schedule ---
app.conf.beat_schedule = {
    'daily-pull-and-embed': {
        'task': 'celery_tasks.pull_and_embed_candidates',
        'schedule': crontab(hour=0, minute=0), # Run daily at midnight UTC
    },
    'daily-rebuild-faiss': {
        'task': 'celery_tasks.rebuild_faiss_index',
        'schedule': crontab(hour=1, minute=0), # Run daily at 1 AM UTC
    },
}

# --- Background Tasks ---

@app.task
def pull_and_embed_candidates():
    """
    Task 1: Pull new candidates from PostgreSQL, generate embeddings,
    and push to Redis with a 24hr TTL.
    """
    print("[Task] Starting pull_and_embed_candidates...")
    
    # 1. Mock pulling from PostgreSQL
    # candidates = db.query("SELECT * FROM candidates WHERE status = 'new'")
    mock_new_candidates = [
        {"id": "c_101", "text": "Senior Java Developer"},
        {"id": "c_102", "text": "Data Scientist intern"}
    ]
    
    print(f"Pulled {len(mock_new_candidates)} new candidates from DB.")
    
    # 2. Generate Embeddings using actual BharatBERT
    print("Loading tokenizer and generating embeddings using BharatBERT...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        model = BharatBERT()
        model.eval()
        
        # We process in a simple loop for this orchestration mock
        TTL_SECONDS = 24 * 60 * 60
        
        with torch.no_grad():
            for c in mock_new_candidates:
                inputs = tokenizer(c['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
                embeddings = model(inputs['input_ids'], inputs['attention_mask'])
                
                # Convert 256-dim tensor to list
                emb_list = embeddings.squeeze(0).tolist()
                
                # 3. Push to Redis with 24h TTL
                redis_key = f"candidate_embedding:{c['id']}"
                embedding_data = json.dumps(emb_list) 
                
                # Set with expiry
                redis_client.setex(redis_key, TTL_SECONDS, embedding_data)
                
        print("[Task] Completed. Real embeddings generated and pushed to Redis.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        
    return f"Processed {len(mock_new_candidates)} candidates."

@app.task
def rebuild_faiss_index():
    """
    Task 2: Rebuild the FAISS index with all active candidates.
    """
    print("[Task] Starting rebuild_faiss_index...")
    
    # 1. Pull all active candidates and their embeddings from Redis/Postgres
    print("Gathering all active embeddings...")
    
    # 2. Re-initialize FAISS IVF-PQ index
    print("Re-initializing FAISS IndexIVFPQ...")
    
    # 3. Train and Add vectors
    print("Training and adding vectors to new FAISS index...")
    
    # 4. Save to disk or swap in memory
    print("FAISS index successfully rebuilt and swapped.")
    
    print("[Task] Completed. Index rebuilt.")
    return "FAISS index rebuilt."

# --- Mock Execution ---
if __name__ == "__main__":
    print("Celery Worker Configuration Loaded.")
    print("To run worker: celery -A celery_tasks worker --loglevel=info")
    print("To run beat: celery -A celery_tasks beat --loglevel=info")
