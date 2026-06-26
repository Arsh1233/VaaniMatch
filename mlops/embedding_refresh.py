import faiss
import time
import schedule
import psycopg2
import redis
import numpy as np

# --- Configuration ---
PG_DSN = "dbname=vaanimatch_db user=user password=password host=postgres"
REDIS_URL = "redis://redis:6379/0"
FAISS_INDEX_PATH = "/app/data/karmgraph_embeddings.index"
TEMP_INDEX_PATH = "/app/data/karmgraph_embeddings_temp.index"

# Mock Embedding Generation
def generate_bharatbert_embeddings(texts):
    # Simulates generating 256-dim embeddings for candidates
    print(f"Generating embeddings for {len(texts)} candidates...")
    return np.random.random((len(texts), 256)).astype('float32')

def perform_embedding_refresh():
    print(f"[{time.ctime()}] Starting 6-hour embedding refresh cycle...")
    
    # 1. Query PostgreSQL for new candidates (last 6 hours)
    try:
        # conn = psycopg2.connect(PG_DSN)
        # cur = conn.cursor()
        # cur.execute("SELECT id, text FROM candidates WHERE created_at >= NOW() - INTERVAL '6 hours'")
        # new_candidates = cur.fetchall()
        print("Querying Postgres for new candidates in the last 6 hours...")
        
        # Mock data
        new_candidates = [(101, "Java developer with Spring Boot"), (102, "Frontend engineer React")]
    except Exception as e:
        print(f"DB Error: {e}")
        return

    if not new_candidates:
        print("No new candidates found. Exiting refresh cycle.")
        return

    # 2. Generate Embeddings
    texts = [c[1] for c in new_candidates]
    ids = [c[0] for c in new_candidates]
    new_embeddings = generate_bharatbert_embeddings(texts)

    # 3. Incremental FAISS Update with Graceful Degradation
    try:
        print("Loading current FAISS index into memory buffer...")
        # To ensure the API isn't blocked, we load the index from disk,
        # update it in memory, and then overwrite the disk file atomically.
        # Alternatively, if using an in-memory server, we'd update a shadow index.
        
        # Mocking the load:
        # index = faiss.read_index(FAISS_INDEX_PATH)
        
        # For demonstration, we create a mock index
        d = 256
        index = faiss.IndexFlatL2(d)
        
        print("Adding new embeddings via faiss.Index.add_with_ids (Incremental)...")
        # index.add_with_ids(new_embeddings, np.array(ids))
        index.add(new_embeddings) # Using add for FlatL2 mock
        
        print("Writing updated index to temporary file...")
        # faiss.write_index(index, TEMP_INDEX_PATH)
        
        print("Swapping indices atomically to maintain graceful degradation...")
        # os.rename(TEMP_INDEX_PATH, FAISS_INDEX_PATH)
        print("Index updated successfully.")
        
    except Exception as e:
        print(f"FAISS Update Error: {e}")
        # If the update fails, the old index file remains untouched (Graceful Degradation)
        return

    # 4. Update Redis Cache
    try:
        r = redis.Redis.from_url(REDIS_URL)
        print("Updating Redis cache with new candidate profiles...")
        for cid, text in new_candidates:
            r.set(f"candidate:{cid}", text, ex=86400) # 24 hr TTL
    except Exception as e:
        print(f"Redis Error: {e}")

    print(f"[{time.ctime()}] Embedding refresh cycle completed successfully.")

def start_scheduler():
    print("Scheduling embedding refresh to run every 6 hours...")
    schedule.every(6).hours.do(perform_embedding_refresh)
    
    # Run once immediately for testing purposes
    perform_embedding_refresh()

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()
