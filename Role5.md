### Role 5: MLOps & Evaluation Engineer (Performance & Testing)
*Focus: Latency, Precision/Recall, and Containerization.*

**Prompt 5.1 (Comprehensive Evaluation Dashboard):**
> "Write a Jupyter notebook script that loads a test dataset of 500 labeled JDs. Calculate the following metrics for VaaniMatch vs a Keyword-Baseline: Precision@10, Recall@50, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG). Use `matplotlib` and `seaborn` to generate a bright, modern comparison bar chart with value labels on top of each bar."

**Prompt 5.2 (Docker Compose Production Setup):**
> "Generate a `docker-compose.yml` file with 5 services: (1) `fastapi-app` (Uvicorn), (2) `postgres` (with pgvector extension), (3) `redis` (for cache/Celery broker), (4) `celery-worker` (for background indexing), and (5) `nginx` (reverse proxy). Include health checks to ensure the FAISS index is loaded into memory before the API starts serving traffic. Ensure the startup time is optimized to under 30 seconds."

**Prompt 5.3 (Latency Stress Test Script):**
> "Write a `locustfile.py` for load testing. Simulate 50 concurrent recruiters submitting JDs concurrently. Assert that the 95th percentile latency for the `/rank` endpoint is under 500ms. Log the time taken for ANN search vs. Cross-encoder reranking separately to identify bottlenecks."

**Prompt 5.4 (Embedding Refresh Cron Job):**
> "Write a Python background script that runs every 6 hours. It queries PostgreSQL for new candidates added in the last 6 hours, generates their BharatBERT embeddings, updates the FAISS GPU index (without rebuilding entirely, using `faiss.merge` if possible), and updates the Redis cache. Implement a graceful degradation so the API still serves old embeddings during the update."