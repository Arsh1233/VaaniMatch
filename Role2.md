### Role 2: Backend & Retrieval Engineer (API & Ranking)
*Focus: FAISS indexing, FastAPI, and dynamic scoring.*

**Prompt 2.1 (Multi-Stage Retrieval Pipeline):**
> "Implement a multi-stage retrieval class in Python. Stage 1: Use FAISS with IVF-PQ indexing to retrieve the Top 1000 candidates from a pool of 100k+ embeddings (under 100ms). Stage 2: Use a `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank those 1000 candidates into a Top 100 list. Write the code to merge these lists with the KarmGraph expanded candidates (second-degree connections)."

**Prompt 2.2 (FastAPI Dynamic Ranking Endpoint):**
> "Write a FastAPI asynchronous endpoint `/rank`. It accepts a JSON body with `jd_text` and `industry_type`. It must classify the JD as 'Technical', 'Leadership', or 'Gig' using a zero-shot classifier. Based on the classification, dynamically apply the following weight sets (α=Semantic, β=Trajectory, γ=Behavior, δ=Graph, ε=Verification): Technical=[0.4, 0.25, 0.15, 0.15, 0.05]; Leadership=[0.25, 0.35, 0.20, 0.15, 0.05]; Gig=[0.20, 0.10, 0.25, 0.15, 0.30]. Return the ranked list with normalized total scores."

**Prompt 2.3 (Voice-to-Profile Mapper):**
> "Create a Python service using WebRTC and the Google Cloud Speech-to-Text API with Indic language support. Write a prompt chain for LangChain that takes the transcribed Hindi/Tamil voice input and maps it to structured JSON (Name, Skills, Experience, Location) using a slot-filling approach. If a slot is empty, prompt the voice assistant to ask a follow-up question dynamically."

**Prompt 2.4 (Redis Caching & Celery Tasks):**
> "Write the code to set up Celery beat schedules that run daily. Task 1: Pull new candidates from PostgreSQL, generate their embeddings, and push them to a Redis cache with TTL 24hrs. Task 2: Rebuild the FAISS index. Provide the Redis connection pool configuration and the background task implementation."
