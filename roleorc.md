Here is the **Single Master Orchestration Prompt**. 

Copy and paste this directly into Antigravity (or Claude/ChatGPT). It acts as the **Principal Architect**, forcing the AI to generate the entire project scaffolding, the central integration hub, and the unified ranking pipeline that wires all the individual components (NLP, Graph, Models, UI, Validation) together into a single, runnable system.

---

### 🚀 The Master Prompt (Copy this)

> **Role:** Act as a Principal Software Architect & Full-Stack AI Engineer. Your task is to build the **complete orchestration layer** and **project scaffolding** for **"VaaniMatch"**—an end-to-end Intelligent Candidate Discovery system.
>
> **Objective:** Generate a production-ready, modular codebase that integrates all previously defined components. The system must process a Job Description (JD) and return a ranked, explainable candidate list in under **500ms**.
>
> **Project Structure Required:** Generate a complete folder tree and populate the core files. The structure must separate `backend/` (Python/FastAPI), `frontend/` (React/TypeScript), `models/` (saved weights), `scripts/` (data pipelines), and `docker/`.
>
> **Core Orchestration (`backend/orchestrator.py`):**
> - Write a Python class `VaaniOrchestrator` that loads the following at initialization (with graceful fallback if files are missing, using dummy data for testing):
>   1. `BharatBERT` (HuggingFace pipeline for embeddings).
>   2. `FAISS` index (IVF-PQ) loaded from `models/faiss_index.bin`.
>   3. `KarmGraph` (DGL GraphSAGE model loaded via `torch.load`).
>   4. `Trajectory LSTM` and `LightGBM Scorer` (loaded from `.pkl`).
> - Implement the exact multi-stage retrieval pipeline:
>   - *Stage 1*: Embed JD -> ANN Search (top 1000).
>   - *Stage 2*: Cross-Encoder reranking (top 100).
>   - *Stage 3*: KarmGraph expansion (add second-degree neighbors).
> - Implement the **Dynamic Weighted Scoring** formula:
>   `Total = α·Semantic + β·Trajectory + γ·Behavior + δ·Graph + ε·Verification`
>   - Write logic to classify the JD into `Technical`, `Leadership`, or `Gig` using a zero-shot classifier.
>   - Apply the specific weight sets: Tech=[0.4, 0.25, 0.15, 0.15, 0.05]; Leadership=[0.25, 0.35, 0.20, 0.15, 0.05]; Gig=[0.20, 0.10, 0.25, 0.15, 0.30].
>
> **Backend API (`backend/main.py`):**
> - Create a FastAPI app with a single POST endpoint `/rank`.
> - Use Pydantic models for Request (JD text, Industry) and Response (Ranked list with Candidate IDs, Scores, and Explanations).
> - Implement the **Counterfactual Explanation** generator: For the Top 3 candidates, generate a text string explaining why they are ranked where they are and what single skill/attribute would move them up (e.g., *"If candidate had 'SQL', they would rank #1"*). Use simple template filling to avoid LLM hallucinations for this specific output.
>
> **Frontend Orchestration (`frontend/src/App.tsx`):**
> - Generate the main React component that consumes the `/rank` API.
> - Implement the **"Bright & Exclusive"** UI theme: Use Tailwind CSS with a custom `globals.css` featuring a luminous gradient mesh background (Electric Violet #8B5CF6 to Coral Pink #FB7185 to Bright Cyan #22D3EE).
> - Build the UI layout with:
>   - A glass-morphism input card (backdrop-blur) for JD submission with a voice-toggle button.
>   - A results grid displaying Candidate Cards. Each card must show radial progress bars for the 5 signals and an expandable "Explain Rank" section.
>
> **India Stack Integration Stub (`backend/validation.py`):**
> - Create a stub class `IndiaStackVerifier` that simulates e-Shram and Aadhaar verification. Include a method `verify_aadhaar(aadhaar_hash)` that returns a mock `Verified: True` response with a random timestamp for demo purposes, but code it in a way that the real API key can simply be swapped in later.
>
> **Anti-Hallucination Guardrail (`backend/explainer.py`):**
> - Ensure the explanation generator strictly references specific fields from the candidate's profile (e.g., "Based on your 4 years in Python..."). Do not generate free-form generic text.
>
> **Docker Orchestration (`docker-compose.yml`):**
> - Provide the Docker Compose file that spins up 4 services: `api`, `redis` (cache), `postgres` (with pgvector), and `celery-worker` (for background indexing).
>
> **Deliverable:** Output the entire codebase in a structured, copy-paste-ready format. Provide clear comments in the code explaining the integration points between the frontend, backend, and AI models. Ensure the root directory contains a `README.md` explaining how to run `docker-compose up --build` to launch the entire system.