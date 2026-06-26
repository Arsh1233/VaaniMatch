### Role 4: Data Validation & India Stack Engineer (Anti-Hallucination & Verification)
*Focus: e-Shram API, RAG grounding, and fraud detection.*

**Prompt 4.1 (e-Shram & NCS API Integration):**
> "Write a secure Python microservice using `httpx` to integrate with India Stack's e-Shram and National Career Service (NCS) sandbox APIs. Implement OAuth 2.0 client credentials flow. Write a function `fetch_candidate_by_aadhaar(aadhaar_hash)` that retrieves the candidate's verified name, age, and occupational skills from the government database and upserts them into our PostgreSQL, flagging them as 'Government Verified'."

**Prompt 4.2 (RAG-based Hallucination Guardrail):**
> "Implement a 'Hallucination Prevention' middleware using LangChain. Before the LLM generates the 'Explainability' text, inject a retriever that fetches the Top 3 text chunks from the candidate's original resume and the Top 3 from the JD. Force the LLM to strictly cite these chunks using `[Source: Resume Line 4]`. Add a self-consistency checker that validates the generated explanation against the retrieved facts; if consistency < 85%, retry with a lower temperature."

**Prompt 4.3 (Fraud & Profile Inconsistency Detection):**
> "Write a Python heuristic engine that scans candidate profiles. Implement algorithms for: (1) Temporal Overlap Detection (flag overlapping employment dates), (2) Skill Inflation Detection (compare self-reported skill level against project descriptions using NER co-occurrence), and (3) Deduplication using fuzzy matching (Jaro-Winkler on Name+Phone+Skill combo). Output an 'Authenticity Score' (0-100) and a 'Suspicion Report' JSON."

**Prompt 4.4 (Blockchain Verification Microservice):**
> "Write a simple Flask microservice that acts as a blockchain anchoring layer. Use SHA-256 hashing of the candidate's certificate hash and store it in a local SQLite 'ledger' mimicking a Hyperledger structure. Provide an endpoint `/verify` that takes a certificate hash, recomputes the block hash, and returns a `Verified: True/False` response with the block timestamp."
