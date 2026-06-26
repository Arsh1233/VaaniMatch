# VaaniMatch — Role 4: Data Validation & India Stack Backend

This package contains four Python microservices implementing anti-hallucination guardrails, India Stack integration, fraud detection, and blockchain verification.

## Structure

```
backend/
├── india_stack/          # Prompt 4.1 — e-Shram & NCS API Integration
│   └── eshram_client.py
├── rag_guardrail/        # Prompt 4.2 — Hallucination Prevention Middleware
│   └── hallucination_guard.py
├── fraud_detection/      # Prompt 4.3 — Fraud & Profile Inconsistency Engine
│   └── fraud_engine.py
├── blockchain/           # Prompt 4.4 — Blockchain Anchoring Microservice
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Services

### e-Shram Integration (used as a library/module)
```bash
python -m india_stack.eshram_client
```

### RAG Hallucination Guardrail (used as a library/module)
```bash
python -m rag_guardrail.hallucination_guard
```

### Fraud Detection Engine
```bash
python -m fraud_detection.fraud_engine
```

### Blockchain Verification API
```bash
python blockchain/app.py
# Runs on http://localhost:5000
```

## Environment Variables (create a `.env` file)

```
# e-Shram / NCS OAuth 2.0
ESHRAM_CLIENT_ID=your_client_id
ESHRAM_CLIENT_SECRET=your_client_secret
ESHRAM_TOKEN_URL=https://sandbox.eshram.gov.in/oauth/token
ESHRAM_API_URL=https://sandbox.eshram.gov.in/api/v1
NCS_CLIENT_ID=your_ncs_client_id
NCS_CLIENT_SECRET=your_ncs_client_secret
NCS_TOKEN_URL=https://sandbox.ncs.gov.in/oauth/token
NCS_API_URL=https://sandbox.ncs.gov.in/api/v1

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/vaanimatch

# LLM
OPENAI_API_KEY=your_openai_key
```
