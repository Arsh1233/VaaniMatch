# ============================================================
# Prompt 4.2 — RAG-based Hallucination Guardrail
# Focus: LangChain retriever, chunk citation enforcement,
#        self-consistency checker with temperature retry.
# ============================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONSISTENCY_THRESHOLD: float = 0.85   # re-generate if score < this
MAX_RETRIES: int = 3
FALLBACK_TEMPERATURE: float = 0.0     # temperature after a failed consistency check
TOP_K_CHUNKS: int = 3                 # number of chunks to retrieve from each source


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroundedExplanation:
    """Container for the final grounded explanation output."""
    explanation: str
    consistency_score: float
    cited_chunks: list[str] = field(default_factory=list)
    retries: int = 0
    passed_consistency: bool = True


# ---------------------------------------------------------------------------
# Vector store builder
# ---------------------------------------------------------------------------

def _build_vectorstore(texts: list[str], source_label: str) -> FAISS:
    """
    Split raw text into chunks, embed them, and return a FAISS vector store.
    Each chunk is tagged with `source_label` in its metadata so the LLM can
    cite it as `[Source: <source_label> Line <n>]`.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""],
    )
    docs: list[Document] = []
    for raw_text in texts:
        chunks = splitter.split_text(raw_text)
        for idx, chunk in enumerate(chunks, start=1):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": source_label, "line": idx},
                )
            )

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


# ---------------------------------------------------------------------------
# Citation extractor
# ---------------------------------------------------------------------------

_CITATION_PATTERN = re.compile(r"\[Source:\s*[^\]]+\]")


def _extract_citations(text: str) -> list[str]:
    return _CITATION_PATTERN.findall(text)


def _count_citation_coverage(explanation: str, chunks: list[Document]) -> float:
    """
    Rough heuristic: score = (number of cited chunks) / (total retrieved chunks).
    Returns a float in [0, 1].
    """
    cited = _extract_citations(explanation)
    if not chunks:
        return 0.0
    # Check how many retrieved chunks appear to be referenced
    referenced = sum(
        1
        for doc in chunks
        if any(
            doc.metadata["source"] in c and str(doc.metadata["line"]) in c
            for c in cited
        )
    )
    return referenced / len(chunks)


# ---------------------------------------------------------------------------
# Self-consistency checker
# ---------------------------------------------------------------------------

def _check_self_consistency(
    explanation: str,
    retrieved_chunks: list[Document],
    llm: ChatOpenAI,
) -> float:
    """
    Ask the LLM to score consistency between the explanation and the source
    chunks.  Returns a float in [0, 1].
    """
    chunks_text = "\n---\n".join(
        f"[{doc.metadata['source']} Line {doc.metadata['line']}]: {doc.page_content}"
        for doc in retrieved_chunks
    )
    prompt = (
        "You are a strict fact-checker. "
        "Given the SOURCE CHUNKS and the GENERATED EXPLANATION below, "
        "reply with ONLY a decimal number between 0 and 1 representing how "
        "factually consistent the explanation is with the source chunks. "
        "1.0 means fully grounded, 0.0 means completely hallucinated.\n\n"
        f"SOURCE CHUNKS:\n{chunks_text}\n\n"
        f"GENERATED EXPLANATION:\n{explanation}\n\n"
        "Consistency score (0-1):"
    )
    response = llm([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip().split()[0])
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        logger.warning("Could not parse consistency score; defaulting to 0.0")
        return 0.0


# ---------------------------------------------------------------------------
# Core middleware
# ---------------------------------------------------------------------------

def generate_grounded_explanation(
    candidate_resume_text: str,
    job_description_text: str,
    ranking_context: dict[str, Any],
    temperature: float = 0.7,
) -> GroundedExplanation:
    """
    Generate an explainability narrative for a candidate's ranking, grounded
    in factual evidence retrieved from the resume and JD.

    This function:
    1. Builds FAISS vector stores for the resume and JD.
    2. Retrieves the Top-K most relevant chunks for the current ranking context.
    3. Forces the LLM to cite each chunk as [Source: Resume Line N] or
       [Source: JD Line N].
    4. Runs a self-consistency check; if score < 85%, retries with a lower
       temperature (down to FALLBACK_TEMPERATURE).

    Parameters
    ----------
    candidate_resume_text : str
        Raw text of the candidate's resume.
    job_description_text : str
        Raw text of the job description.
    ranking_context : dict
        Scores and metadata from the ranking pipeline (Semantic, Trajectory, …).
    temperature : float
        Initial LLM temperature.

    Returns
    -------
    GroundedExplanation
    """
    # Build per-session vector stores
    resume_store = _build_vectorstore([candidate_resume_text], source_label="Resume")
    jd_store = _build_vectorstore([job_description_text], source_label="JD")

    # Formulate a concise query from the ranking context
    query = (
        f"Explain why this candidate scored "
        f"Semantic={ranking_context.get('semantic', 'N/A')}, "
        f"Trajectory={ranking_context.get('trajectory', 'N/A')}, "
        f"Behavior={ranking_context.get('behavior', 'N/A')}, "
        f"Graph={ranking_context.get('graph', 'N/A')}, "
        f"Verification={ranking_context.get('verification', 'N/A')} "
        "for this role."
    )

    # Retrieve top-K chunks from each store
    resume_docs: list[Document] = resume_store.similarity_search(query, k=TOP_K_CHUNKS)
    jd_docs: list[Document] = jd_store.similarity_search(query, k=TOP_K_CHUNKS)
    all_chunks = resume_docs + jd_docs

    chunks_block = "\n\n".join(
        f"[Source: {doc.metadata['source']} Line {doc.metadata['line']}]\n{doc.page_content}"
        for doc in all_chunks
    )

    system_prompt = (
        "You are an AI explainability assistant for a recruitment platform. "
        "You MUST base your explanation STRICTLY on the provided SOURCE CHUNKS. "
        "For every claim you make, you MUST inline a citation like "
        "`[Source: Resume Line 4]` or `[Source: JD Line 2]`. "
        "Do NOT fabricate any information not present in the chunks."
    )
    user_prompt = (
        f"SOURCE CHUNKS:\n{chunks_block}\n\n"
        f"RANKING CONTEXT:\n{ranking_context}\n\n"
        f"TASK: {query}\n\n"
        "Write a concise 3-paragraph explanation with inline citations."
    )

    retries = 0
    current_temperature = temperature
    explanation_text = ""
    consistency_score = 0.0

    while retries <= MAX_RETRIES:
        llm = ChatOpenAI(temperature=current_temperature, model_name="gpt-4o-mini")
        response = llm([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        explanation_text = response.content

        # Self-consistency check
        consistency_score = _check_self_consistency(explanation_text, all_chunks, llm)
        logger.info(
            "Consistency check — score=%.2f (threshold=%.2f), retry=%d",
            consistency_score,
            CONSISTENCY_THRESHOLD,
            retries,
        )

        if consistency_score >= CONSISTENCY_THRESHOLD:
            break

        retries += 1
        # Lower temperature on retry to reduce creativity/hallucination
        current_temperature = max(
            FALLBACK_TEMPERATURE,
            current_temperature - (temperature / MAX_RETRIES),
        )
        logger.warning(
            "Consistency below threshold — retrying with temperature=%.2f",
            current_temperature,
        )

    cited_chunks = _extract_citations(explanation_text)
    return GroundedExplanation(
        explanation=explanation_text,
        consistency_score=consistency_score,
        cited_chunks=cited_chunks,
        retries=retries,
        passed_consistency=consistency_score >= CONSISTENCY_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_resume = (
        "Aryan Kapoor — Senior Frontend Engineer\n"
        "Experience: 5 years building React applications at scale.\n"
        "Skills: React, TypeScript, GraphQL, Node.js, Docker.\n"
        "Education: B.Tech Computer Science, IIT Delhi, 2019.\n"
        "Projects: Led migration of legacy jQuery app to React at Infosys (2021).\n"
    )
    sample_jd = (
        "Role: Senior Frontend Engineer\n"
        "Requirements: 4+ years React experience, TypeScript, strong CSS skills.\n"
        "Preferred: GraphQL, CI/CD experience, open-source contributions.\n"
        "Location: Bangalore, hybrid.\n"
    )
    context = {
        "semantic": 85,
        "trajectory": 92,
        "behavior": 78,
        "graph": 88,
        "verification": 100,
    }

    result = generate_grounded_explanation(sample_resume, sample_jd, context)
    print("=== Grounded Explanation ===")
    print(result.explanation)
    print(f"\nConsistency Score: {result.consistency_score:.2f}")
    print(f"Retries: {result.retries}")
    print(f"Passed Consistency: {result.passed_consistency}")
    print(f"Citations found: {result.cited_chunks}")
