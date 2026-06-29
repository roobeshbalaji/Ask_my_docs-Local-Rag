from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from config import OLLAMA_MODEL, NUM_CTX


def build_llm() -> ChatOllama:
    """Create the Ollama LLM instance."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_ctx=NUM_CTX,
        num_gpu=99,
    )


def generate_answer(question: str, top_docs: list[Document], llm: ChatOllama) -> dict:
    """
    Build the evidence prompt and generate an answer.

    Returns dict with keys: answer, sources, top_docs.
    """
    context_blocks = []
    for i, doc in enumerate(top_docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[{i}] Source: {source}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""You are a precise financial and technical research assistant.
Your job is to fully answer the user's question using ONLY the evidence chunks
provided below.

How to answer:
- SYNTHESISE across ALL relevant evidence chunks. Do not just echo back the one
  sentence that happens to contain the question's keywords — read every chunk and
  combine the facts into one coherent explanation.
- Actually EXPLAIN the answer: state what the method/concept is, how it works, and
  any specifics the evidence gives (names, components, mechanisms, numbers). Aim
  for a thorough 3-6 sentence answer when the evidence supports it.
- Address the user's question directly. If the question asks "what method", name
  the method AND describe it; do not stop at a single noun phrase.

Hard rules:
1. Base every claim strictly on the evidence. Do not use outside knowledge.
2. Write in complete, self-contained sentences. State the answer in words, then
   attach the citation. Never reply with a bare citation or a single quoted phrase.
3. After each factual claim, cite its source using [1], [2], [3], or [4]. You may
   cite multiple chunks (e.g. [1][3]) when combining evidence.
4. If — and only if — the answer genuinely cannot be found in the evidence, respond
   with exactly: "I don't have enough information in the provided documents to
   answer that."
5. Never invent numbers, percentages, dates, or proper nouns. But DO elaborate on
   what the evidence actually says rather than under-answering.

Evidence:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print("💬 Generating answer...")
    response = llm.invoke(messages)
    answer = response.content

    sources = list({doc.metadata.get("source", "unknown") for doc in top_docs})

    return {
        "answer": answer,
        "sources": sources,
        "top_docs": top_docs,
    }


def generate_direct_answer(question: str, llm: ChatOllama) -> dict:
    """Answer a question directly without retrieval (for adaptive routing)."""
    system_prompt = (
        "You are a helpful research assistant. Answer the following question "
        "from general knowledge. Be concise and accurate."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print("💬 Generating direct answer (no retrieval)...")
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [],
        "top_docs": [],
    }
