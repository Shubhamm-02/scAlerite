"""
main.py – FastAPI Backend for scAlerite (with Gemini LLM)
==========================================================

🧠 THE FULL RAG PIPELINE:
──────────────────────────
┌────────────┐  HTTP POST   ┌────────────┐  FAISS    ┌─────────┐  Gemini   ┌─────────┐
│ Telegram   │ ────────────▶│ FastAPI    │ ────────▶ │ Chunks  │ ────────▶│ Natural │
│ Bot        │ ◀────────────│ /query     │ ◀──────── │ Found   │ ◀──────── │ Answer  │
└────────────┘  JSON         └────────────┘           └─────────┘           └─────────┘
"""

import os
import threading
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

from app.utils import process_directory
from app.vector_store import VectorStore

# ─────────────────────────────────
# Configure Gemini (new SDK)
# ─────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini LLM configured.")
else:
    client = None
    print("⚠️ No GEMINI_API_KEY found. Bot will return raw chunks.")

# ─────────────────────────────────
# FastAPI App
# ─────────────────────────────────
app = FastAPI(title="scAlerite API", description="College academic chatbot backend")
store = VectorStore()
_store_ready = False   # set to True once PDFs are loaded


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.on_event("startup")
async def load_data():
    """Start PDF loading in a background thread so the port opens instantly."""
    threading.Thread(target=_load_pdfs, daemon=True).start()


def _load_pdfs():
    global _store_ready
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    print(f"\n📂 Loading PDFs from: {data_dir}")
    chunk_dicts = process_directory(data_dir)

    if chunk_dicts:
        store.add_chunks(chunk_dicts)
        print(f"✅ Vector store ready with {len(chunk_dicts)} chunks.\n")
    else:
        print("⚠️ No chunks found. Add PDFs to the data/ folder.\n")
    _store_ready = True


# ─────────────────────────────────
# Generate answer with Gemini
# ─────────────────────────────────
def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    🧠 NLP NOTE — Prompt Engineering:
    The quality of this prompt DIRECTLY controls answer quality.
    We tell Gemini its role, give it context, and set strict rules.
    """
    if not client:
        parts = []
        for i, c in enumerate(context_chunks, 1):
            parts.append(f"Result {i} (from {c['source']}):\n{c['chunk']}")
        return "\n\n".join(parts)

    # Build context from retrieved chunks
    context_text = ""
    for i, c in enumerate(context_chunks, 1):
        context_text += f"\n--- Document {i} (Source: {c['source']}) ---\n{c['chunk']}\n"

    prompt = f"""You are scAlerite, a helpful academic assistant for a college.
Your job is to answer student questions ONLY using the provided context from official college documents.

RULES:
- Answer in a clear, friendly, and concise manner.
- If the context contains the answer, provide it with relevant details.
- Always mention which source document the information comes from.
- If the context does NOT contain enough information to answer, say:
  "I couldn't find specific information about this in our policies. Please contact the administration for more details."
- Do NOT make up information. Only use what's in the context.
- Format your answer nicely with bullet points or numbered lists when appropriate.
- Keep the answer short and to the point (3-5 sentences max, unless the question needs more detail).

CONTEXT FROM COLLEGE DOCUMENTS:
{context_text}

STUDENT QUESTION: {question}

ANSWER:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        parts = []
        for i, c in enumerate(context_chunks, 1):
            parts.append(f"Result {i} (from {c['source']}):\n{c['chunk'][:300]}...")
        return "⚠️ AI generation failed. Here are the raw results:\n\n" + "\n\n".join(parts)


# ─────────────────────────────────
# Query Endpoint
# ─────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    results = store.search(request.question, top_k=request.top_k)

    if not results:
        return QueryResponse(
            answer="Sorry, I couldn't find any relevant information in the policies.",
            sources=[]
        )

    answer = generate_answer(request.question, results)
    source_files = list(set(r["source"] for r in results))

    return QueryResponse(answer=answer, sources=source_files)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": _store_ready,
        "chunks_loaded": len(store.chunks),
        "gemini_enabled": client is not None,
    }


@app.get("/debug")
async def debug():
    import glob
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    return {
        "data_dir": data_dir,
        "data_dir_exists": os.path.isdir(data_dir),
        "pdf_count": len(pdf_files),
        "pdf_files": [os.path.basename(f) for f in pdf_files],
        "cwd": os.getcwd(),
        "__file__": __file__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
