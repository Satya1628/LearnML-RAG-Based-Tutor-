"""Build the local semantic search index for LearnML."""
import os
from pathlib import Path

from dotenv import load_dotenv
from learnml_core import build_index

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY in a local .env file before building the index.")

from google import genai
client = genai.Client(api_key=api_key)
result = build_index(
    client=client,
    documents_dir=ROOT / "knowledge_base" / "documents",
    storage_dir=ROOT / "storage",
    embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    progress_callback=lambda p, message: print(f"[{p:3d}%] {message}"),
)
print(f"Done. Indexed {result['chunks']} chunks from {result['documents']} documents.")
