from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import fitz

SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED_FILES = SUPPORTED_TEXT | {".pdf"}
CHUNK_CHARS = 2200
CHUNK_OVERLAP = 300
EMBED_BATCH_SIZE = 24
EMBEDDING_DIMENSION = 768


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be greater than overlap")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                piece = paragraph[start : start + chunk_chars]
                if piece.strip():
                    chunks.append(piece.strip())
                if start + chunk_chars >= len(paragraph):
                    break
                start += chunk_chars - overlap
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            carry = current[-overlap:] if current else ""
            current = f"{carry}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 80]


def source_details(filename: str) -> dict[str, str]:
    name = filename.lower()
    if "dive_into_deep_learning" in name or name.startswith("d2l"):
        return {
            "source_url": "https://d2l.ai/",
            "license": "CC BY-SA 4.0",
        }
    if "modern_statistics" in name or name.startswith("ims"):
        return {
            "source_url": "https://openintrostat.github.io/ims/",
            "license": "CC BY-SA 3.0 Unported",
        }
    if "openintro" in name:
        return {
            "source_url": "https://www.openintro.org/book/os/",
            "license": "CC BY-SA 3.0 for most Statistics resources; verify the specific file",
        }
    if "learnml_starter_notes" in name:
        return {
            "source_url": "",
            "license": "CC BY 4.0 (LearnML original starter notes)",
        }
    return {
        "source_url": "",
        "license": "User-provided / not verified by LearnML; confirm usage rights",
    }


def load_documents(documents_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    paths: list[Path] = []
    if documents_dir.exists():
        paths.extend(sorted(p for p in documents_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_FILES))

    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_text = clean_text(page.get_text("text"))
                    if not page_text:
                        continue
                    for chunk_number, chunk in enumerate(chunk_text(page_text), start=1):
                        docs.append({
                            "source": path.name,
                            "page": page_number,
                            "chunk_number": chunk_number,
                            "subject": infer_subject(path.name, chunk),
                            "text": chunk,
                            **source_details(path.name),
                        })
        else:
            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            for chunk_number, chunk in enumerate(chunk_text(text), start=1):
                docs.append({
                    "source": path.name,
                    "page": None,
                    "chunk_number": chunk_number,
                    "subject": infer_subject(path.name, chunk),
                    "text": chunk,
                    **source_details(path.name),
                })
    return docs


def infer_subject(filename: str, text: str) -> str:
    value = f"{filename} {text[:500]}".lower()
    rules = [
        ("Deep Learning", ["neural network", "deep learning", "convolution", "transformer", "backpropagation"]),
        ("Machine Learning", ["regression", "classification", "clustering", "machine learning", "overfitting"]),
        ("Statistics", ["probability", "hypothesis test", "confidence interval", "distribution", "statistics"]),
        ("Data Science", ["data science", "data cleaning", "exploratory data", "visualization", "reproducibility"]),
        ("NLP", ["natural language", "word embedding", "tokenization", "language model", "attention"]),
    ]
    for subject, keywords in rules:
        if any(word in value for word in keywords):
            return subject
    return "General"


def _embedding_values(response: Any) -> np.ndarray:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is None:
        raise RuntimeError("Gemini returned no embeddings.")
    rows = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values")
        if values is None:
            raise RuntimeError("Gemini returned an embedding in an unexpected format.")
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def embed_texts(
    client: Any,
    texts: list[str],
    model: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = EMBED_BATCH_SIZE,
    progress_callback: Callable[[int, str], None] | None = None,
) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    all_vectors: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                from google.genai import types
                config = types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSION,
                    task_type=task_type,
                )
                result = client.models.embed_content(model=model, contents=batch, config=config)
                vectors = _embedding_values(result)
                if vectors.shape[0] != len(batch):
                    raise RuntimeError("The embedding API returned a different number of vectors than inputs.")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
                all_vectors.append(vectors)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
        if last_error is not None:
            raise RuntimeError(f"Embedding request failed after retries: {last_error}")
        if progress_callback:
            percent = int(min(95, ((start + len(batch)) / total) * 95))
            progress_callback(percent, f"Embedding chunks {min(start + len(batch), total)} / {total}...")
    return np.vstack(all_vectors).astype(np.float32)


def build_index(
    client: Any,
    documents_dir: Path,
    storage_dir: Path,
    embedding_model: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, int]:
    chunks = load_documents(documents_dir)
    if not chunks:
        raise ValueError("No readable documents were found. Add a PDF, TXT or Markdown file first.")
    if progress_callback:
        progress_callback(2, f"Extracted {len(chunks)} text chunks from the documents.")
    vectors = embed_texts(
        client,
        [item["text"] for item in chunks],
        embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
        progress_callback=progress_callback,
    )
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(storage_dir / "embeddings.npy", vectors)
    if progress_callback:
        progress_callback(100, "Knowledge index saved successfully.")
    return {"chunks": len(chunks), "documents": len({c["source"] for c in chunks})}


def load_index(storage_dir: Path) -> dict[str, Any]:
    chunks_file = storage_dir / "chunks.json"
    vectors_file = storage_dir / "embeddings.npy"
    if not chunks_file.exists() or not vectors_file.exists():
        return {"chunks": [], "embeddings": np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)}
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    vectors = np.load(vectors_file).astype(np.float32)
    if len(chunks) != len(vectors):
        raise ValueError("The saved chunks and embedding files do not match. Rebuild the knowledge index.")
    return {"chunks": chunks, "embeddings": vectors}


def retrieve_chunks(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not chunks or embeddings.size == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = embeddings @ query.T
    count = min(top_k, len(chunks))
    indices = np.argsort(scores[:, 0])[::-1][:count]
    results: list[dict[str, Any]] = []
    for idx in indices:
        result = dict(chunks[int(idx)])
        result["score"] = float(scores[int(idx), 0])
        results.append(result)
    return results

