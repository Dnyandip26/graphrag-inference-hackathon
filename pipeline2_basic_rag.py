import time
import json
import hashlib
import numpy as np
import faiss

from pathlib import Path

from llm_client import (
    chat,
    embed,
    embed_query
)

# Approximate token cost
COST_PER_TOKEN = 0.000002

CACHE_FILE = Path("data/embedding_cache.json")
INDEX_FILE = Path("data/faiss_index.bin")
CHUNKS_FILE = Path("data/chunks_cache.json")


def _load_cache():

    if CACHE_FILE.exists():

        try:
            text = CACHE_FILE.read_text()

            if text.strip():
                return json.loads(text)

        except (
            json.JSONDecodeError,
            ValueError
        ):

            print(
                "Warning: Cache file corrupted, "
                "starting fresh"
            )

            CACHE_FILE.unlink(
                missing_ok=True
            )

    return {}


def _save_cache(cache):

    try:
        CACHE_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        CACHE_FILE.write_text(
            json.dumps(cache)
        )

    except Exception as e:
        print(
            f"Warning: Failed to save cache: {e}"
        )


def embed_with_cache(
    text: str
) -> np.ndarray:

    cache = _load_cache()

    key = hashlib.md5(
        text.encode()
    ).hexdigest()

    if key in cache:
        return np.array(
            cache[key],
            dtype="float32"
        )

    vec = embed(text)

    cache[key] = vec

    _save_cache(cache)

    return np.array(
        vec,
        dtype="float32"
    )


def load_and_chunk(
    filepath: str,
    chunk_size: int = 20
) -> list[str]:

    if CHUNKS_FILE.exists():

        try:
            text = CHUNKS_FILE.read_text()

            if text.strip():

                chunks = json.loads(text)

                print(
                    "Loading chunks from cache..."
                )

                return chunks

        except (
            json.JSONDecodeError,
            ValueError
        ):

            print(
                "Warning: Chunks cache corrupted, "
                "rebuilding"
            )

            CHUNKS_FILE.unlink(
                missing_ok=True
            )

    text = Path(filepath).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    sentences = [
        s.strip()
        for s in text.split("\n")
        if s.strip()
    ]

    chunks = []

    for i in range(
        0,
        len(sentences),
        chunk_size
    ):

        chunk = " ".join(
            sentences[i:i + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    CHUNKS_FILE.write_text(
        json.dumps(chunks)
    )

    return chunks


def build_faiss_index(
    chunks: list[str]
):

    if INDEX_FILE.exists():

        try:
            print(
                "Loading FAISS index from cache..."
            )

            index = faiss.read_index(
                str(INDEX_FILE)
            )

            print(
                f"Index loaded instantly! "
                f"Total vectors: {index.ntotal}"
            )

            return index

        except Exception as e:

            print(
                f"Cache load failed: {e}"
            )

            INDEX_FILE.unlink(
                missing_ok=True
            )

    print(
        f"Building FAISS index for "
        f"{len(chunks)} chunks..."
    )

    first_vec = embed_with_cache(
        chunks[0]
    )

    dim = len(first_vec)

    index = faiss.IndexFlatL2(dim)

    batch_size = 64

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            i:i + batch_size
        ]

        batch_vecs = np.stack([
            embed_with_cache(c)
            for c in batch
        ]).astype("float32")

        index.add(batch_vecs)

        print(
            f"Indexed "
            f"{min(i + batch_size, len(chunks))}"
            f"/{len(chunks)}"
        )

    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print("Index saved successfully!")

    print(
        f"Total vectors: {index.ntotal}"
    )

    return index


def run_basic_rag(
    query: str,
    chunks: list[str],
    index,
    top_k: int = 5
) -> dict:

    q_vec = np.array(
        embed_query(query),
        dtype="float32"
    ).reshape(1, -1)

    _, indices = index.search(
        q_vec,
        top_k
    )

    retrieved = [
        chunks[i]
        for i in indices[0]
        if i < len(chunks)
    ]

    context = "\n\n".join(
        retrieved
    )

    context = context[:2000]

    system = (
        "You are a helpful assistant. "
        "Use the provided context to "
        "answer questions accurately."
    )

    prompt = f"""
Context:

{context}

Question: {query}

Answer:
"""

    start = time.time()

    answer, in_tok, out_tok = chat(
        prompt,
        system=system
    )

    latency_ms = round(
        (time.time() - start) * 1000
    )

    total_tokens = (
        in_tok + out_tok
    )

    return {
        "pipeline": "Basic RAG",
        "answer": answer,
        "tokens": total_tokens,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "latency_ms": latency_ms,
        "cost_usd": round(
            total_tokens * COST_PER_TOKEN,
            6
        ),
        "retrieved_chunks": len(
            retrieved
        ),
        "context_chars": len(
            context
        ),
    }
