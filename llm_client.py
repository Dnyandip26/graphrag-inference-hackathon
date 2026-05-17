from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit as st

# Streamlit Secret
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Models
MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq Client
_client = Groq(api_key=GROQ_API_KEY)

# Embedding Model
_embedder = SentenceTransformer(EMBEDDING_MODEL)


def chat(prompt: str, system: str = "") -> tuple[str, int, int]:
    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    answer = response.choices[0].message.content
    in_tok = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens

    return answer, in_tok, out_tok


def embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def embed_query(text: str) -> list[float]:
    return _embedder.encode(text).tolist()
