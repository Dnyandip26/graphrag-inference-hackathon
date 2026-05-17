from groq import Groq
from sentence_transformers import SentenceTransformer


_client = Groq(api_key=GROQ_API_KEY)
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer('data/local_model')

def chat(prompt: str, system: str = "") -> tuple[str, int, int]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    answer = response.choices[0].message.content
    in_tok  = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    return answer, in_tok, out_tok


def embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def embed_query(text: str) -> list[float]:
    return _embedder.encode(text).tolist()
