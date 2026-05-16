import time
from llm_client import chat
from config import COST_PER_TOKEN


def run_llm_only(query: str) -> dict:
    system = "You are a helpful assistant. Answer the question accurately and concisely."
    start = time.time()
    answer, in_tok, out_tok = chat(query, system=system)
    latency_ms = round((time.time() - start) * 1000)
    total_tokens = in_tok + out_tok
    return {
        "pipeline": "LLM Only",
        "answer": answer,
        "tokens": total_tokens,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "latency_ms": latency_ms,
        "cost_usd": round(total_tokens * COST_PER_TOKEN, 6),
    }


if __name__ == "__main__":
    result = run_llm_only("What are the main causes of climate change?")
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Latency: {result['latency_ms']}ms")