import json
from llm_client import chat


def llm_judge(question: str, answer: str, reference: str) -> dict:
    prompt = f"""You are an expert evaluator. Score the answer vs reference.

Question: {question}
Reference: {reference}
Answer to Evaluate: {answer}

Rate each 1-10:
1. Accuracy: factually correct?
2. Completeness: covers key points?
3. Conciseness: appropriately brief?

Respond ONLY in this exact JSON (no markdown, no extra text):
{{"accuracy": 8, "completeness": 7, "conciseness": 9, "overall": 8, "feedback": "reason here"}}"""

    answer_text, _, _ = chat(prompt)
    try:
        clean = answer_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {"accuracy": 5, "completeness": 5, "conciseness": 5, "overall": 5, "feedback": "parse error"}


def bertscore_eval(reference: str, candidate: str) -> float:
    """BERTScore F1 — falls back to word overlap if bert_score not installed."""
    try:
        from bert_score import score
        P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
        return round(float(F1[0]) * 100, 2)
    except Exception:
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        overlap = len(ref_words & cand_words)
        p = overlap / len(cand_words) if cand_words else 0
        r = overlap / len(ref_words) if ref_words else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return round(f1 * 100, 2)


def evaluate_all(question: str, results: list[dict], reference: str) -> list[dict]:
    print("\nRunning evaluation...")
    for r in results:
        print(f"  Evaluating {r['pipeline']}...")
        r["bertscore"] = bertscore_eval(reference, r["answer"])
        judge = llm_judge(question, r["answer"], reference)
        r["llm_judge"] = judge
        r["accuracy_score"] = judge.get("overall", 5)
    return results


def print_comparison_table(results: list[dict]):
    print("\n" + "="*65)
    print(f"{'Metric':<20} {'LLM Only':>13} {'Basic RAG':>13} {'GraphRAG':>13}")
    print("="*65)

    metrics = [
        ("Tokens", "tokens", ""),
        ("Latency (ms)", "latency_ms", ""),
        ("BERTScore (%)", "bertscore", ""),
        ("LLM Judge (/10)", "accuracy_score", ""),
    ]
    for label, key, prefix in metrics:
        vals = [f"{prefix}{r.get(key, '—')}" for r in results]
        print(f"{label:<20} {vals[0]:>13} {vals[1]:>13} {vals[2]:>13}")

    print("="*65)
    if len(results) == 3:
        t1, t3 = results[0]["tokens"], results[2]["tokens"]
        if t1 > 0:
            saved = round(((t1 - t3) / t1) * 100, 1)
            print(f"\nGraphRAG token reduction vs LLM-only: {saved}%")
