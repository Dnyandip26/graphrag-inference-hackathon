"""
main.py - Run all 3 pipelines on a query and compare results.
Usage: python main.py
       python main.py --query "Your custom question here"
       python main.py --all
"""
import argparse
from pipeline1_llm_only import run_llm_only
from pipeline2_basic_rag import load_and_chunk, build_faiss_index, run_basic_rag
from pipeline3_graphrag import TigerGraphManager, build_graph_from_docs, run_graphrag
from evaluator import evaluate_all, print_comparison_table

DATA_FILE = "data/dataset.txt"

REFERENCE_ANSWERS = {
    "What are the main causes of climate change?":
        "The main causes of climate change are human activities, primarily burning fossil fuels (coal, oil, gas) which releases CO2 and greenhouse gases. Other causes include deforestation, agriculture (methane from livestock), and industrial processes.",
    "What is the Paris Agreement?":
        "The Paris Agreement is a 2015 international treaty aimed at limiting global warming to well below 2°C above pre-industrial levels through nationally determined contributions from participating countries.",
    "How does deforestation contribute to climate change?":
        "Deforestation contributes to climate change by removing forests that act as carbon sinks. When trees are cut down, the stored carbon dioxide is released back into the atmosphere.",
    "What is machine learning?":
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed, using algorithms to parse data and make decisions.",
    "What caused World War II?":
        "World War II was caused by multiple factors including the rise of fascism in Europe, Adolf Hitler's aggressive expansionist policies, the Great Depression's economic impacts, and the failure of appeasement policies.",
}

DEFAULT_QUERY = "What are the main causes of climate change?"

TEST_QUERIES = list(REFERENCE_ANSWERS.keys())


def run_comparison(query: str, chunks=None, index=None, kg=None, tg=None, verbose: bool = True):
    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print('='*65)

    # Load data once if not passed
    if chunks is None:
        print("Loading dataset...")
        chunks = load_and_chunk(DATA_FILE)
        index = build_faiss_index(chunks)
    if kg is None:
        tg = TigerGraphManager()
        kg, tg = build_graph_from_docs(DATA_FILE, tg)

    print("\nRunning Pipeline 1: LLM Only...")
    r1 = run_llm_only(query)

    print("Running Pipeline 2: Basic RAG...")
    r2 = run_basic_rag(query, chunks, index)

    print("Running Pipeline 3: GraphRAG...")
    r3 = run_graphrag(query, kg, tg)

    results = [r1, r2, r3]

    reference = REFERENCE_ANSWERS.get(query)
    if reference:
        results = evaluate_all(query, results, reference)

    if verbose:
        for r in results:
            print(f"\n--- {r['pipeline']} ---")
            print(f"Answer: {r['answer'][:200]}...")
        print_comparison_table(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--all", action="store_true", help="Run all test queries")
    args = parser.parse_args()

    # Load once, reuse
    print("Loading dataset and building indexes...")
    chunks = load_and_chunk(DATA_FILE)
    index = build_faiss_index(chunks)
    tg = TigerGraphManager()
    kg, tg = build_graph_from_docs(DATA_FILE, tg)

    if args.all:
        for q in TEST_QUERIES:
            run_comparison(q, chunks, index, kg, tg)
    else:
        run_comparison(args.query, chunks, index, kg, tg)