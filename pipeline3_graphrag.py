import time
import pickle
from collections import defaultdict
from pathlib import Path
from llm_client import chat, embed_query
from config import COST_PER_TOKEN
import numpy as np

KG_CACHE = Path("data/kg_cache.pkl")

class SimpleKnowledgeGraph:
    def __init__(self):
        self.entities: dict[str, list[str]] = defaultdict(list)
        self.relations: list[tuple] = []

    def add_entity(self, entity_id: str, chunk: str):
        self.entities[entity_id].append(chunk)

    def add_relation(self, a: str, relation: str, b: str):
        self.relations.append((a, relation, b))

    def multi_hop_search(self, query_entities: list[str], hops: int = 2) -> list[str]:
        visited = set()
        frontier = set(query_entities)
        results = []
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                if node in self.entities and self.entities[node]:
                    results.append(self.entities[node][0])
                for (a, rel, b) in self.relations:
                    if a == node and b not in visited:
                        next_frontier.add(b)
                    elif b == node and a not in visited:
                        next_frontier.add(a)
            frontier = next_frontier
        return results


DOMAIN_KEYWORDS = [
    "climate change", "global warming", "greenhouse gas", "carbon dioxide", "CO2",
    "methane", "fossil fuels", "deforestation", "renewable energy", "solar energy",
    "wind power", "Paris Agreement", "net-zero", "carbon capture", "ocean acidification",
    "sea level", "arctic", "permafrost", "emissions", "temperature", "atmosphere",
    "carbon cycle", "ozone layer", "air pollution", "biodiversity", "ecosystem",
    "photosynthesis", "water cycle", "glacier", "drought", "flood", "hurricane",
    "wildfire", "coral reef", "rainforest", "Amazon", "Kyoto Protocol", "IPCC",
    "sustainability", "green energy", "electric vehicle", "hydrogen fuel",
    "nuclear power", "coal", "petroleum", "natural gas", "biomass", "hydroelectric",
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "natural language processing", "large language model", "transformer", "GPT",
    "BERT", "computer vision", "reinforcement learning", "generative AI",
    "ChatGPT", "OpenAI", "Anthropic", "Google", "DeepMind", "Meta",
    "data science", "big data", "cloud computing", "quantum computing",
    "blockchain", "cryptocurrency", "bitcoin", "ethereum", "cybersecurity",
    "internet of things", "5G", "semiconductor", "microchip", "robotics",
    "autonomous vehicle", "self-driving", "drone", "virtual reality",
    "Google", "Microsoft", "Apple", "Amazon", "Tesla", "NVIDIA", "IBM",
    "Samsung", "Intel", "AMD", "SpaceX", "Netflix", "Uber",
    "United Nations", "World Bank", "WHO", "NATO", "European Union",
    "Greenpeace", "WWF", "UNICEF",
    "World War II", "World War I", "Cold War", "French Revolution",
    "Industrial Revolution", "Roman Empire", "British Empire",
    "American Revolution", "Civil War", "Holocaust", "democracy",
    "communism", "capitalism", "socialism", "fascism",
    "United States", "China", "Russia", "India", "Europe", "Africa",
    "black hole", "quantum mechanics", "theory of relativity", "Big Bang",
    "solar system", "Mars", "Moon", "NASA", "SpaceX", "DNA", "gene",
    "CRISPR", "evolution", "atom", "gravity", "dark matter",
    "inflation", "GDP", "stock market", "recession", "interest rate",
    "Federal Reserve", "supply chain", "globalization", "trade",
    "unemployment", "poverty", "inequality", "investment",
    "COVID-19", "pandemic", "vaccine", "antibiotic", "cancer", "diabetes",
    "heart disease", "mental health", "depression", "anxiety", "Alzheimer",
    "nutrition", "obesity",
    "Elon Musk", "Bill Gates", "Steve Jobs", "Jeff Bezos", "Mark Zuckerberg",
    "Barack Obama", "Donald Trump", "Vladimir Putin", "Xi Jinping",
    "Albert Einstein", "Isaac Newton", "Charles Darwin", "Marie Curie",
    "Nikola Tesla", "Alan Turing", "Stephen Hawking",
]

def extract_entities_from_chunk(chunk: str) -> list[str]:
    chunk_lower = chunk.lower()
    return [kw for kw in DOMAIN_KEYWORDS if kw.lower() in chunk_lower]

def identify_query_entities(query: str) -> list[str]:
    query_lower = query.lower()
    return [kw for kw in DOMAIN_KEYWORDS if kw.lower() in query_lower]


def build_graph_from_docs(filepath: str, tg_manager=None) -> tuple:
    if KG_CACHE.exists():
        with open(KG_CACHE, "rb") as f:
            kg = pickle.load(f)
        print(f"KG loaded: {len(kg.entities)} entities, {len(kg.relations)} relations")
        return kg, tg_manager

    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    chunks = [s.strip() for s in text.split("\n") if s.strip() and len(s.strip()) > 100]

    kg = SimpleKnowledgeGraph()
    for chunk in chunks:
        entities = extract_entities_from_chunk(chunk)
        for ent in entities:
            kg.add_entity(ent, chunk)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                kg.add_relation(entities[i], "co-occurs-with", entities[j])

    with open(KG_CACHE, "wb") as f:
        pickle.dump(kg, f)

    print(f"Graph built: {len(kg.entities)} entities, {len(kg.relations)} relations")
    return kg, tg_manager


def semantic_fallback(query: str, chunks: list[str], top_k: int = 2) -> list[str]:
    """
    When graph has no matching entities, use semantic search as fallback.
    This ensures GraphRAG always gives an answer.
    """
    try:
        import faiss
        q_vec = np.array(embed_query(query), dtype="float32").reshape(1, -1)
        
        # Build mini index from sample chunks
        sample = chunks[:500] if len(chunks) > 500 else chunks
        vecs = np.stack([np.array(embed_query(c), dtype="float32") for c in sample[:50]])
        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        _, indices = index.search(q_vec, top_k)
        return [sample[i] for i in indices[0] if i < len(sample)]
    except Exception:
        return chunks[:top_k] if chunks else []


def run_graphrag(query: str, kg: SimpleKnowledgeGraph, tg_manager=None, chunks: list = None) -> dict:
    query_entities = identify_query_entities(query)
    
    graph_context = []
    if query_entities:
        graph_context = kg.multi_hop_search(query_entities, hops=2)

    # Deduplicate
    seen = set()
    unique_context = []
    for ctx in graph_context:
        if ctx not in seen and ctx:
            seen.add(ctx)
            unique_context.append(ctx)

    # If graph found nothing, use FAISS fallback on chunks
    fallback_used = False
    if not unique_context and chunks:
        unique_context = semantic_fallback(query, chunks, top_k=3)
        fallback_used = True
        if not query_entities:
            query_entities = ["general query"]

    # Limit context — KEY to token reduction
    filtered_context = unique_context[:2]
    context = "\n---\n".join([c[:500] for c in filtered_context])
    entity_list = ", ".join(query_entities[:5])

    system = "You are a precise assistant. Answer accurately and concisely using only the provided context."
    prompt = f"""Graph-retrieved context (entities: {entity_list}):
{context}

Question: {query}

Answer concisely and accurately:"""

    start = time.time()
    answer, in_tok, out_tok = chat(prompt, system=system)
    latency_ms = round((time.time() - start) * 1000)
    total_tokens = in_tok + out_tok

    return {
        "pipeline": "GraphRAG",
        "answer": answer,
        "tokens": total_tokens,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "latency_ms": latency_ms,
        "cost_usd": round(total_tokens * COST_PER_TOKEN, 6),
        "entities_found": query_entities,
        "context_chunks": len(filtered_context),
        "graph_nodes": len(kg.entities),
        "graph_edges": len(kg.relations),
        "graph_source": "TigerGraph Savanna" if (tg_manager and getattr(tg_manager, 'using_real_tg', False)) else "in-memory",
        "fallback_used": fallback_used,
    }


class TigerGraphManager:
    def __init__(self):
        self.using_real_tg = False
        print("Using optimized in-memory graph")


if __name__ == "__main__":
    tg = TigerGraphManager()
    KG_CACHE.unlink(missing_ok=True)
    kg, tg = build_graph_from_docs("data/dataset.txt", tg)
    result = run_graphrag("What are the main causes of climate change?", kg, tg)
    print(f"Entities: {result['entities_found']}")
    print(f"Answer: {result['answer'][:300]}")
    print(f"Tokens: {result['tokens']} | Chunks: {result['context_chunks']}")