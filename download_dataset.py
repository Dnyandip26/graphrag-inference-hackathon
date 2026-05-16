"""
download_dataset.py

Wikipedia se large interconnected dataset generate karta hai
for GraphRAG Inference Hackathon.

Features:
- 2M+ token target
- Linked article crawling
- Duplicate prevention
- Memory-safe page limits
- Better entity relationships for GraphRAG
"""

import wikipediaapi
from pathlib import Path

# 🔥 High-value interconnected topics
TOPICS = [
    # Climate & Environment
    "Climate change",
    "Global warming",
    "Greenhouse gas",
    "Carbon dioxide",
    "Deforestation",
    "Renewable energy",
    "Solar energy",
    "Wind power",
    "Paris Agreement",
    "Kyoto Protocol",
    "Carbon capture",
    "Ocean acidification",
    "Arctic ice",
    "Sea level rise",
    "Methane",
    "Fossil fuel",

    # Energy
    "Electric vehicle",
    "Battery electric vehicle",
    "Hydrogen fuel cell",
    "Nuclear power",
    "Coal",
    "Natural gas",
    "Petroleum",
    "Biomass",

    # Science
    "Atmosphere of Earth",
    "Carbon cycle",
    "Water cycle",
    "Biodiversity",
    "Ecosystem",
    "Photosynthesis",
    "Ozone layer",
    "Air pollution",

    # Organizations
    "United Nations",
    "IPCC",
    "Greenpeace",
    "World Wildlife Fund",
    "European Green Deal",
    "Net zero emissions",

    # AI & Technology
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Large language model",
    "Transformer model",
    "Neural network",
    "Computer vision",
    "Reinforcement learning",
    "GPT",
    "BERT",
    "Attention mechanism",

    # Companies
    "Google",
    "Microsoft",
    "OpenAI",
    "Meta Platforms",
    "Amazon",
    "Tesla",
    "Apple Inc",
    "NVIDIA",
    "IBM",
    "DeepMind",

    # Health
    "COVID-19 pandemic",
    "Vaccine",
    "Antibiotic resistance",
    "Cancer",
    "Diabetes",
    "Mental health",
    "Nutrition",

    # Economics
    "Inflation",
    "Gross domestic product",
    "Stock market",
    "Cryptocurrency",
    "Bitcoin",
    "Blockchain technology",
    "Supply chain",
    "Globalization",
    "International trade",

    # Additional connected topics
    "United States",
    "China",
    "India",
    "European Union",
    "World Bank",
    "Sustainable development",
    "Environmental policy",
    "Data science",
    "Cloud computing",
    "Quantum computing",
    "Cybersecurity",
    
    # History
    "World War II", "World War I", "Cold War", "French Revolution",
    "Industrial Revolution", "Roman Empire", "Ancient Egypt",
    "American Revolution", "British Empire", "Mongol Empire",

    # Space & Physics
    "Black hole", "Quantum mechanics", "Theory of relativity",
    "Big Bang", "Solar System", "Mars", "International Space Station",
    "James Webb Space Telescope", "SpaceX", "NASA",

    # Medicine
    "Alzheimer's disease", "Heart disease", "Immune system",
    "DNA", "Gene therapy", "CRISPR", "Stem cell",

    # Geography
    "Amazon rainforest", "Sahara desert", "Himalayas",
    "Pacific Ocean", "Atlantic Ocean", "Great Barrier Reef",
]


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation
    1 token ≈ 4 chars
    """
    return len(text) // 4


def get_linked_titles(page, limit=5):
    """
    Fetch linked Wikipedia article titles
    """
    try:
        links = list(page.links.keys())
        return links[:limit]
    except Exception:
        return []


def clean_text(text: str, max_chars: int = 50000) -> str:
    """
    Clean and limit text size
    """
    text = text.replace("\n\n\n", "\n\n")
    return text[:max_chars]


def download_wikipedia_dataset(
    output_dir: str = "data",
    target_tokens: int = 2_100_000
):
    """
    Build large GraphRAG-friendly dataset
    """

    Path(output_dir).mkdir(exist_ok=True)

    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='GraphRAG-Hackathon/1.0'
    )

    all_text = []
    total_tokens = 0
    articles_done = 0

    # Prevent duplicates
    visited_titles = set()

    print("=" * 70)
    print("🌍 Building GraphRAG Wikipedia Dataset")
    print("=" * 70)
    print(f"🎯 Target Tokens: {target_tokens:,}\n")

    for topic in TOPICS:

        if total_tokens >= target_tokens:
            break

        if topic in visited_titles:
            continue

        page = wiki.page(topic)

        if not page.exists():
            print(f"❌ SKIP: {topic}")
            continue

        visited_titles.add(topic)

        text = clean_text(page.text)

        tokens = estimate_tokens(text)

        total_tokens += tokens
        articles_done += 1

        all_text.append(f"\n=== {topic} ===\n{text}\n")

        print(
            f"✅ [{articles_done}] {topic} "
            f"| ~{tokens:,} tokens "
            f"| Total: ~{total_tokens:,}"
        )

        # 🔥 Crawl linked pages for better graph relationships
        linked_titles = get_linked_titles(page, limit=5)

        for linked_topic in linked_titles:

            if total_tokens >= target_tokens:
                break

            if linked_topic in visited_titles:
                continue

            linked_page = wiki.page(linked_topic)

            if not linked_page.exists():
                continue

            visited_titles.add(linked_topic)

            linked_text = clean_text(linked_page.text, max_chars=15000)

            linked_tokens = estimate_tokens(linked_text)

            total_tokens += linked_tokens
            articles_done += 1

            all_text.append(
                f"\n=== {linked_topic} ===\n{linked_text}\n"
            )

            print(
                f"   ↳ Linked: {linked_topic} "
                f"| ~{linked_tokens:,} tokens"
            )

    # Save dataset
    output_file = Path(output_dir) / "dataset.txt"

    output_file.write_text(
        "\n".join(all_text),
        encoding="utf-8"
    )

    size_mb = output_file.stat().st_size / 1024 / 1024

    print("\n" + "=" * 70)
    print("✅ DATASET BUILD COMPLETE")
    print("=" * 70)

    print(f"📚 Articles Downloaded : {articles_done}")
    print(f"🧠 Estimated Tokens    : ~{total_tokens:,}")
    print(f"💾 Dataset Size        : {size_mb:.2f} MB")
    print(f"📁 Saved To            : {output_file}")

    return str(output_file)


if __name__ == "__main__":
    download_wikipedia_dataset()