"""
dashboard.py - GraphRAG Inference Dashboard (Neon Cyberpunk UI)
Optimized: All indexes cached on disk — instant load every time!
Run: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import time
import pickle
import json
from pathlib import Path
import os

st.set_page_config(
    page_title="GraphRAG Inference Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');
:root {
    --neon-blue: #00f5ff; --neon-purple: #bf00ff; --neon-green: #00ff88;
    --neon-orange: #ff6b00; --neon-pink: #ff00aa;
    --dark-bg: #030712; --card-bg: #0a0f1e; --card-border: #1a2444;
}
.stApp {
    background: var(--dark-bg) !important;
    background-image: 
        radial-gradient(ellipse at 20% 20%, rgba(0,245,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(191,0,255,0.04) 0%, transparent 50%),
        linear-gradient(rgba(0,245,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.02) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px !important;
}
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif !important; color: #c8d8f0 !important; }
.hero-title {
    font-family: 'Orbitron', monospace; font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple), var(--neon-pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; letter-spacing: 4px; text-transform: uppercase;
    animation: titleGlow 3s ease-in-out infinite alternate; margin-bottom: 0;
}
@keyframes titleGlow {
    from { filter: drop-shadow(0 0 10px rgba(0,245,255,0.5)); }
    to { filter: drop-shadow(0 0 25px rgba(191,0,255,0.8)); }
}
.hero-sub { font-family: 'Share Tech Mono', monospace; color: var(--neon-blue); text-align: center; font-size: 0.85rem; letter-spacing: 3px; margin-top: 4px; opacity: 0.8; }
.pipeline-card { background: var(--card-bg); border-radius: 16px; padding: 24px; margin: 8px 0; position: relative; overflow: hidden; }
.pipeline-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 16px 16px 0 0; }
.card-llm { border: 1px solid rgba(255,107,0,0.3); }
.card-llm::before { background: linear-gradient(90deg, var(--neon-orange), transparent); }
.card-rag { border: 1px solid rgba(0,245,255,0.3); }
.card-rag::before { background: linear-gradient(90deg, var(--neon-blue), transparent); }
.card-graph { border: 1px solid rgba(0,255,136,0.3); box-shadow: 0 0 30px rgba(0,255,136,0.08); }
.card-graph::before { background: linear-gradient(90deg, var(--neon-green), var(--neon-blue)); }
.card-title { font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; }
.card-llm .card-title { color: var(--neon-orange); }
.card-rag .card-title { color: var(--neon-blue); }
.card-graph .card-title { color: var(--neon-green); }
.pipeline-badge { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; margin-bottom: 14px; display: inline-block; letter-spacing: 1px; }
.badge-llm { background: rgba(255,107,0,0.15); border: 1px solid rgba(255,107,0,0.4); color: var(--neon-orange); }
.badge-rag { background: rgba(0,245,255,0.1); border: 1px solid rgba(0,245,255,0.3); color: var(--neon-blue); }
.badge-graph { background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: var(--neon-green); }
.answer-box { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 14px; font-size: 0.92rem; line-height: 1.7; color: #d0e4ff; margin: 12px 0; min-height: 120px; border-left: 3px solid; }
.answer-llm { border-color: var(--neon-orange); }
.answer-rag { border-color: var(--neon-blue); }
.answer-graph { border-color: var(--neon-green); }
.metric-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.metric-pill { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 8px 14px; flex: 1; min-width: 80px; text-align: center; border: 1px solid rgba(255,255,255,0.06); }
.metric-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: #5a7a9a; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 4px; }
.metric-value { font-family: 'Orbitron', monospace; font-size: 1.1rem; font-weight: 700; }
.mv-orange { color: var(--neon-orange); } .mv-blue { color: var(--neon-blue); } .mv-green { color: var(--neon-green); }
.token-section { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 28px; margin: 20px 0; }
.section-title { font-family: 'Orbitron', monospace; font-size: 0.85rem; letter-spacing: 3px; text-transform: uppercase; color: #5a7a9a; margin-bottom: 24px; }
.bar-row { margin-bottom: 20px; }
.bar-label { font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; margin-bottom: 6px; display: flex; justify-content: space-between; }
.bar-track { height: 12px; background: rgba(255,255,255,0.04); border-radius: 6px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; }
.bar-llm { background: linear-gradient(90deg, #ff4400, var(--neon-orange)); }
.bar-rag { background: linear-gradient(90deg, #0066ff, var(--neon-blue)); }
.bar-graph { background: linear-gradient(90deg, #00aa44, var(--neon-green)); box-shadow: 0 0 15px rgba(0,255,136,0.3); }
.win-badge { background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,245,255,0.1)); border: 1px solid var(--neon-green); border-radius: 12px; padding: 20px 28px; text-align: center; font-family: 'Orbitron', monospace; box-shadow: 0 0 30px rgba(0,255,136,0.15); margin-top: 16px; }
.win-number { font-size: 3rem; font-weight: 900; color: var(--neon-green); text-shadow: 0 0 20px rgba(0,255,136,0.8); display: block; }
.win-text { font-size: 0.75rem; letter-spacing: 3px; color: #5a9a7a; text-transform: uppercase; }
.explain-card { border-radius: 12px; padding: 20px 24px; margin: 8px 0; }
.explain-llm { background: rgba(255,107,0,0.06); border: 1px solid rgba(255,107,0,0.2); }
.explain-rag { background: rgba(0,245,255,0.05); border: 1px solid rgba(0,245,255,0.2); }
.explain-graph { background: rgba(0,255,136,0.05); border: 1px solid rgba(0,255,136,0.2); }
.explain-title { font-family: 'Orbitron', monospace; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
.explain-text { font-size: 1rem; line-height: 1.6; color: #a0b8d0; }
section[data-testid="stSidebar"] { background: #050a18 !important; border-right: 1px solid rgba(0,245,255,0.1) !important; }
.stButton > button { font-family: 'Orbitron', monospace !important; letter-spacing: 2px !important; font-size: 0.8rem !important; text-transform: uppercase !important; background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(191,0,255,0.15)) !important; border: 1px solid rgba(0,245,255,0.4) !important; color: var(--neon-blue) !important; border-radius: 8px !important; box-shadow: 0 0 15px rgba(0,245,255,0.1) !important; }
.stTextInput input { background: rgba(0,245,255,0.04) !important; border: 1px solid rgba(0,245,255,0.2) !important; border-radius: 8px !important; color: #d0e4ff !important; font-family: 'Rajdhani', sans-serif !important; font-size: 1rem !important; }
hr { border-color: rgba(0,245,255,0.08) !important; }
#MainMenu, footer, header { visibility: hidden; }
.scanning-line { width: 100%; height: 1px; background: linear-gradient(90deg, transparent, var(--neon-blue), transparent); animation: scan 3s ease-in-out infinite; margin: 8px 0; }
@keyframes scan { 0% { opacity: 0; transform: scaleX(0); } 50% { opacity: 1; transform: scaleX(1); } 100% { opacity: 0; transform: scaleX(0); } }
</style>
""", unsafe_allow_html=True)

# ── DISK CACHE PATHS ──
DATA_DIR = Path("data")
FAISS_PATH = DATA_DIR / "faiss_index.bin"
CHUNKS_PATH = DATA_DIR / "chunks_cache.json"
KG_PATH = DATA_DIR / "kg_cache.pkl"

# ── LOAD ALL DATA FROM DISK (instant after first run) ──

os.environ["TOKENIZERS_PARALLELISM"] = "false"
@st.cache_resource(show_spinner=False)
def load_all_data():
    import faiss
    import numpy as np
    from pipeline3_graphrag import TigerGraphManager, build_graph_from_docs, SimpleKnowledgeGraph

    # Load chunks
    if CHUNKS_PATH.exists():
        chunks = json.loads(CHUNKS_PATH.read_text())
    else:
        from pipeline2_basic_rag import load_and_chunk
        chunks = load_and_chunk("data/dataset.txt")
        CHUNKS_PATH.write_text(json.dumps(chunks))

    # Load FAISS index
    if FAISS_PATH.exists() and FAISS_PATH.stat().st_size > 1000:
        index = faiss.read_index(str(FAISS_PATH))
    else:
        from pipeline2_basic_rag import build_faiss_index
        index = build_faiss_index(chunks)

    # Load KG
    if KG_PATH.exists():
        with open(KG_PATH, "rb") as f:
            kg = pickle.load(f)
        tg = None
    else:
        tg = TigerGraphManager()
        kg, tg = build_graph_from_docs("data/dataset.txt", tg)
        with open(KG_PATH, "wb") as f:
            pickle.dump(kg, f)

    return chunks, index, kg, tg

# ── PRE-LOAD ON STARTUP ──
with st.spinner("⚡ Loading from cache..."):
    chunks, index, kg, tg = load_all_data()

# ── Hero Header ──
st.markdown("""
<div style="padding: 24px 0 16px 0;">
    <div class="hero-title">⚡ GraphRAG Inference</div>
    <div class="hero-sub">// TIGERGRAPH HACKATHON · BENCHMARK DASHBOARD · v2.0 //</div>
    <div class="scanning-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div style="font-family:\'Orbitron\',monospace;color:#00f5ff;font-size:0.85rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">⚙ Control Panel</div>', unsafe_allow_html=True)
    top_k = st.slider("RAG — Top-K Chunks", 1, 10, 5)
    graph_hops = st.slider("GraphRAG — Hop Depth", 1, 3, 2)
    st.divider()
    
    faiss_ok = FAISS_PATH.exists() and FAISS_PATH.stat().st_size > 1000
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#5a7a9a;
                line-height:2;background:rgba(0,245,255,0.04);padding:10px;
                border-radius:6px;border-left:2px solid rgba(0,245,255,0.3);">
    {'✅' if CHUNKS_PATH.exists() else '⭕'} Chunks Cache ({len(chunks)} chunks)<br>
    {'✅' if faiss_ok else '⭕'} FAISS Index<br>
    {'✅' if KG_PATH.exists() else '⭕'} Knowledge Graph ({len(kg.entities)} nodes)
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:#2a4a6a;line-height:1.8;">PIPELINE 1 · LLM ONLY<br>Raw model, zero retrieval<br><br>PIPELINE 2 · BASIC RAG<br>Vector search + LLM<br><br>PIPELINE 3 · GRAPHRAG<br>Graph traversal + LLM</div>', unsafe_allow_html=True)

# ── Query Input ──
col_q, col_btn = st.columns([4, 1])
with col_q:
    query = st.text_input("Enter your question", value="What are the main causes of climate change?", placeholder="Enter your question...")
with col_btn:
    run_btn = st.button("⚡ EXECUTE", use_container_width=True)

st.markdown("---")

if run_btn and query:
    from pipeline1_llm_only import run_llm_only
    from pipeline2_basic_rag import run_basic_rag
    from pipeline3_graphrag import run_graphrag

    progress = st.progress(0, text="Running Pipeline 1: LLM Only...")
    r1 = run_llm_only(query)
    progress.progress(33, text="Running Pipeline 2: Basic RAG...")
    r2 = run_basic_rag(query, chunks, index, top_k=top_k)
    progress.progress(66, text="Running Pipeline 3: GraphRAG...")
    r3 = run_graphrag(query, kg, tg, chunks=chunks)
    progress.progress(100, text="✅ Complete!")
    time.sleep(0.3)
    progress.empty()

    # SECTION 1
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.75rem;letter-spacing:4px;color:#3a5a7a;text-transform:uppercase;margin:8px 0 20px 0;">▸ Section 01 — Pipeline Responses</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="pipeline-card card-llm"><div class="card-title">🤖 LLM Only</div><span class="pipeline-badge badge-llm">BASELINE · NO RETRIEVAL</span><div class="answer-box answer-llm">{r1["answer"][:400]}{"..." if len(r1["answer"])>400 else ""}</div><div class="metric-row"><div class="metric-pill"><span class="metric-label">Tokens</span><span class="metric-value mv-orange">{r1["tokens"]}</span></div><div class="metric-pill"><span class="metric-label">Latency</span><span class="metric-value mv-orange">{r1["latency_ms"]}ms</span></div></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="pipeline-card card-rag"><div class="card-title">📄 Basic RAG</div><span class="pipeline-badge badge-rag">VECTOR SEARCH + LLM</span><div class="answer-box answer-rag">{r2["answer"][:400]}{"..." if len(r2["answer"])>400 else ""}</div><div class="metric-row"><div class="metric-pill"><span class="metric-label">Tokens</span><span class="metric-value mv-blue">{r2["tokens"]}</span></div><div class="metric-pill"><span class="metric-label">Latency</span><span class="metric-value mv-blue">{r2["latency_ms"]}ms</span></div></div></div>', unsafe_allow_html=True)
    with col3:
        graph_src = r3.get('graph_source', 'in-memory')
        st.markdown(f'<div class="pipeline-card card-graph"><div class="card-title">🔗 GraphRAG</div><span class="pipeline-badge badge-graph">GRAPH TRAVERSAL + LLM · {graph_src.upper()}</span><div class="answer-box answer-graph">{r3["answer"][:400]}{"..." if len(r3["answer"])>400 else ""}</div><div class="metric-row"><div class="metric-pill"><span class="metric-label">Tokens</span><span class="metric-value mv-green">{r3["tokens"]}</span></div><div class="metric-pill"><span class="metric-label">Latency</span><span class="metric-value mv-green">{r3["latency_ms"]}ms</span></div></div></div>', unsafe_allow_html=True)

    # SECTION 2
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.75rem;letter-spacing:4px;color:#3a5a7a;text-transform:uppercase;margin:32px 0 20px 0;">▸ Section 02 — How Each Pipeline Works</div>', unsafe_allow_html=True)
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown('<div class="explain-card explain-llm"><div class="explain-title" style="color:#ff6b00;">🤖 Pipeline 1 — LLM Only</div><div class="explain-text">Like asking a friend who hasn\'t read anything about the topic.<br><br><b>What it does:</b> Sends your question directly to AI with no extra info.<br><br><b>Problem:</b> AI guesses. Uses the MOST tokens.</div></div>', unsafe_allow_html=True)
    with exp2:
        st.markdown(f'<div class="explain-card explain-rag"><div class="explain-title" style="color:#00f5ff;">📄 Pipeline 2 — Basic RAG</div><div class="explain-text">Like Googling and pasting top results into your question.<br><br><b>What it does:</b> Searches {len(chunks)} chunks, picks similar ones, sends to AI.<br><br><b>Problem:</b> Dumps too much context — wastes tokens.</div></div>', unsafe_allow_html=True)
    with exp3:
        st.markdown(f'<div class="explain-card explain-graph"><div class="explain-title" style="color:#00ff88;">🔗 Pipeline 3 — GraphRAG</div><div class="explain-text">Like a detective connecting dots between clues.<br><br><b>What it does:</b> Graph of {len(kg.entities)} entities & {len(kg.relations)} relationships, {graph_hops}-hop traversal for only relevant context.<br><br><b>Result:</b> FEWER tokens, SMARTER answers. ✅</div></div>', unsafe_allow_html=True)

    # SECTION 3
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.75rem;letter-spacing:4px;color:#3a5a7a;text-transform:uppercase;margin:32px 0 20px 0;">▸ Section 03 — Token Battle</div>', unsafe_allow_html=True)
    max_tok = max(r1['tokens'], r2['tokens'], r3['tokens'])
    bar1_w = round(r1['tokens']/max_tok*100)
    bar2_w = round(r2['tokens']/max_tok*100)
    bar3_w = round(r3['tokens']/max_tok*100)
    saved = round(((r1['tokens']-r3['tokens'])/r1['tokens'])*100,1) if r1['tokens']>0 else 0
    col_bars, col_win = st.columns([2,1])
    with col_bars:
        st.markdown(f'<div class="token-section"><div class="section-title">// Token Usage Per Query</div><div class="bar-row"><div class="bar-label"><span style="color:#ff6b00;">🤖 LLM Only</span><span style="color:#ff6b00;font-family:\'Orbitron\',monospace;">{r1["tokens"]} tokens</span></div><div class="bar-track"><div class="bar-fill bar-llm" style="width:{bar1_w}%"></div></div></div><div class="bar-row"><div class="bar-label"><span style="color:#00f5ff;">📄 Basic RAG</span><span style="color:#00f5ff;font-family:\'Orbitron\',monospace;">{r2["tokens"]} tokens</span></div><div class="bar-track"><div class="bar-fill bar-rag" style="width:{bar2_w}%"></div></div></div><div class="bar-row"><div class="bar-label"><span style="color:#00ff88;">🔗 GraphRAG</span><span style="color:#00ff88;font-family:\'Orbitron\',monospace;">{r3["tokens"]} tokens</span></div><div class="bar-track"><div class="bar-fill bar-graph" style="width:{bar3_w}%"></div></div></div></div>', unsafe_allow_html=True)
    with col_win:
        st.markdown(f'<div class="win-badge"><span class="win-number">{saved}%</span><span class="win-text">Token Reduction<br>GraphRAG vs LLM Only</span></div>', unsafe_allow_html=True)

    # SECTION 4
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.75rem;letter-spacing:4px;color:#3a5a7a;text-transform:uppercase;margin:32px 0 20px 0;">▸ Section 04 — Full Benchmark Report</div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "Pipeline": ["🤖 LLM Only","📄 Basic RAG","🔗 GraphRAG"],
        "Total Tokens": [r1['tokens'],r2['tokens'],r3['tokens']],
        "Prompt Tokens": [r1['prompt_tokens'],r2['prompt_tokens'],r3['prompt_tokens']],
        "Completion Tokens": [r1['completion_tokens'],r2['completion_tokens'],r3['completion_tokens']],
        "Latency (ms)": [r1['latency_ms'],r2['latency_ms'],r3['latency_ms']],
    }).set_index("Pipeline")
    st.dataframe(df.style.highlight_min(subset=["Total Tokens"],color="#0a2a1a").highlight_min(subset=["Latency (ms)"],color="#0a1a2a").format({"Latency (ms)":"{:.0f} ms"}), use_container_width=True)

    # SECTION 5
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.75rem;letter-spacing:4px;color:#3a5a7a;text-transform:uppercase;margin:32px 0 20px 0;">▸ Section 05 — Graph Intelligence</div>', unsafe_allow_html=True)
    g1,g2,g3,g4 = st.columns(4)
    with g1: st.metric("Graph Nodes", len(kg.entities))
    with g2: st.metric("Graph Edges", len(kg.relations))
    with g3: st.metric("Entities Matched", len(r3.get('entities_found',[])))
    with g4: st.metric("Traversal Hops", graph_hops)
    if r3.get('entities_found'):
        st.markdown(f'<div style="background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.2);border-radius:10px;padding:14px 20px;margin-top:8px;"><span style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#3a9a6a;letter-spacing:1px;">ENTITIES FOUND → </span><span style="color:#00ff88;font-family:\'Rajdhani\',sans-serif;font-size:0.95rem;">{" · ".join(r3["entities_found"][:8])}</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;padding:40px 0 20px 0;"><div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;letter-spacing:4px;color:#2a4a6a;text-transform:uppercase;">// GraphRAG — Smarter · Faster · Cheaper · Built for TigerGraph Hackathon //</div></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;color:#2a4a6a;letter-spacing:3px;margin-bottom:40px;">
            ENTER A QUERY AND HIT EXECUTE TO BEGIN BENCHMARK
        </div>
        <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">
            <div style="background:rgba(255,107,0,0.06);border:1px solid rgba(255,107,0,0.2);border-radius:12px;padding:24px 32px;min-width:180px;">
                <div style="font-family:'Orbitron',monospace;color:#ff6b00;font-size:1.5rem;margin-bottom:8px;">01</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#7a4a2a;letter-spacing:1px;">LLM ONLY</div>
                <div style="font-size:0.85rem;color:#5a4a3a;margin-top:8px;">No retrieval<br>Baseline</div>
            </div>
            <div style="background:rgba(0,245,255,0.05);border:1px solid rgba(0,245,255,0.2);border-radius:12px;padding:24px 32px;min-width:180px;">
                <div style="font-family:'Orbitron',monospace;color:#00f5ff;font-size:1.5rem;margin-bottom:8px;">02</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#2a6a7a;letter-spacing:1px;">BASIC RAG</div>
                <div style="font-size:0.85rem;color:#2a5a6a;margin-top:8px;">Vector search<br>Industry standard</div>
            </div>
            <div style="background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.2);border-radius:12px;padding:24px 32px;min-width:180px;box-shadow:0 0 20px rgba(0,255,136,0.08);">
                <div style="font-family:'Orbitron',monospace;color:#00ff88;font-size:1.5rem;margin-bottom:8px;">03</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#2a7a5a;letter-spacing:1px;">GRAPHRAG ⚡</div>
                <div style="font-size:0.85rem;color:#2a6a4a;margin-top:8px;">Graph traversal<br>Next generation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)