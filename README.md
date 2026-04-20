# 🤖 Transformer Intelligence Desk

> A production-ready, modular Agentic AI system for **Research Paper Q&A** — grounded on *"Attention Is All You Need"* (Vaswani et al., 2017).

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Overview

**Transformer Intelligence Desk** is a 6-node LangGraph agent that answers precise, fact-grounded questions about the Transformer architecture paper. It serves ML engineers, students, and researchers who want fast access to specific details — architecture hyperparameters, attention formulas, BLEU scores, and training setup — without re-reading 15 pages of dense mathematics.

Every answer is:
- **Grounded** — retrieved from a curated 14-document knowledge base, never from hallucination
- **Evaluated** — scored for faithfulness (≥ 0.7 threshold) before being shown to the user
- **Correctable** — the agent retries answers that fail the faithfulness gate (up to 2 retries)
- **Multi-modal** — routes between KB retrieval, conversation memory, and a calculator tool

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────┐    ┌─────────┐    ┌──────────┐
│  memory │───▶│  router │───▶│ retrieve │
└─────────┘    └─────────┘    └──────────┘
                    │               │
                    ├──── skip ─────┤
                    │               │
                    └──── tool ─────┤
                                    ▼
                              ┌──────────┐
                              │  answer  │
                              └──────────┘
                                    │
                              ┌─────▼────┐
                              │   eval   │◀─── retry if faithfulness < 0.7
                              └──────────┘
                                    │
                              ┌─────▼────┐
                              │   save   │
                              └──────────┘
                                    │
                              Final Answer
```

### Node Summary

| Node | Role |
|------|------|
| `memory` | Appends user question to sliding-window conversation history (last 6 messages) |
| `router` | LLM-based 3-way router: `retrieve` / `memory_only` / `tool` |
| `retrieve` | Hybrid multi-query semantic + lexical retrieval from ChromaDB (top-7 chunks) |
| `skip` | Bypasses retrieval for conversational follow-ups |
| `tool` | Safe calculator — LLM extracts expression, regex sanitizes, Python evaluates |
| `answer` | Grounded LLM answer with strict system prompt and retry reinforcement |
| `eval` | LLM-scored faithfulness gate (0.0–1.0); retries if score < 0.7 |
| `save` | Appends assistant answer to conversation history |

---

## ✨ Features

- **14-document knowledge base** — hand-authored, section-by-section coverage of the paper (Abstract → Conclusion)
- **Hybrid retrieval** — combines semantic similarity (sentence-transformers) and lexical overlap scoring across multi-query variants
- **Faithfulness gating** — every KB-retrieved answer is scored; low-quality answers are retried with stricter grounding instructions
- **Calculator tool** — handles arithmetic questions (BLEU deltas, attention head math, parameter counts)
- **Conversation memory** — MemorySaver persistence across turns within a session
- **Red-team robustness** — correctly refuses out-of-scope questions and corrects false-premise questions
- **Model fallback** — automatically switches between `llama-3.3-70b-versatile` → `llama-3.1-8b-instant` → `llama3-8b-8192` on rate limits
- **Premium Streamlit UI** — glassmorphism design, custom navbar, animated chat, source citations

---

## 📊 Evaluation Results

### RAGAS Baseline (5 ground-truth Q&A pairs)

| Metric | Score |
|--------|-------|
| Faithfulness | **0.860** |
| Answer Relevance | **0.512** |
| Context Precision | **0.919** |

### Test Suite (10 questions: 8 domain + 2 red-team)

| Test Type | Result |
|-----------|--------|
| All node isolated tests | ✅ PASS |
| 3-turn conversation memory | ✅ PASS |
| Router (3 routes) | ✅ PASS |
| Calculator tool | ✅ PASS |
| Out-of-scope refusal | ✅ PASS |
| False-premise correction | ✅ PASS |

---

## 🗂️ Project Structure

```
├── agent.py                  # Shared agent module (build_graph, load_agent)
├── capstone_streamlit.py     # Streamlit chat UI — run this to launch the app
├── day13_capstone.ipynb      # Full capstone notebook (8-part workflow)
├── requirements.txt          # Python dependencies
├── .env                      # GROQ_API_KEY (not committed)
├── .streamlit/
│   └── config.toml           # Streamlit theme config
└── code files/               # Course reference notebooks (Day 1–12)
```

---

## 📚 Knowledge Base Coverage

14 documents, one per major section of the paper:

| # | Topic |
|---|-------|
| 1 | Abstract — What the Transformer Is |
| 2 | Introduction — Why the Transformer Was Needed |
| 3 | Background — Prior Work and Self-Attention |
| 4 | Model Architecture Overview — Encoder-Decoder Structure |
| 5 | Encoder Stack — Structure and Sub-Layers |
| 6 | Decoder Stack — Structure, Masking, and Encoder-Decoder Attention |
| 7 | Scaled Dot-Product Attention — Formula and Why Scaling Matters |
| 8 | Multi-Head Attention — h Heads, Dimensions, and Three Use Cases |
| 9 | Position-wise Feed-Forward Networks — FFN Structure and Dimensions |
| 10 | Positional Encoding — How the Transformer Knows Token Order |
| 11 | Training Setup — Data, Hardware, Optimizer, and Regularization |
| 12 | Results and Benchmarks — BLEU Scores and Model Configurations |
| 13 | Ablation Study — Effect of Architecture Choices (Table 3) |
| 14 | Conclusion and Impact — Summary of Contributions and Future Work |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### 1. Clone the repository

```bash
git clone https://github.com/Adityaroy000/Transformer-Intelligence-Desk.git
cd Transformer-Intelligence-Desk
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the App

```bash
streamlit run capstone_streamlit.py
```

The app will open at `http://localhost:8501`.

### Example Questions to Try

```
# Knowledge base questions
How many attention heads does the base Transformer use and what is d_k?
What is the scaled dot-product attention formula?
Which optimizer was used and what were beta1 and beta2?
What BLEU score did Transformer big achieve on English-to-French?

# Calculator questions
What is 28.4 minus 26.36?
What is 512 divided by 8?

# Red-team questions (tests robustness)
The Transformer uses recurrence and only 4 heads, right?
What is the capital of France?
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — `llama-3.3-70b-versatile` + fallback models |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (in-memory) |
| Agent framework | LangGraph (StateGraph + MemorySaver) |
| LLM client | LangChain + `langchain-groq` |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |
| UI | Streamlit with custom CSS glassmorphism |
| Env management | python-dotenv |

---

## 📖 Paper Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
> **Attention Is All You Need.**
> *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
> [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---