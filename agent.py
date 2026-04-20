"""Shared agent module for the Day 13 capstone app."""

from datetime import datetime, timezone
import re
from typing import Any, List, TypedDict

import chromadb
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int


def build_graph(llm, embedder, collection):
    model_candidates = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"]
    llm_pool = {model_candidates[0]: llm}
    active_model = {"name": model_candidates[0]}

    def invoke_llm(payload):
        """Invoke with automatic model fallback to survive per-model rate limits."""
        ordered_models = [active_model["name"]] + [m for m in model_candidates if m != active_model["name"]]
        last_error = None

        for model_name in ordered_models:
            try:
                if model_name not in llm_pool:
                    llm_pool[model_name] = ChatGroq(model=model_name, temperature=0)
                response = llm_pool[model_name].invoke(payload)
                active_model["name"] = model_name
                return response
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            "All configured Groq models are currently unavailable or rate-limited. "
            "Please retry in a few minutes."
        ) from last_error

    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        q = question.lower().strip()

        time_triggers = [
            "utc time", "current utc", "time right now", "current time", "what time is it",
            "date and time", "current date", "today's date", "todays date"
        ]
        math_triggers = [
            "calculate", "plus", "minus", "multiplied", "times", "divided", "subtract", "add", "sum"
        ]
        memory_triggers = [
            "what is my name", "who am i", "remember", "what did i ask", "what did you just say",
            "based on what i asked", "based on our conversation", "earlier", "from above",
            "summarize our conversation", "summarise our conversation", "study plan"
        ]

        if any(t in q for t in time_triggers):
            return {"route": "tool"}

        if re.search(r"\d+\s*[+\-*/]\s*\d+", q) or any(t in q for t in math_triggers):
            return {"route": "tool"}

        if any(t in q for t in memory_triggers) or q in {"hi", "hello", "hey", "thanks", "thank you"}:
            return {"route": "memory_only"}

        return {"route": "retrieve"}

    def retrieval_node(state: CapstoneState) -> dict:
        question = state["question"]
        stopwords = {
            "the", "is", "are", "was", "were", "and", "or", "to", "of", "in", "on", "for",
            "a", "an", "it", "this", "that", "with", "as", "by", "at", "from", "right",
            "only", "just", "please"
        }

        q_tokens = {
            t for t in re.findall(r"[a-z0-9]+", question.lower())
            if len(t) > 0 and t not in stopwords
        }

        raw_parts = re.split(r"\b(?:and|or|but|while|whereas)\b|[?.,;:!]", question, flags=re.IGNORECASE)
        clause_queries = [p.strip() for p in raw_parts if len(p.strip().split()) >= 2]
        token_query = " ".join([t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in stopwords][:10])

        query_variants = [question] + clause_queries
        if token_query:
            query_variants.append(token_query)

        seen = set()
        deduped_queries = []
        for q in query_variants:
            q_norm = q.lower().strip()
            if q_norm and q_norm not in seen:
                seen.add(q_norm)
                deduped_queries.append(q)

        total_docs = max(1, collection.count())
        per_query_k = min(total_docs, 8)
        merged = {}

        for q_index, q in enumerate(deduped_queries):
            q_emb = embedder.encode([q]).tolist()
            results = collection.query(query_embeddings=q_emb, n_results=per_query_k)

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            for rank, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
                topic = (meta or {}).get("topic", "Unknown")
                text = f"{topic} {doc}".lower()
                d_tokens = set(re.findall(r"[a-z0-9]+", text))

                overlap = len(q_tokens & d_tokens)
                lexical_score = overlap / max(1, len(q_tokens))
                semantic_score = 1.0 / rank

                query_weight = 1.0 if q_index == 0 else 0.85
                combined = query_weight * (0.75 * semantic_score + 0.25 * lexical_score)

                prev = merged.get(doc_id)
                if prev is None or combined > prev[0]:
                    merged[doc_id] = (combined, topic, doc)

        ranked = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        top_k = ranked[:7]

        topics = [item[1] for item in top_k]
        chunks = [item[2] for item in top_k]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))

        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        q = question.lower()

        if any(t in q for t in ["utc time", "current utc", "time right now", "current time", "what time is it"]):
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return {"tool_result": f"Current UTC time: {now_utc}", "retrieved": "", "sources": []}

        normalized = q
        replacements = {
            "multiplied by": "*",
            "times": "*",
            "x": "*",
            "divided by": "/",
            "over": "/",
            "plus": "+",
            "minus": "-",
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)

        direct_match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", normalized)
        if direct_match:
            expression = f"{direct_match.group(1)} {direct_match.group(2)} {direct_match.group(3)}"
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                formatted = round(result, 4) if isinstance(result, float) else result
                return {"tool_result": f"Calculation: {expression} = {formatted}", "retrieved": "", "sources": []}
            except Exception as e:
                return {"tool_result": f"Calculator error: {str(e)}", "retrieved": "", "sources": []}

        extract_prompt = f"""Extract the mathematical expression from this question.
Output ONLY the math expression using operators: +, -, *, /
If there is no calculable math expression, output: none
Question: {question}"""

        try:
            raw = invoke_llm(extract_prompt).content.strip()
            if raw.lower() == "none" or not raw:
                return {"tool_result": "No arithmetic expression detected in this question.", "retrieved": "", "sources": []}

            expression = re.sub(r"[^0-9+\-*/().\s]", "", raw).strip()
            if not expression:
                return {"tool_result": f"Could not parse a valid math expression from: '{raw}'", "retrieved": "", "sources": []}

            result = eval(expression, {"__builtins__": {}}, {})
            if not isinstance(result, (int, float)):
                raise ValueError("Expression did not produce a numeric result")

            formatted = round(result, 4) if isinstance(result, float) else result
            tool_result = f"Calculation: {expression} = {formatted}"
        except Exception as e:
            tool_result = f"Calculator error: {str(e)}"

        return {"tool_result": tool_result, "retrieved": "", "sources": []}

    def answer_node(state: CapstoneState) -> dict:
        question = state["question"]
        route = state.get("route", "retrieve")
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        sources = state.get("sources", [])

        if route != "tool":
            tool_result = ""
        if route == "tool":
            retrieved = ""

        if route == "tool" and tool_result:
            return {"answer": tool_result}

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        source_line = ", ".join(sources) if sources else "None"

        if context:
            system_content = f"""You are an expert assistant for the research paper Attention Is All You Need.
1) Answer only from provided context.
2) If missing, say exactly: I don't have that information in my knowledge base.
3) Do not use outside knowledge.
4) If useful, cite relevant source topics from: {source_line}
5) If the question has a false premise and context contains correction, explicitly correct it.

{context}"""
        else:
            system_content = """You are a helpful assistant.
Use conversation history to answer memory/follow-up questions.
If asked for plans or recommendations based on earlier messages, synthesize a practical response from those messages.
If truly missing from conversation history, say exactly: I don't have that information in my knowledge base."""

        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Use only explicitly stated context facts."

        lc_msgs: List[Any] = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = invoke_llm(lc_msgs)
        return {"answer": response.content}

    def eval_node(state: CapstoneState) -> dict:
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:2500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""You are grading faithfulness only.
Rate from 0.0 to 1.0 based only on whether the answer is supported by context.
Return ONLY one number.
Context: {context}
Answer: {answer[:500]}"""

        raw = invoke_llm(prompt).content.strip()
        try:
            token = raw.split()[0].replace(",", ".")
            score = float(token)
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    def route_decision(state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        if route == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    graph = StateGraph(CapstoneState)
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())


def load_agent(documents: List[dict], collection_name: str = "capstone_kb"):
    """Initialize llm, embedding model, vector store, and compiled graph app."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    texts = [d["text"] for d in documents]
    ids = [d["id"] for d in documents]
    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in documents],
    )

    app = build_graph(llm, embedder, collection)
    topics = [d["topic"] for d in documents]
    return app, collection, topics
