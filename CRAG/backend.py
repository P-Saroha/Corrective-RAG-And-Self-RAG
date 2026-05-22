from typing import List, TypedDict
import re
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

# =========================================================
# LANGCHAIN
# =========================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# GEMINI
# =========================================================

from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================================
# LANGGRAPH
# =========================================================

from langgraph.graph import StateGraph, START, END

# =========================================================
# TAVILY
# =========================================================

from langchain_tavily import TavilySearch

# =========================================================
# ENV
# =========================================================

load_dotenv()

# =========================================================
# FAISS CACHE
# =========================================================

FAISS_INDEX_DIR = "./faiss_index"

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def _load_documents() -> List[Document]:

    return (
        PyPDFLoader("./documents/book1.pdf").load()
        + PyPDFLoader("./documents/book2.pdf").load()
        + PyPDFLoader("./documents/book3.pdf").load()
    )

def _split_documents(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    for d in chunks:
        d.page_content = (
            d.page_content
            .encode("utf-8", "ignore")
            .decode("utf-8", "ignore")
        )

    return chunks

@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:

    if os.path.isdir(FAISS_INDEX_DIR):
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            get_embedding_model(),
            allow_dangerous_deserialization=True
        )

    docs = _load_documents()
    chunks = _split_documents(docs)

    store = FAISS.from_documents(
        chunks,
        get_embedding_model()
    )

    store.save_local(FAISS_INDEX_DIR)

    return store

@lru_cache(maxsize=1)
def get_retriever():

    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

# =========================================================
# GEMINI 2.5 FLASH
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# =========================================================
# RETRY HANDLING
# =========================================================

def _extract_retry_seconds(error_text: str) -> float:

    match = re.search(r"retryDelay': '([0-9.]+)s'", error_text)

    if match:
        return float(match.group(1))

    return 5.0

def invoke_with_retry(chain, payload, max_attempts: int = 3):

    for attempt in range(1, max_attempts + 1):

        try:
            return chain.invoke(payload)

        except Exception as exc:

            error_text = str(exc)

            if "RESOURCE_EXHAUSTED" not in error_text:
                raise

            if attempt == max_attempts:
                raise

            delay = _extract_retry_seconds(error_text)
            time.sleep(delay)

# =========================================================
# RETRIEVAL THRESHOLDS
# =========================================================

UPPER_TH = 0.7
LOWER_TH = 0.3

# =========================================================
# STATE
# =========================================================

class State(TypedDict):

    question: str

    docs: List[Document]

    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]

    kept_strips: List[str]

    refined_context: str

    web_query: str

    web_docs: List[Document]

    answer: str

# =========================================================
# RETRIEVE NODE
# =========================================================

def retrieve_node(state: State) -> State:

    q = state["question"]

    retrieved_docs = get_retriever().invoke(q)

    return {
        "docs": retrieved_docs
    }

# =========================================================
# RETRIEVAL EVALUATION
# =========================================================

class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval evaluator for RAG.

You will be given ONE retrieved chunk and a question.

Return a relevance score in [0.0, 1.0].

Scoring:

1.0 = chunk alone can answer the question
0.0 = irrelevant chunk

Be conservative with high scores.

Return JSON only.
"""
        ),
        (
            "human",
            """
Question:
{question}

Chunk:
{chunk}
"""
        )
    ]
)

doc_eval_chain = (
    doc_eval_prompt
    | llm.with_structured_output(DocEvalScore)
)

# =========================================================
# RETRIEVAL EVALUATION NODE
# =========================================================

def eval_each_doc_node(state: State) -> State:

    q = state["question"]

    scores = []

    good_docs = []

    for doc in state["docs"]:

        result = invoke_with_retry(doc_eval_chain, {
            "question": q,
            "chunk": doc.page_content
        })

        scores.append(result.score)

        if result.score > LOWER_TH:
            good_docs.append(doc)

    # =====================================================
    # CORRECT
    # =====================================================

    if any(score > UPPER_TH for score in scores):

        return {
            "good_docs": good_docs,
            "verdict": "CORRECT",
            "reason": "At least one retrieved chunk is highly relevant."
        }

    # =====================================================
    # INCORRECT
    # =====================================================

    if len(scores) > 0 and all(score < LOWER_TH for score in scores):

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "All retrieved chunks are irrelevant."
        }

    # =====================================================
    # AMBIGUOUS
    # =====================================================

    return {
        "good_docs": good_docs,
        "verdict": "AMBIGUOUS",
        "reason": "Mixed retrieval quality."
    }

# =========================================================
# DECOMPOSITION
# =========================================================

def decompose_to_sentences(text: str) -> List[str]:

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]

# =========================================================
# SENTENCE FILTER
# =========================================================

class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict relevance filter.

Return keep=true only if the sentence
directly helps answer the question.

Return JSON only.
"""
        ),
        (
            "human",
            """
Question:
{question}

Sentence:
{sentence}
"""
        )
    ]
)

filter_chain = (
    filter_prompt
    | llm.with_structured_output(KeepOrDrop)
)

# =========================================================
# QUERY REWRITE FOR WEB SEARCH
# =========================================================

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the user question into a short web search query.

Rules:
- Keep it concise
- Use keywords only
- Add recency if needed
- Do NOT answer question

Return JSON only.
"""
        ),
        (
            "human",
            """
Question:
{question}
"""
        )
    ]
)

rewrite_chain = (
    rewrite_prompt
    | llm.with_structured_output(WebQuery)
)

def rewrite_query_node(state: State) -> State:

    result = invoke_with_retry(rewrite_chain, {
        "question": state["question"]
    })

    return {
        "web_query": result.query
    }

# =========================================================
# WEB SEARCH
# =========================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilySearch(
    max_results=5,
    tavily_api_key=TAVILY_API_KEY
)

def web_search_node(state: State) -> State:

    if not TAVILY_API_KEY:
        raise RuntimeError("Missing TAVILY_API_KEY in environment or .env file.")

    q = state.get("web_query") or state["question"]

    results = tavily.invoke(q)

    web_docs = []

    for r in results:

        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")

        text = f"""
TITLE: {title}

URL: {url}

CONTENT:
{content}
"""

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "url": url
                }
            )
        )

    return {
        "web_docs": web_docs
    }

# =========================================================
# REFINE NODE
# =========================================================

def refine(state: State) -> State:

    q = state["question"]

    # =====================================================
    # CORRECT
    # =====================================================

    if state["verdict"] == "CORRECT":

        docs_to_use = state["good_docs"]

    # =====================================================
    # INCORRECT
    # =====================================================

    elif state["verdict"] == "INCORRECT":

        docs_to_use = state["web_docs"]

    # =====================================================
    # AMBIGUOUS
    # =====================================================

    else:

        docs_to_use = (
            state["good_docs"]
            + state["web_docs"]
        )

    context = "\n\n".join(
        d.page_content
        for d in docs_to_use
    ).strip()

    # STEP 1 — DECOMPOSE
    strips = decompose_to_sentences(context)

    # STEP 2 — FILTER
    kept = []

    for sentence in strips:

        result = invoke_with_retry(filter_chain, {
            "question": q,
            "sentence": sentence
        })

        if result.keep:
            kept.append(sentence)

    # STEP 3 — RECOMPOSE
    refined_context = "\n".join(kept)

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context
    }

# =========================================================
# GENERATION
# =========================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI tutor.

Answer ONLY using the provided context.

If context is insufficient,
say:

"I don't know."
"""
        ),
        (
            "human",
            """
Question:
{question}

Context:
{context}
"""
        )
    ]
)

def generate(state: State) -> State:

    chain = answer_prompt | llm

    result = invoke_with_retry(chain, {
        "question": state["question"],
        "context": state["refined_context"]
    })

    return {
        "answer": result.content
    }

# =========================================================
# ROUTER
# =========================================================

def route_after_eval(state: State):

    if state["verdict"] == "CORRECT":

        return "refine"

    else:

        return "rewrite_query"

# =========================================================
# LANGGRAPH
# =========================================================

graph = StateGraph(State)

# =========================================================
# NODES
# =========================================================

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "evaluate",
    eval_each_doc_node
)

graph.add_node(
    "rewrite_query",
    rewrite_query_node
)

graph.add_node(
    "web_search",
    web_search_node
)

graph.add_node(
    "refine",
    refine
)

graph.add_node(
    "generate",
    generate
)

# =========================================================
# FLOW
# =========================================================

graph.add_edge(
    START,
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "evaluate"
)

graph.add_conditional_edges(
    "evaluate",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query"
    }
)

graph.add_edge(
    "rewrite_query",
    "web_search"
)

graph.add_edge(
    "web_search",
    "refine"
)

graph.add_edge(
    "refine",
    "generate"
)

graph.add_edge(
    "generate",
    END
)

# =========================================================
# COMPILE
# =========================================================

app = graph.compile()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    while True:

        question = input("\nAsk Question: ")

        if question.lower() == "exit":
            break

        result = app.invoke({

            "question": question,

            "docs": [],

            "good_docs": [],

            "verdict": "",
            "reason": "",

            "strips": [],

            "kept_strips": [],

            "refined_context": "",

            "web_query": "",

            "web_docs": [],

            "answer": ""
        })

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(result["answer"])

        print("\n" + "=" * 60)
        print("RETRIEVAL VERDICT")
        print("=" * 60)

        print(result["verdict"])

        print("\n" + "=" * 60)
        print("REASON")
        print("=" * 60)

        print(result["reason"])