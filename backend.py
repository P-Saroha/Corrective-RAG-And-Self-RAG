from typing import List, TypedDict
import re
import os

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

load_dotenv()

# =========================================================
# LOAD DOCUMENTS
# =========================================================

docs = (
    PyPDFLoader("./documents/book1.pdf").load()
    + PyPDFLoader("./documents/book2.pdf").load()
    + PyPDFLoader("./documents/book3.pdf").load()
)

# =========================================================
# TEXT SPLITTING
# =========================================================

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

# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# FAISS VECTOR STORE
# =========================================================

vector_store = FAISS.from_documents(
    chunks,
    embedding_model
)

retriever = vector_store.as_retriever(
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
# RETRIEVAL EVALUATION THRESHOLDS
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

    answer: str

# =========================================================
# RETRIEVE NODE
# =========================================================

def retrieve(state: State) -> State:

    q = state["question"]

    retrieved_docs = retriever.invoke(q)

    return {
        "docs": retrieved_docs
    }

# =========================================================
# RETRIEVAL EVALUATION MODEL
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
1.0 = chunk alone can answer question
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

def evaluate_retrieval(state: State) -> State:

    q = state["question"]

    scores = []

    good_docs = []

    for doc in state["docs"]:

        result = doc_eval_chain.invoke({
            "question": q,
            "chunk": doc.page_content
        })

        scores.append(result.score)

        # Keep moderately relevant docs
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
# SENTENCE FILTER MODEL
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
# REFINE NODE
# =========================================================

def refine(state: State) -> State:

    q = state["question"]

    # Combine only good docs
    context = "\n\n".join(
        d.page_content
        for d in state["good_docs"]
    ).strip()

    # STEP 1 — DECOMPOSE
    strips = decompose_to_sentences(context)

    # STEP 2 — FILTER
    kept = []

    for sentence in strips:

        result = filter_chain.invoke({
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
# GENERATION PROMPT
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

"I don't know based on the provided books."
"""
        ),
        (
            "human",
            """
Question:
{question}

Context:
{refined_context}
"""
        )
    ]
)

# =========================================================
# GENERATE NODE
# =========================================================

def generate(state: State) -> State:

    chain = answer_prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "refined_context": state["refined_context"]
    })

    return {
        "answer": result.content
    }

# =========================================================
# FAIL NODE
# =========================================================

def fail_node(state: State) -> State:

    return {
        "answer": f"""
I could not find relevant information.

Reason:
{state['reason']}
"""
    }

# =========================================================
# AMBIGUOUS NODE
# =========================================================

def ambiguous_node(state: State) -> State:

    return {
        "answer": f"""
Retrieved information is ambiguous.

Reason:
{state['reason']}
"""
    }

# =========================================================
# ROUTER
# =========================================================

def route_after_eval(state: State):

    if state["verdict"] == "CORRECT":
        return "refine"

    elif state["verdict"] == "INCORRECT":
        return "fail"

    else:
        return "ambiguous"

# =========================================================
# LANGGRAPH
# =========================================================

graph = StateGraph(State)

# Nodes
graph.add_node("retrieve", retrieve)

graph.add_node("evaluate", evaluate_retrieval)

graph.add_node("refine", refine)

graph.add_node("generate", generate)

graph.add_node("fail", fail_node)

graph.add_node("ambiguous", ambiguous_node)

# Flow
graph.add_edge(START, "retrieve")

graph.add_edge("retrieve", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_after_eval,
    {
        "refine": "refine",
        "fail": "fail",
        "ambiguous": "ambiguous"
    }
)

graph.add_edge("refine", "generate")

graph.add_edge("generate", END)

graph.add_edge("fail", END)

graph.add_edge("ambiguous", END)

# Compile
app = graph.compile()

# =========================================================
# RUN
# =========================================================

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