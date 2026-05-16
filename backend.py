from typing import List, TypedDict
import re
import os

from dotenv import load_dotenv
from pydantic import BaseModel

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph
from langgraph.graph import StateGraph, START, END

load_dotenv()

# =========================================================
# LOAD PDF DOCUMENTS
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
# FREE EMBEDDING MODEL
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
# STATE
# =========================================================

class State(TypedDict):
    question: str
    docs: List[Document]

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
# FILTER MODEL
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

Output JSON only.
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

    context = "\n\n".join(
        d.page_content
        for d in state["docs"]
    ).strip()

    # STEP 1: DECOMPOSE
    strips = decompose_to_sentences(context)

    # STEP 2: FILTER
    kept = []

    for sentence in strips:

        result = filter_chain.invoke({
            "question": q,
            "sentence": sentence
        })

        if result.keep:
            kept.append(sentence)

    # STEP 3: RECOMPOSE
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
# LANGGRAPH
# =========================================================

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("refine", refine)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "refine")
graph.add_edge("refine", "generate")
graph.add_edge("generate", END)

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
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "answer": ""
    })

    print("\nANSWER:\n")
    print(result["answer"])