from typing import List, TypedDict, Literal
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# =========================================================
# ENV + MODELS
# =========================================================

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# LOAD DOCUMENTS
# =========================================================

docs = (
    PyPDFLoader("./documents/Company_Policies.pdf").load()
    + PyPDFLoader("./documents/Company_Profile.pdf").load()
    + PyPDFLoader("./documents/Product_and_Pricing.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
).split_documents(docs)

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# =========================================================
# STATE
# =========================================================

class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str

# =========================================================
# HELPERS
# =========================================================

def build_chain(prompt, schema):
    return prompt | llm.with_structured_output(schema)

# =========================================================
# RETRIEVAL DECISION
# =========================================================

class RetrieveDecision(BaseModel):
    should_retrieve: bool

decide_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Decide if retrieval is needed. "
     "Use retrieval for company-specific questions. "
     "Return JSON only."),
    ("human", "{question}")
])

retrieve_chain = build_chain(decide_prompt, RetrieveDecision)

def decide_retrieval(state: State):
    result = retrieve_chain.invoke({
        "question": state["question"]
    })
    return {"need_retrieval": result.should_retrieve}

def route_after_decide(state: State):
    return "retrieve" if state["need_retrieval"] else "generate_direct"

# =========================================================
# DIRECT GENERATION
# =========================================================

direct_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using general knowledge only. "
     "If company-specific info is required say "
     "'I don't know based on general knowledge.'"),
    ("human", "{question}")
])

def generate_direct(state: State):
    result = llm.invoke(
        direct_prompt.format_messages(
            question=state["question"]
        )
    )
    return {"answer": result.content}

# =========================================================
# RETRIEVE
# =========================================================

def retrieve(state: State):
    query = state["retrieval_query"] or state["question"]
    docs = retriever.invoke(query)
    return {"docs": docs}

# =========================================================
# RELEVANCE FILTER
# =========================================================

class RelevanceDecision(BaseModel):
    is_relevant: bool

relevance_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Judge whether document is relevant to question. "
     "Use topic-level relevance. Return JSON only."),
    ("human",
     "Question:\n{question}\n\nDocument:\n{document}")
])

relevance_chain = build_chain(
    relevance_prompt,
    RelevanceDecision
)

def is_relevant(state: State):

    relevant_docs = []

    for doc in state["docs"]:

        result = relevance_chain.invoke({
            "question": state["question"],
            "document": doc.page_content
        })

        if result.is_relevant:
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: State):
    return (
        "generate_from_context"
        if state["relevant_docs"]
        else "no_answer_found"
    )

# =========================================================
# GENERATE FROM CONTEXT
# =========================================================

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from provided context. "
     "Do not mention context."),
    ("human",
     "Question:\n{question}\n\nContext:\n{context}")
])

def generate_from_context(state: State):

    context = "\n\n".join(
        d.page_content
        for d in state["relevant_docs"]
    )

    if not context:
        return {
            "answer": "No answer found.",
            "context": ""
        }

    result = llm.invoke(
        rag_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )

    return {
        "answer": result.content,
        "context": context
    }

def no_answer_found(state: State):
    return {
        "answer": "No answer found.",
        "context": ""
    }

# =========================================================
# ISSUP
# =========================================================

class IsSUPDecision(BaseModel):
    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support"
    ]
    evidence: List[str]

issup_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Verify whether answer is supported by context. "
     "Return fully_supported, partially_supported, "
     "or no_support. Return JSON only."),
    ("human",
     "Question:\n{question}\n\n"
     "Answer:\n{answer}\n\n"
     "Context:\n{context}")
])

issup_chain = build_chain(
    issup_prompt,
    IsSUPDecision
)

def is_sup(state: State):

    result = issup_chain.invoke({
        "question": state["question"],
        "answer": state["answer"],
        "context": state["context"]
    })

    return {
        "issup": result.issup,
        "evidence": result.evidence
    }

MAX_RETRIES = 5

def route_after_issup(state: State):

    if state["issup"] == "fully_supported":
        return "accept_answer"

    if state["retries"] >= MAX_RETRIES:
        return "accept_answer"

    return "revise_answer"

# =========================================================
# REVISE
# =========================================================

revise_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Revise answer strictly using supported context."),
    ("human",
     "Question:\n{question}\n\n"
     "Answer:\n{answer}\n\n"
     "Context:\n{context}")
])

def revise_answer(state: State):

    result = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            context=state["context"]
        )
    )

    return {
        "answer": result.content,
        "retries": state["retries"] + 1
    }

def accept_answer(state: State):
    return {}

# =========================================================
# ISUSE
# =========================================================

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str

isuse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Judge whether answer is useful for the question. "
     "Return useful or not_useful. Return JSON only."),
    ("human",
     "Question:\n{question}\n\nAnswer:\n{answer}")
])

isuse_chain = build_chain(
    isuse_prompt,
    IsUSEDecision
)

def is_use(state: State):

    result = isuse_chain.invoke({
        "question": state["question"],
        "answer": state["answer"]
    })

    return {
        "isuse": result.isuse,
        "use_reason": result.reason
    }

# =========================================================
# QUERY REWRITE
# =========================================================

class RewriteDecision(BaseModel):
    retrieval_query: str

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite question for vector retrieval using "
     "keywords and entities. Return JSON only."),
    ("human",
     "Question:\n{question}\n\n"
     "Previous Query:\n{retrieval_query}\n\n"
     "Answer:\n{answer}")
])

rewrite_chain = build_chain(
    rewrite_prompt,
    RewriteDecision
)

def rewrite_question(state: State):

    result = rewrite_chain.invoke({
        "question": state["question"],
        "retrieval_query": state["retrieval_query"],
        "answer": state["answer"]
    })

    return {
        "retrieval_query": result.retrieval_query,
        "rewrite_tries": state["rewrite_tries"] + 1,
        "docs": [],
        "relevant_docs": [],
        "context": ""
    }

MAX_REWRITE_TRIES = 3

def route_after_isuse(state: State):

    if state["isuse"] == "useful":
        return "END"

    if state["rewrite_tries"] >= MAX_REWRITE_TRIES:
        return "no_answer_found"

    return "rewrite_question"

# =========================================================
# GRAPH
# =========================================================

g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_answer_found", no_answer_found)
g.add_node("is_sup", is_sup)
g.add_node("revise_answer", revise_answer)
g.add_node("is_use", is_use)
g.add_node("rewrite_question", rewrite_question)

g.add_edge(START, "decide_retrieval")

g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "retrieve": "retrieve",
        "generate_direct": "generate_direct"
    }
)

g.add_edge("generate_direct", END)

g.add_edge("retrieve", "is_relevant")

g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found"
    }
)

g.add_edge("no_answer_found", END)

g.add_edge("generate_from_context", "is_sup")

g.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "accept_answer": "is_use",
        "revise_answer": "revise_answer"
    }
)

g.add_edge("revise_answer", "is_sup")

g.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found"
    }
)

g.add_edge("rewrite_question", "retrieve")

# =========================================================
# COMPILE
# =========================================================

app = g.compile()

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = app.invoke({
        "question": "Describe NexaAI company culture.",
        "retrieval_query": "",
        "rewrite_tries": 0,
        "need_retrieval": True,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": ""
    })

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nISSUP:", result["issup"])
    print("ISUSE:", result["isuse"])