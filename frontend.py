import streamlit as st
from backend import app

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stTextInput > div > div > input {
    font-size: 18px;
}

.answer-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #444;
    color: white;
    font-size: 16px;
    line-height: 1.7;
}

.verdict-box {
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    font-size: 18px;
}

.correct {
    background-color: #198754;
    color: white;
}

.ambiguous {
    background-color: #ffc107;
    color: black;
}

.incorrect {
    background-color: #dc3545;
    color: white;
}

.small-box {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #444;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🤖 Advanced RAG Assistant")

st.markdown("""
Production-style RAG system using:

- Gemini 2.5 Flash
- FAISS
- Sentence Transformers
- LangGraph
- Tavily Web Search
- Retrieval Evaluation
""")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System Features")

    st.markdown("""
- PDF-based RAG
- Retrieval Evaluation
- Ambiguous Query Handling
- Web Search Fallback
- Sentence Filtering
- Gemini 2.5 Flash
- FAISS Vector Search
""")

# =========================================================
# USER INPUT
# =========================================================

question = st.text_input(
    "Ask Your Question"
)

# =========================================================
# BUTTON
# =========================================================

if st.button("Generate Answer"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating response..."):

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

        # =================================================
        # ANSWER
        # =================================================

        st.subheader("📌 Answer")

        st.markdown(
            f"""
<div class="answer-box">
{result["answer"]}
</div>
""",
            unsafe_allow_html=True
        )

        # =================================================
        # VERDICT
        # =================================================

        st.subheader("📊 Retrieval Verdict")

        verdict = result["verdict"]

        if verdict == "CORRECT":

            css_class = "correct"

        elif verdict == "AMBIGUOUS":

            css_class = "ambiguous"

        else:

            css_class = "incorrect"

        st.markdown(
            f"""
<div class="verdict-box {css_class}">
{verdict}
</div>
""",
            unsafe_allow_html=True
        )

        # =================================================
        # REASON
        # =================================================

        st.subheader("🧠 Reason")

        st.markdown(
            f"""
<div class="small-box">
{result["reason"]}
</div>
""",
            unsafe_allow_html=True
        )

        # =================================================
        # CONTEXT
        # =================================================

        with st.expander("📚 View Refined Context"):

            st.write(result["refined_context"])

        # =================================================
        # KEPT SENTENCES
        # =================================================

        with st.expander("✂️ View Filtered Sentences"):

            for idx, sentence in enumerate(
                result["kept_strips"],
                start=1
            ):

                st.markdown(f"**{idx}.** {sentence}")