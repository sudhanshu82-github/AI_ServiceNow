import streamlit as st
import faiss
import pickle
import numpy as np

from offline_embedder import OfflineEmbedder
from offline_llm import OfflineLLM

# ============================
# Load FAISS + Metadata
# ============================

INDEX_PATH = "faiss_index/unified.index"
METADATA_PATH = "faiss_index/metadata.pkl"

try:
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        documents = pickle.load(f)
    st.sidebar.success(f"AI-Powered Incident Resolution Assistant Loaded With {index.ntotal} entries.")
except Exception as e:
    st.sidebar.error(f"Error loading FAISS index or metadata: {e}")
    st.stop()

embedder = OfflineEmbedder()
llm = OfflineLLM()


# ============================
# Helper Functions
# ============================

def build_context(hits, k=5):
    ctx = []
    for idx in hits[:k]:
        doc = documents[idx]

        if doc["source"] == "KB":
            ctx.append(
                f"[KB]\n"
                f"KB_ID: {doc.get('id','')}\n"
                f"Title: {doc.get('title','')}\n"
                f"Content: {doc.get('content','')}\n"
            )
        else:
            ctx.append(
                f"[INCIDENT]\n"
                f"INC_ID: {doc.get('id','')}\n"
                f"Short description: {doc.get('title','')}\n"
                f"Issue: {doc.get('content','')}\n"
                f"Linked_KB: {doc.get('kb_ref','')}\n"
                f"Resolution: {doc.get('resolution','')}\n"
            )

    return "\n\n".join(ctx)


def retrieve_evidence(short_description, top_k=5):
    try:
        query_vec = embedder.embed([short_description]).astype("float32")
        distances, indices = index.search(query_vec, top_k)
        hits = indices[0]

        context = build_context(hits, k=top_k)
        return context, hits
    except Exception as e:
        return f"Error during evidence retrieval: {e}", []


def generate_analysis(short_description, context):
    try:
        prompt = f"""
        You are an incident resolution assistant for a banking microservice platform.

        Use ONLY the information in the context to answer.
        If the answer is not clearly present, say: "I don't know based on current data."

        Incident short description:
        \"\"\"{short_description}\"\"\"

        Context:
        \"\"\"{context}\"\"\"

        Provide:
        - Likely root cause
        - Recommended resolution steps
        - Any linked KB or change references
        """

        answer = llm.generate(prompt)
        return answer
    except Exception as e:
        return f"Error during analysis generation: {e}"


# ============================
# Streamlit UI (Enhanced)
# ============================

st.set_page_config(
    page_title="AI-Powered Incident Resolution Assistant",
    layout="wide"
)

# ===== Sidebar =====
st.sidebar.title("Navigation")
st.sidebar.markdown("""
**Modules**
- Incident Analysis  
- Knowledge Base Lookup  
- Change History  
- System Health  
""")

top_k = st.sidebar.slider("FAISS Top‑K Results", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Offline RAG powered by FAISS + MiniLM + Mistral LLM")


# ===== Header =====
st.markdown("""
<div style="
    background-color:#1f2937;
    padding: 18px;
    border-radius: 6px;
    color: white;
    font-size: 24px;
    font-weight: 500;">
🚀 AI-Powered Incident Resolution Assistant
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("### Incident Analysis Console")


# ===== Input Section =====
st.markdown("#### Enter Incident Short Description")

query = st.text_input(
    "Short Description",
    placeholder="Example: Microservice gateway node down on csxraa01",
    help="Provide a brief description of the incident for analysis."
)

# Buttons for Evidence and Analysis
col1, col2, col3 = st.columns([1, 1, 1])
generate_evidence_btn = col1.button("📂 Generate Evidence")
generate_analysis_btn = col2.button("🧠 Generate Analysis")
reset_btn = col3.button("🔄 Reset")

# ===== Output Section =====
if reset_btn:
    st.session_state.clear()
    st.experimental_rerun()

if generate_evidence_btn:
    if not query.strip():
        st.error("Please enter a short description.")
    else:
        with st.spinner("🔍 Retrieving evidence..."):
            context, hits = retrieve_evidence(query, top_k)
            st.session_state["context"] = context
            st.session_state["hits"] = hits

        # Evidence Section
        st.markdown("## 📂 Evidence Retrieved")
        st.markdown("Relevant KB and Incident records identified from FAISS vector search.")
        st.code(context, language="text")

        st.markdown("#### Raw FAISS Hit Indexes")
        st.write(hits)

if generate_analysis_btn:
    if not query.strip():
        st.error("Please enter a short description.")
    elif "context" not in st.session_state:
        st.error("Please generate evidence first.")
    else:
        with st.spinner("🤖 Generating AI-powered analysis..."):
            answer = generate_analysis(query, st.session_state["context"])

        # Analysis Section
        st.markdown("## 🧠 Analysis Summary")
        st.markdown("AI-powered analysis based on the retrieved evidence:")
        st.write(answer)

        st.markdown("---")
        st.caption("Generated using offline vector search and local LLM inference.")