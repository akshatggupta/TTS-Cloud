import streamlit as st
from rag_pipeline import process_pdf, answer_question, init_pinecone

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talk to Syllabus",
    page_icon="📚",
    layout="centered"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Dark academic theme */
.stApp {
    background: #0f0e0c;
    color: #e8e0d0;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0c060;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.subtitle {
    color: #8a7f70;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 2rem;
}

.chat-bubble-user {
    background: #1e1c18;
    border: 1px solid #3a3530;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8e0d0;
}
.chat-bubble-ai {
    background: #1a1f14;
    border: 1px solid #2e3a28;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #c8ddb8;
}
.source-tag {
    font-size: 0.72rem;
    color: #6a6050;
    margin-top: 6px;
}
.status-pill {
    display: inline-block;
    background: #1e2a18;
    color: #7acc50;
    border: 1px solid #3a5030;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 Talk to Syllabus</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any course syllabus PDF and ask questions about it</div>', unsafe_allow_html=True)

# ─── Sidebar: API Keys ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API Keys")
    st.caption("Keys are used only for your session and never stored.")

    groq_key = st.text_input("Groq API Key", type="password",
                              help="Get free key at console.groq.com")
    pinecone_key = st.text_input("Pinecone API Key", type="password",
                                  help="Get free key at pinecone.io")
    pinecone_index = st.text_input("Pinecone Index Name",
                                    value="syllabus-rag",
                                    help="Create an index with dimension=384 in Pinecone")

    st.divider()
    st.markdown("### ℹ️ Setup Guide")
    st.markdown("""
1. **Groq**: [console.groq.com](https://console.groq.com) → Free API key
2. **Pinecone**: [pinecone.io](https://pinecone.io) → Create index  
   - Dimension: `384`  
   - Metric: `cosine`
3. Upload your syllabus PDF
4. Ask away! 🎓
    """)

# ─── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "index" not in st.session_state:
    st.session_state.index = None

# ─── PDF Upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"])

if uploaded_file and groq_key and pinecone_key and pinecone_index:
    if not st.session_state.pdf_processed:
        with st.spinner("🔍 Reading & indexing your syllabus..."):
            try:
                index = init_pinecone(pinecone_key, pinecone_index)
                st.session_state.index = index
                num_chunks = process_pdf(uploaded_file.read(), index)
                st.session_state.pdf_processed = True
                st.markdown(f'<div class="status-pill">✅ Indexed {num_chunks} chunks from "{uploaded_file.name}"</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
    else:
        st.markdown(f'<div class="status-pill">✅ Syllabus ready — "{uploaded_file.name}"</div>',
                    unsafe_allow_html=True)

elif not groq_key or not pinecone_key:
    st.info("👈 Add your API keys in the sidebar to get started.")

# ─── Chat Interface ─────────────────────────────────────────────────────────
if st.session_state.pdf_processed:
    st.divider()
    st.markdown("### 💬 Ask a Question")

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True)
            if msg.get("sources"):
                st.markdown(f'<div class="source-tag">📎 Sources: {msg["sources"]}</div>',
                            unsafe_allow_html=True)

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        suggestions = [
            "What are the prerequisites for this course?",
            "What topics are covered in Unit 3?",
            "How is the final grade calculated?",
            "What are the assignment deadlines?"
        ]
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion

    # Handle suggested question click
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                question, st.session_state.index, groq_key
            )
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources
        })
        st.rerun()

    # Text input
    question = st.chat_input("Ask anything about your syllabus...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                question, st.session_state.index, groq_key
            )
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources
        })
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
