import streamlit as st
import tempfile
import os
from rag_pipeline import load_pdf, get_chunks, build_vectorstore, build_qa_chain

st.set_page_config(page_title="DocChat — RAG Chatbot", page_icon="📄")
st.title("📄 DocChat")
st.caption("Upload a PDF and ask questions about it")

# Sidebar — file upload
with st.sidebar:
    st.header("Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if "vectorstore" not in st.session_state or \
           st.session_state.get("last_file") != uploaded_file.name:

            with st.spinner("Reading and indexing your document..."):
                # Save upload to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Build pipeline
                text = load_pdf(tmp_path)
                chunks = get_chunks(text)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.qa_chain = build_qa_chain(st.session_state.vectorstore)
                st.session_state.last_file = uploaded_file.name
                st.session_state.messages = []
                os.unlink(tmp_path)

            st.success(f"Indexed {len(chunks)} chunks!")
            st.info(f"Model: LLaMA 3 (Groq)")

# Chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
if question := st.chat_input("Ask a question about your document..."):
    if "qa_chain" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})