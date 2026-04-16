import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Agentic RAG Pipeline", layout="wide")

st.title("🤖 Gemma-4 Agentic RAG")
st.markdown("Ask questions based on your ingested PDF documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_logs" not in st.session_state:
    st.session_state.latest_logs = []

# Sidebar for Workflow Logs
with st.sidebar:
    st.header("⚙️ Server Logs")
    st.markdown("Transparency from the FastAPI Backend.")
    if st.session_state.latest_logs:
        for log in st.session_state.latest_logs:
            with st.expander(log.split("\n")[0], expanded=False):
                st.markdown(log)
    else:
        st.info("Logs will appear here after a query.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("E.g., Compare supervised and unsupervised learning..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Connecting to FastAPI Backend..."):
            try:
                # Make HTTP Request to FastAPI
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status() # Check for HTTP errors
                
                # Parse JSON response
                data = response.json()
                answer = data["final_answer"]
                logs = data["workflow_logs"]
                
                st.markdown(answer)
                
                # Update state
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.latest_logs = logs
                st.rerun()
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to connect to backend. Is the FastAPI server running?")
            except requests.exceptions.HTTPError as err:
                st.error(f"❌ API Error: {err}")