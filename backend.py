import os
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback

load_dotenv()

# --- Initialize Global Clients ---
chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH"))
collection = chroma_client.get_collection(name=os.getenv("COLLECTION_NAME"))
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it", 
    temperature=0,
    max_retries=5
)

# --- Define State ---
class RAGState(TypedDict):
    question: str
    retrieved_chunks: List[Dict[str, str]]
    reasoning_process: str
    final_answer: str
    workflow_logs: List[str]

# --- Helper to handle complex LLM responses ---
def extract_text(content) -> str:
    """Extracts string from LangChain message content, handling reasoning blocks."""
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        extracted = ""
        for block in content:
            if isinstance(block, dict):
                # If the model shares its hidden thinking process
                if "thought" in block:
                    extracted += f"🤔 **Internal Thought:**\n{block['thought']}\n\n"
                # The actual text output
                if "text" in block:
                    extracted += block["text"]
            elif isinstance(block, str):
                extracted += block
        return extracted
        
    return str(content)

# --- Node 2: Retrieve Nodes ---
def retrieve_node(state: RAGState):
    question = state["question"]
    query_embedding = embedding_model.encode(question).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved_data = []
    log_details = f"🔍 **Retrieval Phase:** Searched ChromaDB for '{question}'.\n"
    
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        
        retrieved_data.append({"content": text, "page": meta['page']})
        log_details += f"- Found Chunk {i+1} (Page {meta['page']}) | Distance: {dist:.4f}\n"
        
    return {
        "retrieved_chunks": retrieved_data,
        "workflow_logs": state.get("workflow_logs", []) + [log_details]
    }

# --- Node 2: Reasoning ---
def reason_node(state: RAGState):
    question = state["question"]
    chunks = state["retrieved_chunks"]
    
    # Format chunks to explicitly include Source and Page
    context_str = "\n\n".join([f"[Source: texas_AI_book.pdf, Page {c['page']}]:\n{c['content']}" for c in chunks])
    
    reasoning_prompt = f"""
    You are an internal analytical engine. You must strictly follow these rules:
    1. ONLY use the provided context to answer the question. 
    2. Do NOT use outside knowledge or external information.
    3. If the context does not contain the answer, you must state: "The provided document does not contain this information."
    
    Analyze the context against the user's question. Write down your thought process, identify the key points, and explicitly note which page the data comes from.
    
    Question: {question}
    
    Context:
    {context_str}
    
    Internal Reasoning:
    """
    
    response = llm.invoke(reasoning_prompt)
    reasoning_text = extract_text(response.content)

    log_details = f"🧠 **Reasoning Phase (Strict Grounding):**\n{reasoning_text}"
    
    return {
        "reasoning_process": reasoning_text,
        "workflow_logs": state.get("workflow_logs", []) + [log_details]
    }

# --- Node 3: Summarization ( with Citations) ---
def summarize_node(state: RAGState):
    question = state["question"]
    reasoning = state["reasoning_process"]
    
    summary_prompt = f"""
    You are an expert AI assistant answering questions based strictly on the provided internal reasoning.
    
    Rules for your response:
    1. Formulate a clear, professional answer.
    2. You MUST cite your sources at the end of every relevant claim or paragraph.
    3. Format your citations exactly like this: (Source: texas_AI_book.pdf, Page X). 
    4. Try to mention the section or topic if it is obvious from the reasoning.
    5. If the reasoning states the information is not in the document, inform the user politely without making up an answer.
    
    User Question: {question}
    
    Internal Reasoning:
    {reasoning}
    
    Final Answer:
    """
    
    response = llm.invoke(summary_prompt)
    final_answer = extract_text(response.content)
    log_details = "📝 **Summarization Phase:** Final answer formulated with strict citations."
    
    return {
        "final_answer": final_answer,
        "workflow_logs": state.get("workflow_logs", []) + [log_details]
    }

# --- Compile Graph ---
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("reason", reason_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "summarize")
workflow.add_edge("summarize", END)

app_graph = workflow.compile()

def process_query(question: str):
    initial_state = {"question": question, "workflow_logs": []}
    return app_graph.invoke(initial_state)

# ==========================================
# FastAPI Implementation
# ==========================================

app = FastAPI(title="Agentic RAG API", version="1.0")

# Request and Response Data Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    final_answer: str
    workflow_logs: List[str]

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    try:
        # Run the LangGraph workflow
        result = process_query(request.question)
        
        return QueryResponse(
            final_answer=result["final_answer"],
            workflow_logs=result["workflow_logs"]
        )
    except Exception as e:
        # This is the critical part: Print the full error to your terminal
        print("❌ ERROR DETECTED IN BACKEND:")
        traceback.print_exc() 
        
        # Also send the specific error message back to Streamlit
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)