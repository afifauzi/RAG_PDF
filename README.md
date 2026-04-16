# Agentic PDF RAG (AI Russell Norvig Book - Texas)

A stateful RAG system using **LangGraph** for reasoning and **FastAPI** for the backend.

## 🚀 Architecture
- **Orchestration:** LangGraph (Retrieve -> Reason -> Summarize)
- **LLM:** Gemma-4 31B (Google AI Studio)
- **Vector DB:** ChromaDB
- **Frontend:** Streamlit

## 🛠️ How to Run
1. Add your `AI_Russell_Norvig.pdf` to the `/data` folder.
2. Create a `.env` file with your `GOOGLE_API_KEY`,`CHROMA_PATH`,and `COLLECTION_NAME=ai_knowledge_base`.
3. [Skip this because I already run] Run `run_ingest.bat` to vectorize the document.
4. Run `start.bat` to launch the FastAPI server and Streamlit UI.

## Example Output
![](chatbot.png)

## Retrieval Phase
![](retrieval_phase.png)

## Reasoning Phase
![](reasoning_phase_1.png)
![](reasoning_phase_2.png)
