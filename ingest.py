import os
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

def ingest_pdf(file_path: str):
    print(f"Loading {file_path}...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    print("Initializing ChromaDB and Embedding Model...")
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH"))
    collection = client.get_or_create_collection(name=os.getenv("COLLECTION_NAME"))
    model = SentenceTransformer('all-MiniLM-L12-v2')

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to ChromaDB"):
        batch = chunks[i : i + batch_size]
        
        texts = [c.page_content for c in batch]
        # Store page number and source for logging
        # PyMuPDF is 0-indexed. Add +1 for human-readable page numbers.
        metadatas = [
            {
                "page": c.metadata.get("page", 0) + 1, 
                "source": file_path
            } 
            for c in batch
        ]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        
        embeddings = model.encode(texts).tolist()
        
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    # Ensure you have a PDF in the data folder
    ingest_pdf("data/AI_Russell_Norvig.pdf")