from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os
import pickle

app = FastAPI()

# Define input and output models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Function to load documents from PDFs
def load_documents(directory_path):
    pdf_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".pdf")
    ]
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    return documents

# Function to load or save preprocessed data
def load_preprocessed_data(directory_path):
    emb_file = os.path.join(directory_path, 'embeddings.pkl')
    chunk_file = os.path.join(directory_path, 'chunks.pkl')
    if os.path.exists(emb_file) and os.path.exists(chunk_file):
        with open(emb_file, 'rb') as f:
            embeddings_data = pickle.load(f)
        with open(chunk_file, 'rb') as f:
            chunks_data = pickle.load(f)
        return embeddings_data, chunks_data
    return None, None

def save_preprocessed_data(directory_path, embeddings_data, chunks_data):
    os.makedirs(directory_path, exist_ok=True)
    emb_file = os.path.join(directory_path, 'embeddings.pkl')
    chunk_file = os.path.join(directory_path, 'chunks.pkl')
    with open(emb_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunks_data, f)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="LazarusNLP/all-indo-e5-small-v4")

# Path to preprocessed data
preprocessed_data_dir = './preprocessed_data'

# Load or preprocess data
embeddings_data, chunks_data = load_preprocessed_data(preprocessed_data_dir)
if embeddings_data is None or chunks_data is None:
    documents = load_documents("docs/")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [text_splitter.split_documents([doc]) for doc in documents]
    chunks = [chunk for sublist in chunks for chunk in sublist]
    embeddings_data = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    save_preprocessed_data(preprocessed_data_dir, embeddings_data, chunks)
else:
    chunks = chunks_data

vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_store")
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# Initialize the LLM
llm = OllamaLLM(model="llama3")

# Microservices check
@app.get("/check")   
def health_check():
    return {"status": "ok"}

# API endpoint
@app.post("/rag", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    query = request.query

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # Ensure diverse retrieval
    source_count = {}
    diversified_docs = []
    for doc in retrieved_docs:
        source = doc.metadata['source']
        if source_count.get(source, 0) < 5:
            diversified_docs.append(doc)
            source_count[source] = source_count.get(source, 0) + 1

    # Assess query complexity
    complex_keywords = {"kenapa", "mengapa", "bagaimana", "jelaskan"}
    is_complex = len(query.split()) > 7 or any(word in query.lower() for word in complex_keywords)

    if is_complex:
        # Late chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = [text_splitter.split_documents([doc]) for doc in diversified_docs]
        chunks = [chunk for sublist in chunks for chunk in sublist]
    else:
        # Use pre-chunked documents
        chunks = diversified_docs

    # Combine context
    context = "\n\n".join([chunk.page_content for chunk in chunks])

    # Prepare prompt
    prompt = f"""
    You are an expert medical AI assistant. Use the provided context and your own knowledge to answer the question in a clear, concise, and professional manner. Remember, always answer in Bahasa Indonesia.

    ### Instructions:
    1. First, prioritize using the context to provide the answer.
    2. If additional information is needed, supplement your response with your own knowledge (use Bahasa Indonesia).
    4. If the context does not answer the question and you rely solely on your own knowledge, clearly state that no external sources were used (use Bahasa Indonesia).

    ### Context:
    {context}

    ### Question:
    {query}

    ### Answer:
    """

    # Generate response
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(response=response)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
