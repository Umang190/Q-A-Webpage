from flask import Flask, request, jsonify
from langchain_community.llms import HuggingFacePipeline  # Updated import for HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated FAISS import
import requests
from bs4 import BeautifulSoup
from transformers import pipeline 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document  # Import Document class

# Hugging Face API configuration for embeddings
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": "Bearer "+ API_KEY}  # Replace with your Hugging Face API key

# Step 1: Fetch webpage content with User-Agent header
def fetch_webpage(url):
    headers = {"User-Agent": "MyLangChainBot/1.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("Webpage fetched successfully!")
        return response.text
    else:
        raise Exception(f"Failed to fetch webpage. Status code: {response.status_code}")

# Step 2: Extract data using BeautifulSoup
def extract_data(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(strip=True)
    documents = [text]  # Treat the full webpage text as one document
    print("Data extracted successfully!")
    return documents

# Step 3: Process data and create chunks
def process_data(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(documents[0])  # Split the single document into chunks
    print(f"Data split into {len(chunks)} chunks.")
    return chunks

# Step 4: Generate embeddings using Hugging Face API
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        payload = {"inputs": {"source_sentence": chunk, "sentences": [chunk]}}
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")

        if response.status_code == 200:
            embeddings.append(response.json()[0])  # Extract the embedding vector
        else:
            print(f"Error generating embedding for chunk: {chunk[:50]}... (status code: {response.status_code})")
            raise Exception(f"Failed to generate embeddings. Response: {response.text}")

    print("Embeddings generated successfully!")
    return embeddings

# Step 5: Store embeddings in vector store
def store_embeddings(chunks, embeddings):
    # Create HuggingFace embeddings object
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Wrap the text chunks in Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, hf_embeddings)
    vector_store.save_local("vector_store")
    print("Vector store created and saved successfully!")
    return vector_store, hf_embeddings

# Step 6: Load the vector store with embeddings
def load_vector_store(embeddings):
    vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully!")
    return vector_store

# Step 7: Create a QA chain using HuggingFacePipeline
def create_qa_chain(vector_store):
    # Use a Hugging Face pipeline for the question-answering model
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    llm = HuggingFacePipeline(pipeline=qa_model)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    return qa_chain

app = Flask(__name__)

# Entry point for running the Flask app
if __name__ == "__main__":
    # Step 1: Fetch webpage content
    url = "https://brainlox.com/courses/category/technical"
    html_content = fetch_webpage(url)

    # Step 2: Extract data
    documents = extract_data(html_content)

    # Step 3: Process data
    chunks = process_data(documents)

    # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 5: Store embeddings in vector store
    vector_store, hf_embeddings = store_embeddings(chunks, embeddings)

    # Step 6: Load the vector store with embeddings
    vector_store = load_vector_store(hf_embeddings)

    # Step 7: Create a QA chain
    qa_chain = create_qa_chain(vector_store)

    # Define the Flask API
    @app.route("/chat", methods=["POST"])
    def chat():
        # Get the user input
        user_input = request.json.get("message", "")

        if not user_input:
            return jsonify({"error": "No message provided!"}), 400

        # Use the QA chain to generate a response
        response = qa_chain.run(user_input)

        # Return the response as JSON
        return jsonify({"response": response})

    # Run the Flask app
    app.run(debug=True)
