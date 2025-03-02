import openai
import faiss
import numpy as np
import tiktoken

# OpenAI API key (replace with your actual key)
OPENAI_API_KEY = "your-api-key"

# Sample documents (these can be from PDFs, articles, etc.)
documents = [
    "Quantum computing is a type of computation that harnesses the power of quantum mechanics.",
    "Neural networks are a subset of machine learning algorithms used for deep learning.",
    "The capital of France is Paris, known for its rich history and culture.",
]

# Function to get embeddings from OpenAI
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    return np.array(response["data"][0]["embedding"])

# Convert all documents to embeddings
doc_embeddings = np.array([get_embedding(doc) for doc in documents])

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Function to retrieve top-k relevant documents
def retrieve_documents(query, top_k=2):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Function to generate response using OpenAI GPT-4
def generate_response(query):
    retrieved_docs = retrieve_documents(query)
    context = "\n".join(retrieved_docs)

    prompt = f"Using the following retrieved information:\n{context}\nAnswer the question: {query}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )

    return response["choices"][0]["message"]["content"]

# Example Query
query = "What is quantum computing?"
response = generate_response(query)
print(response)
