import os
from openai import OpenAI
import faiss
import numpy as np

# OpenAI API key (replace with your actual key)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI()

# Sample documents (these can be from PDFs, articles, etc.)
documents = [
    "Product X for Capital Partners provides AI-enabled tools that expedite sourcing, evaluating, and due diligence on new opportunities.",
    "Product X for Climate Companies provides a platform to organize and present your investment opportunity in the best way possible for capital partners.",
]

# Function to get embeddings from OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input = [text], model=model).data[0].embedding
    return np.array(response)

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
