import faiss
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
import os


app = FastAPI(title="University Chatbot API")

# DATA_DIR = "../data"
# FAISS_INDEX_PATH = os.path.join(DATA_DIR, "university_chatbot_faiss.index")
# QUERY_RESPONSE_PATH = os.path.join(DATA_DIR, "university_chatbot_queries_responses.csv")

#Trial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of api.py
DATA_DIR = os.path.join(BASE_DIR, "../data")  # Adjust based on actual structure
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "university_chatbot_faiss.index")
QUERY_RESPONSE_PATH = os.path.join(DATA_DIR, "university_chatbot_queries_responses.csv")
FAISS_INDEX_PATH = os.path.abspath(FAISS_INDEX_PATH)
QUERY_RESPONSE_PATH = os.path.abspath(QUERY_RESPONSE_PATH)

print(f"FAISS index expected at: {QUERY_RESPONSE_PATH}")
print(f"Exists: {os.path.exists(QUERY_RESPONSE_PATH)}")


try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    # df = pd.DataFrame(QUERY_RESPONSE_PATH)
    df = pd.read_csv(QUERY_RESPONSE_PATH)
    print("FAISS index and query-response dataset loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index or dataset: {e}")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

def retrieve_response(user_query):
    query_embedding = get_embedding(user_query).reshape(1, -1)
    _, indices = index.search(query_embedding, k=1)

    matched_index = indices[0][0]
    response = df.iloc[matched_index]['response']
    return response

@app.get("/")
def home():
    return {"message": "Welcome to the University Chatbot API"}

@app.post("/chat/")
def chatbot(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    response = retrieve_response(query)
    return {"query": query, "response": response}
