import faiss
import pandas as pd
import torch
from numpy.ma.core import indices
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

from src.preprocess import tokenizer

DATA_DIR = "../data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "university_chatbot_faiss.index")
QUERY_RESPONSE_PATH = os.path.join(DATA_DIR, "university_chatbot_queries_responses.csv")

index = faiss.read_index(FAISS_INDEX_PATH)
df = pd.read_csv(QUERY_RESPONSE_PATH)

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

if __name__ == "__main__":
    user_input = input("Ask something: ")
    bot_response = retrieve_response(user_input)
    print("Bot:", bot_response)