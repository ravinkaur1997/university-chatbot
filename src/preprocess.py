import json
from operator import index
from pickle import FALSE

import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import os

DATA_DIR = "../data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "intents.json")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_chatbot_data.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "university_chatbot_faiss.index")
QUERY_RESPONSE_PATH = os.path.join(DATA_DIR, "university_chatbot_queries_responses.csv")

device = torch.device("cpu")
faiss.omp_set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

with open(RAW_DATA_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

intents = data['intents']

queries, responses, intent_labels = [], [], []

for intent in intents:
    for text in intent['text']:
        queries.append(text.lower().strip())
        responses.append(intent['responses'][0])
        intent_labels.append(intent['intent'])

df = pd.DataFrame({'query': queries, 'response': responses, 'intent': intent_labels})

df.to_csv(CLEANED_DATA_PATH, index=False)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu()
    return embedding.squeeze().numpy()

embeddings = np.array([get_embedding(text) for text in df['query']])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)

df.to_csv(QUERY_RESPONSE_PATH, index=False)