import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import torch
import faiss
import lime.lime_text
from lime.lime_text import LimeTextExplainer
import time
import json
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BATCH_SIZE = 256
TOP_K = 5
TIME_LIMIT = 300  # 5 minutes

# Start Timer
start_time = time.time()

# Load Data
clinical_trials = pd.read_csv('usecase_1_.csv', engine='python', on_bad_lines='skip')
criteria = pd.read_csv('eligibilities.txt', sep='\t', header=None, names=['nct_id', 'Criteria'], engine='python')

# Preprocess Data
clinical_trials.fillna('', inplace=True)
criteria.fillna('', inplace=True)
clinical_trials = clinical_trials.merge(criteria, left_on='NCT Number', right_on='nct_id', how='left')

# Print column names to verify
print("Clinical Trials DataFrame Columns:", clinical_trials.columns)

# Vectorized Text Preprocessing
def preprocess_text(df, columns):
    for column in columns:
        df[column] = df[column].str.lower().replace(r'[^a-z0-9\s]', '', regex=True).replace(r'\s+', ' ', regex=True).fillna('')
    return df

# Ensure correct column names
columns_to_preprocess = ['Study Title', 'Primary Outcome Measures', 'Secondary Outcome Measures', 'Criteria']
clinical_trials = preprocess_text(clinical_trials, columns_to_preprocess)

# Sentence Embedding Model
model = SentenceTransformer('allenai/scibert_scivocab_uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Batch Embedding with Time Limit
def create_embeddings(texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        if time.time() - start_time > TIME_LIMIT:
            break
        batch = texts[i:i + BATCH_SIZE]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

# Prepare Text Data
text_data = clinical_trials.apply(
    lambda row: f"{row['Study Title']} {row['Primary Outcome Measures']} {row['Secondary Outcome Measures']} {row['Criteria']}",
    axis=1
).tolist()

embedding_matrix = create_embeddings(text_data)

# Build FAISS Index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
faiss.normalize_L2(embedding_matrix)  # Normalize for cosine similarity
index.add(embedding_matrix)

# Similarity Search
def search_similar_trials(query_text, Top_K=TOP_K):
    query_embedding = model.encode([query_text], convert_to_tensor=True, device=device).cpu().numpy()
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, Top_K)
    return [(idx, clinical_trials.iloc[idx]['NCT Number'], clinical_trials.iloc[idx]['Study Title'], distances[0][i]) for i, idx in enumerate(indices[0])]

# Cosine Similarity Evaluation
def evaluate_cosine_similarity(query_text, results):
    query_embedding = model.encode([query_text], convert_to_tensor=True, device=device).cpu().numpy()
    trial_embeddings = [embedding_matrix[idx] for idx, _, _, _ in results]
    similarities = cosine_similarity(query_embedding, trial_embeddings)
    return similarities.flatten()

# Recall@K Calculation
def calculate_recall_at_k(results, ground_truth_ids, k=TOP_K):
    top_k_ids = [res[0] for res in results[:k]]
    relevant_ids = set(ground_truth_ids)
    retrieved_relevant = len(set(top_k_ids) & relevant_ids)
    total_relevant = len(relevant_ids)
    recall_at_k = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
    return recall_at_k

# Visualization Functions
def visualize_similarity(results, cosine_similarities):
    study_titles = [res[2] for res in results]
    sns.barplot(x=cosine_similarities, y=study_titles, palette='viridis')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Study Title')
    plt.title('Similarity of Top Results')
    plt.tight_layout()
    plt.show()

def visualize_recall(recall_value):
    plt.figure(figsize=(6, 4))
    plt.bar(['Recall@K'], [recall_value], color='orange')
    plt.ylim(0, 1)
    plt.title('Recall@K')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

# LimeTextExplainer Integration
explainer = LimeTextExplainer(class_names=["Relevant", "Irrelevant"])

def lime_explain(query_text):
    explanation = explainer.explain_instance(
        query_text,
        classifier_fn=lambda x: model.encode(x, convert_to_tensor=True, device=device).cpu().numpy(),
        num_features=10
    )
    return explanation

# Query & Display Results
user_query = input("Enter your clinical trial search query: ")
results = search_similar_trials(user_query)

# Display Similar Trials
print("Similar Trials:")
for result in results:
    print(f"Index: {result[0]}, NCT ID: {result[1]}, Study Title: {result[2]}, Distance: {result[3]:.4f}")

# Cosine Similarity Evaluation
cosine_similarities = evaluate_cosine_similarity(user_query, results)
visualize_similarity(results, cosine_similarities)

# Recall@K Calculation (Assume ground_truth_ids are provided for the test query)
ground_truth_ids = []  # Replace with actual relevant NCT IDs for the query
recall_at_k = calculate_recall_at_k(results, ground_truth_ids)
print(f"\nRecall@{TOP_K}: {recall_at_k:.4f}")
visualize_recall(recall_at_k)

# Lime Explanation for User Query
print("\nLime Explanation for Query:")
explanation = lime_explain(user_query)
explanation.show_in_notebook(text=True)