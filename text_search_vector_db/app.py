import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Assuming your dataset and embeddings are now locally available
# Update these paths to where you have stored them locally
dataset_path = 'metadata.csv'
embeddings_path = 'document_embeddings.npy'

df = pd.read_csv(dataset_path)
embeddings = np.load(embeddings_path)

st_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def get_query_embedding(question, st_model):
    return st_model.encode([question])[0]

def find_top_n_relevant_docs(query_embedding, embeddings, n=10):
    from scipy.spatial.distance import cdist
    distances = cdist(query_embedding.reshape(1, -1), embeddings, 'cosine')[0]
    indices = distances.argsort()[:n]
    return indices

def answer_question(question, context, qa_tokenizer, qa_model):
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = qa_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# Streamlit UI
st.title('QnA System with BERT-SQuAD and Vector DB')

question = st.text_input('Ask a question about Parkinsons Disease:')

if st.button('Search Answer'):
    if question:
        query_embedding = get_query_embedding(question, st_model)
        top_docs_indices = find_top_n_relevant_docs(query_embedding, embeddings, n=10)

        for idx in top_docs_indices:
            doc_details = df.iloc[idx]  # Get the document details
            context = doc_details['abstract']  # Assuming 'abstract' contains the context
            answer = answer_question(question, context, qa_tokenizer, qa_model)
            
            # Now also display pubmed_id, title, and abstract along with the answer
            st.markdown(f"**Pubmed ID:** {doc_details['pubmed_id']}")
            st.markdown(f"**Title:** {doc_details['title']}")
            st.markdown(f"**Abstract:** {context}")
            st.markdown(f"**Answer:** {answer}")
            st.markdown("---")  # Add a separator for readability
    else:
        st.write("Please enter a question.")