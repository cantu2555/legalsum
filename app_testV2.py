import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from collections import Counter

# Specify the directory where punkt is already downloaded
nltk_data_dir = r"D:/Data/OneDrive/Ccantu/OneDrive - CFTC/Documents/Python Scripts/punkt"
nltk.data.path.append(nltk_data_dir)

# Load Legal-BERT model
@st.cache_resource
def load_local_legal_bert():
    model_path = r"D:/Data/OneDrive/Ccantu/OneDrive - CFTC\Documents/Python Scripts/BERT-Legal"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model

# Load GPT-2 model
@st.cache_resource
def load_local_gpt2():
    gpt2_model_path = r"D:/Data/OneDrive/Ccantu/OneDrive - CFTC/Documents/Python Scripts/GPT2"
    model = AutoModelForCausalLM.from_pretrained(gpt2_model_path)
    tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
    return model, tokenizer
# A previous version using regrex
# Definition extraction
# def extract_definitions(text):
#     definition_patterns = [
#         r"(?P<term>\w+)\s+(?:is|means|refers to|is defined as)\s+(?P<definition>.*?)[;.]",
#         r"(?:the term\s+)?['\"](?P<term>\w+)['\"](?:\s+is|means|refers to|is defined as)\s+(?P<definition>.*?)[;.]",
#         r"(?P<definition>.*?)\s+(?:is|means|refers to|is defined as)\s+['\"](?P<term>\w+)['\"][;.]"
#     ]
#     definitions = {}
#     for pattern in definition_patterns:
#         for match in re.finditer(pattern, text, re.IGNORECASE):
#             term = match.group("term").lower()
#             definition = match.group("definition").strip()
#             definitions[term] = definition
#     return definitions

def extract_definitions(text):
    # Improved regex pattern to catch definitions more precisely
    definition_pattern = r'(?:^|\n)(?P<term>[\w\s-]+?)(?:\s+|:\s*)(?:means|is defined as|refers to)\s+(?P<definition>(?:(?!(?:\n\w+(?:\s+|:\s*)(?:means|is defined as|refers to))).)+)'
    
    # Find all matches in the text
    matches = re.finditer(definition_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    # Dictionary to store the definitions
    definitions = {}
    
    for match in matches:
        term = match.group('term').strip().lower()
        definition = match.group('definition').strip()
        
        # Clean up the definition
        definition = re.sub(r'\s+', ' ', definition)  # Replace multiple spaces with single space
        definition = definition.rstrip('.')  # Remove trailing period if present
        
        # Remove any trailing partial sentences
        definition = re.sub(r'\s+[A-Z][^.]*$', '', definition)
        
        # Store in the dictionary if the definition is not empty
        if definition:
            definitions[term] = definition
    
    return definitions

# Sentence scoring
def score_sentence(sentence, definitions, tfidf_matrix, sentence_similarities, sentence_index):
    score = 0
    sentence_lower = sentence.lower()
    
    # Score based on definitions
    for term, definition in definitions.items():
        if term in sentence_lower or definition.lower() in sentence_lower:
            score += 1
    
    # Score based on TF-IDF similarity
    score += np.mean(sentence_similarities[sentence_index])
    
    # Score based on position (favor sentences at the beginning and end)
    if sentence_index == 0 or sentence_index == len(sentence_similarities) - 1:
        score += 0.5
    
    # Score based on sentence length (avoid very short sentences)
    if len(sentence.split()) > 5:
        score += 0.2
    
    return score

# Extractive summarization
def extractive_summarize(text, num_sentences=3):
    sentences = nltk.sent_tokenize(text)
    definitions = extract_definitions(text)

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Compute sentence similarities
    sentence_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Rank sentences based on improved scoring
    sentence_scores = [score_sentence(sentence, definitions, tfidf_matrix, sentence_similarities, i) for i, sentence in enumerate(sentences)]
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[::-1]]
    
    # Select top sentences while maintaining order
    summary_sentences = []
    for sentence in sentences:
        if sentence in ranked_sentences[:num_sentences]:
            summary_sentences.append(sentence)
        if len(summary_sentences) == num_sentences:
            break
    
    summary = ' '.join(summary_sentences)
    return summary

# Process with Legal-BERT
def process_with_legal_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Simplify summary for layperson
def simplify_summary_for_layperson(summary, gpt2_model, gpt2_tokenizer):
    input_text = f"Simplify this legal text for a layperson: {summary}"
    inputs = gpt2_tokenizer(input_text, return_tensors='pt')
    
    outputs = gpt2_model.generate(
        **inputs,
        max_length=len(inputs['input_ids'][0]) + 200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    simplified_summary = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    simplified_summary = simplified_summary.replace(input_text, "").strip()
    
    return simplified_summary

# Streamlit app
def main():
    st.title("CFTC LawLogic Pro")
    st.subheader("Legal Text Summarizer and Simplifier")
    
    # Load models
    legal_bert_tokenizer, legal_bert_model = load_local_legal_bert()
    gpt2_model, gpt2_tokenizer = load_local_gpt2()

    # Text input
    text = st.text_area("Enter the legal text:", height=200)

    if st.button("Process Text"):
        if text:
            st.subheader("Original Text")
            st.write(text)

            # Extract definitions
            definitions = extract_definitions(text)
            st.subheader("Definitions Found")
            for term, definition in definitions.items():
                st.write(f"**{term}**: {definition}")

            # Process with Legal-BERT
            bert_output = process_with_legal_bert(text, legal_bert_tokenizer, legal_bert_model)
            st.subheader("Legal-BERT Processing")
            #st.write(f"Output shape: {bert_output.shape}")

            # Generate summary
            summary = extractive_summarize(text)
            st.subheader("Generated Summary")
            st.write(summary)

            # Simplify summary
            simplified_summary = simplify_summary_for_layperson(summary, gpt2_model, gpt2_tokenizer)
            st.subheader("Simplified Summary")
            st.write(simplified_summary)
        else:
            st.warning("Please enter some text to process.")

if __name__ == "__main__":
    main()

# to run 
#cd "C:\Users\Ccantu\Documents\python_scripts"
#streamlit run app_testV2.py