import streamlit as st
import pandas as pd
import nltk
import re
from collections import defaultdict, Counter
from nltk.util import ngrams
from docx import Document
from transformers import pipeline, MarianMTModel, MarianTokenizer
import torch
import math

# ----------------------
# 1. Sentiment Analysis
# ----------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# ----------------------
# 2. Next Word Prediction
# ----------------------
@st.cache_resource
def load_bigrams(doc_path="data.docx"):
    nltk.download('punkt', quiet=True)
    doc = Document(doc_path)
    sentences = [para.text for para in doc.paragraphs if para.text.strip()]
    
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    
    processed_sentences = [preprocess(sent) for sent in sentences]
    bigram_counts = defaultdict(Counter)
    for sentence in processed_sentences:
        tokens = nltk.word_tokenize(sentence)
        for w1, w2 in ngrams(tokens, 2):
            bigram_counts[w1][w2] += 1
    return bigram_counts

def predict_next_word(bigram_counts, word, n_predictions=3):
    word = word.lower()
    if word in bigram_counts:
        next_words = bigram_counts[word]
        total = sum(next_words.values())
        sorted_words = sorted(next_words.items(), key=lambda x: x[1]/total, reverse=True)
        return [(w, c/total) for w, c in sorted_words[:n_predictions]]
    else:
        return [("No prediction found", 0)]

# ----------------------
# 3. Sentence Translation
# ----------------------
@st.cache_resource
def load_translation_model(model_name="Helsinki-NLP/opus-mt-en-fr"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_sentence(sentence, tokenizer, model):
    inputs = tokenizer([sentence], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translation_confidence(sentence, tokenizer, model):
    inputs = tokenizer([sentence], return_tensors="pt", padding=True)
    output = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
    log_prob = float(output.sequences_scores[0].detach())
    confidence = math.exp(log_prob)
    return confidence

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="NLP Projects", layout="wide")
st.title("🚀 NLP Projects Deployment")

project = st.sidebar.selectbox("Select Project", 
                               ["Sentiment Analysis", "Next Word Prediction", "Sentence Translation"])

# ----------------------
# Project 1: Sentiment Analysis
# ----------------------
if project == "Sentiment Analysis":
    st.header("Sentiment Analysis (DistilBERT)")
    sentiment_model = load_sentiment_model()
    user_text = st.text_area("Enter text for sentiment analysis:", height=150)
    
    if st.button("Predict Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = sentiment_model(user_text)[0]
            label = result["label"]
            confidence = result["score"]
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence Score: {confidence:.4f}")

# ----------------------
# Project 2: Next Word Prediction
# ----------------------
elif project == "Next Word Prediction":
    st.header("Next Word Prediction (Bigram Model)")
    bigram_counts = load_bigrams()
    user_word = st.text_input("Enter a word to predict the next word:")
    
    if st.button("Predict Next Words"):
        if user_word.strip() == "":
            st.warning("Please enter a word.")
        else:
            predictions = predict_next_word(bigram_counts, user_word)
            for i, (w, prob) in enumerate(predictions, start=1):
                st.write(f"{i}. {w} (Probability: {prob:.2f})")

# ----------------------
# Project 3: Sentence Translation
# ----------------------
elif project == "Sentence Translation":
    st.header("Sentence Translation (English → French)")
    tokenizer, model = load_translation_model()
    user_text = st.text_area("Enter text to translate:", height=150)
    
    if st.button("Translate"):
        if user_text.strip() == "":
            st.warning("Please enter text to translate.")
        else:
            translation = translate_sentence(user_text, tokenizer, model)
            confidence = translation_confidence(user_text, tokenizer, model)
            st.success(f"Translation: {translation}")
            st.info(f"Confidence Score: {confidence:.4f}")
