import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from bs4 import BeautifulSoup

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Load the pre-trained models
word2vec_model = Word2Vec.load('word2vec_model.model')

with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing functions
def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9\s]+', '', text)  # Remove special characters
    text = re.sub(r'(http|https|ftp|ssh)://[\w_-]+(?:\.[\w_-]+)+[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-]?', '', text)  # Remove URLs
    text = BeautifulSoup(text, 'lxml').get_text()  # Remove HTML tags
    text = " ".join([word for word in text.split() if word.lower() not in stopwords.words('english')])  # Remove stopwords
    text = " ".join(text.split())  # Remove any additional spaces
    return text
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
def average_w2vec(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv.index_to_key]
    if not word_vectors:  # If no words are found in the model's vocabulary
        return np.zeros(model.vector_size)  # Return a zero vector of appropriate size
    return np.mean(word_vectors, axis=0)


def average_w2vec(text, model):
    words = text.split()
    return np.mean([model.wv[word] for word in words if word in model.wv.index_to_key], axis=0)

# Streamlit app
st.title('Sentiment Analysis App')
st.write('Enter a review text to predict if it is positive or negative.')

input_text = st.text_area('Review Text')

if st.button('Predict'):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    lemmatized_text = lemmatize_text(preprocessed_text)
    
    # Transform the text to word2vec vector
    try:
        text_vector = average_w2vec(lemmatized_text, word2vec_model).reshape(1, -1)
        
        # Predict the sentiment using the Random Forest model
        prediction = rf_model.predict(text_vector)
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        
        st.write(f'The sentiment of the review is: {sentiment}')
    except ValueError:
        st.write('The input text is too short or does not contain recognizable words.')

# Save and run this Streamlit app with the following command:
# streamlit run app.py
