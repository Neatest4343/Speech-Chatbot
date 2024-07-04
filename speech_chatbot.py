import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
import speech_recognition as sr
from threading import Thread
import time

# Downloading NLTK resources 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    with open("C:\\Users\\HP\\Desktop\\Pandora\\pride_and_prejudice.txt.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize words using WordNet
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to transcribe speech into text
def transcribe_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Say something...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Speech recognition could not understand audio.")
    except sr.RequestError:
        st.error("Could not request results from Google Speech Recognition service.")
    
    return None

# Function to perform chatbot operations
def chatbot(query, text):
    if query.startswith('transcribe:'):
        # Transcribe speech input
        speech_input = transcribe_speech()
        if speech_input:
            query = query.replace('transcribe:', '').strip() + ' ' + speech_input
        else:
            return "Error transcribing speech input. Please try again."
    
    # Preprocess the query
    query = preprocess_text(query)
    
    # Preprocess each sentence in the text
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit transform the preprocessed sentences
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    # Transform the query
    query_tfidf = vectorizer.transform([query])
    
    # Compute cosine similarity between query and each sentence
    similarities = []
    for i in range(len(sentences)):
        similarity = cosine_similarity(query_tfidf, tfidf_matrix[i])
        similarities.append((sentences[i], similarity))
    
    # Sort sentences by descending similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the most relevant sentence
    most_relevant_sentence = similarities[0][0]
    
    return most_relevant_sentence

# Streamlit app
def main():
    st.title("Vic Speech-Enabled Chatbot")
    st.write("You can input text or use speech recognition by starting with 'transcribe:'")

    # Read the text file
    with open('pride_and_prejudice.txt.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # User input for question
    user_input = st.text_input("Enter your question or say 'transcribe:' for speech input")

    if user_input:
        # Get the chatbot's response
        bot_response = chatbot(user_input, text)

        # Display the response
        st.write("User:", user_input)
        st.write("Chatbot:", bot_response)

if __name__ == "__main__":
    main()
