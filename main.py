import streamlit as st
import joblib
import time
import numpy as np
from cleaning import clean, compound_words, lemmatize_words

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')

# Load your trained model and TF-IDF vectorizer
model = joblib.load("aspect_sentiment_model.pkl")
tfidf = joblib.load('tfidfVectorizer.pkl')

def predict_sentiment(review_tfidf):
    # Make predictions using the trained model
    sentiments = model.predict(review_tfidf)

    # If the model is a multi-label classifier, it will return a 2D array
    # Assuming the model returns sentiment for each aspect (e.g., overall, camera, price, interaction)
    aspects = ["overall", "camera", "price", "interaction"]
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.02)  # Simulate delay while predicting sentiment

    # Return predictions as a dictionary
    return dict(zip(aspects, sentiments[0]))  # Ensure sentiments[0] is the correct format


st.set_page_config(page_title="Analysing iPhone Reviews")
st.markdown("<h1 style='text-align:center;'>Perfecting the Next iPhone:<br> Aspect-Based Sentiment Analysis</h1>",
            unsafe_allow_html=True)

st.write("Enter a review below, and the model will classify the sentiment for the overall experience, camera, price, "
         "and interaction.")

# User input
review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    progress_bar = st.progress(0)
    if review.strip():
        review = clean(review)
        review = lemmatize_words(review)

        # Transform the review using TF-IDF
        review_tfidf = tfidf.transform([review])

        # Predict sentiment for each aspect
        sentiment_results = predict_sentiment(review_tfidf)

        st.subheader("Sentiment Results")

        # Displaying sentiment results in a horizontal list with emojis
        emojis = {
            "positive": "😊",
            "negative": "😡",
            "neutral": "😐"
        }

        # Create columns for each aspect
        cols = st.columns(4)
        for i, (aspect, sentiment) in enumerate(sentiment_results.items()):
            emoji = emojis.get(sentiment, "😶")
            with cols[i]:

                st.markdown(f"<h3 style='text-align:center; font-size: 50px; margin-left: 20px;'>{emoji}</h3>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='text-align:center'>{aspect.capitalize()} Sentiment:<br><strong>{sentiment.capitalize()}</strong></p>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("Please enter a review before submitting.")

