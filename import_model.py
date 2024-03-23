
import streamlit as st
import pickle
import re

tfidf_2, clf_nb = pickle.load(
    open('imdb-senti-718_naive.pckl', 'rb'))

st.title('Review Sentiment Analysis')

review = st.text_area('Enter your review here:')
if st.button('Predict'):
    review = tfidf_2.transform([review])
    prediction = clf_nb.predict(review)
    if prediction[0] == 'pos':
        st.write('Positive review')
    else:
        st.write('Negative review')