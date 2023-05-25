import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

cv=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def transform_text(text):
    # Lower case
    text=text.lower()
    # Tokenization
    text=nltk.word_tokenize(text)
    # Special character removal
    target=[]
    for i in text:
        if i.isalnum():
            target.append(i)
    # Stop word and punctuation removal
    text=target[:]
    target.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            target.append(i)
    # Stemming
    text=target[:]
    target.clear()
    for i in text:
        target.append(ps.stem(i))
    return " ".join(target)

st.title("Email/SMS Spam Classifier")
msg = st.text_area('Enter the message')
if st.button('Predict'):
    # Preprocessing
    transformed_txt=transform_text(msg)
    # vectorization
    vectored_txt=cv.transform([transformed_txt])
    # Predict
    result=model.predict(vectored_txt)[0]
    # Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")