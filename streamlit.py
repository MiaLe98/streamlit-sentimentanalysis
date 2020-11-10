import streamlit as st
import pandas as pd 
import joblib
import nltk
import numpy as np
import os 

def main():
    st.markdown("""<style>
                        .st-eb {
                            background-color:#F9786F
                        } </style>""", unsafe_allow_html=True)

    # Heading
    st.header("Positive, Neutral or Negative")

    # Text area for user input
    user_input = st.text_area("Enter your text here")
    if st.button("Let's Go!"):
        with st.spinner('Processing your text ... '):
            prediction = classifier.predict(format_sentence(user_input))
            if prediction == 'pos': 
                st.markdown('**Positive**')
            elif prediction == 'neg': 
                st.markdown('**Negative**')
            elif prediction == 'neu': 
                st.markdown('**Neutral**')
    
        

@st.cache(show_spinner=False)
def nltk_download(): 
    return nltk.download('punkt')

nltk_download()

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(): 
    with open('model-73acc.cls', 'rb') as f:
        classifier = joblib.load(f)
    return classifier

classifier = get_model()


def format_sentence(sent):
    return np.array([sent])


if __name__ == "__main__":
    main()



