import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
# Download required NLTK data files if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text, language='english')

    # Remove non-alphanumeric tokens
    y = [i for i in tokens if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load the TF-IDF vectorizer and classification model
tfidf = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

# Streamlit app UI
st.title('Spam Classification')

# User input
input_sms = st.text_input('Enter a message')

if st.button('Classify'):
    if input_sms:
        # Preprocess the input
        transform_sms = transform_text(input_sms)

        # Vectorize the input
        vector_input = tfidf.transform([transform_sms])

        # Predict using the loaded model
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
