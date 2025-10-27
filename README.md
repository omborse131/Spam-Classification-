# Spam-Classification-
This project classifies messages as Spam or Not Spam using Natural Language Processing (NLP) and Machine Learning. It involves data cleaning, feature extraction (TF-IDF), model training, and deployment using a Streamlit web app and the Naive Bayes Theorem.

Spam_Classification_Project/
│
├── data/
│   ├── spam.csv                  # Dataset used for training
│
├── notebooks/
│   ├── data_cleaning.ipynb       # Jupyter notebook for cleaning and preprocessing
│
├── models/
│   ├── model.pkl                 # Trained Naive Bayes model
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│
├── app.py                        # Streamlit app for spam detection
├── requirements.txt              # Required Python packages
├── README.md                     # Project documentation
└── utils.py                      # Text preprocessing and helper functions
Create a Virtual Environment
python -m venv .venv

2️⃣ Activate the Virtual Environment

Windows:

.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

3️⃣ Install Required Packages
pip install -r requirements.txt


Example requirements.txt:

pandas
numpy
scikit-learn
nltk
streamlit
pickle-mixin

🧹 Data Cleaning and Preprocessing

Open the Jupyter Notebook:

jupyter notebook


Open notebooks/data_cleaning.ipynb.

In this notebook:

Load the dataset (spam.csv)

Clean text data (remove punctuation, stopwords, lowercase)

Apply stemming using PorterStemmer

Convert text into features using TF-IDF Vectorizer

Train and save model (model.pkl) and vectorizer (vectorizer.pkl) using pickle

🧪 Running the Project in PyCharm

Open the project folder in PyCharm.

Make sure the virtual environment is selected (.venv).

Run the Streamlit app using:

streamlit run app.py


This will open your app in the browser.
Enter any message and the model will predict if it’s Spam or Not Spam.

🧰 Tools and Libraries Used

Python 3.10+

Pandas, NumPy

NLTK for text preprocessing

Scikit-learn for TF-IDF and model training

Streamlit for web interface

PyCharm for IDE

Jupyter Notebook for data cleaning

📈 Model Used

Algorithm: Multinomial Naive Bayes

Feature Extraction: TF-IDF Vectorizer

Accuracy: ~95–98% (depending on dataset split)

🚀 How It Works

User Input: You enter a message in the web app.

Preprocessing: Text is cleaned and transformed using the saved vectorizer.

Prediction: The trained Naive Bayes model classifies the message.

Output: Displays “Spam” or “Not Spam”.

🧾 Example Usage

Input:

Congratulations! You’ve won a $500 gift card! Click the link below.


Output:

Prediction: SPAM
