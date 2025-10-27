import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text, language='english')
    y = [i for i in tokens if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load your dataset (change path if needed)
df = pd.read_csv(r"C:\Users\HP\OneDrive\Downloads\archive (1)\spam.csv", encoding='latin-1')
df = df[['Category', 'Message']]  # use actual column names from your CSV
df.columns = ['label', 'message']  # rename for consistency


df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['transformed'] = df['message'].apply(transform_text)


# Split data
X_train, X_test, y_train, y_test = train_test_split(df['transformed'], df['label'], test_size=0.2, random_state=2)

# Train TF-IDF Vectorizer and Model
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save files
pickle.dump(tfidf, open('vectorizer (1).pkl', 'wb'))
pickle.dump(model, open('model (1).pkl', 'wb'))

# Check accuracy
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)
print("Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, y_pred))
