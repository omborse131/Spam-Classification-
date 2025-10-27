import nltk
import ssl

# NLTK sometimes runs into SSL certificate issues.
# This block attempts to fix it by using a default SSL context.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Starting NLTK resource download...")

# 1. Download the specific resource requested in the error message
try:
    nltk.download('punkt_tab')
    print("Downloaded 'punkt_tab'.")
except Exception as e:
    print(f"Could not download 'punkt_tab'. Error: {e}")
    print("This resource may be obsolete. Proceeding with standard 'punkt'.")


# 2. Download the standard 'punkt' (Sentence Tokenizer) and 'stopwords'
nltk.download('punkt')
nltk.download('stopwords')
print("Downloaded 'punkt' and 'stopwords'.")

print("NLTK data download complete. You can now delete this script.")