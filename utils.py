import re
import string
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def transform_text(text):
    # Convert to string
    text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML Tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove Email Addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # 5. Remove Numbers
    text = re.sub(r'\d+', ' ', text)

    # 6. Remove Special Characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 7. Remove Extra Whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Tokenization
    tokens = word_tokenize(text)

    # 9. Remove Stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 10. Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # 11. Join Tokens
    return " ".join(tokens)