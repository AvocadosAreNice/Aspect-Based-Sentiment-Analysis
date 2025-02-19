# Text Preprocessing
import joblib
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import stopwords, words, wordnet
from nltk.stem import WordNetLemmatizer
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# Cleaning Functions
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
valid_words = set(words.words())


def clean(text):
    punc_free = ''.join([ch for ch in text.lower() if ch not in exclude])
    stop_free = ' '.join([i for i in punc_free.split() if i not in stop])
    no_numbers = ' '.join([word for word in stop_free.split() if not any(ch.isdigit() for ch in word)])
    valid = ' '.join([word for word in no_numbers.split() if word in valid_words])

    final = ' '.join([word for word in valid.split() if len(word) >= 3])
    return final


def compound_words(text):
    compound_words_only = joblib.load('compound_words_only.pkl')
    mwe_tokenizer = MWETokenizer(compound_words_only, separator='_')
    final = ' '.join(mwe_tokenizer.tokenize(word_tokenize(text)))
    return final

