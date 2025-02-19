# Text Preprocessing
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords, words, wordnet
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')


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

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens) 
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized_tokens)


def compound_words(text):
    compound_words_only = joblib.load('compound_words_only.pkl')
    mwe_tokenizer = MWETokenizer(compound_words_only, separator='_')
    final = ' '.join(mwe_tokenizer.tokenize(word_tokenize(text)))
    return final

