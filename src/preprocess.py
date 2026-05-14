import re
import nltk
import spacy
from nltk.corpus import stopwords

# Téléchargement des ressources NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Chargement du modèle spaCy
nlp = spacy.load('en_core_web_sm')

# Stop words anglais + mots spécifiques Twitter/tech à ignorer
STOP_WORDS = set(stopwords.words('english'))
CUSTOM_STOP_WORDS = {
    'chatgpt', 'gpt', 'ai', 'openai', 'chat', 'via', 'rt',
    'amp', 'would', 'could', 'also', 'get', 'got', 'said',
    'nft', 'nftart', 'nftjapan', 'nftarti', 'nftar',
    'aiart', 'aipad', 'aiartwork', 'web', 'bingai',
    'https', 'http', 'co', 'de', 'en', 'la', 'le'
}
STOP_WORDS.update(CUSTOM_STOP_WORDS)


def remove_urls(text):
    """Supprime les URLs (http, https, t.co...)"""
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


def remove_mentions(text):
    """Supprime les mentions @user"""
    return re.sub(r'@\w+', '', text)


def remove_hashtags(text):
    """Garde le mot du hashtag, supprime le #"""
    return re.sub(r'#(\w+)', r'\1', text)


def remove_special_chars(text):
    """Supprime les caractères spéciaux et encodages bizarres (â, ð, etc.)"""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # non-ASCII
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)      # garde seulement lettres
    return text


def normalize(text):
    """Lowercase + espaces multiples"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_and_filter(text):
    """Lemmatisation avec spaCy + suppression stop words"""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in STOP_WORDS
        and not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return ' '.join(tokens)


def preprocess(text):
    """Pipeline complet"""
    if not isinstance(text, str):
        return ''
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_special_chars(text)
    text = normalize(text)
    text = lemmatize_and_filter(text)
    return text