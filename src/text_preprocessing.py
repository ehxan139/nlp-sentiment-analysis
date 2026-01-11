"""
Text Preprocessing Utilities

Clean and preprocess text for sentiment analysis.
"""

import re
import string
from html import unescape


class TextPreprocessor:
    """
    Text preprocessing for sentiment analysis.
    
    Parameters
    ----------
    lowercase : bool, default=True
        Convert text to lowercase
    remove_urls : bool, default=True
        Remove URLs
    remove_html : bool, default=True
        Remove HTML tags
    remove_mentions : bool, default=False
        Remove @mentions (for social media)
    remove_hashtags : bool, default=False
        Remove #hashtags
    remove_punctuation : bool, default=False
        Remove punctuation
    remove_numbers : bool, default=False
        Remove numbers
    remove_extra_spaces : bool, default=True
        Remove extra whitespace
    """
    
    def __init__(self, lowercase=True, remove_urls=True, remove_html=True,
                 remove_mentions=False, remove_hashtags=False,
                 remove_punctuation=False, remove_numbers=False,
                 remove_extra_spaces=True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces
    
    def preprocess(self, text):
        """
        Preprocess single text.
        
        Parameters
        ----------
        text : str
            Input text
        
        Returns
        -------
        cleaned : str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # HTML entities
        if self.remove_html:
            text = unescape(text)
            text = re.sub(r'<[^>]+>', '', text)
        
        # URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Hashtags
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Extra spaces
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts):
        """
        Preprocess batch of texts.
        
        Parameters
        ----------
        texts : list of str
            Input texts
        
        Returns
        -------
        cleaned : list of str
            Preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


def clean_text_for_bert(text):
    """
    Minimal cleaning for BERT (BERT handles most preprocessing).
    
    Parameters
    ----------
    text : str
        Input text
    
    Returns
    -------
    cleaned : str
        Cleaned text
    """
    # Remove excessive newlines
    text = re.sub(r'\n+', ' ', text)
    
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def normalize_contractions(text):
    """
    Expand English contractions.
    
    Parameters
    ----------
    text : str
        Input text
    
    Returns
    -------
    expanded : str
        Text with expanded contractions
    """
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    # Create pattern from contractions
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b', re.IGNORECASE)
    
    def replace(match):
        return contractions[match.group(0).lower()]
    
    return pattern.sub(replace, text)


def extract_features(text):
    """
    Extract text features for analysis.
    
    Parameters
    ----------
    text : str
        Input text
    
    Returns
    -------
    features : dict
        Text features
    """
    return {
        'length': len(text),
        'num_words': len(text.split()),
        'num_sentences': len(re.split(r'[.!?]+', text)),
        'num_uppercase': sum(1 for c in text if c.isupper()),
        'num_punctuation': sum(1 for c in text if c in string.punctuation),
        'num_exclamation': text.count('!'),
        'num_question': text.count('?'),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'has_url': bool(re.search(r'http\S+|www\.\S+', text)),
        'has_mention': bool(re.search(r'@\w+', text)),
        'has_hashtag': bool(re.search(r'#\w+', text))
    }


import numpy as np
