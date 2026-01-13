"""
Utility functions for NLP tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


def generate_wordcloud(texts, sentiment=None, max_words=100):
    """
    Generate word cloud from texts.

    Parameters
    ----------
    texts : list of str
        Input texts
    sentiment : str, optional
        Filter by sentiment ('positive', 'negative', etc.)
    max_words : int
        Maximum words to display
    """
    # Combine texts
    combined_text = ' '.join(texts)

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis'
    ).generate(combined_text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    if sentiment:
        plt.title(f'Word Cloud - {sentiment.title()} Sentiment', fontsize=16)
    else:
        plt.title('Word Cloud', fontsize=16)

    plt.tight_layout()

    return plt.gcf()


def get_most_common_words(texts, n=20, exclude_stopwords=True):
    """
    Get most common words in texts.

    Parameters
    ----------
    texts : list of str
        Input texts
    n : int
        Number of top words
    exclude_stopwords : bool
        Whether to exclude common stopwords

    Returns
    -------
    common_words : list of tuples
        (word, count) pairs
    """
    from nltk.corpus import stopwords
    import nltk

    try:
        stop_words = set(stopwords.words('english')) if exclude_stopwords else set()
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english')) if exclude_stopwords else set()

    # Tokenize and count
    all_words = []
    for text in texts:
        words = text.lower().split()
        words = [w for w in words if w.isalpha() and (not exclude_stopwords or w not in stop_words)]
        all_words.extend(words)

    return Counter(all_words).most_common(n)


def plot_sentiment_distribution(sentiments):
    """
    Plot distribution of sentiments.

    Parameters
    ----------
    sentiments : array-like
        Sentiment labels
    """
    sentiment_counts = Counter(sentiments)

    plt.figure(figsize=(10, 6))
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(),
            color=['green' if 'pos' in str(k).lower() else 'red' if 'neg' in str(k).lower() else 'gray'
                   for k in sentiment_counts.keys()])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return plt.gcf()


def compute_sentiment_metrics_by_group(texts, sentiments, group_by_feature):
    """
    Compute sentiment metrics grouped by a feature.

    Parameters
    ----------
    texts : list of str
        Input texts
    sentiments : array-like
        Sentiment labels
    group_by_feature : function
        Function to extract grouping feature from text

    Returns
    -------
    metrics : dict
        Metrics by group
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for text, sentiment in zip(texts, sentiments):
        group = group_by_feature(text)
        groups[group].append(sentiment)

    metrics = {}
    for group, group_sentiments in groups.items():
        sentiment_dist = Counter(group_sentiments)
        metrics[group] = {
            'count': len(group_sentiments),
            'distribution': dict(sentiment_dist),
            'positive_ratio': sentiment_dist.get('POSITIVE', 0) / len(group_sentiments)
        }

    return metrics


def extract_key_phrases(texts, n=10):
    """
    Extract key phrases from texts using simple n-grams.

    Parameters
    ----------
    texts : list of str
        Input texts
    n : int
        Number of top phrases

    Returns
    -------
    phrases : list of tuples
        (phrase, count) pairs
    """
    from collections import Counter

    bigrams = []
    trigrams = []

    for text in texts:
        words = text.lower().split()

        # Bigrams
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")

        # Trigrams
        for i in range(len(words) - 2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    # Combine and get most common
    all_phrases = bigrams + trigrams
    return Counter(all_phrases).most_common(n)


def save_predictions_to_csv(texts, predictions, probabilities, output_path):
    """
    Save predictions to CSV file.

    Parameters
    ----------
    texts : list of str
        Input texts
    predictions : array-like
        Predicted labels
    probabilities : array-like
        Prediction probabilities
    output_path : str
        Output file path
    """
    import pandas as pd

    df = pd.DataFrame({
        'text': texts,
        'prediction': predictions,
        'confidence': np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else probabilities
    })

    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def batch_process_files(input_files, analyzer, output_dir):
    """
    Process multiple files in batch.

    Parameters
    ----------
    input_files : list of str
        List of input file paths
    analyzer : SentimentAnalyzer
        Sentiment analyzer instance
    output_dir : str
        Output directory
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        # Read file
        df = pd.read_csv(input_file)
        texts = df['text'].tolist()  # Assumes 'text' column

        # Predict
        results = analyzer.predict_batch(texts)

        # Save results
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_results.csv'))

        df['sentiment'] = [r['label'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]

        df.to_csv(output_file, index=False)
        print(f"Processed {input_file} -> {output_file}")
