"""
Sentiment Analyzer

High-level API for sentiment analysis using pre-trained transformer models.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class SentimentAnalyzer:
    """
    Easy-to-use sentiment analyzer with pre-trained models.

    Parameters
    ----------
    model_name : str, default='distilbert-base-uncased-finetuned-sst-2-english'
        Pre-trained model name from Hugging Face
    device : int, default=-1
        Device to use (-1 for CPU, 0+ for GPU)
    """

    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english', device=-1):
        self.model_name = model_name
        self.device = device

        # Load pipeline
        self.pipeline = pipeline(
            'sentiment-analysis',
            model=model_name,
            device=device,
            truncation=True,
            max_length=512
        )

    def predict(self, text):
        """
        Predict sentiment for single text.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        result : dict
            Sentiment label and confidence score
        """
        result = self.pipeline(text)[0]

        return {
            'label': result['label'],
            'confidence': result['score'],
            'text': text
        }

    def predict_batch(self, texts, batch_size=32):
        """
        Predict sentiment for batch of texts.

        Parameters
        ----------
        texts : list of str
            Input texts
        batch_size : int
            Batch size for processing

        Returns
        -------
        results : list of dict
            Sentiment predictions
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.pipeline(batch)

            for text, result in zip(batch, batch_results):
                results.append({
                    'label': result['label'],
                    'confidence': result['score'],
                    'text': text
                })

        return results

    def analyze_with_aspects(self, text, aspects=None):
        """
        Analyze sentiment for specific aspects in text.

        Parameters
        ----------
        text : str
            Input text
        aspects : list of str, optional
            Aspects to analyze (e.g., ['service', 'food', 'price'])

        Returns
        -------
        results : dict
            Overall sentiment and aspect-level sentiments
        """
        # Overall sentiment
        overall = self.predict(text)

        results = {
            'overall': overall,
            'aspects': {}
        }

        # Aspect-level sentiment
        if aspects:
            text_lower = text.lower()
            for aspect in aspects:
                # Find sentences containing aspect
                sentences = text.split('.')
                aspect_sentences = [s for s in sentences if aspect.lower() in s.lower()]

                if aspect_sentences:
                    aspect_text = '. '.join(aspect_sentences)
                    aspect_sentiment = self.predict(aspect_text)
                    results['aspects'][aspect] = aspect_sentiment

        return results

    def get_confidence_threshold_predictions(self, texts, threshold=0.9):
        """
        Get predictions with confidence above threshold.

        Parameters
        ----------
        texts : list of str
            Input texts
        threshold : float
            Minimum confidence threshold

        Returns
        -------
        high_confidence : list
            Predictions with confidence >= threshold
        low_confidence : list
            Predictions with confidence < threshold
        """
        results = self.predict_batch(texts)

        high_confidence = [r for r in results if r['confidence'] >= threshold]
        low_confidence = [r for r in results if r['confidence'] < threshold]

        return high_confidence, low_confidence


class MultiClassSentimentAnalyzer(SentimentAnalyzer):
    """
    Multi-class sentiment analyzer (positive, neutral, negative).
    """

    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment', device=-1):
        super().__init__(model_name=model_name, device=device)

        # Label mapping
        self.label_map = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE'
        }

    def predict(self, text):
        """Predict with mapped labels."""
        result = super().predict(text)

        # Map label
        if result['label'] in self.label_map:
            result['label'] = self.label_map[result['label']]

        return result

    def predict_batch(self, texts, batch_size=32):
        """Predict batch with mapped labels."""
        results = super().predict_batch(texts, batch_size)

        # Map labels
        for result in results:
            if result['label'] in self.label_map:
                result['label'] = self.label_map[result['label']]

        return results


class EmotionAnalyzer:
    """
    Emotion detection analyzer.

    Detects emotions: joy, sadness, anger, fear, love, surprise
    """

    def __init__(self, model_name='bhadresh-savani/distilbert-base-uncased-emotion', device=-1):
        self.pipeline = pipeline(
            'text-classification',
            model=model_name,
            device=device,
            top_k=None
        )

    def predict(self, text):
        """
        Predict emotions in text.

        Returns
        -------
        emotions : list of dict
            All emotions with confidence scores
        """
        results = self.pipeline(text)[0]

        # Sort by score
        emotions = sorted(results, key=lambda x: x['score'], reverse=True)

        return {
            'text': text,
            'emotions': emotions,
            'dominant_emotion': emotions[0]['label']
        }

    def predict_batch(self, texts, batch_size=32):
        """Predict emotions for batch."""
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.pipeline(batch)

            for text, results in zip(batch, batch_results):
                emotions = sorted(results, key=lambda x: x['score'], reverse=True)
                all_results.append({
                    'text': text,
                    'emotions': emotions,
                    'dominant_emotion': emotions[0]['label']
                })

        return all_results


def compare_models(text, model_names=None):
    """
    Compare sentiment predictions across multiple models.

    Parameters
    ----------
    text : str
        Input text
    model_names : list of str, optional
        Models to compare

    Returns
    -------
    comparisons : dict
        Predictions from each model
    """
    if model_names is None:
        model_names = [
            'distilbert-base-uncased-finetuned-sst-2-english',
            'cardiffnlp/twitter-roberta-base-sentiment',
            'nlptown/bert-base-multilingual-uncased-sentiment'
        ]

    comparisons = {}

    for model_name in model_names:
        try:
            analyzer = SentimentAnalyzer(model_name=model_name)
            result = analyzer.predict(text)
            comparisons[model_name] = result
        except Exception as e:
            comparisons[model_name] = {'error': str(e)}

    return comparisons
