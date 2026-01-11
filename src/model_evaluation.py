"""
Model Evaluation Utilities

Comprehensive evaluation metrics and visualization for sentiment models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)


class SentimentEvaluator:
    """
    Comprehensive evaluation for sentiment models.
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba=None):
        """
        Calculate comprehensive metrics.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like, optional
            Prediction probabilities
        
        Returns
        -------
        metrics : dict
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        # Add ROC AUC for binary classification
        if y_proba is not None and len(np.unique(y_true)) == 2:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                # Use probability of positive class
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_positive)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Class names for labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm))
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, class_names=None):
        """
        Plot ROC curve for binary or multi-class classification.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_proba : array-like
            Prediction probabilities
        class_names : list, optional
            Class names
        """
        plt.figure(figsize=(10, 8))
        
        # Binary classification
        if len(np.unique(y_true)) == 2:
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Multi-class
        else:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                label = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_proba):
        """
        Plot precision-recall curve for binary classification.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_proba : array-like
            Prediction probabilities
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def analyze_confidence_distribution(y_proba, y_pred, y_true):
        """
        Analyze confidence score distribution.
        
        Parameters
        ----------
        y_proba : array-like
            Prediction probabilities
        y_pred : array-like
            Predicted labels
        y_true : array-like
            True labels
        """
        # Get max confidence for each prediction
        if len(y_proba.shape) > 1:
            confidences = np.max(y_proba, axis=1)
        else:
            confidences = y_proba
        
        correct = (y_pred == y_true)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall confidence distribution
        axes[0].hist(confidences, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by correctness
        axes[1].hist(confidences[correct], bins=30, alpha=0.7, label='Correct', edgecolor='black')
        axes[1].hist(confidences[~correct], bins=30, alpha=0.7, label='Incorrect', edgecolor='black')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence by Prediction Correctness')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def get_classification_report(y_true, y_pred, class_names=None):
        """
        Generate detailed classification report.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Class names
        
        Returns
        -------
        report : str
            Classification report
        """
        return classification_report(y_true, y_pred, target_names=class_names)
    
    @staticmethod
    def analyze_errors(texts, y_true, y_pred, y_proba, n_examples=10):
        """
        Analyze misclassified examples.
        
        Parameters
        ----------
        texts : list of str
            Input texts
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like
            Prediction probabilities
        n_examples : int
            Number of examples to show
        
        Returns
        -------
        errors : list of dict
            Misclassified examples with details
        """
        # Find errors
        errors_idx = np.where(y_true != y_pred)[0]
        
        # Get confidence for predicted class
        if len(y_proba.shape) > 1:
            confidences = np.max(y_proba, axis=1)
        else:
            confidences = y_proba
        
        # Sort by confidence (most confident errors first)
        sorted_idx = errors_idx[np.argsort(-confidences[errors_idx])]
        
        errors = []
        for idx in sorted_idx[:n_examples]:
            errors.append({
                'text': texts[idx],
                'true_label': y_true[idx],
                'predicted_label': y_pred[idx],
                'confidence': confidences[idx],
                'probabilities': y_proba[idx] if len(y_proba.shape) > 1 else [1-y_proba[idx], y_proba[idx]]
            })
        
        return errors
