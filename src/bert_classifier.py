"""
BERT Classifier for Custom Fine-tuning

Fine-tune BERT models on custom sentiment datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    """
    Dataset for text classification.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier:
    """
    BERT-based text classifier with fine-tuning support.
    
    Parameters
    ----------
    model_name : str, default='bert-base-uncased'
        Pre-trained model name
    num_classes : int, default=2
        Number of output classes
    max_length : int, default=128
        Maximum sequence length
    device : str, optional
        Device to use ('cuda' or 'cpu')
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, 
                 max_length=128, device=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        ).to(self.device)
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, texts, labels, val_size=0.2, epochs=3, batch_size=16,
              learning_rate=2e-5, warmup_ratio=0.1):
        """
        Fine-tune model on training data.
        
        Parameters
        ----------
        texts : list of str
            Training texts
        labels : list of int
            Training labels
        val_size : float
            Validation split ratio
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        learning_rate : float
            Learning rate
        warmup_ratio : float
            Warmup ratio for learning rate schedule
        
        Returns
        -------
        history : dict
            Training history
        """
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        return self.history
    
    def _train_epoch(self, loader, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc='Training'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(loader), correct / total
    
    def _validate_epoch(self, loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(loader), correct / total
    
    def predict(self, texts, batch_size=32):
        """
        Predict sentiment for texts.
        
        Parameters
        ----------
        texts : list of str or str
            Input text(s)
        batch_size : int
            Batch size for prediction
        
        Returns
        -------
        predictions : np.ndarray
            Predicted class indices
        probabilities : np.ndarray
            Class probabilities
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Create dataset and loader
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = TextDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate(self, texts, labels, batch_size=32):
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        texts : list of str
            Test texts
        labels : list of int
            True labels
        batch_size : int
            Batch size
        
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        predictions, probabilities = self.predict(texts, batch_size)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': classification_report(labels, predictions)
        }
    
    def save_model(self, path):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path):
        """Load model and tokenizer."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
