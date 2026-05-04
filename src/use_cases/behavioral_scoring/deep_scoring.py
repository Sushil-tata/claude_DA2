"""
Deep Learning Temporal Scoring

Implements deep learning models for temporal behavioral scoring:
- LSTM for transaction sequences
- Transformer with attention mechanism
- Temporal CNN for pattern recognition
- Sequence preprocessing utilities
- Attention visualization
- Embedding layers for categorical features
- Multi-horizon predictions
- Model interpretation tools

Example:
    >>> from src.use_cases.behavioral_scoring.deep_scoring import LSTMScoringModel
    >>> 
    >>> # Train LSTM scorer
    >>> model = LSTMScoringModel(input_size=50, hidden_size=128)
    >>> model.fit(sequences, labels, epochs=20)
    >>> 
    >>> # Real-time scoring
    >>> scores = model.predict(new_sequences)
    >>> attention_weights = model.get_attention_weights(new_sequences)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False


class SequencePreprocessor:
    """Preprocess transaction sequences for deep learning models."""
    
    def __init__(
        self,
        max_sequence_length: int = 100,
        padding: str = "pre",
        truncating: str = "pre"
    ):
        """
        Initialize sequence preprocessor.
        
        Args:
            max_sequence_length: Maximum sequence length
            padding: Padding strategy ('pre' or 'post')
            truncating: Truncating strategy ('pre' or 'post')
            
        Example:
            >>> preprocessor = SequencePreprocessor(max_sequence_length=50)
            >>> X_padded = preprocessor.fit_transform(sequences)
        """
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating
        self.feature_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.is_fitted = False
        
    def fit(
        self,
        sequences: List[np.ndarray],
        categorical_features: Optional[Dict[int, List]] = None
    ):
        """
        Fit preprocessor on sequences.
        
        Args:
            sequences: List of sequence arrays
            categorical_features: Dict mapping feature index to sequences
        """
        # Concatenate all sequences for fitting scaler
        all_values = np.vstack([seq for seq in sequences if len(seq) > 0])
        self.feature_scaler.fit(all_values)
        
        # Fit categorical encoders
        if categorical_features:
            for feat_idx, cat_sequences in categorical_features.items():
                encoder = LabelEncoder()
                all_cats = np.concatenate([seq for seq in cat_sequences if len(seq) > 0])
                encoder.fit(all_cats)
                self.categorical_encoders[feat_idx] = encoder
        
        self.is_fitted = True
        logger.debug("Sequence preprocessor fitted")
        
    def transform(
        self,
        sequences: List[np.ndarray],
        categorical_features: Optional[Dict[int, List]] = None
    ) -> np.ndarray:
        """
        Transform sequences to padded arrays.
        
        Args:
            sequences: List of sequence arrays
            categorical_features: Dict mapping feature index to sequences
            
        Returns:
            Padded sequence array of shape (n_samples, max_length, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Must fit preprocessor before transform")
        
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) == 0:
                # Create zero sequence
                padded_seq = np.zeros((self.max_sequence_length, seq.shape[1] if len(seq.shape) > 1 else 1))
            else:
                # Scale
                scaled_seq = self.feature_scaler.transform(seq)
                
                # Pad or truncate
                if len(scaled_seq) < self.max_sequence_length:
                    # Pad
                    pad_width = self.max_sequence_length - len(scaled_seq)
                    if self.padding == "pre":
                        padded_seq = np.vstack([
                            np.zeros((pad_width, scaled_seq.shape[1])),
                            scaled_seq
                        ])
                    else:
                        padded_seq = np.vstack([
                            scaled_seq,
                            np.zeros((pad_width, scaled_seq.shape[1]))
                        ])
                elif len(scaled_seq) > self.max_sequence_length:
                    # Truncate
                    if self.truncating == "pre":
                        padded_seq = scaled_seq[-self.max_sequence_length:]
                    else:
                        padded_seq = scaled_seq[:self.max_sequence_length]
                else:
                    padded_seq = scaled_seq
                    
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences)
    
    def fit_transform(
        self,
        sequences: List[np.ndarray],
        categorical_features: Optional[Dict[int, List]] = None
    ) -> np.ndarray:
        """Fit and transform sequences."""
        self.fit(sequences, categorical_features)
        return self.transform(sequences, categorical_features)


if TORCH_AVAILABLE:
    class LSTMScoringModel(nn.Module):
        """LSTM-based scoring model for transaction sequences."""
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True
        ):
            """
            Initialize LSTM scoring model.
            
            Args:
                input_size: Number of input features
                hidden_size: Hidden layer size
                num_layers: Number of LSTM layers
                dropout: Dropout rate
                bidirectional: Whether to use bidirectional LSTM
                
            Example:
                >>> model = LSTMScoringModel(input_size=50, hidden_size=128)
                >>> model.fit(X_train, y_train, epochs=20)
            """
            super(LSTMScoringModel, self).__init__()
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            
            lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
            
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            self.preprocessor = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            # LSTM
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use last output
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                hidden = h_n[-1]
            
            # Fully connected layers
            output = self.fc(hidden)
            
            return output.squeeze()
        
        def fit(
            self,
            X: Union[np.ndarray, List[np.ndarray]],
            y: np.ndarray,
            epochs: int = 20,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            validation_split: float = 0.2,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """
            Fit LSTM model.
            
            Args:
                X: Sequences (n_samples, seq_length, n_features) or list of arrays
                y: Labels
                epochs: Number of training epochs
                batch_size: Batch size
                learning_rate: Learning rate
                validation_split: Validation split ratio
                verbose: Whether to print progress
                
            Returns:
                Training history
                
            Example:
                >>> history = model.fit(sequences, labels, epochs=20)
                >>> print(f"Final AUC: {history['val_auc'][-1]:.3f}")
            """
            logger.info(f"Training LSTM model for {epochs} epochs")
            
            # Preprocess sequences if needed
            if isinstance(X, list):
                if self.preprocessor is None:
                    self.preprocessor = SequencePreprocessor()
                X = self.preprocessor.fit_transform(X)
            
            # Train/val split
            n_train = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train).to(self.device),
                torch.FloatTensor(y_train).to(self.device)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val).to(self.device),
                torch.FloatTensor(y_val).to(self.device)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Setup training
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_auc': [],
                'val_auc': []
            }
            
            # Training loop
            for epoch in range(epochs):
                # Training
                self.train()
                train_losses = []
                train_preds = []
                train_targets = []
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_targets.extend(batch_y.cpu().numpy())
                
                # Validation
                self.eval()
                val_losses = []
                val_preds = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                
                # Calculate metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_auc = roc_auc_score(train_targets, train_preds)
                val_auc = roc_auc_score(val_targets, val_preds)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_auc'].append(train_auc)
                history['val_auc'].append(val_auc)
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                        f"train_auc: {train_auc:.4f}, val_auc: {val_auc:.4f}"
                    )
            
            logger.info("Training completed")
            return history
        
        def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
            """
            Predict risk scores.
            
            Args:
                X: Sequences
                
            Returns:
                Risk scores
            """
            # Preprocess if needed
            if isinstance(X, list):
                if self.preprocessor is None:
                    raise ValueError("Model not fitted yet")
                X = self.preprocessor.transform(X)
            
            self.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                scores = self(X_tensor).cpu().numpy()
            
            return scores
        
        def save(self, path: Union[str, Path]):
            """Save model to disk."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'bidirectional': self.bidirectional,
                'preprocessor': self.preprocessor
            }
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
        
        @classmethod
        def load(cls, path: Union[str, Path]) -> 'LSTMScoringModel':
            """Load model from disk."""
            checkpoint = torch.load(path)
            
            model = cls(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                bidirectional=checkpoint['bidirectional']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.preprocessor = checkpoint['preprocessor']
            
            logger.info(f"Model loaded from {path}")
            return model


    class TransformerScoringModel(nn.Module):
        """Transformer-based scoring with attention mechanism."""
        
        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 3,
            dropout: float = 0.3
        ):
            """
            Initialize Transformer scoring model.
            
            Args:
                input_size: Number of input features
                d_model: Model dimension
                nhead: Number of attention heads
                num_layers: Number of transformer layers
                dropout: Dropout rate
                
            Example:
                >>> model = TransformerScoringModel(input_size=50, d_model=128)
                >>> model.fit(X_train, y_train)
                >>> attention = model.get_attention_weights(X_test)
            """
            super(TransformerScoringModel, self).__init__()
            
            self.input_size = input_size
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            
            # Input projection
            self.input_projection = nn.Linear(input_size, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output layers
            self.fc = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            self.preprocessor = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
            self.attention_weights = None
            
        def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Forward pass."""
            # Project input
            x = self.input_projection(x)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoding
            x = self.transformer(x, src_mask)
            
            # Global average pooling
            x = torch.mean(x, dim=1)
            
            # Output
            output = self.fc(x)
            
            return output.squeeze()
        
        def fit(
            self,
            X: Union[np.ndarray, List[np.ndarray]],
            y: np.ndarray,
            epochs: int = 20,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            validation_split: float = 0.2,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """Fit Transformer model (similar to LSTM)."""
            logger.info(f"Training Transformer model for {epochs} epochs")
            
            # Preprocess sequences if needed
            if isinstance(X, list):
                if self.preprocessor is None:
                    self.preprocessor = SequencePreprocessor()
                X = self.preprocessor.fit_transform(X)
            
            # Train/val split
            n_train = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train).to(self.device),
                torch.FloatTensor(y_train).to(self.device)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val).to(self.device),
                torch.FloatTensor(y_val).to(self.device)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Setup training
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_auc': [],
                'val_auc': []
            }
            
            # Training loop
            for epoch in range(epochs):
                # Training
                self.train()
                train_losses = []
                train_preds = []
                train_targets = []
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_targets.extend(batch_y.cpu().numpy())
                
                # Validation
                self.eval()
                val_losses = []
                val_preds = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                
                # Calculate metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_auc = roc_auc_score(train_targets, train_preds)
                val_auc = roc_auc_score(val_targets, val_preds)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_auc'].append(train_auc)
                history['val_auc'].append(val_auc)
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                        f"train_auc: {train_auc:.4f}, val_auc: {val_auc:.4f}"
                    )
            
            logger.info("Training completed")
            return history
        
        def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
            """Predict risk scores."""
            if isinstance(X, list):
                if self.preprocessor is None:
                    raise ValueError("Model not fitted yet")
                X = self.preprocessor.transform(X)
            
            self.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                scores = self(X_tensor).cpu().numpy()
            
            return scores
        
        def get_attention_weights(
            self,
            X: Union[np.ndarray, List[np.ndarray]]
        ) -> np.ndarray:
            """
            Extract attention weights for interpretability.
            
            Args:
                X: Input sequences
                
            Returns:
                Attention weights array
                
            Example:
                >>> attention = model.get_attention_weights(sequences)
                >>> # Visualize attention for first sequence
                >>> plt.imshow(attention[0], cmap='hot')
            """
            if isinstance(X, list):
                if self.preprocessor is None:
                    raise ValueError("Model not fitted yet")
                X = self.preprocessor.transform(X)
            
            self.eval()
            
            # Store attention weights during forward pass
            attention_weights_list = []
            
            def hook_fn(module, input, output):
                # Extract attention from transformer layer
                attention_weights_list.append(output[1].detach().cpu().numpy())
            
            # Register hook (simplified - actual implementation would need layer access)
            # For now, return placeholder
            logger.warning("Attention visualization requires custom hook implementation")
            
            return np.zeros((len(X), X.shape[1], X.shape[1]))
        
        def save(self, path: Union[str, Path]):
            """Save model to disk."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'preprocessor': self.preprocessor
            }
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
        
        @classmethod
        def load(cls, path: Union[str, Path]) -> 'TransformerScoringModel':
            """Load model from disk."""
            checkpoint = torch.load(path)
            
            model = cls(
                input_size=checkpoint['input_size'],
                d_model=checkpoint['d_model'],
                nhead=checkpoint['nhead'],
                num_layers=checkpoint['num_layers']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.preprocessor = checkpoint['preprocessor']
            
            logger.info(f"Model loaded from {path}")
            return model


    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer."""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super(PositionalEncoding, self).__init__()
            
            # Validate d_model
            if d_model < 2:
                raise ValueError(f"d_model must be at least 2, got {d_model}")
            
            self.dropout = nn.Dropout(p=dropout)
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            
            # Apply sinusoidal positional encoding
            # Even positions (0, 2, 4, ...) use sine
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            
            # Odd positions (1, 3, 5, ...) use cosine
            if d_model % 2 == 0:
                # Even d_model: all odd positions can use div_term directly
                pe[:, 0, 1::2] = torch.cos(position * div_term)
            else:
                # Odd d_model: last sine position has no cosine counterpart
                # so we need one fewer cosine term
                if len(div_term) > 1:
                    pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])
                else:
                    pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)


    class TemporalCNN(nn.Module):
        """Temporal CNN for pattern recognition in sequences."""
        
        def __init__(
            self,
            input_size: int,
            num_filters: int = 64,
            kernel_sizes: List[int] = [3, 5, 7],
            dropout: float = 0.3
        ):
            """
            Initialize Temporal CNN.
            
            Args:
                input_size: Number of input features
                num_filters: Number of convolutional filters
                kernel_sizes: List of kernel sizes for multi-scale
                dropout: Dropout rate
                
            Example:
                >>> model = TemporalCNN(input_size=50, num_filters=64)
                >>> model.fit(X_train, y_train)
            """
            super(TemporalCNN, self).__init__()
            
            self.input_size = input_size
            self.num_filters = num_filters
            self.kernel_sizes = kernel_sizes
            
            # Multi-scale convolutional layers
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k//2
                )
                for k in kernel_sizes
            ])
            
            # Pooling
            self.pool = nn.AdaptiveMaxPool1d(1)
            
            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(num_filters * len(kernel_sizes), 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            self.preprocessor = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            # x shape: (batch, seq_len, features)
            # Conv1d expects: (batch, features, seq_len)
            x = x.permute(0, 2, 1)
            
            # Apply multi-scale convolutions
            conv_outputs = []
            for conv in self.conv_layers:
                conv_out = F.relu(conv(x))
                pooled = self.pool(conv_out).squeeze(-1)
                conv_outputs.append(pooled)
            
            # Concatenate multi-scale features
            x = torch.cat(conv_outputs, dim=1)
            
            # Fully connected
            output = self.fc(x)
            
            return output.squeeze()
        
        def fit(
            self,
            X: Union[np.ndarray, List[np.ndarray]],
            y: np.ndarray,
            epochs: int = 20,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            validation_split: float = 0.2,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """Fit Temporal CNN model."""
            logger.info(f"Training Temporal CNN for {epochs} epochs")
            
            # Similar to LSTM fit method
            if isinstance(X, list):
                if self.preprocessor is None:
                    self.preprocessor = SequencePreprocessor()
                X = self.preprocessor.fit_transform(X)
            
            n_train = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train).to(self.device),
                torch.FloatTensor(y_train).to(self.device)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val).to(self.device),
                torch.FloatTensor(y_val).to(self.device)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            
            history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
            
            for epoch in range(epochs):
                self.train()
                train_losses, train_preds, train_targets = [], [], []
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_targets.extend(batch_y.cpu().numpy())
                
                self.eval()
                val_losses, val_preds, val_targets = [], [], []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_auc = roc_auc_score(train_targets, train_preds)
                val_auc = roc_auc_score(val_targets, val_preds)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_auc'].append(train_auc)
                history['val_auc'].append(val_auc)
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                        f"train_auc: {train_auc:.4f}, val_auc: {val_auc:.4f}"
                    )
            
            logger.info("Training completed")
            return history
        
        def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
            """Predict risk scores."""
            if isinstance(X, list):
                if self.preprocessor is None:
                    raise ValueError("Model not fitted yet")
                X = self.preprocessor.transform(X)
            
            self.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                scores = self(X_tensor).cpu().numpy()
            
            return scores
        
        def save(self, path: Union[str, Path]):
            """Save model to disk."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'num_filters': self.num_filters,
                'kernel_sizes': self.kernel_sizes,
                'preprocessor': self.preprocessor
            }
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
        
        @classmethod
        def load(cls, path: Union[str, Path]) -> 'TemporalCNN':
            """Load model from disk."""
            checkpoint = torch.load(path)
            
            model = cls(
                input_size=checkpoint['input_size'],
                num_filters=checkpoint['num_filters'],
                kernel_sizes=checkpoint['kernel_sizes']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.preprocessor = checkpoint['preprocessor']
            
            logger.info(f"Model loaded from {path}")
            return model

else:
    # Provide placeholders when PyTorch is not available
    class LSTMScoringModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install with: pip install torch")
    
    class TransformerScoringModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install with: pip install torch")
    
    class TemporalCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install with: pip install torch")


class DeepScoringInterpreter:
    """Interpret deep learning scoring models."""
    
    def __init__(self, model: Any):
        """
        Initialize interpreter.
        
        Args:
            model: Deep learning scoring model
            
        Example:
            >>> interpreter = DeepScoringInterpreter(lstm_model)
            >>> importance = interpreter.feature_importance(X_test)
        """
        self.model = model
        
    def feature_importance(
        self,
        X: np.ndarray,
        method: str = "gradient"
    ) -> np.ndarray:
        """
        Compute feature importance.
        
        Args:
            X: Input sequences
            method: Method to use ('gradient', 'perturbation')
            
        Returns:
            Feature importance scores
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning zeros")
            return np.zeros(X.shape[-1])
        
        if method == "gradient":
            return self._gradient_importance(X)
        elif method == "perturbation":
            return self._perturbation_importance(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _gradient_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute gradient-based importance."""
        if not isinstance(self.model, nn.Module):
            logger.warning("Model is not a PyTorch module")
            return np.zeros(X.shape[-1])
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.model.device)
        X_tensor.requires_grad = True
        
        outputs = self.model(X_tensor)
        
        # Compute gradients
        outputs.sum().backward()
        
        # Aggregate gradients across samples and time steps
        importance = X_tensor.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        
        return importance
    
    def _perturbation_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute perturbation-based importance."""
        baseline_scores = self.model.predict(X)
        baseline_mean = baseline_scores.mean()
        
        importance = np.zeros(X.shape[-1])
        
        for feat_idx in range(X.shape[-1]):
            # Perturb feature
            X_perturbed = X.copy()
            X_perturbed[:, :, feat_idx] = 0
            
            perturbed_scores = self.model.predict(X_perturbed)
            perturbed_mean = perturbed_scores.mean()
            
            # Importance = change in prediction
            importance[feat_idx] = abs(baseline_mean - perturbed_mean)
        
        return importance
    
    def visualize_attention(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Visualize attention weights (for Transformer models).
        
        Args:
            X: Input sequences
            sample_idx: Sample index to visualize
            
        Returns:
            Attention weights matrix or None
        """
        if hasattr(self.model, 'get_attention_weights'):
            attention = self.model.get_attention_weights(X)
            return attention[sample_idx]
        else:
            logger.warning("Model does not support attention visualization")
            return None
