"""
Temporal Encoder Implementation

Supports both LSTM and Transformer architectures for encoding temporal sequences.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerBlock(nn.Module):
    """Single Transformer block with MultiHeadAttention and Feed-Forward Network."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            attn_mask: Optional attention mask [seq_len, seq_len] (2D) or 
                      [batch*num_heads, seq_len, seq_len] (3D)
                      True = mask out (don't attend), False = allow attention
        
        Returns:
            [batch, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TemporalEncoder(nn.Module):
    """
    Temporal Encoder supporting both LSTM and Transformer architectures.
    
    Encodes temporal sequences [batch, seq_len, input_dim] into a temporal feature vector.
    - If input_dim=1: Only v_t (highlight vector)
    - If input_dim=2: Both v_t and c_t (highlight vector + task feature)
    """
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_dim: int = 2,
        # Transformer parameters
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        # LSTM parameters
        lstm_hidden_sizes: list = [256, 64, 8],
        lstm_dropout: float = 0.2,
    ):
        """
        Initialize Temporal Encoder.
        
        Args:
            encoder_type: 'lstm' or 'transformer'
            input_dim: Input feature dimension (1 for v_t only, 2 for v_t + c_t)
            embed_dim: Embedding dimension for Transformer
            num_heads: Number of attention heads for Transformer
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward dimension for Transformer
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            lstm_hidden_sizes: List of hidden sizes for LSTM layers [256, 64, 8]
            lstm_dropout: Dropout rate for LSTM
        """
        super(TemporalEncoder, self).__init__()
        self.encoder_type = encoder_type.lower()
        self.input_dim = input_dim
        
        if self.encoder_type == 'transformer':
            self._build_transformer(embed_dim, num_heads, num_layers, ff_dim, dropout, max_seq_len)
            self.output_dim = embed_dim
        elif self.encoder_type == 'lstm':
            self._build_lstm(lstm_hidden_sizes, lstm_dropout)
            self.output_dim = lstm_hidden_sizes[-1]
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'lstm' or 'transformer'")
    
    def _build_transformer(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_seq_len: int
    ):
        """Build Transformer encoder."""
        # Project input from input_dim features to embed_dim
        self.input_projection = nn.Linear(self.input_dim, embed_dim)
        
        # Positional encoding (learnable embeddings)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Layer normalization after projection + positional encoding
        self.norm = nn.LayerNorm(embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global average pooling (will be done in forward)
    
    def _build_lstm(self, hidden_sizes: list, dropout: float):
        """Build LSTM encoder."""
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        input_size = self.input_dim  # v_t (and optionally c_t)
        for i, hidden_size in enumerate(hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0  # We'll apply dropout manually between layers
                )
            )
            # Add dropout after each LSTM except the last one
            if i < len(hidden_sizes) - 1:
                self.dropout_layers.append(nn.Dropout(dropout))
            else:
                self.dropout_layers.append(nn.Identity())  # No dropout after last layer
            input_size = hidden_size
    
    def _create_attention_mask_2d(self, sequence_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create 2D attention mask for MultiheadAttention.
        
        Args:
            sequence_lengths: [batch] tensor with actual sequence lengths
            max_len: Maximum sequence length in batch
        
        Returns:
            [seq_len, seq_len] boolean mask (True for positions to mask out)
            For batch_first=True MultiheadAttention, this 2D mask is broadcasted
        """
        # For simplicity, create mask based on first batch item
        # For multiple batches with different lengths, we'd need 3D mask
        # [batch*num_heads, seq_len, seq_len]
        seq_len_actual = sequence_lengths[0].item() if len(sequence_lengths) > 0 else max_len
        
        # Create 2D mask: [seq_len, seq_len]
        # mask[i, j] = True means position i cannot attend to position j
        mask = torch.zeros(max_len, max_len, dtype=torch.bool, device=sequence_lengths.device)
        
        # Mask out positions beyond actual sequence length
        # Positions >= seq_len_actual cannot attend to anything
        mask[seq_len_actual:, :] = True
        # Nothing can attend to positions >= seq_len_actual
        mask[:, seq_len_actual:] = True
        
        return mask
    
    def _create_valid_positions_mask(self, sequence_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create 1D mask for valid positions (used in pooling).
        
        Args:
            sequence_lengths: [batch] tensor with actual sequence lengths
            max_len: Maximum sequence length in batch
        
        Returns:
            [batch, max_len] boolean mask (True for valid positions, False for padding)
        """
        batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_len, device=sequence_lengths.device).expand(
            batch_size, max_len
        ) < sequence_lengths.unsqueeze(1)
        return mask
    
    def forward(
        self,
        temporal_features: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal encoder.
        
        Args:
            temporal_features: [batch, seq_len, input_dim] tensor
              - If input_dim=1: [v_t] per timestep
              - If input_dim=2: [v_t, c_t] per timestep
            sequence_lengths: Optional [batch] tensor with actual sequence lengths
                            (for handling variable-length sequences)
        
        Returns:
            Temporal feature vector [batch, output_dim]
        """
        if self.encoder_type == 'transformer':
            return self._forward_transformer(temporal_features, sequence_lengths)
        else:  # LSTM
            return self._forward_lstm(temporal_features, sequence_lengths)
    
    def _forward_transformer(
        self,
        temporal_features: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass for Transformer encoder."""
        batch_size, seq_len, _ = temporal_features.shape
        
        # Project input: [batch, seq_len, 2] → [batch, seq_len, embed_dim]
        x = self.input_projection(temporal_features)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=temporal_features.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # Layer normalization
        x = self.norm(x)
        
        # Create attention mask if sequence_lengths provided
        attn_mask = None
        if sequence_lengths is not None:
            # Create 2D mask: [seq_len, seq_len]
            # True = mask out (don't attend), False = allow attention
            attn_mask = self._create_attention_mask_2d(sequence_lengths, seq_len)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        
        # Global average pooling over sequence dimension
        # If sequence_lengths provided, mask out padding before pooling
        if sequence_lengths is not None:
            # Create 1D mask for valid positions
            valid_mask = self._create_valid_positions_mask(sequence_lengths, seq_len)
            # Set padding positions to zero
            x = x * valid_mask.unsqueeze(-1).float()
            # Sum and divide by actual lengths
            x = x.sum(dim=1) / sequence_lengths.unsqueeze(1).float()
        else:
            # Simple average pooling
            x = x.mean(dim=1)
        
        return x  # [batch, embed_dim]
    
    def _forward_lstm(
        self,
        temporal_features: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass for LSTM encoder."""
        x = temporal_features
        
        # Pass through LSTM layers
        num_lstm_layers = len(self.lstm_layers)
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            is_last_layer = (i == num_lstm_layers - 1)
            
            if sequence_lengths is not None and i == 0:
                # Pack padded sequence for first layer (more efficient)
                x_packed = nn.utils.rnn.pack_padded_sequence(
                    x, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                lstm_out, (h_n, c_n) = lstm(x_packed)
                # Unpack
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out, batch_first=True
                )
            else:
                # Regular forward pass
                lstm_out, (h_n, c_n) = lstm(x)
                # For last layer, use hidden state (return_sequences=False)
                # For other layers, use output (return_sequences=True)
                if is_last_layer:
                    x = h_n.squeeze(0)  # [batch, hidden_size]
                    break
                else:
                    x = lstm_out
            
            # Apply dropout (except after last layer)
            if not is_last_layer:
                x = dropout(x)
        
        return x  # [batch, output_dim]
