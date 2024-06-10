"""Defining the GPT2 model and training loop."""
from dataclasses import dataclass
import torch
import torch.nn as nn # This is the module that contains the neural network layers
from torch.nn import functional as F # This is the module that contains the activation functions

# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 256       # Number of tokens in each block (or sequence length)
    vocab_size: int = 65        # Number of unique tokens in the vocabulary
    n_layer: int = 6            # Number of layers in the transformer
    n_nead: int = 6             # Number of attention heads
    n_embed: int = 384          # Embedding size for each token (otherwise known as channels)

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Transformer Architecture (naming convention copied from GPT2)
        self.transformer = nn.ModuleDict( # ModuleDict is a dictionary that contains nn.Module objects
            dict(
                # Embedding Layer
                wte = nn.Embedding(config.vocab_size, config.n_embed),          # Vocab -> Embedding
                wpe = nn.Embedding(config.block_size, config.n_embed),          # Position -> Embedding
                # Transformer Blocks
                h = nn.ModuleList( # ModuleList is a list that contains nn.Module objects (indexable)
                    [GPTBlock(config) for _ in range(config.n_layer)]           # n_layer Transformer Blocks
                ),
                ln_f = nn.LayerNorm(config.n_embed),                            # Layer Normalization (for final output)
                # Output Layer
                head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # Embedding -> Vocab
            )
        )

class GPTBlock(nn.Module):
    
    def __init__(self, config):
        super.__init__()
        self.config = config

        # Multi-Head Self-Attention Layer
        self.ln_1 = nn.LayerNorm(config.n_embed) # Layer Normalization (before attention)
        self.attn = CausalSelfAttention(config) # Self-Attention Layer
        self.ln_2 = nn.LayerNorm(config.n_embed)
        # Feed-Forward Layer
        self.mpl = MLP(config)

    def forward(self, x):
        # Add & Norm: x -> Attention -> x -> Feed-Forward -> x
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x