"""Defining the GPT2 model and training loop."""
from dataclasses import dataclass
import torch
import torch.nn as nn # This is the module that contains the neural network layers
from torch.nn import functional as F # This is the module that contains the activation functions
import math

# ----------------------------------------------------------------------------------------------------------------------

def compare_state_dicts(state_dict, hf_state_dict):
    all_keys = set(state_dict.keys()).union(set(hf_state_dict.keys()))
    
    for key in all_keys:
        state_shape = state_dict.get(key)
        hf_shape = hf_state_dict.get(key)
        
        if state_shape is None:
            print(f"{key} is missing in state_dict")
        elif hf_shape is None:
            print(f"{key} is missing in hf_state_dict")
        elif state_shape != hf_shape:
            print(f"{key} shape mismatch: state_dict {state_shape}, hf_state_dict {hf_shape}")
        else:
            print(f"{key} matches")
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        assert self.n_embed % self.n_head == 0, "n_embed must be divisible by n_head"

        """
        Splitting the Query, Key, and Value Matrices:
        
        The attention mechanism is a matrix operation that is applied to the Query, Key, and Value matrices.
        The Query, Key, and Value matrices are multiplied by the attention weights to produce the output.
        
        It can be done as a batch operation. This is done by stacking the Query, Key, and Value matrices along the batch
        dimension. This allows us to process multiple inputs at the same time.
        """

        # Query, Key, and Value Matrices (in a batch)
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed)
        # Output Matrix
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        # Regularization
        self.register_buffer("bias", 
            torch.tril(
                torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        # B = batch size
        # T = sequence length
        # C = number of channels (embedding size)
        assert C == self.n_embed, "Input size does not match n_embed"

        """
        Method for splitting the QKV matrix:
        
        The QKV matrix is a 3D tensor with shape (B, T, 3 * C), where B is the batch size, T is the sequence length,
        and C is the number of channels (embedding size). The QKV matrix is split along the last dimension (dim=2) into
        three matrices: Q, K, and V. The Q matrix contains the Query vectors, the K matrix contains the Key vectors,
        and the V matrix contains the Value vectors.
        
        The split operation is done using the split() method. The split() method takes two arguments: the size of the
        split dimension and the dimension along which to split the tensor.

        The view() method is used to reshape the QKV matrix into three separate matrices: Q, K, and V. The view() method
        takes the new shape of the tensor as its argument. The new shape is specified as a tuple of the dimensions of the
        reshaped tensor.
        """
        qkv = self.c_attn(x)
        # Splitting the QKV matrix
        q, k, v = qkv.split(self.n_embed, dim=2) 
        # Reshaping the QKV matrix
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # Self-Attention
        y = self.self_attention(q, k, v)
        # Output Matrix
        y = self.c_proj(y)

        return y

    def self_attention(self, q, k, v):
        B, T, C = q.size()
        # Attention Weights
        att = (q @ k.transpose(-2, -1)) / (1.0 / math.sqrt(k.size(-1)))
        # Masking the Attention Weights
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # Softmax Activation
        att = F.softmax(att, dim=-1)
        # Weighted Sum
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # Reshaping the Output
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Fully-Connected Layers
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

        '''
        GELU: Gaussian Error Linear Unit
        
        The approximate eq:
            GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))\
        
        The exact eq:
            GELU(x) = x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            
        The reason for the approximation is that the exact GELU function is computationally expensive.
        In GPT2, the approximation is used. However, in GPT3, the exact GELU function is used. This is because
        GPT3 is trained on a supercomputer with 175 billion parameters, so the computational cost is not an issue.
        Additionally, the exact GELU function is more accurate than the approximate GELU function and is now easier
        to compute with the advent of faster hardware.'''

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class GPTBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # Multi-Head Self-Attention Layer
        self.ln_1 = nn.LayerNorm(config.n_embed) # Layer Normalization (before attention)
        self.attn = CausalSelfAttention(config) # Self-Attention Layer
        self.ln_2 = nn.LayerNorm(config.n_embed)
        # Feed-Forward Layer
        self.mlp = MLP(config)

    def forward(self, x):
        # Add & Norm: x -> Attention -> x -> Feed-Forward -> x
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024      # Number of tokens in each block (or sequence length)
    vocab_size: int = 50257     # Number of unique tokens in the vocabulary
    n_layer: int = 12           # Number of layers in the transformer
    n_head: int = 12            # Number of attention heads
    n_embed: int = 768          # Embedding size for each token (otherwise known as channels)

    """
    The vocab size for GPT is built from:
    
    - 50,000 BPE merges
    - 256 byte tokens
    - 1<|endoftext|> token that both starts and ends the text
    """

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Transformer Architecture (naming convention copied from GPT2)
        self.transformer = nn.ModuleDict( # ModuleDict is a dictionary that contains nn.Module objects
            dict(
                # Embedding Layer
                wte = nn.Embedding(config.vocab_size, config.n_embed),              # Vocab -> Embedding
                wpe = nn.Embedding(config.block_size, config.n_embed),              # Position -> Embedding
                # Transformer Blocks
                h = nn.ModuleList( # ModuleList is a list that contains nn.Module objects (indexable)
                    [GPTBlock(config) for _ in range(config.n_layer)]               # n_layer Transformer Blocks
                ),
                ln_f = nn.LayerNorm(config.n_embed),                                # Layer Normalization (for final output)
            )
        )

        # Output Layer
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)     # Embedding -> Vocab

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pre-trained model weights from Hugging Face."""
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], "Invalid model type"

        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pre-trained model: {model_type}")

        # Specify the configuration arguments for each model type
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embed=768),  # 124M parameters
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embed=1024), # 350M parameters
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embed=1280), # 774M parameters
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M parameters
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # Initialize the GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith(".attn.bias")] # Not required as used for auto-regressive mask

        # Initialise a pre-trained GPT2 model from Hugging Face
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_state_dict = hf_model.state_dict()
        
        # Check if the state dict keys match
        hf_state_dict_keys = hf_state_dict.keys()
        hf_state_dict_keys = [k for k in hf_state_dict_keys if not k.endswith(".attn.masked_bias")]
        hf_state_dict_keys = [k for k in hf_state_dict_keys if not k.endswith(".attn.bias")]

        # Transpose the required weights - why?
        #       The openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        #       this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(hf_state_dict_keys) == len(state_dict_keys), f"mismatched keys: {len(hf_state_dict_keys)} != {len(state_dict_keys)}"
        for k in hf_state_dict_keys:
            hf_state_dict_item = hf_state_dict[k]
            state_dict_item = state_dict[k]
            if any(k.endswith(w) for w in transposed):
                assert hf_state_dict_item.shape[::-1] == state_dict_item.shape, f"{k}: {hf_state_dict_item.shape} -> {state_dict_item.shape}"
                with torch.no_grad():
                    state_dict_item.copy_(hf_state_dict_item.t())
            else:
                assert hf_state_dict_item.shape == state_dict_item.shape, f"{k}: {hf_state_dict_item.shape} -> {state_dict_item.shape}"
                with torch.no_grad():
                    state_dict_item.copy_(hf_state_dict_item)

        return model
    

if __name__ == "__main__":
    model = GPT.from_pretrained("gpt2")
    print(model)
    print("Model loaded successfully!")