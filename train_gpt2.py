"""Defining the GPT2 model and training loop."""
import os
from dataclasses import dataclass
import math
import inspect
from time import time

# Importing the tokenization library
import tiktoken
import numpy as np

# Importing the PyTorch libraries
import torch
import torch.nn as nn # This is the module that contains the neural network layers
from torch.nn import functional as F # This is the module that contains the activation functions
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Importing the hellaswag script for evaluation
from hellaswag import render_example, iterate_examples


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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Scale the initialization of the output matrix (to prevent exploding gradients)
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
        # Self-Attention (flash-attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Reshaping the Output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output Matrix
        y = self.c_proj(y)

        return y

    def self_attention(self, q, k, v):
        B, nh, T, hs = q.size()
        C = nh * hs
        # Attention Weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Masking the Attention Weights
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # Softmax Activation
        att = F.softmax(att, dim=-1)
        # Weighted Sum
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

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
    vocab_size: int = 50304     # Number of unique tokens in the vocabulary
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

        # Weight sharing scheme (Token embeddings and output embeddings are shared)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            """
            Scales the STD by the number of layers to prevent exploding gradients:

            STD grows inside the residual stream i.e. as you go deeper into the network
            and you sum the input with the output, the STD grows by the number of layers.

            We use 2 because there are both an attention and a feed-forward layer in each block.

            We square root the number of layers because the STD is squared in the forward pass.
            """
            std += (2 * self.config.n_layer) ** -0.5 
        """Initializes the weights of the model."""
        # If the module is a Linear or Embedding layer
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Length {T} exceeds block size {self.config.block_size}"

        # Embedding Layer
        position = torch.arange(T, dtype=torch.long, device=idx.device)
        position_embed = self.transformer.wpe(position)
        token_embed = self.transformer.wte(idx)
        x = token_embed + position_embed

        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)

        # Output Layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Loss Calculation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                # We flatten the logits and the targets to (B * T, vocab_size) as required by cross_entropy in PyTorch
                logits.view(-1, self.config.vocab_size), 
                targets.view(-1) # Note that this is not OHE
            )

        return logits, loss

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
        config_args["vocab_size"] = 50304
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
    
    def configure_optimizers(self, weight_decay, lr, device):
        """Configures the optimizer."""
        # All candidate parameters of the model that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Create optim groups
        # Parameters that are 2D will be weight decayed, else no weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"Number of decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters.")
        print(f"Number of non-decayed parameter tensors: {len(no_decay_params)} with {num_no_decay_params:,} parameters.")
        # Create the optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"Using {'fused' if use_fused else 'unfused'} AdamW.")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
class DataLoaderLite:

    def __init__(self, B, T, process_rank=0, num_processes=1, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Check if the split is valid
        assert split in ("train", "val"), f"Invalid split: {split}"

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"Loaded {len(self.tokens)} tokens")
            print(f"Found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def load_tokens(filename):
        """Loads the tokens from a file."""
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self):
        """Returns the next batch of data."""
        B, T = self.B, self.T

        # Get the next batch
        batch = self.tokens[self.current_position : self.current_position + B*T + 1]
        batch_x = batch[:-1].view(B, T)
        batch_y = batch[1:].view(B, T)

        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position += B * T * self.num_processes
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return batch_x, batch_y
        

class LRScheduler:

    def __init__(self, max_lr = 3e-4, warmup_steps = 10, max_steps = 50):
        self.max_lr = max_lr
        self.min_lr = max_lr * 0.1
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step):
        """Linear warmup and cosine decay."""
        # Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * ((step + 1) / self.warmup_steps)
        # Cosine decay
        elif step < self.max_steps:
            decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + coefficient * (self.max_lr - self.min_lr)
        # Minimum learning rate
        else:
            return self.min_lr
        
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


if __name__ == "__main__":

    """ ---------- Hyperparameters ---------- """
    seed = 1337
    max_lr = 6e-4
    warmup_steps = 715
    max_steps = 19073
    weight_decay = 0.1
    vocab_size = 50304
    total_batch_size = 524288 # 2**19 ~0.5M tokens
    batch_size = 64
    sequence_length = 1024
    val_loss_steps = 20
    checkpoint_steps = 5000


    """ ---------- Setting up for DDP (Distributed Data Parallel) ---------- """
    ddp = int(os.environ.get('RANK', -1)) != -1 # Checks if this is a ddp run
    if ddp:
        # Use of DDP demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backends="nccl")
        ddp_rank = int(os.environ["RANK"]) # Rank of the current process (i.e. 0 = 1st process, 1 = 2nd process, etc.) potentially across multiple nodes
        ddp_local_rank = int(os.environ["LOCAL_RANK"]) # Used in multi-node settings, it is the rank of the current process on the current node
        ddp_world_size = int(os.environ["WORLD_SIZE"]) # Number of processes
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device) # Set the device for the current process
        master_process = ddp_rank == 0 # Check if this is the master process (arbitrarily chosen as rank 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cpu"
        if torch.cuda.is_available(): # Check if GPU is available
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check if MPS is available (for M1 chips)
            device = "mps"
        print(f"Using device: {device}")
        master_process = True

    """ ---------- Setting the seed ---------- """
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)

    """ ---------- Lower precision -> Faster Computation ---------- 
    Range (Exponent):
        The exponent in floating-point numbers determines the range of values the number can 
        represent. A larger exponent allows for a wider range of values. For instance, both 
        float32 and TF32 have an 8-bit exponent, providing a range from approximately 10^-38 
        to 10^38. BF16 also uses an 8-bit exponent, offering a similar range. FP16, with a 
        5-bit exponent, has a more limited range from approximately 10^-5 to 10^5.

    Mantissa (Significand):
        The mantissa (or significand) determines the precision of the floating-point number. 
        More bits in the mantissa mean higher precision but slower computations. The mantissa 
        captures the significant digits of the number, with fewer bits leading to faster but 
        less precise calculations.

    Precision:
      - FP32 (float32) uses a 23-bit mantissa and an 8-bit exponent. It offers high precision 
        and a wide range, suitable for tasks requiring detailed calculations but is slower due 
        to its high precision.
      - TF32 uses a 10-bit mantissa and the same 8-bit exponent as FP32. It balances speed 
        and precision, making it faster than FP32 while maintaining sufficient precision for 
        many deep learning tasks.
      - FP16 (float16) has a 10-bit mantissa and a 5-bit exponent. It provides lower 
        precision and a more limited range, making it suitable for specific tasks where speed 
        is critical, and precision requirements are lower.
      - BF16 (bfloat16) has a 7-bit mantissa and an 8-bit exponent. It offers a wide range 
        like FP32 and TF32 but with lower precision, providing fast computations while still 
        covering a broad range of values. BF16 is efficient for training large models where extreme precision is less crucial.
    """
    torch.set_float32_matmul_precision('high') # Highest, High or Medium

    """ ---------- Loading the model ---------- """
    # Loading the GPT2 model
    # model = GPT.from_pretrained("gpt2")
    # Randomly initializing the model 
    config = GPTConfig()
    model = GPT(config)
    model.to(device)
    print(model)
    print("Model loaded successfully!")

    """ ---------- Compiling the model ---------- 
    
    `torch.compile(model)` speeds up computation by optimizing the model's execution 
    through several techniques:

      - Fusion: Combines multiple operations into a single operation, reducing overhead.
      - Kernel Optimization: Utilizes optimized low-level GPU kernels for faster execution.
      - Graph Optimization: Converts dynamic models to static computation graphs, allowing 
        for better optimization.
      - Memory Optimization: Reduces redundant memory operations and improves cache usage.
    
    These optimizations streamline the model's execution, making it faster and more efficient.

    In simple terms:
    
    Using torch.compile(model) makes your model run faster because it transforms the model's 
    instructions into a super-efficient version that the computer can understand and execute 
    more quickly. Imagine giving a chef a recipe that's already perfectly planned out, 
    with all the steps combined in the best order and using the fewest pots and pans. 
    This way, the chef can cook the meal much faster. Similarly, torch.compile reorganizes 
    and optimizes the model's instructions, allowing the computer to process them more 
    quickly and efficiently.

    """
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)

    if ddp:
        """ Used for backwards pass synchronization.
        
        The DistributedDataParallel module is used to parallelize the training of the model across multiple GPUs.
        It is a wrapper that enables the model to be trained on multiple GPUs in a distributed manner.

        It essentially averages the gradients across all GPUs and then updates the model with the averaged gradients.
        """
        model = DDP(model, device_ids=[ddp_local_rank], output_device=device)
    # Get the raw model (without DDP)
    raw_model = model.module if ddp else model

    """ ---------- Learning Rate Scheduler ---------- """
    lr_scheduler = LRScheduler(
        max_lr=max_lr, 
        warmup_steps=warmup_steps, 
        max_steps=max_steps)
    print("Learning rate scheduler loaded successfully!")

    """ ---------- Optimizer ---------- """
    optimizer = raw_model.configure_optimizers(
        lr=max_lr,
        weight_decay=weight_decay,
        device=device,
    )
    print("Optimizer loaded successfully!")

    """ ---------- Logging ---------- """
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass


    """ ---------- Micro-Batching ---------- 

    For each forward pass we accumulate 16 * 64 * WORLD_SIZE tokens before we do a backward pass.
    This is called gradient accumulation. This is done because the model is too large to fit on a single GPU.
    """
    total_batch_size = total_batch_size # 2**19 ~0.5M tokens
    # total_batch_size = 16384 # 2**14 ~16K tokens to make it managaable
    B = batch_size
    T = sequence_length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "Total batch size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")

    """ ---------- Data Loader ---------- """
    train_loader = DataLoaderLite(
        B=B, 
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train"
    )
    val_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val"
    )
    print("Data loaders (train and val) loaded successfully!")

    """ ---------- Tokenization ---------- """
    enc = tiktoken.get_encoding("gpt2")


    """ ---------- Training the model ---------- """
    t = time()
    print("Training the model...")
    for step in range(max_steps):
        t0 = time()
        last_step = (step == max_steps - 1)

        # Evaluation
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % checkpoint_steps == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        # once in a while evaluate hellaswag
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # Inference
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    logits, loss = model(xgen) # (B, T, vocab_size)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")
            

        # Training
        optimizer.zero_grad() # Zero the gradients - Why? Because PyTorch accumulates the gradients on subsequent backward passes
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            """ This is the training loop.
            
            For each micro step required to accumulate the gradients, we:
                - Load the next batch of data
                - Forward pass
                - Calculate the loss
                - Backward pass

            To calculate the loss, we must remember to divide by the number of gradient accumulation steps.
            """
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                # Synchronize the gradients across all processes when a full batch is accumulated
                # This is a default pytorch variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            # Synchronize the gradients across all processes
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize() # Wait for the GPU to finish
        ''' !!! Use `watch -n 0.1 nvidia-smi` in the terminal to monitor the GPU usage !!! '''
        t1 = time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (dt / 1000)
        if master_process:
            print(f"Step {step + 1:5d} | LR: {lr:.4e} | Norm: {norm:.4f} | Loss: {loss_accum.item():.6f} | Time: {dt:.2f}ms | Tokens/sec: {tokens_per_sec:.0f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    t = time() - t
    print(f"Training took {t:.2f}s.")
