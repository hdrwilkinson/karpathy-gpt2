"""Defining the GPT2 model and training loop."""
import os
from dataclasses import dataclass
import math
import inspect
from time import time

# Importing the tokenization library
import tiktoken

# Importing the PyTorch libraries
import torch
import torch.nn as nn # This is the module that contains the neural network layers
from torch.nn import functional as F # This is the module that contains the activation functions
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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

    def __init__(self, B, T, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Load the data
        with open("data/input.txt", "r") as f:
            text = f.read()

        # Tokenize the data
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Number of tokens: {len(self.tokens)} pre mini batch.")
        print(f"Number of mini batches: {len(self.tokens) // (B * T)} per Epoch.")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Returns the next batch of data."""
        B, T = self.B, self.T

        # Get the next batch
        batch = self.tokens[self.current_position : self.current_position + B*T + 1]
        batch_x = batch[:-1].view(B, T)
        batch_y = batch[1:].view(B, T)

        # Update the current position
        self.current_position += B * T * self.num_processes
        if self.current_position + ((B * T * self.num_processes) + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

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


if __name__ == "__main__":
    """ ---------- Setting up for DDP (Distributed Data Parallel) ---------- """
    ddp = int(os.environ.get('RANK', -1)) != -1 # Checks if this is a ddp run
    if ddp:
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
    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)

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

    """ ---------- Optimizer ---------- """
    optimizer = raw_model.configure_optimizers(
        lr=3e-4,
        weight_decay=0.1,
        device=device,
    )
    print("Optimizer loaded successfully!")

    """ ---------- Learning Rate Scheduler ---------- """
    lr_scheduler = LRScheduler(max_lr=6e-4, warmup_steps=10, max_steps=50)
    print("Learning rate scheduler loaded successfully!")

    """ ---------- Micro-Batching ---------- 

    For each forward pass we accumulate 16 * 64 = 1024 tokens before we do a backward pass.
    To achieve the desired batch size of 524288 tokens, we need to accumulate the gradients
    for 512 steps. This is called gradient accumulation.
    """
    # total_batch_size = 524288 # 2**19 ~0.5M tokens
    total_batch_size = 16384
    B = 16
    T = 64
    assert total_batch_size % (B * T * ddp_world_size) == 0, "Total batch size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
    train_loader = DataLoaderLite(
        B=B, 
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
    )

    """ ---------- Training the model ---------- """
    t = time()
    print("Training the model...")
    for i in range(50):
        t0 = time()
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
        lr = lr_scheduler.get_lr(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize() # Wait for the GPU to finish
        ''' !!! Use `watch -n 0.1 nvidia-smi` in the terminal to monitor the GPU usage !!! '''
        t1 = time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (dt / 1000)
        print(f"Step {i + 1} of 50 | LR: {lr:.4e} | Norm: {norm:.4f} | Loss: {loss_accum.item():.6f} | Time: {dt:.2f}ms | Tokens/sec: {tokens_per_sec:.0f}")
    t = time() - t
    print(f"Training took {t:.2f}s.")



    import sys; sys.exit() # Ignores the rest of the code

    """ ---------- Inferencing the model ---------- """
    # Inference parameters
    num_return_sequences = 5
    max_length = 30

    # Setting model to eval and to cuda (if available)
    model.eval()

    # Prefix tokens
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    # Inference
    torch.manual_seed(42)
    # torch.cuda.manual_seed(42)  
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
            # Take the logits at the last position
            logits = logits [:, -1, :] # (B, vocab_size)
            # Get the probabilities
            probs = F.softmax(logits, dim=-1)
            # Do top-k sampling of 50 (HF pipeline default)
            topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1) # (B, 50) and (B, 50)
            # Sample from the top-k
            ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
            # Gather the corresponding indices
            xcol = torch.gather(topk_indicies, -1, ix) # (B, 1)
            # Append to sequence
            x = torch.cat([x, xcol], dim=1)

    # Decode the tokens
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)