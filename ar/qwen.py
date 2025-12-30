import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM, AutoTokenizer

# ---- Compatibility shim for cut_cross_entropy import-time decorators ----
_orig_torch_compile = torch.compile

def _compile_decorator_compat(model=None, /, *args, **kwargs):  # type: ignore[no-redef]
    if model is None:
        def _decorator(fn):
            return _orig_torch_compile(fn, *args, **kwargs)
        return _decorator
    return _orig_torch_compile(model, *args, **kwargs)

try:
    torch.compile = _compile_decorator_compat  # type: ignore[assignment]
    from cut_cross_entropy import linear_cross_entropy
    LINEAR_CROSS_ENTROPY_AVAILABLE = True
except Exception:
    LINEAR_CROSS_ENTROPY_AVAILABLE = False
finally:
    torch.compile = _orig_torch_compile  # type: ignore[assignment]
    
from torch._dynamo import disable

torch._dynamo.config.optimize_ddp = False

# Configure logger for distributed training
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Only show logs from the main process
rank = os.environ.get("RANK")
if rank is not None and int(rank) > 0:
    logger.setLevel(logging.WARNING)


class Qwen3LM(torch.nn.Module):
    def __init__(
        self, 
        pretrain_path, 
        load_weights=True, 
        attn_implementation='flash_attention_2',
        use_linear_cross_entropy=False
    ):
        super().__init__()
        self.use_linear_cross_entropy = use_linear_cross_entropy
        
        if use_linear_cross_entropy and not LINEAR_CROSS_ENTROPY_AVAILABLE:
            logger.warning(
                "linear_cross_entropy requested but not available. "
                "Falling back to standard PyTorch cross_entropy."
            )
            self.use_linear_cross_entropy = False
        
        logger.info(f"Using {'linear' if self.use_linear_cross_entropy else 'standard'} cross entropy")
        
        ckpt = Path(pretrain_path) / 'model.safetensors'
        if not ckpt.is_file() and load_weights:
            logger.warning(f'No Qwen3LM pretrained model found at {pretrain_path}')
            load_weights = False

        if load_weights:
            # Load model without device_map to let Lightning/DDP handle device placement
            # device_map="auto" conflicts with Lightning's DDP strategy in multi-GPU training
            self.model = Qwen3ForCausalLM.from_pretrained(
                pretrain_path,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
            logger.info(f"Loaded Qwen pretrained model from {pretrain_path}")
        else:
            # Initialize from config when not loading weights
            config = Qwen3ForCausalLM.config_class.from_pretrained(pretrain_path)
            config.attn_implementation = attn_implementation  # Specify attention implementation in config
            self.model = Qwen3ForCausalLM(
                config=config,
            )
            logger.info(f"Initialized Qwen model with config only (no weights loaded)")

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.gradient_checkpointing_enable()

        assert self.model.lm_head.bias is None, "QwenLM lm_head should have no bias"

    @property
    def config(self):
        return self.model.config

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resize token embeddings to match the tokenizer vocabulary size if needed.
        
        This should be called after adding special tokens to the tokenizer.
        The method only resizes when new_num_tokens > current vocab size (expansion).
        If embedding matrix already has redundancy (old size >= new size), no resize is performed.
        
        Args:
            new_num_tokens: New vocabulary size (typically len(tokenizer))
        """
        old_num_tokens = self.model.config.vocab_size
        
        if new_num_tokens <= old_num_tokens:
            logger.info(
                f"Token embeddings size {old_num_tokens} >= tokenizer size {new_num_tokens}, "
                f"no resize needed (embeddings have redundancy)"
            )
            return
        
        logger.info(f"Resizing token embeddings from {old_num_tokens} to {new_num_tokens}")
        
        # Use transformers' built-in resize method which properly handles:
        # - Input embeddings (model.embed_tokens)
        # - Output embeddings (lm_head)
        # - Copying old weights and initializing new tokens
        self.model.resize_token_embeddings(new_num_tokens)
        
        # Update config
        self.model.config.vocab_size = new_num_tokens
        
        logger.info(f"Successfully resized embeddings to {new_num_tokens} tokens")

    def forward_one_step(self, xs, cache=None, cache_position=None):
        """
        Performs one forward step for autoregressive decoding, correctly handling the attention mask for the KV cache.
        """
        batch_size, seq_len_new = xs.shape[:2]

        # If a cache is used, the attention mask must cover the entire sequence length (past + new).
        if cache is not None and cache_position is not None:
            past_len = cache_position[0]
            total_len = past_len + seq_len_new
            # Create a mask of all ones for the full sequence length.
            attention_mask = torch.ones((batch_size, total_len), dtype=torch.long, device=xs.device)
        else:
            # No cache (prefill step), the mask only needs to cover the current input.
            attention_mask = torch.ones((batch_size, seq_len_new), dtype=torch.long, device=xs.device)

        outs = self.model(
            inputs_embeds=xs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
            cache_position=cache_position,
        )
        new_cache = outs.past_key_values
        return outs, new_cache

    @torch.compile
    def forward_lm(self, *args, **kwargs):
        return self.model.model(*args, **kwargs)  # forward LM.

    @disable
    def forward_lm_head(
            self, hidden_states, labels, ignore_index=-1, reduction='mean'):
        """
        Compute cross entropy loss using either linear or standard implementation.
        
        Args:
            hidden_states: Hidden states from the model
            labels: Target labels
            ignore_index: Index to ignore in loss computation
            reduction: Loss reduction method ('mean' or 'none')
            
        Returns:
            Cross entropy loss
        """
        assert reduction in ['mean', 'none']

        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        labels = labels.reshape(-1)

        if self.use_linear_cross_entropy:
            # Memory-efficient linear cross entropy implementation
            # Computes loss directly from hidden states without materializing full logits
            loss = linear_cross_entropy(
                hidden_states.to(torch.bfloat16), 
                self.model.lm_head.weight.to(torch.bfloat16),
                labels.view(-1), 
                ignore_index=ignore_index, 
                reduction=reduction, 
                impl="cce_exact"
            )
        else:
            # Standard Transformers implementation:
            # 1. Get logits through lm_head
            logits = self.model.lm_head(hidden_states)
            
            # 2. Compute cross entropy loss
            loss = F.cross_entropy(
                logits.float(),  # Convert to float32 for numerical stability
                labels,
                ignore_index=ignore_index,
                reduction=reduction
            )
        
        return loss
