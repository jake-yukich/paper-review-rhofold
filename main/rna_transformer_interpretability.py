"""
RNA Transformer Model with Mechanistic Interpretability Tools

This module implements an RNA language model based on the RhoFold RNA-FM architecture,
designed for mechanistic interpretability studies using TransformerLens-style hooks.

Architecture:
- 12 transformer encoder blocks
- 640-dimensional embeddings
- 20 attention heads per block
- 5120-dimensional feed-forward hidden size
- Bidirectional self-attention (BERT-style)
- Masked language modeling objective

Features:

1. Architecture matching RhoFold's RNA-FM:
    - 12 transformer encoder blocks
    - 640-dimensional embeddings
    - 20 attention heads (32 dims per head)
    - 5120-dimensional FFN hidden size
    - BERT-style bidirectional attention
    - Pre-norm architecture with residual connections
  2. RNA-specific tokenizer:
    - Handles standard nucleotides (A, C, G, U)
    - Supports ambiguity codes (R, Y, K, M, etc.)
    - Special tokens for padding, masking, etc.
  3. Interpretability hooks throughout:
    - Attention weight caching
    - MLP activation tracking
    - Residual stream decomposition
    - Layer-wise hidden state extraction
  4. Analysis utilities:
    - visualize_attention_patterns() - Plot attention heatmaps
    - analyze_neuron_activations() - Find most active neurons
    - get_residual_stream_contributions() - Decompose layer contributions
    - run_with_hooks() - Custom hook insertion
  5. Hook points available:
    - Embeddings
    - Layer normalizations (pre/post)
    - Attention Q/K/V matrices
    - Attention weights
    - MLP pre/post activations
    - Residual connections

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import einops
from fancy_einsum import einsum


@dataclass
class RNATransformerConfig:
    """Configuration for RNA Transformer model"""
    vocab_size: int = 25  # RNA tokens + special tokens
    n_layers: int = 12
    d_model: int = 640
    n_heads: int = 20
    d_head: int = 32  # d_model // n_heads
    d_mlp: int = 5120
    n_ctx: int = 1024  # max sequence length
    act_fn: str = "gelu"
    normalization_type: str = "LN"
    dropout: float = 0.1
    use_attn_bias: bool = True
    use_mlp_bias: bool = True
    
    # RNA-specific tokens (matching RhoFold alphabet)
    cls_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    mask_token_id: int = 24
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


class RNATokenizer:
    """Simple RNA tokenizer compatible with RhoFold's alphabet"""
    
    def __init__(self):
        # Match RhoFold's exact token order:
        # ['<cls>', '<pad>', '<eos>', '<unk>', 'A', 'C', 'G', 'U', 'R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N', '-', '<null_1>', '<null_2>', '<null_3>', '<null_4>', '<mask>']
        self.base_tokens = ['A', 'C', 'G', 'U', 'R', 'Y', 'K', 'M', 
                           'S', 'W', 'B', 'D', 'H', 'V', 'N', '-']
        
        # Build vocab exactly like RhoFold
        self.vocab = ['<cls>', '<pad>', '<eos>', '<unk>'] + self.base_tokens + ['<null_1>', '<null_2>', '<null_3>', '<null_4>', '<mask>']
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 24
    
    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """Encode RNA sequence to token IDs"""
        tokens = []
        if add_special_tokens:
            tokens.append(self.cls_token_id)
        
        for char in sequence.upper():
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to RNA sequence"""
        special_tokens = ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>', '<null_1>', '<null_2>', '<null_3>', '<null_4>']
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if skip_special_tokens and token in special_tokens:
                    continue
                tokens.append(token)
        return ''.join(tokens)


class HookedTransformerBlock(nn.Module):
    """Single transformer block with interpretability hooks"""
    
    def __init__(self, cfg: RNATransformerConfig, block_idx: int):
        super().__init__()
        self.cfg = cfg
        self.block_idx = block_idx
        
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        
        self.attn = HookedMultiHeadAttention(cfg, block_idx)
        self.mlp = HookedMLP(cfg, block_idx)
        
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.hooks = defaultdict(list)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                cache_activations: bool = False) -> torch.Tensor:
        """
        Forward pass through transformer block - RhoFold compatible (T,B,E) format
        
        Args:
            x: Input tensor [seq_len, batch, d_model]
            attention_mask: Optional padding mask [batch, seq_len] where True = padding
            cache_activations: Whether to cache intermediate activations
        
        Returns:
            Output tensor [seq_len, batch, d_model]
        """
        residual = x
        x_normed = self.ln1(x)
        
        if cache_activations:
            self.hooks[f"block{self.block_idx}.ln1"].append(x_normed.detach())
        
        attn_out = self.attn(x_normed, attention_mask=attention_mask, 
                            cache_activations=cache_activations)
        x = residual + self.dropout(attn_out)
        
        if cache_activations:
            self.hooks[f"block{self.block_idx}.attn_out"].append(x.detach())
        
        residual = x
        x_normed = self.ln2(x)
        
        if cache_activations:
            self.hooks[f"block{self.block_idx}.ln2"].append(x_normed.detach())
        
        mlp_out = self.mlp(x_normed, cache_activations=cache_activations)
        x = residual + self.dropout(mlp_out)
        
        if cache_activations:
            self.hooks[f"block{self.block_idx}.mlp_out"].append(x.detach())
        
        return x


class HookedMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability hooks"""
    
    def __init__(self, cfg: RNATransformerConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.use_attn_bias)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.use_attn_bias)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.use_attn_bias)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.use_attn_bias)
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / np.sqrt(cfg.d_head)
        
        self.hooks = defaultdict(list)
    
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cache_activations: bool = False) -> torch.Tensor:
        """
        Multi-head attention forward pass - RhoFold compatible (T,B,E) format
        
        Args:
            x: Input tensor [seq_len, batch, d_model]
            attention_mask: Optional padding mask [batch, seq_len] where True = padding
            cache_activations: Whether to cache intermediate values
        
        Returns:
            Attention output [seq_len, batch, d_model]
        """
        seq_len, batch_size, _ = x.shape
        
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Rearrange for (T,B,E) input format
        Q = einops.rearrange(Q, 's b (h d) -> b h s d', h=self.cfg.n_heads)
        K = einops.rearrange(K, 's b (h d) -> b h s d', h=self.cfg.n_heads)
        V = einops.rearrange(V, 's b (h d) -> b h s d', h=self.cfg.n_heads)
        
        if cache_activations:
            self.hooks[f"layer{self.layer_idx}.Q"].append(Q.detach())
            self.hooks[f"layer{self.layer_idx}.K"].append(K.detach())
            self.hooks[f"layer{self.layer_idx}.V"].append(V.detach())
        
        scores = einsum('b h q d, b h k d -> b h q k', Q, K) * self.scale
        
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] where True = padding
            # Convert to attention scores mask: [batch, 1, seq_len, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
            mask = mask.expand(-1, -1, seq_len, -1)  # [B, 1, S, S] 
            scores = scores.masked_fill(mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if cache_activations:
            self.hooks[f"layer{self.layer_idx}.attn_weights"].append(attn_weights.detach())
        
        attn_out = einsum('b h q k, b h k d -> b h q d', attn_weights, V)
        attn_out = einops.rearrange(attn_out, 'b h s d -> s b (h d)')  # Back to (T,B,E)
        output = self.W_O(attn_out)
        
        if cache_activations:
            self.hooks[f"layer{self.layer_idx}.attn_output"].append(output.detach())
        
        return output


class HookedMLP(nn.Module):
    """Feed-forward network with interpretability hooks"""
    
    def __init__(self, cfg: RNATransformerConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp, bias=cfg.use_mlp_bias)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model, bias=cfg.use_mlp_bias)
        
        self.act_fn = nn.GELU() if cfg.act_fn == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.hooks = defaultdict(list)
    
    def forward(self, x: torch.Tensor, cache_activations: bool = False) -> torch.Tensor:
        """
        MLP forward pass - works with both (B,T,E) and (T,B,E) formats
        
        Args:
            x: Input tensor [seq_len, batch, d_model] - RhoFold format
            cache_activations: Whether to cache intermediate activations
        
        Returns:
            MLP output [seq_len, batch, d_model]
        """
        hidden = self.W_in(x)
        
        if cache_activations:
            self.hooks[f"layer{self.layer_idx}.mlp_pre_act"].append(hidden.detach())
        
        hidden = self.act_fn(hidden)
        hidden = self.dropout(hidden)
        
        if cache_activations:
            self.hooks[f"layer{self.layer_idx}.mlp_post_act"].append(hidden.detach())
        
        output = self.W_out(hidden)
        
        return output


class RNATransformerWithHooks(nn.Module):
    """
    RNA Transformer model with mechanistic interpretability hooks
    
    This model closely follows the RhoFold RNA-FM architecture but adds
    extensive hooks for interpretability analysis.
    """
    
    def __init__(self, cfg: RNATransformerConfig):
        super().__init__()
        self.cfg = cfg
        
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Add pre-transformer embedding layer norm to match RhoFold architecture
        self.emb_layer_norm_before = nn.LayerNorm(cfg.d_model, eps=1e-5)
        
        self.blocks = nn.ModuleList([
            HookedTransformerBlock(cfg, idx) for idx in range(cfg.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.hooks = defaultdict(list)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following BERT/RoBERTa initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cache_activations: bool = False,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNA transformer
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            cache_activations: Whether to cache all intermediate activations
            return_hidden_states: Whether to return hidden states from all layers
        
        Returns:
            Dictionary containing:
                - logits: Language modeling logits [batch, seq_len, vocab_size]
                - hidden_states: List of hidden states from each layer (if requested)
                - attentions: Cached attention weights (if cache_activations=True)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create padding-aware position embeddings like RhoFold
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            mask = attention_mask.int()
            # Compute cumulative positions for non-padding tokens only
            positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long()
            # Add padding offset (RhoFold uses padding_idx + positions)
            positions = positions + self.cfg.pad_token_id  # pad_token_id = 1
            # Clamp to valid range
            positions = positions.clamp(max=self.cfg.n_ctx - 1)
        else:
            # No mask provided, use sequential positions
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(positions)
        
        x = token_embeds + pos_embeds
        # Apply pre-transformer layer norm like RhoFold
        x = self.emb_layer_norm_before(x)
        
        # Apply padding mask like RhoFold (if attention_mask provided)
        if attention_mask is not None:
            # Convert attention_mask (1 for real tokens, 0 for padding) to padding_mask format
            padding_mask = (attention_mask == 0)  # True for padding tokens
            # Apply mask by zeroing out padding positions
            x = x * (1 - padding_mask.unsqueeze(-1).float())
        
        x = self.dropout(x)
        
        if cache_activations:
            self.hooks["embeddings"].append(x.detach())
        
        # Convert to (T,B,E) format for transformer blocks like RhoFold
        x = x.transpose(0, 1)  # (B,T,E) -> (T,B,E)
        
        # Convert attention mask for RhoFold-style processing
        if attention_mask is not None:
            # Use the padding mask we computed above
            rhofold_padding_mask = padding_mask
        else:
            rhofold_padding_mask = None
        
        # Track hidden states
        hidden_states = []
        if return_hidden_states:
            # Store as (B,T,E) for output compatibility
            hidden_states.append(x.transpose(0, 1))
        
        # Pass through transformer blocks in (T,B,E) format
        for i, block in enumerate(self.blocks):
            x = block(x, attention_mask=rhofold_padding_mask, 
                     cache_activations=cache_activations)
            
            if return_hidden_states:
                if i == len(self.blocks) - 1:  # Last layer (layer 11, which becomes layer 12 in 1-indexed)
                    # First storage: before final layer norm (will be overwritten)
                    hidden_states.append(x.transpose(0, 1))  # Store as (B,T,E)
                else:
                    hidden_states.append(x.transpose(0, 1))  # Store as (B,T,E)
        
        # Apply final layer norm in (T,B,E) format like RhoFold
        x = self.ln_f(x)  # Apply layer norm in (T,B,E) format
        
        # Update the final hidden state to include layer norm (like RhoFold's overwrite)
        if return_hidden_states:
            hidden_states[-1] = x.transpose(0, 1)  # Overwrite with post-layer-norm version
        
        x = x.transpose(0, 1)  # (T,B,E) -> (B,T,E)
        
        if cache_activations:
            self.hooks["ln_final"].append(x.detach())
        
        logits = self.lm_head(x)
        output = {"logits": logits}
        
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        
        if cache_activations:
            all_attentions = []
            for i in range(self.cfg.n_layers):
                attn_key = f"layer{i}.attn_weights"
                for block in self.blocks:
                    if attn_key in block.attn.hooks:
                        all_attentions.extend(block.attn.hooks[attn_key])
            
            if all_attentions:
                output["attentions"] = all_attentions
        
        return output
    
    def get_attention_patterns(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get cached attention patterns for a specific layer"""
        key = f"layer{layer_idx}.attn_weights"
        for block in self.blocks:
            if key in block.attn.hooks and block.attn.hooks[key]:
                return block.attn.hooks[key][-1]  # Return most recent
        return None
    
    def clear_hooks(self):
        """Clear all cached activations"""
        self.hooks.clear()
        for block in self.blocks:
            block.hooks.clear()
            block.attn.hooks.clear()
            block.mlp.hooks.clear()
    
    def get_mlp_activations(self, layer_idx: int, 
                           activation_type: str = "post_act") -> Optional[torch.Tensor]:
        """
        Get MLP activations for a specific layer
        
        Args:
            layer_idx: Which layer to get activations from
            activation_type: "pre_act" or "post_act"
        
        Returns:
            Cached MLP activations if available
        """
        key = f"layer{layer_idx}.mlp_{activation_type}"
        if layer_idx < len(self.blocks):
            block = self.blocks[layer_idx]
            if key in block.mlp.hooks and block.mlp.hooks[key]:
                return block.mlp.hooks[key][-1]
        return None
    
    def run_with_hooks(self, 
                      input_ids: torch.Tensor,
                      hook_points: List[str],
                      hook_fn: Optional[callable] = None) -> Dict[str, torch.Tensor]:
        """
        Run forward pass with custom hooks at specified points
        
        Args:
            input_ids: Input token IDs
            hook_points: List of hook point names (e.g., ["block0.mlp_out", "block5.attn_weights"])
            hook_fn: Optional function to apply at hook points
        
        Returns:
            Forward pass output with hooked values
        """
        output = self.forward(input_ids, cache_activations=True)
        
        hooked_values = {}
        for hook_point in hook_points:
            # Parse hook point
            if "block" in hook_point:
                parts = hook_point.split(".")
                block_idx = int(parts[0].replace("block", ""))
                hook_type = ".".join(parts[1:])
                
                if block_idx < len(self.blocks):
                    block = self.blocks[block_idx]
                    
                    if hook_type in block.hooks:
                        values = block.hooks[hook_type]
                    elif hook_type in block.attn.hooks:
                        values = block.attn.hooks[hook_type]
                    elif hook_type in block.mlp.hooks:
                        values = block.mlp.hooks[hook_type]
                    else:
                        continue
                    
                    if values:
                        hooked_value = values[-1]
                        if hook_fn is not None:
                            hooked_value = hook_fn(hooked_value)
                        hooked_values[hook_point] = hooked_value
            
            elif hook_point in self.hooks:
                values = self.hooks[hook_point]
                if values:
                    hooked_value = values[-1]
                    if hook_fn is not None:
                        hooked_value = hook_fn(hooked_value)
                    hooked_values[hook_point] = hooked_value
        
        output["hooked_values"] = hooked_values
        return output


def create_attention_mask(sequence_lengths: List[int], 
                         max_length: int,
                         device: torch.device) -> torch.Tensor:
    """
    Create attention mask for variable length sequences
    
    Args:
        sequence_lengths: List of actual sequence lengths
        max_length: Maximum sequence length in batch
        device: Device to create tensor on
    
    Returns:
        Attention mask [batch_size, max_length]
    """
    batch_size = len(sequence_lengths)
    mask = torch.zeros(batch_size, max_length, device=device)
    
    for i, length in enumerate(sequence_lengths):
        mask[i, :length] = 1
    
    return mask


def visualize_attention_patterns(attention_weights: torch.Tensor,
                                tokens: List[str],
                                layer_idx: int,
                                head_idx: Optional[int] = None) -> None:
    """
    Visualize attention patterns for interpretability
    
    Args:
        attention_weights: Attention weights [batch, heads, seq_len, seq_len]
        tokens: List of token strings
        layer_idx: Which layer's attention to visualize
        head_idx: Specific head to visualize (None for average)
    """
    import matplotlib.pyplot as plt
    
    attn = attention_weights[0]  # [heads, seq_len, seq_len]
    
    if head_idx is not None:
        attn_to_plot = attn[head_idx]
    else:
        attn_to_plot = attn.mean(dim=0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_to_plot.cpu().numpy(), cmap='Blues', aspect='auto')
    plt.colorbar()
    
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    
    title = f"Layer {layer_idx} Attention"
    if head_idx is not None:
        title += f" (Head {head_idx})"
    else:
        title += " (Average)"
    
    plt.title(title)
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.show()


def analyze_neuron_activations(mlp_activations: torch.Tensor,
                              top_k: int = 10) -> Dict[int, float]:
    """
    Analyze which neurons are most active in MLP layers
    
    Args:
        mlp_activations: MLP activations [batch, seq_len, d_mlp]
        top_k: Number of top neurons to return
    
    Returns:
        Dictionary mapping neuron indices to average activation values
    """
    avg_activations = mlp_activations.mean(dim=[0, 1])  # [d_mlp]
    
    top_values, top_indices = torch.topk(avg_activations.abs(), k=top_k)
    
    neuron_activations = {}
    for idx, value in zip(top_indices.tolist(), top_values.tolist()):
        neuron_activations[idx] = value
    
    return neuron_activations


def get_residual_stream_contributions(model: RNATransformerWithHooks,
                                     input_ids: torch.Tensor,
                                     layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    Decompose residual stream contributions at a specific layer
    
    Args:
        model: RNA transformer model
        input_ids: Input token IDs
        layer_idx: Which layer to analyze
    
    Returns:
        Dictionary with residual stream components
    """
    output = model.forward(input_ids, cache_activations=True, return_hidden_states=True)
    hidden_states = output["hidden_states"]
    
    residual_before = hidden_states[layer_idx]
    residual_after = hidden_states[layer_idx + 1]
    
    layer_contribution = residual_after - residual_before
    
    block = model.blocks[layer_idx]
    attn_out_key = f"block{layer_idx}.attn_out"
    mlp_out_key = f"block{layer_idx}.mlp_out"
    
    contributions = {
        "residual_before": residual_before,
        "residual_after": residual_after,
        "total_contribution": layer_contribution
    }
    
    if attn_out_key in block.hooks and block.hooks[attn_out_key]:
        attn_residual = block.hooks[attn_out_key][-1]
        contributions["attention_contribution"] = attn_residual - residual_before
    
    if mlp_out_key in block.hooks and block.hooks[mlp_out_key]:
        mlp_residual = block.hooks[mlp_out_key][-1]
        contributions["mlp_contribution"] = mlp_residual - attn_residual
    
    return contributions


if __name__ == "__main__":
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    rna_sequence = "AUGCUAGCUAGCUAGCUA"
    
    input_ids = torch.tensor([tokenizer.encode(rna_sequence)], dtype=torch.long)
    
    output = model.forward(input_ids, cache_activations=True, return_hidden_states=True)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Number of hidden states: {len(output['hidden_states'])}")
    
    attn_patterns = model.get_attention_patterns(layer_idx=5)
    if attn_patterns is not None:
        print(f"Attention patterns shape: {attn_patterns.shape}")
    
    mlp_acts = model.get_mlp_activations(layer_idx=5, activation_type="post_act")
    if mlp_acts is not None:
        top_neurons = analyze_neuron_activations(mlp_acts)
        print(f"Top activated neurons in layer 5: {list(top_neurons.keys())[:5]}")
    
    model.clear_hooks()