"""
Load RhoFold RNA-FM weights into hooked transformer for interpretability analysis.

This module provides utilities to:
1. Map weight names between RhoFold's RNA-FM and our hooked transformer
2. Load pretrained weights with proper initialization
3. Verify the loading was successful
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import re
from collections import OrderedDict
import sys

sys.path.append(str(Path(__file__).parent.parent / "rhofold-main"))
from rhofold.model import rna_fm as rna_fm_module
from rhofold.model.rna_fm import Alphabet

from rna_transformer_interpretability import (
    RNATransformerConfig, 
    RNATransformerWithHooks,
    RNATokenizer
)


def create_weight_mapping() -> Dict[str, str]:
    """
    Create mapping between RhoFold RNA-FM parameter names and our hooked model.
    
    RhoFold RNA-FM structure (from rhofold.model.rna_fm):
    - embed_tokens: token embeddings
    - embed_positions: position embeddings
    - layers.{i}: transformer layers
        - self_attn: multi-head attention
        - self_attn_layer_norm: pre-attention layer norm
        - fc1, fc2: FFN layers
        - final_layer_norm: pre-FFN layer norm
    - emb_layer_norm_after: final layer norm
    - lm_head.dense: LM head linear layer
    - lm_head.layer_norm: LM head layer norm
    
    Returns:
        Dictionary mapping RhoFold names to our model names
    """
    mapping = {}
    mapping['embed_tokens.weight'] = 'token_embedding.weight'
    mapping['embed_positions.weight'] = 'pos_embedding.weight'
    mapping['emb_layer_norm_before.weight'] = 'emb_layer_norm_before.weight'
    mapping['emb_layer_norm_before.bias'] = 'emb_layer_norm_before.bias'
    mapping['emb_layer_norm_after.weight'] = 'ln_f.weight'
    mapping['emb_layer_norm_after.bias'] = 'ln_f.bias'
    
    # NOTE: Skipping LM head mapping due to architectural differences
    # RhoFold LM head: hidden → dense[640→640] → gelu → layer_norm → final_proj[640→25] + bias  
    # Our LM head: hidden → layer_norm → direct_proj[640→25] (with weight tying)
    # For interpretability analysis, the transformer backbone is more important than exact LM head replication
    
    for i in range(12):  # RNA-FM has 12 layers
        mapping[f'layers.{i}.self_attn_layer_norm.weight'] = f'blocks.{i}.ln1.weight'
        mapping[f'layers.{i}.self_attn_layer_norm.bias'] = f'blocks.{i}.ln1.bias'
        mapping[f'layers.{i}.final_layer_norm.weight'] = f'blocks.{i}.ln2.weight'
        mapping[f'layers.{i}.final_layer_norm.bias'] = f'blocks.{i}.ln2.bias'

        # Attention weights
        mapping[f'layers.{i}.self_attn.k_proj.weight'] = f'blocks.{i}.attn.W_K.weight'
        mapping[f'layers.{i}.self_attn.v_proj.weight'] = f'blocks.{i}.attn.W_V.weight'
        mapping[f'layers.{i}.self_attn.q_proj.weight'] = f'blocks.{i}.attn.W_Q.weight'
        mapping[f'layers.{i}.self_attn.out_proj.weight'] = f'blocks.{i}.attn.W_O.weight'
        
        # Attention biases
        mapping[f'layers.{i}.self_attn.k_proj.bias'] = f'blocks.{i}.attn.W_K.bias'
        mapping[f'layers.{i}.self_attn.v_proj.bias'] = f'blocks.{i}.attn.W_V.bias'
        mapping[f'layers.{i}.self_attn.q_proj.bias'] = f'blocks.{i}.attn.W_Q.bias'
        mapping[f'layers.{i}.self_attn.out_proj.bias'] = f'blocks.{i}.attn.W_O.bias'
        
        # MLP weights
        mapping[f'layers.{i}.fc1.weight'] = f'blocks.{i}.mlp.W_in.weight'
        mapping[f'layers.{i}.fc1.bias'] = f'blocks.{i}.mlp.W_in.bias'
        mapping[f'layers.{i}.fc2.weight'] = f'blocks.{i}.mlp.W_out.weight'
        mapping[f'layers.{i}.fc2.bias'] = f'blocks.{i}.mlp.W_out.bias'
    
    return mapping


def load_rhofold_weights(
    hooked_model: RNATransformerWithHooks,
    rhofold_checkpoint_path: str,
    strict: bool = False
) -> Tuple[RNATransformerWithHooks, Dict[str, bool]]:
    """
    Load RhoFold RNA-FM weights into our hooked transformer.
    
    Args:
        hooked_model: Our interpretability-focused transformer
        rhofold_checkpoint_path: Path to RhoFold checkpoint
        strict: Whether to require all weights to be loaded
        
    Returns:
        Tuple of (loaded model, loading report dict)
    """
    print(f"Loading RhoFold checkpoint from {rhofold_checkpoint_path}")
    checkpoint = torch.load(rhofold_checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        full_state_dict = checkpoint['model']
    else:
        full_state_dict = checkpoint
    
    rna_fm_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if 'msa_embedder.rna_fm.' in key:
            # Remove the prefix to get RNA-FM internal names
            rna_fm_key = key.replace('msa_embedder.rna_fm.', '')
            rna_fm_state_dict[rna_fm_key] = value
    
    print(f"Found {len(rna_fm_state_dict)} RNA-FM parameters")
    
    # Print LM head related parameters to understand the structure
    lm_head_keys = [k for k in rna_fm_state_dict.keys() if 'lm_head' in k]
    print(f"LM head parameters found: {lm_head_keys}")
    for k in lm_head_keys:
        print(f"  {k}: {rna_fm_state_dict[k].shape}")
    
    mapping = create_weight_mapping()
    
    loaded = {}
    missing_in_mapping = []
    missing_in_checkpoint = []
    shape_mismatches = []
    
    our_state_dict = hooked_model.state_dict()
    our_params_loaded = set()
    
    for rna_fm_name, our_name in mapping.items():
        if rna_fm_name in rna_fm_state_dict:
            if our_name in our_state_dict:
                rna_fm_param = rna_fm_state_dict[rna_fm_name]
                our_param = our_state_dict[our_name]
                
                if rna_fm_param.shape == our_param.shape:
                    our_state_dict[our_name] = rna_fm_param
                    loaded[our_name] = True
                    our_params_loaded.add(our_name)
                elif our_name == 'pos_embedding.weight' and rna_fm_param.shape[0] > our_param.shape[0]:
                    # Handle position embedding size mismatch by truncating 
                    print(f"Truncating position embeddings from {rna_fm_param.shape} to {our_param.shape}")
                    our_state_dict[our_name] = rna_fm_param[:our_param.shape[0]]
                    loaded[our_name] = True
                    our_params_loaded.add(our_name)
                else:
                    shape_mismatches.append(
                        f"{our_name}: RNA-FM {rna_fm_param.shape} vs ours {our_param.shape}"
                    )
                    loaded[our_name] = False
            else:
                missing_in_checkpoint.append(our_name)
                loaded[our_name] = False
        else:
            missing_in_mapping.append(rna_fm_name)
            loaded[our_name] = False
    
    unmapped_rna_fm = []
    for key in rna_fm_state_dict.keys():
        if key not in mapping:
            unmapped_rna_fm.append(key)
    
    unloaded_ours = []
    for key in our_state_dict.keys():
        if key not in our_params_loaded:
            unloaded_ours.append(key)
    
    hooked_model.load_state_dict(our_state_dict, strict=False)
    
    report = {
        'loaded_count': sum(loaded.values()),
        'total_mapped': len(mapping),
        'missing_in_mapping': missing_in_mapping,
        'missing_in_checkpoint': missing_in_checkpoint,
        'shape_mismatches': shape_mismatches,
        'unmapped_rna_fm_params': unmapped_rna_fm,
        'unloaded_our_params': unloaded_ours,
        'loaded_params': loaded
    }
    
    print(f"\nLoading Summary:")
    print(f"Successfully loaded: {report['loaded_count']}/{report['total_mapped']} mapped parameters")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    print(f"Unmapped RNA-FM params: {len(unmapped_rna_fm)}")
    print(f"Unloaded model params: {len(unloaded_ours)}")
    
    if shape_mismatches:
        print("\nShape mismatches:")
        for mismatch in shape_mismatches:
            print(f"  {mismatch}")
            
    if unmapped_rna_fm:
        print("\nUnmapped RNA-FM parameters:")
        for param in unmapped_rna_fm:
            print(f"  {param}")
            
    if unloaded_ours:
        print("\nOur parameters that weren't loaded:")
        for param in unloaded_ours:
            print(f"  {param}")
    
    # Show which parameter failed to load
    failed_loads = [name for name, success in loaded.items() if not success]
    if failed_loads:
        print(f"\nFailed to load these mapped parameters:")
        for name in failed_loads:
            print(f"  {name}")
    
    if strict and (shape_mismatches or missing_in_checkpoint or unloaded_ours):
        raise ValueError("Strict loading failed - not all parameters were loaded successfully")
    
    return hooked_model, report


def verify_weight_loading_simple(
    model: RNATransformerWithHooks,
    rhofold_checkpoint_path: str
) -> Dict[str, bool]:
    """
    Simple verification that weights were loaded correctly by checking individual parameters.
    
    Args:
        model: Our loaded model
        rhofold_checkpoint_path: Path to original checkpoint
        
    Returns:
        Dictionary with parameter loading verification
    """
    checkpoint = torch.load(rhofold_checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        full_state_dict = checkpoint['model']
    else:
        full_state_dict = checkpoint
    
    rna_fm_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if 'msa_embedder.rna_fm.' in key:
            rna_fm_key = key.replace('msa_embedder.rna_fm.', '')
            rna_fm_state_dict[rna_fm_key] = value
            
    mapping = create_weight_mapping()
    our_state_dict = model.state_dict()
    
    verification_results = {}
    
    # Check a few key parameters to see if they loaded correctly
    key_params = [
        ('embed_tokens.weight', 'token_embedding.weight'),
        ('layers.0.self_attn.q_proj.weight', 'blocks.0.attn.W_Q.weight'),
        ('layers.0.self_attn_layer_norm.weight', 'blocks.0.ln1.weight'),
        ('layers.0.fc1.weight', 'blocks.0.mlp.W_in.weight'),
        ('emb_layer_norm_after.weight', 'ln_f.weight'),
    ]
    
    for rhofold_name, our_name in key_params:
        if rhofold_name in rna_fm_state_dict and our_name in our_state_dict:
            rhofold_param = rna_fm_state_dict[rhofold_name]
            our_param = our_state_dict[our_name]
            
            if rhofold_param.shape == our_param.shape:
                diff = torch.max(torch.abs(rhofold_param - our_param))
                verification_results[our_name] = diff.item() < 1e-6
                print(f"{our_name}: diff = {diff.item():.2e}, loaded = {verification_results[our_name]}")
            else:
                verification_results[our_name] = False
                print(f"{our_name}: shape mismatch {rhofold_param.shape} vs {our_param.shape}")
        else:
            verification_results[our_name] = False
            print(f"{our_name}: parameter not found")
    
    return verification_results


def verify_loading(
    model: RNATransformerWithHooks,
    rhofold_checkpoint_path: str,
    test_sequence: str = "AUGCUAGCUAGCUAGCUA"
) -> Dict[str, torch.Tensor]:
    """
    Verify that weights were loaded correctly by comparing outputs.
    
    Args:
        model: Our loaded model
        rhofold_checkpoint_path: Path to original checkpoint
        test_sequence: RNA sequence to test with
        
    Returns:
        Dictionary with comparison results
    """
    original_model, original_alphabet = rna_fm_module.pretrained.esm1b_rna_t12()
    
    checkpoint = torch.load(rhofold_checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        full_state_dict = checkpoint['model']
    else:
        full_state_dict = checkpoint
    
    rna_fm_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if 'msa_embedder.rna_fm.' in key:
            rna_fm_key = key.replace('msa_embedder.rna_fm.', '')
            rna_fm_state_dict[rna_fm_key] = value
            
    original_model.load_state_dict(rna_fm_state_dict, strict=False)
    original_model.eval()
    
    batch_converter = original_alphabet.get_batch_converter()
    _, _, original_tokens = batch_converter([("test", test_sequence)])
    
    our_tokenizer = RNATokenizer()
    our_tokens = torch.tensor([our_tokenizer.encode(test_sequence)], dtype=torch.long)
    
    attention_mask = torch.ones_like(our_tokens)
    
    print(f"Original alphabet tokens: {original_alphabet.all_toks}")
    print(f"Our tokenizer vocab: {our_tokenizer.vocab}")
    print(f"Original tokens shape: {original_tokens.shape}")
    print(f"Our tokens shape: {our_tokens.shape}")
    print(f"Original tokens: {original_tokens[0]}")
    print(f"Our tokens: {our_tokens[0]}")
    print(f"Attention mask: {attention_mask[0]}")
    
    print("Original sequence decoded:", ''.join([original_alphabet.get_tok(t.item()) for t in original_tokens[0]]))
    print("Our sequence decoded:", ''.join([our_tokenizer.decode([t.item()]) for t in our_tokens[0]]))
    
    model.eval()
    with torch.no_grad():
        our_output = model(our_tokens, attention_mask=attention_mask, return_hidden_states=True)
    
    with torch.no_grad():
        original_output = original_model(original_tokens, repr_layers=[0, 6, 8, 10, 11, 12])

    # Skip logits comparison since we're not loading LM head weights
    # Focus on comparing the transformer backbone (hidden states)
    
    # Check intermediate layers to debug where divergence occurs
    layers_to_check = [0, 6, 8, 10, 11, 12]
    
    for layer_idx in layers_to_check:
        if layer_idx < len(our_output['hidden_states']) and layer_idx in original_output['representations']:
            our_layer = our_output['hidden_states'][layer_idx]
            orig_layer = original_output['representations'][layer_idx]
            
            diff = torch.max(torch.abs(our_layer - orig_layer))
            layer_name = "embeddings" if layer_idx == 0 else f"layer {layer_idx}"
            if layer_idx == 12:
                layer_name = "final (after ln_f)"
            print(f"{layer_name} max diff: {diff.item():.6e}")
    
    # Keep the final comparison for the summary
    our_final_hidden = our_output['hidden_states'][-1]
    original_final_hidden = original_output['representations'][12]
    
    print(f"Our hidden shape: {our_final_hidden.shape}")
    print(f"Original hidden shape: {original_final_hidden.shape}")
    
    # Compare only the overlapping sequence length
    min_seq_len = min(our_final_hidden.shape[1], original_final_hidden.shape[1])
    our_hidden_trimmed = our_final_hidden[:, :min_seq_len]
    original_hidden_trimmed = original_final_hidden[:, :min_seq_len]
    
    hidden_diff = torch.max(torch.abs(our_hidden_trimmed - original_hidden_trimmed))
    
    print(f"Max absolute difference in final hidden state: {hidden_diff.item():.6e}")
    print("NOTE: Skipping logits comparison due to LM head architectural differences")
    
    results = {
        'our_hidden_mean': our_hidden_trimmed.mean(),
        'original_hidden_mean': original_hidden_trimmed.mean(),
        'hidden_max_abs_diff': hidden_diff
    }
    
    if hidden_diff > 1e-4:
        print("!! WARNING: Hidden state differences are larger than expected. Transformer backbone loading may be incorrect.")
    else:
        print("++ SUCCESS: Hidden states match closely. Transformer backbone weights loaded correctly.")
        
    return results


def create_hooked_model_from_rhofold(
    rhofold_checkpoint_path: str,
    device: torch.device = torch.device('cpu')
) -> RNATransformerWithHooks:
    """
    Create a hooked transformer with weights loaded from RhoFold.
    
    Args:
        rhofold_checkpoint_path: Path to RhoFold checkpoint
        device: Device to load model on
        
    Returns:
        Hooked transformer with loaded weights
    """
    config = RNATransformerConfig(
        vocab_size=25,  # RNA-FM vocab size
        n_layers=12,
        d_model=640,
        n_heads=20,
        d_head=32,
        d_mlp=5120,
        n_ctx=1024,
        act_fn="gelu",
        dropout=0.1,
        use_attn_bias=True,
        use_mlp_bias=True
    )
    
    model = RNATransformerWithHooks(config)
    model = model.to(device)
    
    model, report = load_rhofold_weights(model, rhofold_checkpoint_path)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', 
        type=str,
        default='rhofold-main/pretrained/rhofold_pretrained_params.pt',
        help='Path to RhoFold checkpoint'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run verification test'
    )
    
    args = parser.parse_args()
    model = create_hooked_model_from_rhofold(args.checkpoint)
    
    if args.test:
        print("\nRunning simple weight verification...")
        weight_results = verify_weight_loading_simple(model, args.checkpoint)
        
        print("\nRunning full verification test...")
        results = verify_loading(model, args.checkpoint)
        
        print("\nVerification results:")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value}")