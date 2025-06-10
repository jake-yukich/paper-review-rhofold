"""
Examples of using the RNA Transformer for mechanistic interpretability studies
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from rna_transformer_interpretability import (
    RNATransformerConfig, 
    RNATransformerWithHooks,
    RNATokenizer,
    visualize_attention_patterns,
    analyze_neuron_activations,
    get_residual_stream_contributions
)


def example_1_basic_usage():
    """Basic model usage and forward pass"""
    print("=== Example 1: Basic Usage ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    sequences = [
        "AUGCUAGCUAGCUAGCUA",
        "GCGCGCGCGCGCGCGC",
        "UUUUAAAAUUUUAAAA"
    ]
    
    input_ids = torch.tensor([tokenizer.encode(seq) for seq in sequences])
    print(f"Input shape: {input_ids.shape}")
    
    output = model.forward(input_ids)
    print(f"Logits shape: {output['logits'].shape}")
    
    predictions = torch.argmax(output['logits'], dim=-1)
    print(f"Predictions shape: {predictions.shape}")
    
    for i, seq in enumerate(sequences):
        pred_seq = tokenizer.decode(predictions[i].tolist(), skip_special_tokens=True)
        print(f"Original: {seq}")
        print(f"Predicted: {pred_seq}\n")


def example_2_attention_analysis():
    """Analyze attention patterns across layers"""
    print("\n=== Example 2: Attention Pattern Analysis ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    # Use a sequence with interesting structure
    sequence = "GGGGAAAACCCC"  # Simple hairpin-like pattern
    input_ids = torch.tensor([tokenizer.encode(sequence)])
    tokens = ['<cls>'] + list(sequence) + ['<eos>']
    
    output = model.forward(input_ids, cache_activations=True)
    
    layers_to_analyze = [0, 5, 11]  # Early, middle, late layers
    
    for layer_idx in layers_to_analyze:
        attn_weights = model.get_attention_patterns(layer_idx)
        if attn_weights is not None:
            print(f"\nLayer {layer_idx} attention shape: {attn_weights.shape}")
            
            # Analyze attention entropy (how focused vs distributed)
            # Average over heads and batch
            avg_attn = attn_weights[0].mean(dim=0)  # [seq_len, seq_len]
            entropy = -torch.sum(avg_attn * torch.log(avg_attn + 1e-10), dim=-1)
            print(f"Average attention entropy: {entropy.mean().item():.3f}")
            
            max_attn_indices = torch.argmax(avg_attn, dim=-1)
            for i, token in enumerate(tokens[:len(max_attn_indices)]):
                attending_to = tokens[max_attn_indices[i]]
                print(f"  {token} attends most to {attending_to}")
    
    model.clear_hooks()


def example_3_neuron_analysis():
    """Analyze MLP neuron activations"""
    print("\n=== Example 3: MLP Neuron Analysis ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    sequences = {
        "AU-rich": "AUAUAUAUAUAUAUAU",
        "GC-rich": "GCGCGCGCGCGCGCGC",
        "Mixed": "AUGCAUGCAUGCAUGC",
        "Poly-A": "AAAAAAAAAAAAAAAA"
    }
    
    for seq_type, sequence in sequences.items():
        input_ids = torch.tensor([tokenizer.encode(sequence)])
        output = model.forward(input_ids, cache_activations=True)
        
        print(f"\n{seq_type} sequence: {sequence}")
        
        # Check middle layer MLP activations
        layer_idx = 6
        mlp_acts = model.get_mlp_activations(layer_idx, activation_type="post_act")
        
        if mlp_acts is not None:
            top_neurons = analyze_neuron_activations(mlp_acts, top_k=5)
            print(f"Top 5 activated neurons in layer {layer_idx}:")
            for neuron_idx, activation in top_neurons.items():
                print(f"  Neuron {neuron_idx}: {activation:.3f}")
        
        model.clear_hooks()


def example_4_residual_stream_analysis():
    """Decompose residual stream contributions"""
    print("\n=== Example 4: Residual Stream Analysis ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    # Use a sequence with known secondary structure motif
    sequence = "GGGCUUUUGCCC"  # Stem-loop structure
    input_ids = torch.tensor([tokenizer.encode(sequence)])
    
    print(f"Analyzing sequence: {sequence}")
    
    for layer_idx in [0, 5, 11]:
        contributions = get_residual_stream_contributions(model, input_ids, layer_idx)
        
        print(f"\nLayer {layer_idx} contributions:")
        
        if "attention_contribution" in contributions:
            attn_norm = contributions["attention_contribution"].norm().item()
            print(f"  Attention contribution norm: {attn_norm:.3f}")
        
        if "mlp_contribution" in contributions:
            mlp_norm = contributions["mlp_contribution"].norm().item()
            print(f"  MLP contribution norm: {mlp_norm:.3f}")
        
        total_norm = contributions["total_contribution"].norm().item()
        print(f"  Total layer contribution norm: {total_norm:.3f}")
    
    model.clear_hooks()


def example_5_hook_interventions():
    """Use hooks for causal interventions"""
    print("\n=== Example 5: Hook-Based Interventions ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    sequence = "AUGCAUGCAUGC"
    input_ids = torch.tensor([tokenizer.encode(sequence)])
    
    def zero_attention_head(attn_weights, head_idx=0):
        """Zero out a specific attention head"""
        attn_weights_copy = attn_weights.clone()
        attn_weights_copy[:, head_idx, :, :] = 0
        return attn_weights_copy
    
    def amplify_neurons(mlp_acts, factor=2.0, neuron_indices=[0, 1, 2]):
        """Amplify specific neuron activations"""
        mlp_acts_copy = mlp_acts.clone()
        mlp_acts_copy[:, :, neuron_indices] *= factor
        return mlp_acts_copy
    
    output_normal = model.forward(input_ids)
    logits_normal = output_normal['logits']
    
    model.clear_hooks()
    hook_fn = lambda x: zero_attention_head(x, head_idx=5)
    output_intervened = model.run_with_hooks(
        input_ids, 
        hook_points=["block5.attn_weights"],
        hook_fn=hook_fn
    )
    logits_intervened = output_intervened['logits']
    
    logit_diff = (logits_normal - logits_intervened).abs().mean().item()
    print(f"Average logit difference after zeroing head 5 in layer 5: {logit_diff:.4f}")
    
    position_effects = (logits_normal - logits_intervened).abs().mean(dim=-1)[0]
    most_affected_pos = torch.argmax(position_effects).item()
    tokens = tokenizer.decode(input_ids[0].tolist())
    print(f"Most affected position: {most_affected_pos} (token: {tokens[most_affected_pos]})")
    
    model.clear_hooks()


def example_6_attention_pattern_visualization():
    """Visualize attention patterns for different sequences"""
    print("\n=== Example 6: Attention Pattern Visualization ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    # Create a sequence with complementary regions
    sequence = "GGGAAAACCC"  # Should form base pairs G-C
    input_ids = torch.tensor([tokenizer.encode(sequence)])
    tokens = ['<cls>'] + list(sequence) + ['<eos>']
    
    output = model.forward(input_ids, cache_activations=True)
    
    layer_idx = 11
    attn_weights = model.get_attention_patterns(layer_idx)
    
    if attn_weights is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        heads_to_plot = [0, 5, 10, 15]
        
        for idx, (ax, head_idx) in enumerate(zip(axes.flat, heads_to_plot)):
            attn = attn_weights[0, head_idx].cpu().numpy()
            im = ax.imshow(attn, cmap='Blues', aspect='auto')
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45)
            ax.set_yticklabels(tokens)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
        print(f"Saved attention pattern visualization to 'attention_patterns.png'")
    
    model.clear_hooks()


def example_7_position_embedding_analysis():
    """Analyze learned position embeddings"""
    print("\n=== Example 7: Position Embedding Analysis ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    
    pos_embeddings = model.pos_embedding.weight.detach()  # [n_ctx, d_model]
    print(f"Position embedding shape: {pos_embeddings.shape}")
    
    pos_embeddings_norm = pos_embeddings / pos_embeddings.norm(dim=-1, keepdim=True)
    similarity_matrix = torch.matmul(pos_embeddings_norm, pos_embeddings_norm.T)
    
    # Plot similarity matrix for first 50 positions
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix[:50, :50].cpu().numpy(), cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Position Embedding Similarity Matrix')
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.tight_layout()
    plt.savefig('position_similarity.png', dpi=150, bbox_inches='tight')
    print("Saved position embedding analysis to 'position_similarity.png'")
    
    # Check if embeddings show any periodic patterns
    distances_from_first = []
    for i in range(1, min(100, len(pos_embeddings))):
        dist = (pos_embeddings[i] - pos_embeddings[0]).norm().item()
        distances_from_first.append(dist)
    
    plt.figure(figsize=(10, 5))
    plt.plot(distances_from_first)
    plt.xlabel('Position')
    plt.ylabel('L2 Distance from Position 0')
    plt.title('Position Embedding Distances')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('position_distances.png', dpi=150, bbox_inches='tight')
    print("Saved position distance analysis to 'position_distances.png'")


def example_8_masked_prediction_analysis():
    """Analyze model predictions for masked tokens"""
    print("\n=== Example 8: Masked Token Prediction Analysis ===")
    
    config = RNATransformerConfig()
    model = RNATransformerWithHooks(config)
    tokenizer = RNATokenizer()
    
    sequence = "AUGCUAGCUAGC"
    print(f"Original sequence: {sequence}")
    
    for mask_pos in [3, 6, 9]:  # Mask at different positions
        seq_list = list(sequence)
        original_token = seq_list[mask_pos]
        
        masked_seq = seq_list.copy()
        masked_seq[mask_pos] = '<mask>'
        masked_input = ''.join(masked_seq).replace('<mask>', 'mask')
        
        # Tokenize (manually handle mask token)
        tokens = tokenizer.encode(sequence, add_special_tokens=True)
        tokens[mask_pos + 1] = tokenizer.mask_token_id  # +1 for CLS token
        input_ids = torch.tensor([tokens])
        
        output = model.forward(input_ids)
        logits = output['logits']
        
        masked_logits = logits[0, mask_pos + 1]  # +1 for CLS token
        probs = torch.softmax(masked_logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, k=5)
        
        print(f"\nMasked position {mask_pos} (original: {original_token}):")
        print("Top 5 predictions:")
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.id_to_token.get(idx.item(), '?')
            print(f"  {token}: {prob.item():.3f}")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_attention_analysis()
    example_3_neuron_analysis()
    example_4_residual_stream_analysis()
    example_5_hook_interventions()
    example_6_attention_pattern_visualization()
    example_7_position_embedding_analysis()
    example_8_masked_prediction_analysis()