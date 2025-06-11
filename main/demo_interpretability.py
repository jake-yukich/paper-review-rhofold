"""
Demo script for RNA transformer interpretability analysis

This script demonstrates the main interpretability capabilities:
1. Loading RhoFold weights into hooked transformer
2. Analyzing attention patterns for base pairing
3. Finding specialized attention heads
4. Circuit analysis with activation patching
5. Neuron analysis and feature discovery
6. Linear probing experiments

Run this script to get a quick overview of what the interpretability tools can do.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from rna_transformer_interpretability import RNATokenizer
from load_rhofold_weights import create_hooked_model_from_rhofold
from rna_interpretability_analysis import (
    AttentionPatternAnalyzer,
    CircuitAnalyzer, 
    NeuronAnalyzer,
    ProbingAnalyzer,
    RNASequenceData,
    create_demo_sequences,
    parse_dot_bracket
)


def run_attention_analysis(analyzer: AttentionPatternAnalyzer, sequences: list, demo_sequences: list):
    """Run comprehensive attention pattern analysis"""
    print("=== ATTENTION PATTERN ANALYSIS ===\n")
    
    print("1. Finding base pairing attention heads...")
    pairing_heads = analyzer.find_base_pairing_heads(demo_sequences)
    top_pairing = sorted(pairing_heads.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top base pairing heads:")
    for (layer, head), score in top_pairing:
        print(f"   Layer {layer:2d}, Head {head:2d}: {score:.4f}")
    
    if top_pairing:
        best_layer, best_head = top_pairing[0][0]
        print(f"\nBest base pairing head: Layer {best_layer}, Head {best_head}")
    
    print("\n2. Analyzing attention specialization (entropy)...")
    entropies = analyzer.attention_entropy_analysis(sequences)
    
    sorted_entropies = sorted(entropies.items(), key=lambda x: x[1])
    most_specialized = sorted_entropies[:5]
    least_specialized = sorted_entropies[-5:]
    
    print("Most specialized heads (low entropy - attend to few positions):")
    for (layer, head), entropy in most_specialized:
        print(f"   Layer {layer:2d}, Head {head:2d}: {entropy:.4f}")
    
    print("\nLeast specialized heads (high entropy - attend broadly):")
    for (layer, head), entropy in least_specialized:
        print(f"   Layer {layer:2d}, Head {head:2d}: {entropy:.4f}")
    
    if top_pairing and demo_sequences:
        print(f"\n3. Visualizing best base pairing head...")
        best_layer, best_head = top_pairing[0][0]
        test_rna = demo_sequences[0]
        
        try:
            analyzer.visualize_attention_head(
                test_rna.sequence,
                best_layer,
                best_head,
                test_rna.secondary_structure,
                save_path="attention_visualization.png"
            )
            print("   Saved attention visualization to attention_visualization.png")
        except Exception as e:
            print(f"   Visualization failed: {e}")
    
    return top_pairing


def run_circuit_analysis(analyzer: CircuitAnalyzer, sequences: list):
    """Run circuit analysis to understand information flow"""
    print("\n=== CIRCUIT ANALYSIS ===\n")
    
    if len(sequences) < 2:
        print("Skipping circuit analysis - need at least 2 sequences")
        return
    
    clean_seq = sequences[0]
    corrupted_seq = sequences[1]
    
    print(f"Clean sequence:     {clean_seq[:50]}...")
    print(f"Corrupted sequence: {corrupted_seq[:50]}...")
    
    def metric_fn(output):
        if 'logits' in output:
            logits = output['logits']
            if logits.shape[1] > 10:
                return logits[0, 10, :].max() - logits[0, 10, :].mean()
        return torch.tensor(0.0)
    
    hook_points = [
        "block0.attn_out", "block3.attn_out", "block6.attn_out", "block9.attn_out",
        "block0.mlp_out", "block3.mlp_out", "block6.mlp_out", "block9.mlp_out"
    ]
    
    print("1. Running activation patching experiment...")
    try:
        importance_scores = analyzer.activation_patching(
            clean_seq, corrupted_seq, hook_points, metric_fn
        )
        
        print("Component importance scores:")
        sorted_importance = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        for component, score in sorted_importance:
            print(f"   {component:15s}: {score:7.4f}")
            
    except Exception as e:
        print(f"   Activation patching failed: {e}")
    
    print("\n2. Analyzing residual stream at position 10...")
    try:
        decomp = analyzer.residual_stream_decomposition(clean_seq, 10)
        
        print("Residual stream contributions:")
        for component, vector in decomp.items():
            if isinstance(vector, torch.Tensor):
                norm = vector.norm().item()
                print(f"   {component:12s}: ||v|| = {norm:.4f}")
                
    except Exception as e:
        print(f"   Residual stream analysis failed: {e}")


def run_neuron_analysis(analyzer: NeuronAnalyzer, sequences: list):
    """Analyze individual neuron behavior"""
    print("\n=== NEURON ANALYSIS ===\n")
    
    test_neurons = [
        (3, 100), (3, 500), (3, 1000),  # Layer 3
        (6, 100), (6, 500), (6, 1000),  # Layer 6  
        (9, 100), (9, 500), (9, 1000),  # Layer 9
    ]
    
    print("1. Finding sequences that maximally activate specific neurons...")
    
    for layer_idx, neuron_idx in test_neurons[:3]:  # Just test first 3
        print(f"\nLayer {layer_idx}, Neuron {neuron_idx}:")
        
        try:
            top_sequences = analyzer.find_top_activating_sequences(
                sequences, layer_idx, neuron_idx, top_k=3
            )
            
            if top_sequences:
                for i, (seq, activation, pos) in enumerate(top_sequences):
                    context = seq[max(0, pos-5):pos+6] if pos < len(seq) else "N/A"
                    print(f"   #{i+1}: activation={activation:.3f}, pos={pos}, context='{context}'")
            else:
                print("   No activations found")
                
        except Exception as e:
            print(f"   Analysis failed: {e}")
    
    print(f"\n2. Analyzing features for Layer 6, Neuron 100...")
    try:
        features = analyzer.neuron_feature_analysis(sequences[:10], 6, 100)
        
        print("Base preferences:")
        if 'base_preferences' in features:
            base_prefs = features['base_preferences']
            for base in ['A', 'C', 'G', 'U']:
                if base in base_prefs.index:
                    mean_act = base_prefs.loc[base, 'mean']
                    count = base_prefs.loc[base, 'count']
                    print(f"   {base}: {mean_act:.4f} (n={count})")
        
        print("\nTop 3-mer contexts:")
        if 'context_3_preferences' in features:
            top_contexts = features['context_3_preferences'].head(5)
            for context, row in top_contexts.iterrows():
                print(f"   '{context}': {row['mean']:.4f} (n={row['count']})")
                
    except Exception as e:
        print(f"   Feature analysis failed: {e}")


def run_probing_analysis(analyzer: ProbingAnalyzer, sequences: list):
    """Run linear probing experiments"""
    print("\n=== LINEAR PROBING ANALYSIS ===\n")
    
    print("Training linear probes to predict nucleotide identity...")
    print("This shows how well each layer encodes base information.\n")
    
    test_layers = [0, 2, 4, 6, 8, 10, 11]
    layer_accuracies = []
    
    for layer_idx in test_layers:
        try:
            results = analyzer.train_base_probe(sequences, layer_idx)
            accuracy = results['test_accuracy']
            layer_accuracies.append((layer_idx, accuracy))
            print(f"Layer {layer_idx:2d}: {accuracy:.4f} accuracy")
            
        except Exception as e:
            print(f"Layer {layer_idx:2d}: Failed - {e}")
    
    if layer_accuracies:
        best_layer, best_acc = max(layer_accuracies, key=lambda x: x[1])
        print(f"\nBest layer for base identity: Layer {best_layer} ({best_acc:.4f} accuracy)")
        
        if len(layer_accuracies) > 3:
            try:
                layers, accs = zip(*layer_accuracies)
                plt.figure(figsize=(10, 6))
                plt.plot(layers, accs, 'bo-', linewidth=2, markersize=8)
                plt.xlabel('Layer')
                plt.ylabel('Probe Accuracy')
                plt.title('Base Identity Probing Accuracy Across Layers')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('probing_accuracy.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("Saved probing accuracy plot to probing_accuracy.png")
            except Exception as e:
                print(f"Plotting failed: {e}")


def generate_test_sequences(n_sequences: int = 20) -> list:
    """Generate diverse RNA sequences for testing"""
    sequences = []
    
    demo_seqs = create_demo_sequences()
    sequences.extend([rna.sequence for rna in demo_seqs])
    
    bases = ['A', 'C', 'G', 'U']
    np.random.seed(42)
    
    for _ in range(n_sequences - len(demo_seqs)):
        length = np.random.randint(20, 100)
        seq = ''.join(np.random.choice(bases, length))
        sequences.append(seq)
    
    return sequences


def main():
    parser = argparse.ArgumentParser(description="RNA Transformer Interpretability Demo")
    parser.add_argument(
        '--checkpoint',
        type=str, 
        default='rhofold-main/pretrained/rhofold_pretrained_params.pt',
        help='Path to RhoFold checkpoint'
    )
    parser.add_argument(
        '--n_sequences',
        type=int,
        default=20,
        help='Number of test sequences to generate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run on'
    )
    parser.add_argument(
        '--skip_attention',
        action='store_true',
        help='Skip attention analysis'
    )
    parser.add_argument(
        '--skip_circuits',
        action='store_true', 
        help='Skip circuit analysis'
    )
    parser.add_argument(
        '--skip_neurons',
        action='store_true',
        help='Skip neuron analysis'
    )
    parser.add_argument(
        '--skip_probing',
        action='store_true',
        help='Skip probing analysis'
    )
    
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please provide a valid path to the RhoFold checkpoint")
        return
    
    device = torch.device(args.device)
    
    print("RNA Transformer Interpretability Demo")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Test sequences: {args.n_sequences}")
    print()
    
    print("Loading RhoFold model and creating hooked transformer...")
    try:
        model = create_hooked_model_from_rhofold(args.checkpoint, device)
        tokenizer = RNATokenizer()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print(f"\nGenerating {args.n_sequences} test sequences...")
    sequences = generate_test_sequences(args.n_sequences)
    demo_sequences = create_demo_sequences()
    print(f"✓ Generated {len(sequences)} sequences")
    
    attn_analyzer = AttentionPatternAnalyzer(model, tokenizer)
    circuit_analyzer = CircuitAnalyzer(model, tokenizer)
    neuron_analyzer = NeuronAnalyzer(model, tokenizer)
    probe_analyzer = ProbingAnalyzer(model, tokenizer)
    
    try:
        if not args.skip_attention:
            top_pairing_heads = run_attention_analysis(attn_analyzer, sequences, demo_sequences)
        
        if not args.skip_circuits:
            run_circuit_analysis(circuit_analyzer, sequences)
        
        if not args.skip_neurons:
            run_neuron_analysis(neuron_analyzer, sequences)
        
        if not args.skip_probing:
            run_probing_analysis(probe_analyzer, sequences)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()