"""
RNA Transformer Interpretability Analysis

This module provides analysis tools for studying mechanistic interpretability
of RNA transformers, particularly focusing on:
1. Attention pattern analysis (base pairing, structural motifs)
2. Circuit discovery (information flow between layers)
3. Neuron activation analysis (what features activate specific neurons)
4. Probing analysis (what information is encoded where)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from pathlib import Path

from rna_transformer_interpretability import RNATransformerWithHooks, RNATokenizer
from load_rhofold_weights import create_hooked_model_from_rhofold


@dataclass
class RNASequenceData:
    """Container for RNA sequence and its structural annotations"""
    sequence: str
    secondary_structure: Optional[str] = None  # Dot-bracket notation
    base_pairs: Optional[List[Tuple[int, int]]] = None
    reactivity: Optional[Dict[str, List[float]]] = None  # DMS, SHAPE, etc.
    name: str = ""


class AttentionPatternAnalyzer:
    """Analyze attention patterns in RNA transformers"""
    
    def __init__(self, model: RNATransformerWithHooks, tokenizer: RNATokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def get_attention_patterns(
        self, 
        sequence: str, 
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get attention patterns for all specified layers
        
        Args:
            sequence: RNA sequence
            layers: List of layer indices (default: all layers)
            
        Returns:
            Dict mapping layer_idx -> attention_weights [heads, seq_len, seq_len]
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))
        
        tokens = torch.tensor([self.tokenizer.encode(sequence)], dtype=torch.long)
        
        with torch.no_grad():
            self.model.clear_hooks()
            _ = self.model(tokens, cache_activations=True)
            
            attention_patterns = {}
            for layer_idx in layers:
                attn = self.model.get_attention_patterns(layer_idx)
                if attn is not None:
                    attention_patterns[layer_idx] = attn[0]  # Remove batch dim
            
        return attention_patterns
    
    def find_base_pairing_heads(
        self,
        sequences_with_structure: List[RNASequenceData],
        layers: Optional[List[int]] = None,
        threshold: float = 0.1
    ) -> Dict[Tuple[int, int], float]:
        """
        Find attention heads that attend to base-paired positions
        
        Args:
            sequences_with_structure: List of RNA sequences with known base pairs
            layers: Layers to analyze
            threshold: Minimum attention weight to consider
            
        Returns:
            Dict mapping (layer, head) -> base_pairing_score
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))
        
        head_scores = defaultdict(list)
        
        for rna_data in sequences_with_structure:
            if rna_data.base_pairs is None:
                continue
                
            attention_patterns = self.get_attention_patterns(rna_data.sequence, layers)
            
            for layer_idx, attn in attention_patterns.items():
                n_heads, seq_len, _ = attn.shape
                
                for head_idx in range(n_heads):
                    head_attn = attn[head_idx]
                    
                    pairing_score = 0.0
                    total_pairs = 0
                    
                    for i, j in rna_data.base_pairs:
                        if i < seq_len and j < seq_len:
                            score = (head_attn[i, j] + head_attn[j, i]) / 2
                            if score > threshold:
                                pairing_score += score
                            total_pairs += 1
                    
                    if total_pairs > 0:
                        head_scores[(layer_idx, head_idx)].append(pairing_score / total_pairs)
        
        final_scores = {}
        for key, scores in head_scores.items():
            final_scores[key] = np.mean(scores) if scores else 0.0
        
        return final_scores
    
    def visualize_attention_head(
        self,
        sequence: str,
        layer_idx: int,
        head_idx: int,
        secondary_structure: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Visualize a specific attention head's pattern"""
        attention_patterns = self.get_attention_patterns(sequence, [layer_idx])
        
        if layer_idx not in attention_patterns:
            raise ValueError(f"No attention pattern found for layer {layer_idx}")
        
        attn = attention_patterns[layer_idx][head_idx].cpu().numpy()
        tokens = [t for t in sequence]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        
        plt.colorbar(im, ax=ax)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        
        if secondary_structure:
            self._add_structure_overlay(ax, secondary_structure)
        
        ax.set_title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}")
        ax.set_xlabel("Keys (Attended To)")
        ax.set_ylabel("Queries (Attending From)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _add_structure_overlay(self, ax, secondary_structure: str):
        """Add secondary structure overlay to attention plot"""
        stack = []
        pairs = []
        
        for i, char in enumerate(secondary_structure):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                pairs.append((j, i))
        
        for i, j in pairs:
            ax.plot([i, j], [j, i], 'r-', alpha=0.7, linewidth=2)
            ax.plot([j, i], [i, j], 'r-', alpha=0.7, linewidth=2)
    
    def attention_entropy_analysis(
        self,
        sequences: List[str],
        layers: Optional[List[int]] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate attention entropy for each head to find specialized vs general heads
        
        Lower entropy = more specialized (attends to few positions)
        Higher entropy = more general (attends broadly)
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))
        
        head_entropies = defaultdict(list)
        
        for sequence in sequences:
            attention_patterns = self.get_attention_patterns(sequence, layers)
            
            for layer_idx, attn in attention_patterns.items():
                n_heads, seq_len, _ = attn.shape
                
                for head_idx in range(n_heads):
                    head_attn = attn[head_idx]
                    
                    entropies = []
                    for i in range(seq_len):
                        probs = head_attn[i]
                        probs = probs / probs.sum()  # Normalize
                        entropy = -(probs * torch.log(probs + 1e-8)).sum()
                        entropies.append(entropy.item())
                    
                    avg_entropy = np.mean(entropies)
                    head_entropies[(layer_idx, head_idx)].append(avg_entropy)
        
        final_entropies = {}
        for key, entropies in head_entropies.items():
            final_entropies[key] = np.mean(entropies)
        
        return final_entropies


class CircuitAnalyzer:
    """Analyze information flow and circuits in RNA transformers"""
    
    def __init__(self, model: RNATransformerWithHooks, tokenizer: RNATokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def activation_patching(
        self,
        clean_sequence: str,
        corrupted_sequence: str,
        hook_points: List[str],
        metric_fn: callable
    ) -> Dict[str, float]:
        """
        Perform activation patching to measure component importance
        
        Args:
            clean_sequence: Original sequence
            corrupted_sequence: Corrupted version
            hook_points: Where to patch activations
            metric_fn: Function to compute metric on outputs
            
        Returns:
            Dict mapping hook_point -> importance score
        """
        clean_tokens = torch.tensor([self.tokenizer.encode(clean_sequence)])
        corrupted_tokens = torch.tensor([self.tokenizer.encode(corrupted_sequence)])
        
        with torch.no_grad():
            clean_output = self.model(clean_tokens)
            corrupted_output = self.model(corrupted_tokens)
            
            clean_metric = metric_fn(clean_output)
            corrupted_metric = metric_fn(corrupted_output)
        
        importance_scores = {}
        
        for hook_point in hook_points:
            self.model.clear_hooks()
            clean_result = self.model.run_with_hooks(
                clean_tokens, [hook_point]
            )
            clean_activation = clean_result['hooked_values'][hook_point]
            
            # Patch corrupted run with clean activation
            def patch_hook(x):
                return clean_activation
            
            corrupted_result = self.model.run_with_hooks(
                corrupted_tokens, [hook_point], hook_fn=patch_hook
            )
            
            patched_metric = metric_fn(corrupted_result)
            
            importance = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric + 1e-8)
            importance_scores[hook_point] = importance.item()
        
        return importance_scores
    
    def residual_stream_decomposition(
        self,
        sequence: str,
        position_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose the residual stream at a specific position
        
        Args:
            sequence: RNA sequence
            position_idx: Position to analyze
            
        Returns:
            Dict with residual stream components
        """
        tokens = torch.tensor([self.tokenizer.encode(sequence)])
        
        with torch.no_grad():
            output = self.model(tokens, cache_activations=True, return_hidden_states=True)
            hidden_states = output['hidden_states']
        
        embedding_contrib = hidden_states[0][0, position_idx]
        
        layer_contribs = {}
        for i in range(len(hidden_states) - 1):
            layer_contrib = hidden_states[i+1][0, position_idx] - hidden_states[i][0, position_idx]
            layer_contribs[f'layer_{i}'] = layer_contrib
        
        return {
            'embedding': embedding_contrib,
            **layer_contribs,
            'final': hidden_states[-1][0, position_idx]
        }


class NeuronAnalyzer:
    """Analyze individual neuron activations and their semantic meaning"""
    
    def __init__(self, model: RNATransformerWithHooks, tokenizer: RNATokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def find_top_activating_sequences(
        self,
        sequences: List[str],
        layer_idx: int,
        neuron_idx: int,
        top_k: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Find sequences that most activate a specific neuron
        
        Returns:
            List of (sequence, max_activation, position) tuples
        """
        activations = []
        
        for sequence in sequences:
            tokens = torch.tensor([self.tokenizer.encode(sequence)])
            
            with torch.no_grad():
                self.model.clear_hooks()
                _ = self.model(tokens, cache_activations=True)
                
                mlp_acts = self.model.get_mlp_activations(layer_idx, "post_act")
                if mlp_acts is not None:
                    neuron_acts = mlp_acts[0, :, neuron_idx]  # [seq_len]
                    max_act = neuron_acts.max()
                    max_pos = neuron_acts.argmax()
                    
                    activations.append((sequence, max_act.item(), max_pos.item()))
        
        activations.sort(key=lambda x: x[1], reverse=True)
        return activations[:top_k]
    
    def neuron_feature_analysis(
        self,
        sequences: List[str],
        layer_idx: int,
        neuron_idx: int
    ) -> Dict[str, Any]:
        """
        Analyze what features correlate with neuron activation
        """
        sequence_data = []
        
        for sequence in sequences:
            tokens = torch.tensor([self.tokenizer.encode(sequence)])
            
            with torch.no_grad():
                self.model.clear_hooks()
                _ = self.model(tokens, cache_activations=True)
                
                mlp_acts = self.model.get_mlp_activations(layer_idx, "post_act")
                if mlp_acts is not None:
                    neuron_acts = mlp_acts[0, :, neuron_idx]
                    
                    for pos, activation in enumerate(neuron_acts):
                        if pos < len(sequence):
                            sequence_data.append({
                                'sequence': sequence,
                                'position': pos,
                                'base': sequence[pos],
                                'activation': activation.item(),
                                'context_3': sequence[max(0, pos-1):pos+2],  # 3-mer context
                                'context_5': sequence[max(0, pos-2):pos+3],  # 5-mer context
                            })
        
        df = pd.DataFrame(sequence_data)
        
        base_stats = df.groupby('base')['activation'].agg(['mean', 'std', 'count'])
        
        context_3_stats = df.groupby('context_3')['activation'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        context_5_stats = df.groupby('context_5')['activation'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        return {
            'base_preferences': base_stats,
            'context_3_preferences': context_3_stats.head(20),
            'context_5_preferences': context_5_stats.head(20),
            'activation_distribution': df['activation'].describe()
        }


class ProbingAnalyzer:
    """Train linear probes to understand what information is encoded in representations"""
    
    def __init__(self, model: RNATransformerWithHooks, tokenizer: RNATokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def extract_representations(
        self,
        sequences: List[str],
        layer_idx: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract representations from a specific layer"""
        all_reprs = []
        all_sequences = []
        
        for sequence in sequences:
            tokens = torch.tensor([self.tokenizer.encode(sequence)])
            
            with torch.no_grad():
                output = self.model(tokens, return_hidden_states=True)
                hidden_states = output['hidden_states']
                
                if layer_idx < len(hidden_states):
                    layer_repr = hidden_states[layer_idx][0]  # Remove batch dim
                    
                    # Only keep representations for actual sequence (not special tokens)
                    seq_len = len(sequence)
                    for pos in range(min(seq_len, layer_repr.shape[0])):
                        all_reprs.append(layer_repr[pos])
                        all_sequences.append(sequence)
        
        return torch.stack(all_reprs), all_sequences
    
    def train_base_probe(
        self,
        sequences: List[str],
        layer_idx: int,
        test_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train probe to predict nucleotide identity from representations
        """
        reprs, seq_list = self.extract_representations(sequences, layer_idx)
        
        labels = []
        for i, sequence in enumerate(seq_list):
            pos = i % len(sequence)
            if pos < len(sequence):
                base = sequence[pos]
                label = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}.get(base, 4)
                labels.append(label)
        
        labels = torch.tensor(labels)
        
        n_train = int(len(reprs) * (1 - test_split))
        train_reprs, test_reprs = reprs[:n_train], reprs[n_train:]
        train_labels, test_labels = labels[:n_train], labels[n_train:]
        
        probe = nn.Linear(reprs.shape[1], 5)  # 5 classes: A, C, G, U, other
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
        
        for epoch in range(100):
            logits = probe(train_reprs)
            loss = F.cross_entropy(logits, train_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            train_logits = probe(train_reprs)
            test_logits = probe(test_reprs)
            
            train_acc = (train_logits.argmax(dim=1) == train_labels).float().mean()
            test_acc = (test_logits.argmax(dim=1) == test_labels).float().mean()
        
        return {
            'train_accuracy': train_acc.item(),
            'test_accuracy': test_acc.item(),
            'layer': layer_idx
        }


def create_demo_sequences() -> List[RNASequenceData]:
    """Create some demo RNA sequences with known structures for testing"""
    sequences = [
        RNASequenceData(
            sequence="GGCCAACGUGGUCCCAUGCCGAACAGUGAUGGCCGACAAGGUCGCAAA",
            secondary_structure="((((...((((....))))....((((....))))....))))",
            name="stem_loop_1"
        ),
        RNASequenceData(
            sequence="GCGCAACCGUGGAAAGGCCGCUCACGCUGACCAACGCUUACGCAGGC",
            secondary_structure="((((....((((....))))....((((....))))....))))",
            name="three_way_junction"
        ),
        RNASequenceData(
            sequence="GGGCCCUUAGGCCCAACCCGGGAAACCCUUUGGCCCCAACCCCCC",
            secondary_structure="((((((((....))))..((((....))))....))))",
            name="pseudoknot_like"
        )
    ]
    
    for rna_data in sequences:
        if rna_data.secondary_structure:
            rna_data.base_pairs = parse_dot_bracket(rna_data.secondary_structure)
    
    return sequences


def parse_dot_bracket(structure: str) -> List[Tuple[int, int]]:
    """Parse dot-bracket notation to extract base pairs"""
    stack = []
    pairs = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs.append((j, i))
    
    return pairs


if __name__ == "__main__":
    print("Loading RhoFold model...")
    model = create_hooked_model_from_rhofold(
        "rhofold-main/pretrained/rhofold_pretrained_params.pt"
    )
    tokenizer = RNATokenizer()
    
    attn_analyzer = AttentionPatternAnalyzer(model, tokenizer)
    circuit_analyzer = CircuitAnalyzer(model, tokenizer)
    neuron_analyzer = NeuronAnalyzer(model, tokenizer)
    probe_analyzer = ProbingAnalyzer(model, tokenizer)
    
    demo_sequences = create_demo_sequences()
    sequences = [rna.sequence for rna in demo_sequences]
    
    print("\n=== Attention Pattern Analysis ===")
    print("Finding base pairing heads...")
    pairing_heads = attn_analyzer.find_base_pairing_heads(demo_sequences)
    top_pairing_heads = sorted(pairing_heads.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("Top base pairing heads:")
    for (layer, head), score in top_pairing_heads:
        print(f"  Layer {layer}, Head {head}: {score:.4f}")
    
    print("\nCalculating attention entropy...")
    entropies = attn_analyzer.attention_entropy_analysis(sequences)
    
    sorted_entropies = sorted(entropies.items(), key=lambda x: x[1])
    most_specialized = sorted_entropies[:3]
    least_specialized = sorted_entropies[-3:]
    
    print("Most specialized heads (low entropy):")
    for (layer, head), entropy in most_specialized:
        print(f"  Layer {layer}, Head {head}: {entropy:.4f}")
    
    print("Least specialized heads (high entropy):")
    for (layer, head), entropy in least_specialized:
        print(f"  Layer {layer}, Head {head}: {entropy:.4f}")
    
    print("\n=== Neuron Analysis ===")
    
    for layer_idx in [3, 6, 9]:
        print(f"\nAnalyzing Layer {layer_idx}, Neuron 100...")
        
        top_sequences = neuron_analyzer.find_top_activating_sequences(
            sequences, layer_idx, 100, top_k=3
        )
        
        print("Top activating sequences:")
        for seq, activation, pos in top_sequences:
            print(f"  Activation: {activation:.4f}, Position: {pos}, Sequence: {seq[:30]}...")
    
    print("\n=== Probing Analysis ===")
    
    for layer_idx in [0, 3, 6, 9, 11]:
        results = probe_analyzer.train_base_probe(sequences, layer_idx)
        print(f"Layer {layer_idx} base probe accuracy: {results['test_accuracy']:.4f}")