"""
This file is a scratchpad for experimenting with the RhoFold model.
It is incomplete, and copied from a Kaggle notebook so there are some paths and such that are not correct.
"""
# %%
import subprocess
import sys
import glob

# TODO: install dependencies with uv instead of this deprecated workaraound from kaggle

# wheel_files = glob.glob("/kaggle/input/rhofold-dependencies-for-offline-use/*.whl")
# wheel_files = [w for w in wheel_files if 'numpy' not in w.lower()] # skip numpy to avoid compatibility issues

# if wheel_files:
#     subprocess.check_call([
#         sys.executable, 
#         "-m", 
#         "pip", 
#         "install", 
#         "--no-index",
#         "--find-links",
#         "/kaggle/input/rhofold-dependencies-for-offline-use/"
#     ] + wheel_files)
# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import tempfile
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from torch.utils.data import Dataset, DataLoader

sys.path.append('/kaggle/input/rhofold/pytorch/default/2/RhoFold-main')
from rhofold.config import rhofold_config
from rhofold.relax.relax import AmberRelaxation
from rhofold.rhofold import RhoFold
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.utils.alphabet import get_features
from rhofold.rhofold import RhoFold
# %%
train_seq = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_seq_v2 = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")
train_labels = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_labels_v2 = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")
val_seq = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
val_labels = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
test_seq = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
sample_submission = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")

# dfs = [
#     (train_seq, "train_seq_head.csv"),
#     (train_seq_v2, "train_seq_v2_head.csv"),
#     (train_labels, "train_labels_head.csv"),
#     (train_labels_v2, "train_labels_v2_head.csv"),
#     (val_seq, "val_seq_head.csv"),
#     (val_labels, "val_labels_head.csv"),
#     (test_seq, "test_seq_head.csv"),
#     (sample_submission, "sample_submission_head.csv")
# ]

# for df in dfs:
#     df[0].head().write_csv(df[1])
# %%
class ReactivityDataset(Dataset):
    def __init__(self, csv_path, max_len=206):
        # RhoFold token mapping
        self.base_to_token = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3,  # T->U
            'R': 4, 'Y': 5, 'K': 6, 'M': 7, 'S': 8, 'W': 9,
            'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14, '-': 15
        }
        self.pad_token = 15  # Using '-' as pad
        self.df = pl.read_csv(csv_path)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        seq = row['sequence'].upper().replace('T', 'U')
        
        tokens = [self.base_to_token.get(base, 14) for base in seq]  # N for unknown
        tokens = tokens[:self.max_len]
        tokens += [self.pad_token] * (self.max_len - len(tokens))
        
        reactivity = []
        mask = []
        
        for i in range(1, self.max_len + 1):
            val = row[f'reactivity_{i:04d}']
            
            if i <= len(seq) and val not in [None, 'NULL'] and not (isinstance(val, float) and pl.Series([val]).is_nan()[0]):
                reactivity.append(float(val))
                mask.append(True)
            else:
                reactivity.append(0.0)
                mask.append(False)
        
        return {
            'tokens': torch.LongTensor(tokens),
            'reactivity': torch.FloatTensor(reactivity),
            'mask': torch.BoolTensor(mask),
            'is_dms': row['experiment_type'] == 'DMS_MaP',
            'seq_len': len(seq)
        }

def collate_fn(batch):
    tokens = torch.stack([x['tokens'] for x in batch])
    mask = torch.stack([x['mask'] for x in batch])
    
    dms = torch.zeros_like(tokens, dtype=torch.float)
    shape = torch.zeros_like(tokens, dtype=torch.float)
    
    for i, item in enumerate(batch):
        if item['is_dms']:
            dms[i] = item['reactivity']
        else:
            shape[i] = item['reactivity']
    
    return {
        'tokens': tokens,
        'dms': dms,
        'shape': shape,
        'mask': mask
    }

reactivity_dataset = ReactivityDataset('/kaggle/input/stanford-ribonanza-training-data/reactivity_train_data_sn_filtered.csv')
reactivity_dataloader = DataLoader(reactivity_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# %%
class DualReactivityModel(nn.Module):
    """Stage 1: Pretrain RNA-FM on chemical reactivity data"""
    def __init__(self, rna_fm, freeze_early_layers=True):
        super().__init__()
        self.rna_fm = rna_fm
        
        if freeze_early_layers:
            for i, layer in enumerate(self.rna_fm.layers):
                if i < 6:  # Freeze first 6/12 layers
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Shared projection
        self.shared_proj = nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.LayerNorm(320)
        )
        
        # Task-specific heads
        self.dms_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )
        
        self.shape_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )
        
    def forward(self, tokens):
        # Get RNA-FM representations
        rna_out = self.rna_fm(tokens, repr_layers=[12])
        features = rna_out['representations'][12]
        
        # Shared features
        shared = self.shared_proj(features)
        
        # Predictions
        dms = self.dms_head(shared).squeeze(-1)
        shape = self.shape_head(shared).squeeze(-1)
        
        return {
            'dms': dms,
            'shape': shape,
            'features': features
        }
# %%
model = RhoFold(rhofold_config)
checkpoint = torch.load("/kaggle/input/rhofold/pytorch/default/2/RhoFold-main/pretrained/rhofold_pretrained_params.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])

# ...


def train_reactivity_stage(model, dataloader, epochs=10):
    """Stage 1: Train on reactivity data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            tokens = batch['tokens']
            dms_true = batch['dms']
            shape_true = batch['shape']
            
            outputs = model(tokens)
            
            # Multi-task loss
            loss = (F.mse_loss(outputs['dms'], dms_true) + 
                   F.mse_loss(outputs['shape'], shape_true))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ...
reactivity_model = DualReactivityModel()
train_reactivity_stage()
# %%
class StructureDataset(Dataset):
    def __init__(self, seq_csvs, labels_csvs, max_len=512):
        sequences = []
        labels = []
        
        for seq_csv, label_csv in zip(seq_csvs, labels_csvs):
            sequences.append(pl.read_csv(seq_csv))
            labels.append(pl.read_csv(label_csv))
        
        self.sequences = pl.concat(sequences)
        self.labels = pl.concat(labels)
        
        # Group by ID to get all residues for each structure
        self.label_groups = self.labels.group_by('ID').agg([
            pl.col('resid'),
            pl.col('resname'),
            pl.col('x_1'), pl.col('y_1'), pl.col('z_1')
        ])
        
        self.base_to_token = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3,
            'R': 4, 'Y': 5, 'K': 6, 'M': 7, 'S': 8, 'W': 9,
            'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14, '-': 15
        }
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_row = self.sequences.row(idx, named=True)
        target_id = seq_row['target_id']
        sequence = seq_row['sequence'].upper().replace('T', 'U')
        
        # Tokenize
        tokens = [self.base_to_token.get(base, 14) for base in sequence[:self.max_len]]
        tokens += [15] * (self.max_len - len(tokens))
        
        # Get all residues for this structure
        coords = np.zeros((self.max_len, 3), dtype=np.float32)
        
        structure_data = self.label_groups.filter(pl.col('ID') == target_id)
        if len(structure_data) > 0:
            structure_row = structure_data.row(0, named=True)
            
            # Sort by resid to ensure correct order
            resids = structure_row['resid']
            x_coords = structure_row['x_1']
            y_coords = structure_row['y_1'] 
            z_coords = structure_row['z_1']
            
            # Fill coordinate array
            for resid, x, y, z in zip(resids, x_coords, y_coords, z_coords):
                if 0 < resid <= self.max_len:
                    coords[resid-1] = [x, y, z]
        
        # Mask and distance matrix
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:min(len(sequence), self.max_len)] = True
        
        # Only compute distances for residues with coordinates
        dist_matrix = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        for i in range(self.max_len):
            for j in range(self.max_len):
                if coords[i].any() and coords[j].any():
                    dist_matrix[i,j] = np.linalg.norm(coords[i] - coords[j])
        
        return {
            'tokens': torch.LongTensor(tokens),
            'coords': torch.FloatTensor(coords),  # C1' atoms only
            'distance_matrix': torch.FloatTensor(dist_matrix),
            'mask': torch.BoolTensor(mask),
            'target_id': target_id,
            'seq_len': len(sequence)
        }
# %%
# Load model
model = RhoFold(rhofold_config)
checkpoint = torch.load("/kaggle/input/rhofold/pytorch/default/2/RhoFold-main/pretrained/rhofold_pretrained_params.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

# Test dimensions with dummy data
seq_len = 100
batch_size = 2

print("=== EXPLORING RHOFOLD DIMENSIONS ===\n")

# 1. RNA-FM Component
print("1. RNA-FM (Foundation Model)")
rna_fm = model.msa_embedder.rna_fm
print(f"Embedding vocab size: {rna_fm.embed_tokens.num_embeddings}")
print(f"Embedding dim: {rna_fm.embed_tokens.embedding_dim}")

# Create dummy RNA sequence tokens
dummy_tokens = torch.randint(0, 25, (batch_size, seq_len))
print(f"\nInput tokens shape: {dummy_tokens.shape}")

with torch.no_grad():
    # Test RNA-FM forward
    rna_fm_out = rna_fm(dummy_tokens, repr_layers=[12])
    print(f"RNA-FM output keys: {rna_fm_out.keys()}")
    print(f"RNA-FM logits shape: {rna_fm_out['logits'].shape}")
    print(f"RNA-FM representations shape: {rna_fm_out['representations'][12].shape}")

# 2. Check MSA embedder expectations
print("\n2. MSA Embedder Input Requirements")
print("MSA embedder expects:")
print("  - tokens: [batch, n_seq, seq_len] for MSA")
print("  - rna_fm_tokens: [batch, seq_len] for RNA-FM")

# 3. Test simplified custom model
print("\n=== SIMPLIFIED CUSTOM MODEL ===")

class SimplifiedRhoFold(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # Extract components
        self.rna_fm = original_model.msa_embedder.rna_fm
        self.structure_module = original_model.structure_module
        self.plddt_head = original_model.plddt_head
        
        # New layers for RNA-only input
        self.single_proj = nn.Linear(640, 384)
        self.pair_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # Reactivity heads
        self.dms_head = nn.Linear(640, 1)
        self.shape_head = nn.Linear(640, 1)
        
    def forward(self, tokens):
        # Get RNA-FM representations
        rna_out = self.rna_fm(tokens, repr_layers=[12])
        features = rna_out['representations'][12]  # [B, L, 640]
        
        # Reactivity predictions
        dms = self.dms_head(features).squeeze(-1)  # [B, L]
        shape = self.shape_head(features).squeeze(-1)  # [B, L]
        
        # Create single representation for structure module
        single_rep = self.single_proj(features)  # [B, L, 384]
        
        # Create pair representation
        B, L, D = features.shape
        feat_i = features.unsqueeze(2).expand(B, L, L, D)
        feat_j = features.unsqueeze(1).expand(B, L, L, D)
        pair_feat = torch.cat([feat_i, feat_j], dim=-1)
        pair_rep = self.pair_proj(pair_feat)  # [B, L, L, 128]
        
        # Structure module expects specific format
        # Let's check what it needs
        return {
            'rna_features': features,
            'single_rep': single_rep,
            'pair_rep': pair_rep,
            'dms': dms,
            'shape': shape,
            'shapes': {
                'rna_features': features.shape,
                'single_rep': single_rep.shape,
                'pair_rep': pair_rep.shape,
                'dms': dms.shape,
                'shape': shape.shape
            }
        }

# Test custom model
custom_model = SimplifiedRhoFold(model)
with torch.no_grad():
    outputs = custom_model(dummy_tokens)
    
print("\nCustom model output shapes:")
for k, v in outputs['shapes'].items():
    print(f"  {k}: {v}")

# 4. Inspect structure module input requirements
print("\n=== STRUCTURE MODULE REQUIREMENTS ===")
print("Checking structure module forward signature...")

# Let's trace through a minimal forward pass
print("\nStructure module expects:")
print("  - s: single representation [batch, n_res, c_s]")
print("  - z: pair representation [batch, n_res, n_res, c_z]")
print("  - backbone frames, rotation matrices, etc.")

# 5. Check coordinate outputs
print("\n=== COORDINATE OUTPUT FORMAT ===")
print("Structure module outputs:")
print("  - Backbone coordinates: [batch, n_res, 3] for C3'")
print("  - All atom coordinates: [batch, n_res, n_atoms, 3]")
print("  - pLDDT confidence: [batch, n_res]")

# 6. Training data requirements summary
print("\n=== TRAINING DATA REQUIREMENTS ===")
print("For RNA-FM fine-tuning on reactivity:")
print("  Input:")
print("    - RNA sequences as token ids [batch, seq_len]")
print("    - Tokens: 0=pad, 1=cls, 2-5=ACGU (likely)")
print("  Targets:")
print("    - DMS reactivity: [batch, seq_len] float")
print("    - SHAPE reactivity: [batch, seq_len] float")
print("\nFor structure prediction:")
print("  Input: Same token ids")
print("  Targets:")
print("    - 3D coordinates: [batch, seq_len, 3]")
print("    - Distance matrix: [batch, seq_len, seq_len]")
print("    - Secondary structure: [batch, seq_len] (paired/unpaired)")

# Quick token check
print("\n=== TOKEN MAPPING ===")
print("Testing token vocabulary...")
test_seq = "ACGU"
# You'll need to check RhoFold's alphabet utils for exact mapping
print("Check rhofold.utils.alphabet for exact token mapping")
# %%
class DualReactivityModel(nn.Module):
    """Stage 1: Pretrain RNA-FM on chemical reactivity data"""
    def __init__(self, rna_fm, freeze_early_layers=True):
        super().__init__()
        self.rna_fm = rna_fm
        
        if freeze_early_layers:
            for i, layer in enumerate(self.rna_fm.layers):
                if i < 6:  # Freeze first 6/12 layers
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Shared projection
        self.shared_proj = nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.LayerNorm(320)
        )
        
        # Task-specific heads
        self.dms_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )
        
        self.shape_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )
        
    def forward(self, tokens):
        # Get RNA-FM representations
        rna_out = self.rna_fm(tokens, repr_layers=[12])
        features = rna_out['representations'][12]
        
        # Shared features
        shared = self.shared_proj(features)
        
        # Predictions
        dms = self.dms_head(shared).squeeze(-1)
        shape = self.shape_head(shared).squeeze(-1)
        
        return {
            'dms': dms,
            'shape': shape,
            'features': features  # Keep for later use
        }


class ChemicallyInformedRhoFold(nn.Module):
    """Full model: RNA-FM + Chemical Knowledge + Structure Prediction"""
    def __init__(self, rhofold_model, pretrained_rna_fm=None):
        super().__init__()
        
        # Use pretrained RNA-FM if provided, otherwise use RhoFold's
        if pretrained_rna_fm is not None:
            self.rna_fm = pretrained_rna_fm
        else:
            self.rna_fm = rhofold_model.msa_embedder.rna_fm
            
        # Keep reactivity heads for auxiliary loss during structure training
        self.shared_proj = nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.LayerNorm(320)
        )
        
        self.dms_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )
        
        self.shape_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )
        
        # Projection layers for structure module
        self.to_single = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 384)
        )
        
        # Pair representation with chemical info integration
        self.to_pair = nn.Sequential(
            nn.Linear(1280 + 4, 512),  # +4 for chemical features
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Structure prediction modules from RhoFold
        self.structure_module = rhofold_model.structure_module
        self.plddt_head = rhofold_model.plddt_head
        self.dist_head = rhofold_model.dist_head
        
        # Optional: E2Eformer for refining representations
        self.use_e2eformer = False  # Set True to include
        if self.use_e2eformer:
            self.e2eformer = rhofold_model.e2eformer
        
    def create_pair_features(self, features, dms=None, shape=None):
        """Create pair features with optional chemical reactivity integration"""
        B, L, D = features.shape
        
        # Basic pair features from concatenation
        feat_i = features.unsqueeze(2).expand(B, L, L, D)
        feat_j = features.unsqueeze(1).expand(B, L, L, D)
        pair_feat = torch.cat([feat_i, feat_j], dim=-1)  # [B, L, L, 1280]
        
        # Add chemical reactivity information if available
        if dms is not None and shape is not None:
            # Create reactivity difference features
            dms_i = dms.unsqueeze(2).expand(B, L, L)
            dms_j = dms.unsqueeze(1).expand(B, L, L)
            shape_i = shape.unsqueeze(2).expand(B, L, L)
            shape_j = shape.unsqueeze(1).expand(B, L, L)
            
            # Reactivity differences can indicate pairing
            dms_diff = (dms_i - dms_j).unsqueeze(-1)
            dms_prod = (dms_i * dms_j).unsqueeze(-1)
            shape_diff = (shape_i - shape_j).unsqueeze(-1)
            shape_prod = (shape_i * shape_j).unsqueeze(-1)
            
            chem_features = torch.cat([
                dms_diff, dms_prod, shape_diff, shape_prod
            ], dim=-1)  # [B, L, L, 4]
            
            pair_feat = torch.cat([pair_feat, chem_features], dim=-1)
        
        return self.to_pair(pair_feat)
    
    def forward(self, tokens, return_reactivity=True):
        # RNA-FM encoding
        rna_out = self.rna_fm(tokens, repr_layers=[12])
        features = rna_out['representations'][12]  # [B, L, 640]
        
        # Predict reactivity (auxiliary task)
        shared = self.shared_proj(features)
        dms = self.dms_head(shared).squeeze(-1)
        shape = self.shape_head(shared).squeeze(-1)
        
        # Create structure module inputs
        single_rep = self.to_single(features)  # [B, L, 384]
        pair_rep = self.create_pair_features(
            features, dms.detach(), shape.detach()
        )  # [B, L, L, 128]
        
        # Optional: Run through E2Eformer for refinement
        if self.use_e2eformer:
            # Create dummy MSA (single sequence repeated)
            msa_rep = single_rep.unsqueeze(1).expand(-1, 5, -1, -1)
            msa_rep, pair_rep = self.e2eformer(msa_rep, pair_rep)
            single_rep = msa_rep[:, 0]  # Take first sequence
        
        # Structure prediction
        struct_outputs = self.structure_module(
            single_rep, 
            pair_rep,
            # Initial coordinates can be provided here if available
        )
        
        # Predict confidence
        plddt = self.plddt_head(single_rep)
        
        # Distance prediction
        dist = self.dist_head(pair_rep)
        
        outputs = {
            'coordinates': struct_outputs['positions'],  
            'plddt': plddt,
            'distogram': dist,
        }
        
        if return_reactivity:
            outputs.update({
                'dms': dms,
                'shape': shape
            })
            
        return outputs


# Usage Example:
def create_full_model(rhofold_checkpoint_path, reactivity_checkpoint_path=None):
    """Create the full model with optional pretrained reactivity weights"""
    
    # Load base RhoFold
    from rhofold.config import rhofold_config
    from rhofold.rhofold import RhoFold
    
    rhofold = RhoFold(rhofold_config)
    checkpoint = torch.load(rhofold_checkpoint_path, map_location='cpu')
    rhofold.load_state_dict(checkpoint['model'])
    
    # Create model
    if reactivity_checkpoint_path:
        # Load pretrained reactivity model
        reactivity_model = DualReactivityModel(rhofold.msa_embedder.rna_fm)
        reactivity_checkpoint = torch.load(reactivity_checkpoint_path, map_location='cpu')
        reactivity_model.load_state_dict(reactivity_checkpoint)
        
        # Use pretrained RNA-FM in full model
        model = ChemicallyInformedRhoFold(
            rhofold, 
            pretrained_rna_fm=reactivity_model.rna_fm
        )
        
        # Also load reactivity heads
        model.shared_proj.load_state_dict(reactivity_model.shared_proj.state_dict())
        model.dms_head.load_state_dict(reactivity_model.dms_head.state_dict())
        model.shape_head.load_state_dict(reactivity_model.shape_head.state_dict())
    else:
        # Start from base RhoFold
        model = ChemicallyInformedRhoFold(rhofold)
    
    return model


# Training Example:
def train_reactivity_stage(model, dataloader, epochs=10):
    """Stage 1: Train on reactivity data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            tokens = batch['tokens']
            dms_true = batch['dms']
            shape_true = batch['shape']
            
            outputs = model(tokens)
            
            # Multi-task loss
            loss = (F.mse_loss(outputs['dms'], dms_true) + 
                   F.mse_loss(outputs['shape'], shape_true))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_structure_stage(model, dataloader, epochs=10):
    """Stage 2: Train on structure data with auxiliary reactivity loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            tokens = batch['tokens']
            coords_true = batch['coordinates']
            
            outputs = model(tokens)
            
            # Structure loss
            coord_loss = F.mse_loss(outputs['coordinates'], coords_true)
            
            # Auxiliary reactivity loss if available
            aux_loss = 0
            if 'dms' in batch:
                aux_loss += 0.1 * F.mse_loss(outputs['dms'], batch['dms'])
                aux_loss += 0.1 * F.mse_loss(outputs['shape'], batch['shape'])
            
            loss = coord_loss + aux_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
# %%
