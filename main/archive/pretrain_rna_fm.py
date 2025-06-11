"""
Pretrain RNA-FM model on chemical reactivity data (DMS and 2A3).
This script fine-tunes the RNA-FM component of RhoFold on the Ribonanza dataset
to learn chemical reactivity patterns explicitly and RNA structure implicitly.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
import polars as pl
from tqdm import tqdm
import wandb

sys.path.append(str(Path(__file__).parent.parent / "rhofold-main"))
from rhofold.config import rhofold_config
from rhofold.rhofold import RhoFold
from rhofold.model.rna_fm import model as rna_fm_module


class ReactivityDataset(Dataset):
    """Dataset for RNA reactivity data from Ribonanza."""
    
    def __init__(
        self, 
        csv_path: str, 
        max_len: int = 206,
        cache_dir: Optional[str] = None,
        experiment_filter: Optional[str] = None
    ):
        """
        Args:
            csv_path: Path to reactivity CSV file
            max_len: Maximum sequence length (206 for Ribonanza)
            cache_dir: Directory for caching processed data
            experiment_filter: Filter for specific experiment type ('DMS_MaP' or '2A3_MaP')
        """
        self.max_len = max_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self.base_to_token = {
            'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3,
            'R': 4, 'Y': 5, 'K': 6, 'M': 7, 'S': 8, 'W': 9,
            'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14, '-': 15
        }
        self.pad_token = 15
        
        print(f"Loading data from {csv_path}...")
        self.df = pl.read_csv(csv_path)
        
        if experiment_filter:
            self.df = self.df.filter(pl.col('experiment_type') == experiment_filter)
            print(f"Filtered to {len(self.df)} {experiment_filter} sequences")
        
        self.experiment_types = self.df['experiment_type'].unique().to_list()
        print(f"Experiment types: {self.experiment_types}")
        print(f"Total sequences: {len(self.df)}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.row(idx, named=True)
        
        seq = row['sequence'].upper().replace('T', 'U')
        seq_len = len(seq)
        
        tokens = [self.base_to_token.get(base, 14) for base in seq]  # 14 (N) for unknown
        tokens = tokens[:self.max_len]
        padding_len = self.max_len - len(tokens)
        tokens += [self.pad_token] * padding_len
        
        reactivity = []
        mask = []
        
        for i in range(1, self.max_len + 1):
            col_name = f'reactivity_{i:04d}'
            
            if i <= seq_len and col_name in row:
                val = row[col_name]
                # check for valid values
                if val is not None and val != 'NULL' and not (isinstance(val, float) and np.isnan(val)):
                    reactivity.append(float(val))
                    mask.append(True)
                else:
                    reactivity.append(0.0)
                    mask.append(False)
            else:
                reactivity.append(0.0)
                mask.append(False)
        
        return {
            'tokens': torch.LongTensor(tokens),
            'reactivity': torch.FloatTensor(reactivity),
            'mask': torch.BoolTensor(mask),
            'is_dms': row['experiment_type'] == 'DMS_MaP',
            'is_2a3': row['experiment_type'] == '2A3_MaP',
            'seq_len': seq_len,
            'sequence_id': row['sequence_id']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    tokens = torch.stack([x['tokens'] for x in batch])
    mask = torch.stack([x['mask'] for x in batch])
    seq_lens = torch.LongTensor([x['seq_len'] for x in batch])
    
    batch_size, max_len = tokens.shape
    dms = torch.zeros((batch_size, max_len), dtype=torch.float32)
    shape = torch.zeros((batch_size, max_len), dtype=torch.float32)
    dms_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    shape_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    for i, item in enumerate(batch):
        if item['is_dms']:
            dms[i] = item['reactivity']
            dms_mask[i] = item['mask']
        else:  # 2A3
            shape[i] = item['reactivity']
            shape_mask[i] = item['mask']
    
    return {
        'tokens': tokens,
        'dms': dms,
        'shape': shape,
        'dms_mask': dms_mask,
        'shape_mask': shape_mask,
        'mask': mask,  # Combined mask
        'seq_lens': seq_lens
    }


class DualReactivityModel(nn.Module):
    """Model for predicting both DMS and 2A3 reactivity using RNA-FM."""
    
    def __init__(
        self, 
        rna_fm_model: nn.Module,
        freeze_layers: int = 6,
        hidden_dim: int = 320,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rna_fm = rna_fm_model
        self.freeze_layers = freeze_layers
        
        if freeze_layers > 0:
            for i, layer in enumerate(self.rna_fm.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"Froze first {freeze_layers} layers of RNA-FM")
        
        rna_fm_dim = 640  # RNA-FM hidden size
        
        self.shared_proj = nn.Sequential(
            nn.Linear(rna_fm_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dms_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for new layers."""
        for module in [self.shared_proj, self.dms_head, self.shape_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        tokens: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tokens: Input token IDs [batch_size, seq_len]
            return_embeddings: Whether to return RNA-FM embeddings
            
        Returns:
            Dictionary with predictions and optionally embeddings
        """
        rna_out = self.rna_fm(tokens, repr_layers=[12])
        embeddings = rna_out['representations'][12]  # [B, L, 640]
        
        shared_features = self.shared_proj(embeddings)  # [B, L, hidden_dim]
        dms_pred = self.dms_head(shared_features).squeeze(-1)  # [B, L]
        shape_pred = self.shape_head(shared_features).squeeze(-1)  # [B, L]
        
        outputs = {
            'dms': dms_pred,
            'shape': shape_pred
        }
        
        if return_embeddings:
            outputs['embeddings'] = embeddings
            outputs['shared_features'] = shared_features
            
        return outputs


class ReactivityTrainer:
    """Trainer for RNA-FM pretraining on reactivity data."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.model = self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_T0'],
            T_mult=config['scheduler_Tmult']
        )
        
        self.scaler = GradScaler() if config['use_amp'] else None
        
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_dms_loss = 0
        total_shape_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            tokens = batch['tokens'].to(self.device)
            dms_true = batch['dms'].to(self.device)
            shape_true = batch['shape'].to(self.device)
            dms_mask = batch['dms_mask'].to(self.device)
            shape_mask = batch['shape_mask'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config['use_amp']):
                outputs = self.model(tokens)
                
                dms_loss = self._masked_mse_loss(
                    outputs['dms'], dms_true, dms_mask
                )
                shape_loss = self._masked_mse_loss(
                    outputs['shape'], shape_true, shape_mask
                )
                
                loss = (
                    self.config['dms_weight'] * dms_loss + 
                    self.config['shape_weight'] * shape_loss
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            total_dms_loss += dms_loss.item()
            total_shape_loss += shape_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dms': f"{dms_loss.item():.4f}",
                'shape': f"{shape_loss.item():.4f}"
            })
        
        self.scheduler.step()
        
        return {
            'train_loss': total_loss / num_batches,
            'train_dms_loss': total_dms_loss / num_batches,
            'train_shape_loss': total_shape_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_dms_loss = 0
        total_shape_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                tokens = batch['tokens'].to(self.device)
                dms_true = batch['dms'].to(self.device)
                shape_true = batch['shape'].to(self.device)
                dms_mask = batch['dms_mask'].to(self.device)
                shape_mask = batch['shape_mask'].to(self.device)
                
                outputs = self.model(tokens)
                
                dms_loss = self._masked_mse_loss(
                    outputs['dms'], dms_true, dms_mask
                )
                shape_loss = self._masked_mse_loss(
                    outputs['shape'], shape_true, shape_mask
                )
                
                loss = (
                    self.config['dms_weight'] * dms_loss + 
                    self.config['shape_weight'] * shape_loss
                )
                
                total_loss += loss.item()
                total_dms_loss += dms_loss.item()
                total_shape_loss += shape_loss.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_dms_loss': total_dms_loss / num_batches,
            'val_shape_loss': total_shape_loss / num_batches
        }
    
    def _masked_mse_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss with masking."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        masked_pred = pred[mask]
        masked_target = target[mask]
        return F.mse_loss(masked_pred, masked_target)
    
    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            print(f"\nEpoch {epoch}: " + 
                  " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            
            if wandb.run:
                wandb.log(metrics)
            
            # Early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % self.config['checkpoint_freq'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config['output_dir']) / filename
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain RNA-FM on reactivity data"
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data/ribonanza/reactivity_train_data_sn_filtered.csv',
        help='Path to reactivity data CSV'
    )
    parser.add_argument(
        '--rhofold_path',
        type=str,
        default='rhofold-main/pretrained/rhofold_pretrained_params.pt',
        help='Path to pretrained RhoFold checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/rna_fm_pretrain',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--freeze_layers',
        type=int,
        default=6,
        help='Number of RNA-FM layers to freeze'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'dms_weight': 1.0,
        'shape_weight': 1.0,
        'freeze_layers': args.freeze_layers,
        'patience': 10,
        'checkpoint_freq': 5,
        'scheduler_T0': 10,
        'scheduler_Tmult': 2,
        'use_amp': args.device != 'cpu',
        'output_dir': args.output_dir,
        'seed': 42
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.use_wandb:
        wandb.init(
            project="rna-fm-pretrain",
            config=config,
            name=f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    device = torch.device(args.device)
    if device.type == 'mps':
        torch.backends.mps.enable_math_fallback(True)
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset = ReactivityDataset(args.data_path)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print("Loading RhoFold model...")
    rhofold = RhoFold(rhofold_config)
    checkpoint = torch.load(args.rhofold_path, map_location='cpu')
    rhofold.load_state_dict(checkpoint['model'])
    
    rna_fm = rhofold.msa_embedder.rna_fm
    print(f"RNA-FM loaded with {len(rna_fm.layers)} layers")
    
    model = DualReactivityModel(
        rna_fm_model=rna_fm,
        freeze_layers=config['freeze_layers']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = ReactivityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    print("\nStarting training...")
    trainer.train(config['num_epochs'])
    
    print("\nTraining complete!")
    
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()