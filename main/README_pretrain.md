# RNA-FM Pretraining on Reactivity Data

## Overview

The `pretrain_rna_fm.py` script fine-tunes the RNA-FM model from RhoFold on chemical reactivity data (DMS and 2A3) from the Ribonanza dataset. This pretraining helps the model learn chemical accessibility patterns that can improve downstream RNA structure prediction.

## Features

- **Dual-task learning**: Simultaneously predicts DMS and 2A3 reactivity
- **MPS optimization**: Optimized for Apple Silicon with mixed precision training
- **Efficient data loading**: Handles missing values and dynamic padding
- **Comprehensive logging**: Supports Weights & Biases integration
- **Model checkpointing**: Saves best model and periodic checkpoints
- **Early stopping**: Prevents overfitting with patience-based stopping

## Architecture

The model consists of:
1. **RNA-FM backbone**: 12-layer transformer with 640-dim embeddings
2. **Shared projection**: Maps RNA-FM features to shared representation
3. **Task-specific heads**: Separate prediction heads for DMS and 2A3

## Usage

### Basic usage:
```bash
python main/pretrain_rna_fm.py
```

### With custom parameters:
```bash
python main/pretrain_rna_fm.py \
    --data_path data/ribonanza/reactivity_train_data_sn_filtered.csv \
    --rhofold_path rhofold-main/pretrained/rhofold_pretrained_params.pt \
    --output_dir outputs/rna_fm_pretrain \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --freeze_layers 6 \
    --device mps \
    --use_wandb
```

### Arguments:
- `--data_path`: Path to reactivity CSV file
- `--rhofold_path`: Path to pretrained RhoFold checkpoint
- `--output_dir`: Directory for saving checkpoints
- `--batch_size`: Training batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--freeze_layers`: Number of RNA-FM layers to freeze (default: 6)
- `--device`: Device to use (cpu/cuda/mps, default: mps)
- `--use_wandb`: Enable Weights & Biases logging

## Output

The script saves:
- `best_model.pt`: Best model based on validation loss
- `checkpoint_epoch_N.pt`: Periodic checkpoints
- `config.json`: Training configuration

## Memory Optimization

For Apple Silicon (MPS):
- Mixed precision training enabled by default
- Gradient accumulation can be added if needed
- Batch size can be adjusted based on available memory

## Next Steps

After pretraining, the model can be:
1. Used as a feature extractor for structure prediction
2. Fine-tuned end-to-end with RhoFold's structure module
3. Analyzed for learned reactivity patterns