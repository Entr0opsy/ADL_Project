"""
Training script with DeepSpeed integration for larger batch sizes.
Uses ZeRO Stage 2 for optimizer and gradient sharding to maximize memory efficiency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
import time
import json
import deepspeed
from deepspeed.ops.adam import FusedAdam

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'edgeface'))

# Import your modified timm wrapper
from backbones.timmfr import get_timmfrv2
from data_loader import get_iris_loaders as get_data_loaders
from utils import save_model


class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=0.6, mining='hard'):
        super(ImprovedTripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
        
    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        if self.mining == 'hard':
            losses = torch.relu(pos_dist - neg_dist + self.margin)
        else:
            losses = pos_dist - neg_dist + self.margin
        return losses.mean(), pos_dist.mean(), neg_dist.mean()


class AsymmetricIrisModel(nn.Module):
    """Model with asymmetric convolutions for radial iris images"""
    def __init__(self, model_name='edgenext_x_small', featdim=512, dropout_rate=0.3,
                 use_asymmetric_conv=True, preserve_vertical=True):
        super(AsymmetricIrisModel, self).__init__()
        
        self.model = get_timmfrv2(
            model_name=model_name,
            featdim=featdim,
            use_asymmetric_conv=use_asymmetric_conv,
            preserve_vertical=preserve_vertical
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.model(x)
        return self.dropout(x)


def get_deepspeed_config(
    train_batch_size=1024,
    gradient_accumulation_steps=1,
    learning_rate=0.001,
    weight_decay=1e-4,
    warmup_steps=500,
    total_steps=10000,
    stage=2,
    offload_optimizer=False,
    offload_param=False,
    cpu_offload=False
):
    """
    Generate DeepSpeed configuration for training.
    
    Args:
        train_batch_size: Total batch size across all GPUs
        gradient_accumulation_steps: Accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps
        total_steps: Total training steps
        stage: ZeRO stage (1, 2, or 3)
        offload_optimizer: Offload optimizer states to CPU
        offload_param: Offload parameters to CPU (Stage 3)
        cpu_offload: Enable CPU offloading for even larger batch sizes
    """
    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": train_batch_size // gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": weight_decay
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps
            }
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,  # Dynamic loss scaling
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }
    
    # Add optimizer offload for Stage 2/3
    if offload_optimizer and stage >= 2:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Add parameter offload for Stage 3
    if offload_param and stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Additional CPU offload optimizations
    if cpu_offload:
        config["zero_optimization"]["sub_group_size"] = 1e9
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False
        }
    
    return config


def train_with_deepspeed(
    data_root,
    save_dir,
    model_name='edgenext_x_small',
    num_epochs=100,
    train_batch_size=32,  # Much larger batch size!
    gradient_accumulation_steps=4,
    learning_rate=0.001,
    margin=0.6,
    patience=15,
    dropout_rate=0.3,
    weight_decay=1e-4,
    use_asymmetric_conv=True,
    preserve_vertical=True,
    deepspeed_stage=2,
    offload_optimizer=False,
    cpu_offload=False,
    local_rank=-1
):
    """
    Train with DeepSpeed for maximum batch size and memory efficiency.
    
    Example command to launch:
    deepspeed --num_gpus=2 train_deepspeed_iris.py --data_root=... --save_dir=...
    
    Or with accelerate:
    accelerate launch --config_file deepspeed_config.yaml train_deepspeed_iris.py
    """
    
    # DeepSpeed will set local_rank automatically
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Only rank 0 writes to TensorBoard
    writer = None
    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))
        print("\n" + "="*80)
        print("DEEPSPEED TRAINING WITH ASYMMETRIC IRIS MODEL")
        print("="*80)
        print(f"Batch size per GPU: {train_batch_size // gradient_accumulation_steps}")
        print(f"Total batch size: {train_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"DeepSpeed Stage: {deepspeed_stage}")
        print(f"Optimizer offload: {offload_optimizer}")
        print(f"CPU offload: {cpu_offload}")
        print("="*80 + "\n")
    
    # Data loaders
    micro_batch_size = train_batch_size // gradient_accumulation_steps
    loaders = get_data_loaders(data_root, loader_type='triplet', batch_size=micro_batch_size)
    train_loader = loaders['triplet_train']
    test_loader = loaders['triplet_test']
    
    if local_rank == 0:
        print(f"\nData loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    # Model
    model = AsymmetricIrisModel(
        model_name=model_name,
        featdim=512,
        dropout_rate=dropout_rate,
        use_asymmetric_conv=use_asymmetric_conv,
        preserve_vertical=preserve_vertical
    )
    
    # DeepSpeed configuration
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    
    ds_config = get_deepspeed_config(
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        stage=deepspeed_stage,
        offload_optimizer=offload_optimizer,
        cpu_offload=cpu_offload
    )
    
    # Save DeepSpeed config
    if local_rank == 0:
        config_path = os.path.join(save_dir, 'deepspeed_config.json')
        with open(config_path, 'w') as f:
            json.dump(ds_config, f, indent=4)
        print(f"\nDeepSpeed config saved to: {config_path}")
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        print("Model initialized with DeepSpeed!\n")
    
    # Loss function
    criterion = ImprovedTripletLoss(margin=margin, mining='hard')
    
    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # ============ TRAINING ============
        model_engine.train()
        epoch_loss, epoch_pos, epoch_neg = 0, 0, 0
        
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
        else:
            pbar = train_loader
        
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            # Ensure input tensors have the same dtype as model parameters.
            # DeepSpeed may move/convert model params to fp16; inputs from the
            # dataloader are usually float32 which causes errors like:
            # RuntimeError: Input type (float) and bias type (c10::Half) should be the same
            # To avoid that, cast inputs to the model's parameter dtype before forward.
            model_param_dtype = next(model_engine.module.parameters()).dtype
            anchor = anchor.to(device=model_engine.local_rank, dtype=model_param_dtype)
            positive = positive.to(device=model_engine.local_rank, dtype=model_param_dtype)
            negative = negative.to(device=model_engine.local_rank, dtype=model_param_dtype)
            
            # Forward pass
            embed_a = model_engine(anchor)
            embed_p = model_engine(positive)
            embed_n = model_engine(negative)
            
            # Compute loss
            loss, pos_dist, neg_dist = criterion(embed_a, embed_p, embed_n)
            
            # Backward pass with DeepSpeed
            model_engine.backward(loss)
            model_engine.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_pos += pos_dist.item()
            epoch_neg += neg_dist.item()
            
            if local_rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos': f'{pos_dist.item():.4f}',
                    'neg': f'{neg_dist.item():.4f}'
                })
        
        epoch_loss /= len(train_loader)
        epoch_pos /= len(train_loader)
        epoch_neg /= len(train_loader)
        
        # ============ VALIDATION ============
        model_engine.eval()
        val_loss, val_pos, val_neg = 0, 0, 0
        
        with torch.no_grad():
            if local_rank == 0:
                pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")
            else:
                pbar = test_loader
                
            for anchor, positive, negative in pbar:
                # Cast validation inputs to model dtype as well to avoid dtype mismatch
                model_param_dtype = next(model_engine.module.parameters()).dtype
                anchor = anchor.to(device=model_engine.local_rank, dtype=model_param_dtype)
                positive = positive.to(device=model_engine.local_rank, dtype=model_param_dtype)
                negative = negative.to(device=model_engine.local_rank, dtype=model_param_dtype)
                
                embed_a = model_engine(anchor)
                embed_p = model_engine(positive)
                embed_n = model_engine(negative)
                
                loss, pos_dist, neg_dist = criterion(embed_a, embed_p, embed_n)
                
                val_loss += loss.item()
                val_pos += pos_dist.item()
                val_neg += neg_dist.item()
                
                if local_rank == 0:
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        val_loss /= len(test_loader)
        val_pos /= len(test_loader)
        val_neg /= len(test_loader)
        
        epoch_time = time.time() - epoch_start
        
        # ============ LOGGING (Rank 0 only) ============
        if local_rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Pos: {epoch_pos:.4f} | Train Neg: {epoch_neg:.4f}")
            print(f"  Val Pos: {val_pos:.4f} | Val Neg: {val_neg:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"{'='*80}\n")
            
            writer.add_scalar("Loss/Train", epoch_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Distance/Train_Positive", epoch_pos, epoch)
            writer.add_scalar("Distance/Train_Negative", epoch_neg, epoch)
            writer.add_scalar("Distance/Val_Positive", val_pos, epoch)
            writer.add_scalar("Distance/Val_Negative", val_neg, epoch)
        
        # ============ MODEL CHECKPOINTING ============
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if local_rank == 0:
                print(f"✓ Validation improved! Saving checkpoint...")
                
            # Save DeepSpeed checkpoint
            checkpoint_dir = os.path.join(save_dir, f'checkpoint_best')
            model_engine.save_checkpoint(checkpoint_dir)
            
        else:
            patience_counter += 1
            if local_rank == 0:
                print(f"✗ No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                if local_rank == 0:
                    print("\nEarly stopping triggered!")
                break
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}')
            model_engine.save_checkpoint(checkpoint_dir)
    
    # Final save
    checkpoint_dir = os.path.join(save_dir, 'checkpoint_final')
    model_engine.save_checkpoint(checkpoint_dir)
    
    if local_rank == 0:
        writer.close()
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with DeepSpeed')
    parser.add_argument('--data_root', type=str, default='/home/aryan/Desktop/Adl_assignment_1/masked_dataset')
    parser.add_argument('--save_dir', type=str, default='/home/aryan/Desktop/Adl_assignment_1/model_weights/deepspeed_iris')
    parser.add_argument('--model_name', type=str, default='edgenext_x_small')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2048, help='Total batch size across all GPUs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--margin', type=float, default=0.6)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--deepspeed_stage', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--offload_optimizer', action='store_true', help='Offload optimizer to CPU')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable aggressive CPU offloading')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    train_with_deepspeed(
        data_root=args.data_root,
        save_dir=args.save_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        margin=args.margin,
        patience=args.patience,
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
        use_asymmetric_conv=True,
        preserve_vertical=True,
        deepspeed_stage=args.deepspeed_stage,
        offload_optimizer=args.offload_optimizer,
        cpu_offload=args.cpu_offload,
        local_rank=args.local_rank
    )