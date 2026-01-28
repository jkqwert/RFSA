# -*- coding: utf-8 -*-
'''
Training script for text-to-image mapping module
Freezes vision and text encoders, only trains the mapping MLP and prompt learners
'''

import os
import json
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from mapping_model import CompositeMappingModule
from data_loader import get_data_loader
from cn_clip.clip.model import CLIP
from cn_clip.training.main import convert_models_to_fp32


def setup_logging(output_dir):
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def cosine_similarity_loss(pred_features, target_features, temperature=0.07):
    """
    Compute cosine similarity loss with temperature scaling (contrastive learning style)
    
    This loss expects batch to have unique pairs (no duplicates). If batch contains
    duplicate images, the contrastive learning may not work well.
    
    Args:
        pred_features: Predicted features [batch_size, embed_dim]
        target_features: Target features [batch_size, embed_dim]
        temperature: Temperature parameter for scaling
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize features
    pred_features = nn.functional.normalize(pred_features, p=2, dim=-1)
    target_features = nn.functional.normalize(target_features, p=2, dim=-1)
    
    # Compute cosine similarity matrix
    similarity = torch.matmul(pred_features, target_features.t()) / temperature
    
    # Create labels (diagonal should be 1, others 0)
    labels = torch.arange(pred_features.size(0), device=pred_features.device)
    
    # Cross-entropy loss
    loss = nn.CrossEntropyLoss()(similarity, labels)
    
    return loss


def compute_loss(pred_features, target_features, loss_type='cosine'):
    """
    Compute loss between predicted and target features
    
    Args:
        pred_features: Predicted features [batch_size, embed_dim]
        target_features: Target features [batch_size, embed_dim]
        loss_type: Loss type ('cosine', 'mse', 'l1')
    
    Returns:
        loss: Scalar loss value
    """
    if loss_type == 'cosine':
        # Cosine similarity loss (contrastive learning style)
        return cosine_similarity_loss(pred_features, target_features)
    elif loss_type == 'mse':
        # Mean squared error
        return nn.MSELoss()(pred_features, target_features)
    elif loss_type == 'l1':
        # L1 loss
        return nn.L1Loss()(pred_features, target_features)
    elif loss_type == 'cosine_simple':
        # Simple cosine similarity loss (1 - cosine_sim)
        # This directly optimizes point-to-point matching
        # Note: This may lead to feature collapse if not regularized
        pred_features = nn.functional.normalize(pred_features, p=2, dim=-1)
        target_features = nn.functional.normalize(target_features, p=2, dim=-1)
        # Compute cosine similarity: dot product of normalized vectors
        cosine_sim = (pred_features * target_features).sum(dim=-1)
        # Loss: maximize cosine similarity (minimize 1 - cosine_sim)
        loss = (1 - cosine_sim).mean()
        return loss
    elif loss_type == 'cosine_with_diversity':
        # Cosine similarity loss with diversity regularization
        # This encourages feature diversity to improve retrieval performance
        pred_features = nn.functional.normalize(pred_features, p=2, dim=-1)
        target_features = nn.functional.normalize(target_features, p=2, dim=-1)
        
        # Main loss: maximize cosine similarity
        cosine_sim = (pred_features * target_features).sum(dim=-1)
        main_loss = (1 - cosine_sim).mean()
        
        # Diversity loss: encourage features to be diverse (lower inter-sample similarity)
        pred_sim_matrix = torch.matmul(pred_features, pred_features.t())
        mask = ~torch.eye(pred_sim_matrix.size(0), dtype=torch.bool, device=pred_sim_matrix.device)
        diversity_loss = pred_sim_matrix[mask].mean()  # Penalize high inter-sample similarity
        
        # Combined loss with small diversity weight
        loss = main_loss + 0.1 * diversity_loss
        return loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def load_clip_model(checkpoint_path, device, vision_model='ViT-B-16', text_model='RoBERTa-wwm-ext-base-chinese'):
    """Load and freeze CLIP model"""
    # Load model config
    vision_model_config_file = Path(__file__).parent / f"cn_clip/clip/model_configs/{vision_model.replace('/', '-')}.json"
    text_model_config_file = Path(__file__).parent / f"cn_clip/clip/model_configs/{text_model.replace('/', '-')}.json"
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v
    
    # Create model
    model = CLIP(**model_info)
    from cn_clip.clip.model import convert_weights
    convert_weights(model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    
    # Convert to FP32
    convert_models_to_fp32(model)
    model.to(device)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    return model


def fuse_image_features(image_features_dict, component_types, fusion_method='weighted_sum'):
    """
    Fuse image features from different components
    
    Args:
        image_features_dict: Dictionary of {component_type: feature_tensor}
        component_types: List of component types
        fusion_method: Fusion method ('weighted_sum', 'mean')
    
    Returns:
        Fused image feature [batch_size, embed_dim]
    """
    features_list = []
    for comp_type in component_types:
        if comp_type in image_features_dict:
            features_list.append(image_features_dict[comp_type])
    
    if not features_list:
        # Return zero vector if no features
        return torch.zeros_like(list(image_features_dict.values())[0])
    
    if fusion_method == 'weighted_sum':
        # Equal weights for now (can be made learnable)
        weights = torch.ones(len(features_list), device=features_list[0].device) / len(features_list)
        fused = sum(w * feat for w, feat in zip(weights, features_list))
    elif fusion_method == 'mean':
        fused = torch.stack(features_list).mean(dim=0)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Normalize
    fused = nn.functional.normalize(fused, p=2, dim=-1)
    
    return fused


def train_epoch(mapping_module, train_loader, optimizer, device, args, logger):
    """Train for one epoch"""
    mapping_module.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        text_features = batch['text_features']
        image_features = batch['image_features']
        
        # Move to device
        text_features = {
            k: v.to(device) for k, v in text_features.items()
        }
        image_features = {
            k: v.to(device) for k, v in image_features.items()
        }
        
        # Forward pass: map text features to image feature space
        pred_image_features = mapping_module(text_features)
        
        # Debug: check if output is all zeros or has reasonable values
        if num_batches == 0:
            with torch.no_grad():
                pred_norm_check = pred_image_features.norm(p=2, dim=-1).mean().item()
                pred_max = pred_image_features.abs().max().item()
                logger.info(f"First batch - Pred feature norm: {pred_norm_check:.6f}, max: {pred_max:.6f}")
        
        # Fuse target image features
        target_image_features = fuse_image_features(
            image_features,
            args.component_types,
            fusion_method='mean'
        )
        
        # Debug: check target features
        if num_batches == 0:
            with torch.no_grad():
                target_norm_check = target_image_features.norm(p=2, dim=-1).mean().item()
                target_max = target_image_features.abs().max().item()
                logger.info(f"First batch - Target feature norm: {target_norm_check:.6f}, max: {target_max:.6f}")
        
        # Compute loss
        loss = compute_loss(pred_image_features, target_image_features, loss_type=args.loss_type)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN or Inf loss detected, skipping batch")
            continue
        
        # Check if loss is detached (shouldn't happen but check anyway)
        if not loss.requires_grad:
            logger.error(f"Loss does not require grad! This means no gradients will flow.")
            continue
        
        # Debug: compute additional metrics for first batch
        if num_batches == 0:
            with torch.no_grad():
                # Compute cosine similarity between pred and target
                pred_norm = nn.functional.normalize(pred_image_features, p=2, dim=-1)
                target_norm = nn.functional.normalize(target_image_features, p=2, dim=-1)
                cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()
                
                # Compute feature diversity (std of feature norms)
                pred_norms = pred_image_features.norm(p=2, dim=-1).std().item()
                target_norms = target_image_features.norm(p=2, dim=-1).std().item()
                
                # Compute inter-sample similarity (diversity within batch)
                pred_sim_matrix = torch.matmul(pred_norm, pred_norm.t())
                target_sim_matrix = torch.matmul(target_norm, target_norm.t())
                # Exclude diagonal
                mask = ~torch.eye(pred_sim_matrix.size(0), dtype=torch.bool, device=pred_sim_matrix.device)
                pred_inter_sim = pred_sim_matrix[mask].mean().item()
                target_inter_sim = target_sim_matrix[mask].mean().item()
                
                logger.info(f"First batch diagnostics:")
                logger.info(f"  Pred-Target cosine sim: {cosine_sim:.4f}")
                logger.info(f"  Pred feature norm std: {pred_norms:.4f}")
                logger.info(f"  Target feature norm std: {target_norms:.4f}")
                logger.info(f"  Pred inter-sample sim: {pred_inter_sim:.4f}")
                logger.info(f"  Target inter-sample sim: {target_inter_sim:.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0.0
        param_count = 0
        for param in mapping_module.parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
                else:
                    # If param has no grad, it might not be in the computation graph
                    if num_batches == 0:  # Only warn on first batch
                        logger.warning(f"Parameter {param.shape} has no gradient!")
        if param_count > 0:
            total_grad_norm = total_grad_norm ** 0.5
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mapping_module.parameters(), args.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Compute cosine similarity for monitoring
        with torch.no_grad():
            pred_norm = nn.functional.normalize(pred_image_features, p=2, dim=-1)
            target_norm = nn.functional.normalize(target_image_features, p=2, dim=-1)
            avg_cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cos_sim': f'{avg_cosine_sim:.4f}',
            'grad': f'{total_grad_norm:.4f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(mapping_module, val_loader, device, args, logger):
    """Validate on validation set"""
    mapping_module.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            text_features = batch['text_features']
            image_features = batch['image_features']
            
            # Move to device
            text_features = {
                k: v.to(device) for k, v in text_features.items()
            }
            image_features = {
                k: v.to(device) for k, v in image_features.items()
            }
            
            # Forward pass
            pred_image_features = mapping_module(text_features)
            
            # Fuse target image features
            target_image_features = fuse_image_features(
                image_features,
                args.component_types,
                fusion_method='mean'
            )
            
            # Compute loss
            loss = compute_loss(pred_image_features, target_image_features, loss_type=args.loss_type)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train text-to-image mapping module')
    
    # Data paths
    parser.add_argument('--train-texts', type=str, required=True,
                        help='Path to train_texts.jsonl')
    parser.add_argument('--valid-texts', type=str, required=True,
                        help='Path to valid_texts.jsonl')
    parser.add_argument('--create-jsonl', type=str, default='create.jsonl',
                        help='Path to create.jsonl with component annotations')
    parser.add_argument('--text-features-dir', type=str, default='features',
                        help='Directory containing text feature files')
    parser.add_argument('--image-features-dir', type=str, default='features',
                        help='Directory containing image feature files')
    
    # Model paths
    parser.add_argument('--clip-checkpoint', type=str, default='clip_cn_vit-b-16.pt',
                        help='Path to CLIP checkpoint (for reference, model is frozen)')
    parser.add_argument('--vision-model', type=str, default='ViT-B-16',
                        choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'])
    parser.add_argument('--text-model', type=str, default='RoBERTa-wwm-ext-base-chinese',
                        choices=['RoBERTa-wwm-ext-base-chinese', 'RoBERTa-wwm-ext-large-chinese', 'RBT3-chinese'])
    
    # Mapping module hyperparameters
    parser.add_argument('--embed-dim', type=int, default=512,
                        help='Feature embedding dimension')
    parser.add_argument('--prompt-length', type=int, default=4,
                        help='Length of learnable prompt tokens')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension for MLP')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in MLP')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--fusion-method', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'concat', 'attention'],
                        help='Method to fuse component features')
    parser.add_argument('--component-types', type=str, nargs='+',
                        default=['subject', 'object', 'second', 'relation'],
                        help='Component types')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--loss-type', type=str, default='cosine_simple',
                        choices=['cosine', 'cosine_simple', 'mse', 'l1'],
                        help='Loss type')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/mapping',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Misc
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load CLIP model (frozen, only for reference)
    logger.info('Loading CLIP model (frozen)...')
    clip_model = load_clip_model(
        args.clip_checkpoint,
        device,
        args.vision_model,
        args.text_model
    )
    logger.info('CLIP model loaded and frozen')
    
    # Create mapping module
    logger.info('Creating mapping module...')
    mapping_module = CompositeMappingModule(
        embed_dim=args.embed_dim,
        prompt_length=args.prompt_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        component_types=args.component_types
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in mapping_module.parameters())
    trainable_params = sum(p.numel() for p in mapping_module.parameters() if p.requires_grad)
    logger.info(f'Mapping module parameters: {total_params:,} total, {trainable_params:,} trainable')
    
    # Create data loaders
    logger.info('Creating data loaders...')
    train_loader = get_data_loader(
        texts_jsonl=args.train_texts,
        create_jsonl=args.create_jsonl,
        text_features_dir=args.text_features_dir,
        image_features_dir=args.image_features_dir,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        component_types=args.component_types
    )
    
    val_loader = get_data_loader(
        texts_jsonl=args.valid_texts,
        create_jsonl=args.create_jsonl,
        text_features_dir=args.text_features_dir,
        image_features_dir=args.image_features_dir,
        split='valid',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        component_types=args.component_types
    )
    
    logger.info(f'Train samples: {len(train_loader.dataset)}')
    logger.info(f'Valid samples: {len(val_loader.dataset)}')
    
    # Optimizer
    optimizer = optim.AdamW(
        mapping_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    logger.info('Starting training...')
    for epoch in range(1, args.epochs + 1):
        logger.info(f'Epoch {epoch}/{args.epochs}')
        
        # Train
        train_loss = train_epoch(mapping_module, train_loader, optimizer, device, args, logger)
        logger.info(f'Train loss: {train_loss:.4f}')
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validate
        val_loss = validate(mapping_module, val_loader, device, args, logger)
        logger.info(f'Valid loss: {val_loss:.4f}')
        writer.add_scalar('Loss/Valid', val_loss, epoch)
        
        # Scheduler step
        scheduler.step(val_loss)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': mapping_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                logger.info(f'Saving best model (val_loss: {val_loss:.4f})')
            else:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            
            torch.save(checkpoint, checkpoint_path)
    
    logger.info('Training completed!')
    writer.close()


if __name__ == '__main__':
    main()

