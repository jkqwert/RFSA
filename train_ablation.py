# -*- coding: utf-8 -*-
'''
Ablation Study Training Script
Supports three modules: prompt learning, component fusion, shared space mapping
'''

import os
import sys
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mapping_model_ablation import SingleDirectionMappingModule, SharedMappingModule
from data_loader_ablation import get_data_loader


def setup_logger(log_dir):
    """Setup logger"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    
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
    """Contrastive loss for shared feature space"""
    pred_features = nn.functional.normalize(pred_features, p=2, dim=-1)
    target_features = nn.functional.normalize(target_features, p=2, dim=-1)
    
    similarity = torch.matmul(pred_features, target_features.t()) / temperature
    labels = torch.arange(pred_features.size(0), device=pred_features.device)
    
    loss_t2i = nn.CrossEntropyLoss()(similarity, labels)
    loss_i2t = nn.CrossEntropyLoss()(similarity.t(), labels)
    loss = (loss_t2i + loss_i2t) / 2
    
    return loss


def simple_cosine_loss(pred_features, target_features):
    """Simple cosine similarity loss"""
    pred_features = nn.functional.normalize(pred_features, p=2, dim=-1)
    target_features = nn.functional.normalize(target_features, p=2, dim=-1)
    cosine_sim = (pred_features * target_features).sum(dim=-1)
    loss = (1 - cosine_sim).mean()
    return loss


def fuse_image_features(image_features_dict, use_component, component_types, fusion_method='mean'):
    """Fuse image features"""
    if use_component:
        features_list = []
        for comp_type in component_types:
            if comp_type in image_features_dict:
                features_list.append(image_features_dict[comp_type])
        
        if not features_list:
            return torch.zeros_like(list(image_features_dict.values())[0])
        
        if fusion_method == 'mean':
            fused = torch.stack(features_list).mean(dim=0)
        else:
            weights = torch.ones(len(features_list), device=features_list[0].device) / len(features_list)
            fused = sum(w * feat for w, feat in zip(weights, features_list))
    else:
        # Full feature, no fusion needed
        if 'full' in image_features_dict:
            fused = image_features_dict['full']
        else:
            fused = torch.zeros_like(list(image_features_dict.values())[0])
    
    fused = nn.functional.normalize(fused, p=2, dim=-1)
    return fused


def train_epoch(mapping_module, train_loader, optimizer, device, args, logger, use_shared_space):
    """Train for one epoch"""
    mapping_module.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        text_features = batch['text_features']
        image_features = batch['image_features']
        
        # Move to device
        text_features = {k: v.to(device) for k, v in text_features.items()}
        image_features = {k: v.to(device) for k, v in image_features.items()}
        
        if use_shared_space:
            # Shared space mapping
            text_shared = mapping_module.forward_text(text_features)
            image_shared = mapping_module.forward_image(image_features)
            loss = cosine_similarity_loss(text_shared, image_shared, temperature=args.temperature)
        else:
            # Single direction mapping
            pred_image_features = mapping_module(text_features)
            target_image_features = fuse_image_features(
                image_features,
                args.use_component,
                args.component_types,
                fusion_method='mean'
            )
            
            # Debug: check feature norms
            if num_batches == 0:
                with torch.no_grad():
                    pred_norm = pred_image_features.norm(p=2, dim=-1).mean().item()
                    target_norm = target_image_features.norm(p=2, dim=-1).mean().item()
                    pred_max = pred_image_features.abs().max().item()
                    target_max = target_image_features.abs().max().item()
                    logger.info(f"First batch - Pred norm: {pred_norm:.6f}, max: {pred_max:.6f}")
                    logger.info(f"First batch - Target norm: {target_norm:.6f}, max: {target_max:.6f}")
                    logger.info(f"First batch - Text features keys: {list(text_features.keys())}")
                    logger.info(f"First batch - Image features keys: {list(image_features.keys())}")
                    # Check if features are all zeros
                    if target_norm < 1e-6:
                        logger.warning("WARNING: Target features are all zeros! Check data loading.")
                    if pred_norm < 1e-6:
                        logger.warning("WARNING: Predicted features are all zeros! Check model.")
            
            loss = simple_cosine_loss(pred_image_features, target_image_features)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN or Inf loss detected, skipping batch")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mapping_module.parameters(), args.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        with torch.no_grad():
            if use_shared_space:
                text_norm = nn.functional.normalize(text_shared, p=2, dim=-1)
                image_norm = nn.functional.normalize(image_shared, p=2, dim=-1)
                avg_cosine_sim = (text_norm * image_norm).sum(dim=-1).mean().item()
            else:
                pred_norm = nn.functional.normalize(pred_image_features, p=2, dim=-1)
                target_norm = nn.functional.normalize(target_image_features, p=2, dim=-1)
                avg_cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cos_sim': f'{avg_cosine_sim:.4f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(mapping_module, val_loader, device, args, logger, use_shared_space):
    """Validate on validation set"""
    mapping_module.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            text_features = batch['text_features']
            image_features = batch['image_features']
            
            text_features = {k: v.to(device) for k, v in text_features.items()}
            image_features = {k: v.to(device) for k, v in image_features.items()}
            
            if use_shared_space:
                text_shared = mapping_module.forward_text(text_features)
                image_shared = mapping_module.forward_image(image_features)
                loss = cosine_similarity_loss(text_shared, image_shared, temperature=args.temperature)
            else:
                pred_image_features = mapping_module(text_features)
                target_image_features = fuse_image_features(
                    image_features,
                    args.use_component,
                    args.component_types,
                    fusion_method='mean'
                )
                loss = simple_cosine_loss(pred_image_features, target_image_features)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation study training')
    
    # Ablation study switches
    parser.add_argument('--use-prompt', action='store_true', default=False,
                        help='Use prompt learning mechanism')
    parser.add_argument('--use-component', action='store_true', default=False,
                        help='Use component fusion (if False, use full features)')
    parser.add_argument('--use-shared-space', action='store_true', default=False,
                        help='Use shared space mapping (if False, use single direction)')
    
    # Data arguments
    parser.add_argument('--train-texts', type=str, required=True,
                        help='Path to train_texts.jsonl')
    parser.add_argument('--valid-texts', type=str, required=True,
                        help='Path to valid_texts.jsonl')
    parser.add_argument('--create-jsonl', type=str, required=True,
                        help='Path to create.jsonl')
    parser.add_argument('--text-features-dir', type=str, required=True,
                        help='Directory containing text feature files')
    parser.add_argument('--image-features-dir', type=str, required=True,
                        help='Directory containing image feature files')
    
    # Model arguments
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
                        help='Fusion method for component features')
    parser.add_argument('--component-types', type=str, nargs='+',
                        default=['subject', 'object', 'second', 'relation'],
                        help='Component types')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    
    # Log ablation configuration
    logger.info('=' * 80)
    logger.info('Ablation Study Configuration')
    logger.info('=' * 80)
    logger.info(f'  Use Prompt Learning: {args.use_prompt}')
    logger.info(f'  Use Component Fusion: {args.use_component}')
    logger.info(f'  Use Shared Space: {args.use_shared_space}')
    logger.info('=' * 80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create mapping module based on configuration
    logger.info('Creating mapping module...')
    if args.use_shared_space:
        mapping_module = SharedMappingModule(
            embed_dim=args.embed_dim,
            prompt_length=args.prompt_length,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            fusion_method=args.fusion_method,
            component_types=args.component_types,
            use_prompt=args.use_prompt,
            use_component=args.use_component
        ).to(device)
    else:
        mapping_module = SingleDirectionMappingModule(
            embed_dim=args.embed_dim,
            prompt_length=args.prompt_length,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            fusion_method=args.fusion_method,
            component_types=args.component_types,
            use_prompt=args.use_prompt,
            use_component=args.use_component
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
        component_types=args.component_types,
        use_component=args.use_component
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
        component_types=args.component_types,
        use_component=args.use_component
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
        train_loss = train_epoch(mapping_module, train_loader, optimizer, device, args, logger, args.use_shared_space)
        logger.info(f'Train loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(mapping_module, val_loader, device, args, logger, args.use_shared_space)
        logger.info(f'Valid loss: {val_loss:.4f}')
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': mapping_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f'Saving best model (val_loss: {val_loss:.4f})')
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': mapping_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()

