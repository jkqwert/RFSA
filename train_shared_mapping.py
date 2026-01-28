# -*- coding: utf-8 -*-
'''
Training script for shared feature space mapping
Maps both text and image features to a common feature space
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

from mapping_model_shared import SharedMappingModule
from data_loader import get_data_loader


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


def contrastive_loss(text_features, image_features, temperature=0.07):
    """
    Contrastive loss for shared feature space
    Maps matching text-image pairs closer, non-matching pairs farther
    
    Args:
        text_features: Text features in shared space [batch_size, embed_dim]
        image_features: Image features in shared space [batch_size, embed_dim]
        temperature: Temperature parameter for scaling
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize features
    text_features = nn.functional.normalize(text_features, p=2, dim=-1)
    image_features = nn.functional.normalize(image_features, p=2, dim=-1)
    
    # Compute similarity matrix: text_features @ image_features.T
    # Shape: [batch_size, batch_size]
    similarity = torch.matmul(text_features, image_features.t()) / temperature
    
    # Create labels (diagonal should be 1, others 0)
    # Each text matches with its corresponding image (same index in batch)
    labels = torch.arange(text_features.size(0), device=text_features.device)
    
    # Cross-entropy loss for text-to-image
    loss_t2i = nn.CrossEntropyLoss()(similarity, labels)
    
    # Cross-entropy loss for image-to-text (transpose)
    loss_i2t = nn.CrossEntropyLoss()(similarity.t(), labels)
    
    # Combined loss
    loss = (loss_t2i + loss_i2t) / 2
    
    return loss


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
        
        # Forward pass: map both text and image to shared space
        text_shared = mapping_module.forward_text(text_features)
        image_shared = mapping_module.forward_image(image_features)
        
        # Debug: check if output is reasonable
        if num_batches == 0:
            with torch.no_grad():
                text_norm = text_shared.norm(p=2, dim=-1).mean().item()
                image_norm = image_shared.norm(p=2, dim=-1).mean().item()
                cosine_sim = (text_shared * image_shared).sum(dim=-1).mean().item()
                logger.info(f"First batch - Text shared norm: {text_norm:.6f}, Image shared norm: {image_norm:.6f}")
                logger.info(f"First batch - Text-Image cosine sim: {cosine_sim:.4f}")
        
        # Compute contrastive loss
        loss = contrastive_loss(text_shared, image_shared, temperature=args.temperature)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN or Inf loss detected, skipping batch")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mapping_module.parameters(), args.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Compute cosine similarity for monitoring
        with torch.no_grad():
            text_norm = nn.functional.normalize(text_shared, p=2, dim=-1)
            image_norm = nn.functional.normalize(image_shared, p=2, dim=-1)
            avg_cosine_sim = (text_norm * image_norm).sum(dim=-1).mean().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cos_sim': f'{avg_cosine_sim:.4f}'
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
            text_shared = mapping_module.forward_text(text_features)
            image_shared = mapping_module.forward_image(image_features)
            
            # Compute loss
            loss = contrastive_loss(text_shared, image_shared, temperature=args.temperature)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train shared feature space mapping')
    
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
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create mapping module
    logger.info('Creating shared mapping module...')
    mapping_module = SharedMappingModule(
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
        
        # Validate
        val_loss = validate(mapping_module, val_loader, device, args, logger)
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



