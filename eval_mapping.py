# -*- coding: utf-8 -*-
'''
Evaluation script for text-to-image mapping module
Evaluates retrieval performance using the learned mapping
'''

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from mapping_model import CompositeMappingModule
from data_loader import get_data_loader


def load_checkpoint(checkpoint_path, device, args):
    """Load trained mapping module from checkpoint"""
    mapping_module = CompositeMappingModule(
        embed_dim=args.embed_dim,
        prompt_length=args.prompt_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        component_types=args.component_types
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mapping_module.load_state_dict(checkpoint['model_state_dict'])
    mapping_module.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f}")
    
    return mapping_module


def extract_mapped_features(mapping_module, data_loader, device):
    """Extract mapped features for all samples"""
    mapping_module.eval()
    mapped_features = {}
    text_ids_list = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Extracting mapped features')
        for batch in pbar:
            text_ids = batch['text_ids']
            text_features = batch['text_features']
            
            # Move to device
            text_features = {
                k: v.to(device) for k, v in text_features.items()
            }
            
            # Map text features to image feature space
            mapped_feat = mapping_module(text_features)
            
            # Store features
            for text_id, feat in zip(text_ids, mapped_feat.cpu().numpy()):
                mapped_features[text_id] = feat
                text_ids_list.append(text_id)
    
    return mapped_features, text_ids_list


def load_image_features(image_features_dir, component_types):
    """Load all image features"""
    image_features = {}
    image_type_map = {
        'subject': 'subject',
        'object': 'object',
        'second': 'second_object',
        'relation': 'relation'
    }
    
    # Load features for each component
    all_image_ids = set()
    for comp_type in component_types:
        image_type = image_type_map[comp_type]
        feature_file = os.path.join(image_features_dir, f"{image_type}_features.json")
        
        if os.path.exists(feature_file):
            with open(feature_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
                for img_id, feat in features.items():
                    if img_id not in image_features:
                        image_features[img_id] = {}
                    image_features[img_id][comp_type] = np.array(feat)
                    all_image_ids.add(img_id)
    
    # Fuse image features for each image
    fused_image_features = {}
    for img_id in all_image_ids:
        if img_id in image_features:
            # Average across components
            component_feats = []
            for comp_type in component_types:
                if comp_type in image_features[img_id]:
                    component_feats.append(image_features[img_id][comp_type])
            
            if component_feats:
                fused_feat = np.mean(component_feats, axis=0)
                fused_image_features[img_id] = fused_feat
    
    return fused_image_features


def generate_detailed_report_t2i(predictions, ground_truth, k_values=[1, 5, 10], output_file='t2i_detailed_report.txt'):
    """Generate detailed report with checkmarks for text-to-image retrieval"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('Text-to-Image Retrieval Detailed Report\n')
        f.write('=' * 80 + '\n\n')
        
        for text_id in sorted(predictions.keys()):
            if text_id not in ground_truth:
                continue
            
            gt_images = sorted([str(img_id).zfill(6) for img_id in ground_truth[text_id]])
            pred_images = predictions[text_id]
            
            f.write(f'Text ID: {text_id}\n')
            f.write(f'  Ground Truth Images: {", ".join(gt_images)}\n')
            f.write(f'  Top-10 Predictions: {", ".join(pred_images[:10])}\n')
            
            for k in k_values:
                pred_k = set(pred_images[:k])
                gt_set = set(gt_images)
                intersection = pred_k & gt_set
                hit = len(intersection) > 0
                
                status = '✓' if hit else '✗'
                f.write(f'  Recall@{k}: {status} ')
                if hit:
                    f.write(f'(Found: {", ".join(sorted(intersection))})\n')
                else:
                    f.write(f'(None found)\n')
            f.write('\n')
        
        f.write('=' * 80 + '\n')
        f.write('Summary\n')
        f.write('=' * 80 + '\n')
        for k in k_values:
            hits = 0
            total = 0
            for text_id in sorted(predictions.keys()):
                if text_id not in ground_truth:
                    continue
                total += 1
                gt_images = set([str(img_id).zfill(6) for img_id in ground_truth[text_id]])
                pred_k = set(predictions[text_id][:k])
                if len(pred_k & gt_images) > 0:
                    hits += 1
            recall = hits / total if total > 0 else 0.0
            f.write(f'Recall@{k}: {recall:.4f} ({hits}/{total})\n')
    
    print(f'Generated detailed report: {output_file}')


def generate_detailed_report_i2t(predictions, reverse_gt, k_values=[1, 5, 10], output_file='i2t_detailed_report.txt'):
    """Generate detailed report with checkmarks for image-to-text retrieval"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('Image-to-Text Retrieval Detailed Report\n')
        f.write('=' * 80 + '\n\n')
        
        for image_id in sorted(predictions.keys()):
            if image_id not in reverse_gt:
                continue
            
            gt_texts = reverse_gt[image_id]
            pred_texts = predictions[image_id]
            
            f.write(f'Image ID: {image_id}\n')
            f.write(f'  Ground Truth Text: {gt_texts[0] if gt_texts else "N/A"}\n')
            f.write(f'  Top-10 Predictions: {", ".join(pred_texts[:10])}\n')
            
            for k in k_values:
                pred_k = set(pred_texts[:k])
                gt_text_id = gt_texts[0] if gt_texts else None
                hit = gt_text_id in pred_k if gt_text_id else False
                
                status = '✓' if hit else '✗'
                f.write(f'  Recall@{k}: {status} ')
                if hit:
                    f.write(f'(Found: {gt_text_id})\n')
                else:
                    f.write(f'(Expected: {gt_text_id}, Not found)\n')
            f.write('\n')
        
        f.write('=' * 80 + '\n')
        f.write('Summary\n')
        f.write('=' * 80 + '\n')
        for k in k_values:
            hits = 0
            total = 0
            for image_id in sorted(predictions.keys()):
                if image_id not in reverse_gt:
                    continue
                total += 1
                gt_text_id = reverse_gt[image_id][0] if reverse_gt[image_id] else None
                if gt_text_id and gt_text_id in set(predictions[image_id][:k]):
                    hits += 1
            recall = hits / total if total > 0 else 0.0
            f.write(f'Recall@{k}: {recall:.4f} ({hits}/{total})\n')
    
    print(f'Generated detailed report: {output_file}')


def compute_retrieval_metrics(pred_features, target_features, text_ids, image_ids, ground_truth):
    """
    Compute retrieval metrics (Recall@K)
    
    Args:
        pred_features: Predicted features dictionary {text_id: feature}
        target_features: Target features dictionary {image_id: feature}
        text_ids: List of text IDs
        image_ids: List of image IDs
        ground_truth: Ground truth dictionary {text_id: [image_ids]}
    
    Returns:
        metrics: Dictionary of metrics
        predictions: Dictionary of predictions
    """
    # Convert to numpy arrays
    text_vectors = np.array([pred_features[tid] for tid in text_ids])
    image_vectors = np.array([target_features[iid] for iid in image_ids])
    
    # Create mapping
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_vectors, image_vectors)
    
    # Get top-k predictions for each text
    k_values = [1, 5, 10]
    predictions = {}
    recalls = {f'recall@{k}': [] for k in k_values}
    
    for text_id in text_ids:
        if text_id not in text_id_to_idx:
            continue
        
        text_idx = text_id_to_idx[text_id]
        similarities = similarity_matrix[text_idx]
        
        # Get top-k image indices
        topk_indices = np.argsort(similarities)[::-1]
        
        # Get top-k image IDs with deduplication
        seen_ids = set()
        formatted_image_ids = []
        for idx in topk_indices:
            if len(formatted_image_ids) >= max(k_values):
                break
            img_id = image_ids[idx]
            
            # Format image ID (6-digit string)
            try:
                if isinstance(img_id, str):
                    num = int(img_id)
                else:
                    num = int(img_id)
                formatted_id = f"{num:06d}"
            except:
                formatted_id = str(img_id)
            
            # Only add if not seen before
            if formatted_id not in seen_ids:
                formatted_image_ids.append(formatted_id)
                seen_ids.add(formatted_id)
        
        predictions[text_id] = formatted_image_ids
        
        # Compute recall@K
        if text_id in ground_truth:
            # Format ground truth image IDs
            gt_image_ids = set()
            for img_id in ground_truth[text_id]:
                try:
                    if isinstance(img_id, str):
                        num = int(img_id)
                    else:
                        num = int(img_id)
                    formatted_gt_id = f"{num:06d}"
                    gt_image_ids.add(formatted_gt_id)
                    # Also add original format
                    gt_image_ids.add(str(img_id))
                except:
                    gt_image_ids.add(str(img_id))
            
            for k in k_values:
                pred_k = set(formatted_image_ids[:k])
                intersection = pred_k & gt_image_ids
                # Text-to-Image: one text can map to multiple images (one-to-many)
                # If any ground truth image is found in top-k, it's a hit
                if len(intersection) > 0:
                    recall_k = 1.0  # Hit: found at least one ground truth image
                else:
                    recall_k = 0.0  # Miss: no ground truth image found
                recalls[f'recall@{k}'].append(recall_k)
                
                # Debug print for first few samples
                if len(recalls[f'recall@{k}']) <= 3:
                    print(f'\n  Text-to-Image Recall@{k} calculation for text_id: {text_id}')
                    print(f'    Ground truth images: {sorted(list(gt_image_ids))}')
                    print(f'    Top-{k} predictions: {sorted(list(pred_k))}')
                    print(f'    Intersection: {sorted(list(intersection))}')
                    print(f'    Recall@{k}: {recall_k:.4f} ({"HIT" if recall_k > 0 else "MISS"})')
    
    # Average recalls
    metrics = {}
    for k in k_values:
        if recalls[f'recall@{k}']:
            avg_recall = np.mean(recalls[f'recall@{k}'])
            metrics[f'recall@{k}'] = avg_recall
            print(f'\n  Text-to-Image Recall@{k} summary:')
            print(f'    Total samples: {len(recalls[f"recall@{k}"])}')
            print(f'    Hits: {int(sum(recalls[f"recall@{k}"]))}')
            print(f'    Misses: {len(recalls[f"recall@{k}"]) - int(sum(recalls[f"recall@{k}"]))}')
            print(f'    Average Recall@{k}: {avg_recall:.4f}')
        else:
            metrics[f'recall@{k}'] = 0.0
    
    return metrics, predictions


def compute_reverse_retrieval_metrics(pred_features, target_features, text_ids, image_ids, ground_truth):
    """
    Compute reverse retrieval metrics (Image-to-Text, Recall@K)
    
    Args:
        pred_features: Predicted features dictionary {text_id: feature}
        target_features: Target features dictionary {image_id: feature}
        text_ids: List of text IDs
        image_ids: List of image IDs
        ground_truth: Ground truth dictionary {text_id: [image_ids]}
    
    Returns:
        metrics: Dictionary of metrics
        predictions: Dictionary of predictions
    """
    # Convert to numpy arrays
    text_vectors = np.array([pred_features[tid] for tid in text_ids])
    image_vectors = np.array([target_features[iid] for iid in image_ids])
    
    # Create mapping
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Compute similarity matrix (transpose for image-to-text)
    similarity_matrix = cosine_similarity(image_vectors, text_vectors)
    
    # Build reverse ground truth: image_id -> [text_ids]
    # Use normalized format (6-digit string) to avoid duplicates
    reverse_gt = {}
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
            # Normalize to 6-digit string format
            try:
                if isinstance(img_id, str):
                    num = int(img_id)
                else:
                    num = int(img_id)
                img_id_normalized = f"{num:06d}"
            except:
                img_id_normalized = str(img_id).zfill(6)
            
            # Store only in normalized format (avoid duplicates)
            if img_id_normalized not in reverse_gt:
                reverse_gt[img_id_normalized] = []
            if text_id not in reverse_gt[img_id_normalized]:
                reverse_gt[img_id_normalized].append(text_id)
    
    # Get top-k predictions for each image
    k_values = [1, 5, 10]
    predictions = {}
    recalls = {f'recall@{k}': [] for k in k_values}
    
    for image_id in image_ids:
        # image_id should already be in normalized format (6-digit string)
        # Find the correct index
        image_idx = None
        if image_id in image_id_to_idx:
            image_idx = image_id_to_idx[image_id]
        
        if image_idx is None:
            # Try to find by converting to int (fallback)
            try:
                img_id_num = int(image_id)
                for key in image_id_to_idx.keys():
                    try:
                        if int(key) == img_id_num:
                            image_idx = image_id_to_idx[key]
                            break
                    except:
                        pass
            except:
                pass
        
        if image_idx is None:
            continue
        
        similarities = similarity_matrix[image_idx]
        
        # Get top-k text indices
        topk_indices = np.argsort(similarities)[::-1]
        topk_text_ids = [text_ids[idx] for idx in topk_indices[:max(k_values)]]
        
        # Use normalized image_id (should already be normalized)
        img_id_normalized = image_id
        predictions[img_id_normalized] = topk_text_ids
        
        # Compute recall@K - find ground truth texts for this image (stored as list)
        gt_text_ids = None
        if img_id_normalized in reverse_gt:
            gt_text_ids = reverse_gt[img_id_normalized]
        else:
            # Try to find by converting to int (fallback)
            try:
                img_id_num = int(image_id)
                for key in reverse_gt.keys():
                    try:
                        if int(key) == img_id_num:
                            gt_text_ids = reverse_gt[key]
                            break
                    except:
                        pass
            except:
                pass
        
        if gt_text_ids and len(gt_text_ids) > 0:
            # Image-to-Text: one image maps to one text (one-to-one relationship)
            # gt_text_ids is a list, but should contain only one text_id
            gt_text_id = gt_text_ids[0]  # Get the first (and only) text_id
            # Check if ground truth text is in top-k predictions
            for k in k_values:
                pred_k = set(topk_text_ids[:k])
                if gt_text_id in pred_k:
                    # Hit: ground truth text found in top-k
                    recall_k = 1.0
                else:
                    # Miss: ground truth text not found in top-k
                    recall_k = 0.0
                recalls[f'recall@{k}'].append(recall_k)
                
                # Debug print for first few samples
                if len(recalls[f'recall@{k}']) <= 3:
                    print(f'\n  Image-to-Text Recall@{k} calculation for image_id: {img_id_normalized}')
                    print(f'    Ground truth text: {gt_text_id}')
                    print(f'    Top-{k} predictions: {sorted(list(pred_k))}')
                    print(f'    Recall@{k}: {recall_k:.4f} ({"HIT" if recall_k > 0 else "MISS"})')
    
    # Average recalls
    metrics = {}
    for k in k_values:
        if recalls[f'recall@{k}']:
            avg_recall = np.mean(recalls[f'recall@{k}'])
            metrics[f'recall@{k}'] = avg_recall
            print(f'\n  Image-to-Text Recall@{k} summary:')
            print(f'    Total samples: {len(recalls[f"recall@{k}"])}')
            print(f'    Hits: {int(sum(recalls[f"recall@{k}"]))}')
            print(f'    Misses: {len(recalls[f"recall@{k}"]) - int(sum(recalls[f"recall@{k}"]))}')
            print(f'    Average Recall@{k}: {avg_recall:.4f}')
        else:
            metrics[f'recall@{k}'] = 0.0
    
    return metrics, predictions


def load_ground_truth(test_texts_jsonl):
    """Load ground truth from test_texts.jsonl"""
    ground_truth = {}
    with open(test_texts_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            image_ids = obj.get('image_ids', [])
            ground_truth[text_id] = image_ids
    return ground_truth


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text-to-image mapping module')
    
    # Data paths
    parser.add_argument('--test-texts', type=str, required=True,
                        help='Path to test_texts.jsonl')
    parser.add_argument('--create-jsonl', type=str, default='create.jsonl',
                        help='Path to create.jsonl with component annotations')
    parser.add_argument('--text-features-dir', type=str, default='features',
                        help='Directory containing text feature files')
    parser.add_argument('--image-features-dir', type=str, default='features',
                        help='Directory containing image feature files')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained mapping module checkpoint')
    
    # Mapping module hyperparameters (must match training)
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
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='outputs/eval',
                        help='Output directory for results')
    
    # Misc
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    print('Loading checkpoint...')
    mapping_module = load_checkpoint(args.checkpoint, device, args)
    
    # Create data loader
    print('Creating data loader...')
    test_loader = get_data_loader(
        texts_jsonl=args.test_texts,
        create_jsonl=args.create_jsonl,
        text_features_dir=args.text_features_dir,
        image_features_dir=args.image_features_dir,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        component_types=args.component_types
    )
    
    # Load ground truth first to get test set image and text IDs
    print('Loading ground truth...')
    ground_truth = load_ground_truth(args.test_texts)
    print(f'Loaded ground truth for {len(ground_truth)} texts')
    
    # Extract mapped features (only for test set texts)
    print('Extracting mapped features...')
    mapped_features, text_ids = extract_mapped_features(mapping_module, test_loader, device)
    print(f'Extracted {len(mapped_features)} mapped features from test set')
    
    # Filter to only test set text IDs
    test_text_ids = set(ground_truth.keys())
    filtered_mapped_features = {tid: mapped_features[tid] for tid in text_ids if tid in test_text_ids}
    filtered_text_ids = [tid for tid in text_ids if tid in test_text_ids]
    print(f'Filtered to {len(filtered_mapped_features)} test set text features')
    
    mapped_features = filtered_mapped_features
    text_ids = filtered_text_ids
    
    # Get all image IDs that appear in test set (use normalized format)
    test_image_ids_set = set()
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
            # Normalize to 6-digit string format
            try:
                if isinstance(img_id, str):
                    num = int(img_id)
                else:
                    num = int(img_id)
                img_id_normalized = f"{num:06d}"
                test_image_ids_set.add(img_id_normalized)
            except:
                test_image_ids_set.add(str(img_id).zfill(6))
    
    print(f'Test set contains {len(test_image_ids_set)} unique image IDs')
    
    # Load image features
    print('Loading image features...')
    all_image_features = load_image_features(args.image_features_dir, args.component_types)
    print(f'Loaded {len(all_image_features)} total image features')
    
    # Filter to only test set images
    # Create a mapping from normalized ID to actual feature key
    image_features = {}
    image_id_mapping = {}  # normalized_id -> actual_key_in_features
    
    for normalized_id in test_image_ids_set:
        # Try to find in all features with different formats
        found = False
        try:
            img_id_num = int(normalized_id)
            
            # Try direct match
            if normalized_id in all_image_features:
                image_features[normalized_id] = all_image_features[normalized_id]
                image_id_mapping[normalized_id] = normalized_id
                found = True
            else:
                # Try to find by converting keys to int
                for key in all_image_features.keys():
                    try:
                        if int(key) == img_id_num:
                            image_features[normalized_id] = all_image_features[key]
                            image_id_mapping[normalized_id] = key
                            found = True
                            break
                    except:
                        pass
        except:
            pass
        
        if not found:
            print(f'  Warning: Image ID {normalized_id} not found in features')
    
    image_ids = sorted(list(image_features.keys()))
    print(f'Filtered to {len(image_features)} unique image features from test set')
    print(f'  Expected: {len(test_image_ids_set)}, Found: {len(image_features)}')
    
    # Compute text-to-image retrieval metrics
    print('Computing text-to-image retrieval metrics...')
    
    # Debug: Check if ground truth images are in the retrieval pool
    gt_images_in_pool = 0
    gt_images_total = 0
    for text_id, gt_img_ids in ground_truth.items():
        for img_id in gt_img_ids:
            gt_images_total += 1
            img_id_str = str(img_id).zfill(6)
            img_id_orig = str(img_id)
            if img_id_str in image_features or img_id_orig in image_features:
                gt_images_in_pool += 1
            else:
                # Try to find by converting to int
                try:
                    img_id_num = int(img_id)
                    for key in image_features.keys():
                        try:
                            if int(key) == img_id_num:
                                gt_images_in_pool += 1
                                break
                        except:
                            pass
                except:
                    pass
    
    print(f'  Ground truth images in pool: {gt_images_in_pool}/{gt_images_total} ({gt_images_in_pool/gt_images_total*100:.2f}%)')
    
    # Debug: Check a sample
    sample_text_id = list(ground_truth.keys())[0] if ground_truth else None
    if sample_text_id and sample_text_id in mapped_features:
        print(f'\nDebug - Sample text_id: {sample_text_id}')
        print(f'  Ground truth image_ids: {ground_truth[sample_text_id]}')
        gt_formatted = [str(img_id).zfill(6) for img_id in ground_truth[sample_text_id]]
        print(f'  Formatted GT: {gt_formatted}')
        in_pool = [img_id for img_id in gt_formatted if img_id in image_features]
        print(f'  GT images in pool: {in_pool} ({len(in_pool)}/{len(gt_formatted)})')
    
    metrics_t2i, predictions_t2i = compute_retrieval_metrics(
        mapped_features, image_features, text_ids, image_ids, ground_truth
    )
    
    # Debug: Check predictions
    if sample_text_id and sample_text_id in predictions_t2i:
        print(f'  Top-5 predictions: {predictions_t2i[sample_text_id][:5]}')
        gt_set = set([str(img_id).zfill(6) for img_id in ground_truth[sample_text_id]])
        pred_set = set(predictions_t2i[sample_text_id][:5])
        intersection = gt_set & pred_set
        print(f'  Intersection: {intersection}')
        if len(gt_set) > 0:
            print(f'  Recall@5: {len(intersection) / len(gt_set):.4f}')
    
    print('\nText-to-Image Retrieval Metrics:')
    for k, v in metrics_t2i.items():
        print(f'  {k}: {v:.4f}')
    
    # Compute image-to-text retrieval metrics
    print('\nComputing image-to-text retrieval metrics...')
    metrics_i2t, predictions_i2t = compute_reverse_retrieval_metrics(
        mapped_features, image_features, text_ids, image_ids, ground_truth
    )
    
    print('\nImage-to-Text Retrieval Metrics:')
    for k, v in metrics_i2t.items():
        print(f'  {k}: {v:.4f}')
    
    # Save results
    print('\nSaving results...')
    
    # Save metrics
    all_metrics = {
        'text_to_image': metrics_t2i,
        'image_to_text': metrics_i2t
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f'Saved metrics to {metrics_path}')
    
    # Save predictions (text-to-image)
    predictions_path = os.path.join(args.output_dir, 'test_predictions.jsonl')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for text_id in sorted(predictions_t2i.keys()):
            # Remove duplicates and limit to top-10
            image_ids = predictions_t2i[text_id][:10]
            # Ensure no duplicates (should already be deduplicated, but double-check)
            seen = set()
            unique_image_ids = []
            for img_id in image_ids:
                if img_id not in seen:
                    unique_image_ids.append(img_id)
                    seen.add(img_id)
            
            pred = {
                'text_id': text_id,
                'image_ids': unique_image_ids
            }
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f'Saved text-to-image predictions to {predictions_path}')
    
    # Save predictions (image-to-text)
    predictions_i2t_path = os.path.join(args.output_dir, 'image_to_text_predictions.jsonl')
    with open(predictions_i2t_path, 'w', encoding='utf-8') as f:
        for image_id in sorted(predictions_i2t.keys()):
            pred = {
                'image_id': image_id,
                'text_ids': predictions_i2t[image_id][:10]  # Top-10
            }
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f'Saved image-to-text predictions to {predictions_i2t_path}')
    
    # Generate detailed reports with checkmarks
    print('\nGenerating detailed reports...')
    
    # Build reverse ground truth for i2t report
    reverse_gt_for_report = {}
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
            try:
                if isinstance(img_id, str):
                    num = int(img_id)
                else:
                    num = int(img_id)
                img_id_normalized = f"{num:06d}"
            except:
                img_id_normalized = str(img_id).zfill(6)
            
            if img_id_normalized not in reverse_gt_for_report:
                reverse_gt_for_report[img_id_normalized] = []
            if text_id not in reverse_gt_for_report[img_id_normalized]:
                reverse_gt_for_report[img_id_normalized].append(text_id)
    
    # Generate reports
    t2i_report_path = os.path.join(args.output_dir, 't2i_detailed_report.txt')
    generate_detailed_report_t2i(predictions_t2i, ground_truth, k_values=[1, 5, 10], output_file=t2i_report_path)
    
    i2t_report_path = os.path.join(args.output_dir, 'i2t_detailed_report.txt')
    generate_detailed_report_i2t(predictions_i2t, reverse_gt_for_report, k_values=[1, 5, 10], output_file=i2t_report_path)
    
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()

