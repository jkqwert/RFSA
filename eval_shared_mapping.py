# -*- coding: utf-8 -*-
'''
Evaluation script for shared feature space mapping
Evaluates retrieval performance in the shared feature space
'''

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from mapping_model_shared import SharedMappingModule
from data_loader import get_data_loader


def normalize_feature_key(key):
    """
    Normalize feature keys to handle both old and new formats
    Old format: "0", "1", "2", ...
    New format: "tensor(0)", "tensor(1)", "tensor(2)", ...
    
    Returns normalized integer string
    """
    key_str = str(key)
    
    # Check if it's the new tensor format
    if key_str.startswith('tensor(') and key_str.endswith(')'):
        # Extract the number from "tensor(123)"
        try:
            num_str = key_str[7:-1]  # Remove "tensor(" and ")"
            return num_str
        except:
            pass
    
    # Return as-is for old format or other formats
    return key_str


def load_checkpoint(checkpoint_path, device, args):
    """Load trained mapping module from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    mapping_module = SharedMappingModule(
        embed_dim=args.embed_dim,
        prompt_length=args.prompt_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        component_types=args.component_types
    ).to(device)
    
    mapping_module.load_state_dict(checkpoint['model_state_dict'])
    mapping_module.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, val_loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    return mapping_module


def extract_shared_features(mapping_module, data_loader, device, mode='text'):
    """
    Extract features in shared space
    
    Args:
        mapping_module: Trained mapping module
        data_loader: Data loader
        device: Device
        mode: 'text' or 'image'
    
    Returns:
        features_dict: Dictionary of {id: feature}
        ids_list: List of IDs
    """
    mapping_module.eval()
    features_dict = {}
    ids_list = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'Extracting {mode} features')
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
            
            if mode == 'text':
                ids = batch['text_ids']
                shared_feat = mapping_module.forward_text(text_features)
            else:  # image
                ids = batch['text_ids']  # Use text_id as identifier for image
                shared_feat = mapping_module.forward_image(image_features)
            
            # Store features
            for feat_id, feat in zip(ids, shared_feat.cpu().numpy()):
                features_dict[feat_id] = feat
                ids_list.append(feat_id)
    
    return features_dict, ids_list


def load_all_image_features(image_features_dir, component_types, mapping_module, device):
    """
    Load all image features and map to shared space
    
    Args:
        image_features_dir: Directory containing image feature files
        component_types: List of component types
        mapping_module: Trained mapping module
        device: Device
    
    Returns:
        image_features_dict: Dictionary of {image_id: feature_in_shared_space}
    """
    mapping_module.eval()
    image_features_dict = {}
    image_type_map = {
        'subject': 'subject',
        'object': 'object',
        'second': 'second_object',
        'relation': 'relation'
    }
    
    # Load features for each component
    all_image_features = {}
    all_image_ids = set()
    
    for comp_type in component_types:
        image_type = image_type_map[comp_type]
        feature_file = os.path.join(image_features_dir, f"{image_type}_features.json")
        
        if os.path.exists(feature_file):
            with open(feature_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
                for img_id, feat in features.items():
                    # Normalize the image ID to handle both old and new formats
                    normalized_img_id = normalize_feature_key(img_id)
                    
                    if normalized_img_id not in all_image_features:
                        all_image_features[normalized_img_id] = {}
                    all_image_features[normalized_img_id][comp_type] = np.array(feat)
                    all_image_ids.add(normalized_img_id)
    
    # Map all images to shared space
    print(f'Mapping {len(all_image_ids)} images to shared space...')
    batch_size = 64
    image_ids_list = sorted(list(all_image_ids))
    
    with torch.no_grad():
        for i in range(0, len(image_ids_list), batch_size):
            batch_ids = image_ids_list[i:i+batch_size]
            
            # Prepare batch
            batch_image_features = {}
            for comp_type in component_types:
                batch_feats = []
                for img_id in batch_ids:
                    if img_id in all_image_features and comp_type in all_image_features[img_id]:
                        batch_feats.append(all_image_features[img_id][comp_type])
                    else:
                        # Use zero vector if missing
                        batch_feats.append(np.zeros(512))
                
                batch_image_features[comp_type] = torch.FloatTensor(np.array(batch_feats)).to(device)
            
            # Map to shared space
            image_shared = mapping_module.forward_image(batch_image_features)
            
            # Store features
            for img_id, feat in zip(batch_ids, image_shared.cpu().numpy()):
                image_features_dict[img_id] = feat
    
    print(f'Mapped {len(image_features_dict)} images to shared space')
    return image_features_dict


def load_ground_truth(texts_jsonl):
    """Load ground truth text-image pairs"""
    ground_truth = {}
    with open(texts_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            image_ids = obj.get('image_ids', [])
            ground_truth[text_id] = image_ids
    return ground_truth


def compute_retrieval_metrics(text_features, image_features, text_ids, image_ids, ground_truth):
    """
    Compute retrieval metrics (Recall@K) in shared feature space
    
    Args:
        text_features: Dictionary of {text_id: feature}
        image_features: Dictionary of {image_id: feature}
        text_ids: List of text IDs
        image_ids: List of image IDs
        ground_truth: Ground truth dictionary {text_id: [image_ids]}
    
    Returns:
        metrics: Dictionary of metrics
        predictions: Dictionary of predictions
    """
    # Convert to numpy arrays
    text_vectors = np.array([text_features[tid] for tid in text_ids])
    image_vectors = np.array([image_features[iid] for iid in image_ids])
    
    # Create mapping
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_vectors, image_vectors)
    
    # Get top-k predictions for each text
    k_values = [1, 5, 10]
    predictions_t2i = {}
    recalls_t2i = {f'recall@{k}': [] for k in k_values}
    
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
        
        predictions_t2i[text_id] = formatted_image_ids
        
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
                except:
                    gt_image_ids.add(str(img_id))
            
            for k in k_values:
                pred_k = set(formatted_image_ids[:k])
                intersection = pred_k & gt_image_ids
                # Text-to-Image: one-to-many, if any GT image found, it's a hit
                if len(intersection) > 0:
                    recall_k = 1.0
                else:
                    recall_k = 0.0
                recalls_t2i[f'recall@{k}'].append(recall_k)
    
    # Average recalls
    metrics_t2i = {}
    for k in k_values:
        if recalls_t2i[f'recall@{k}']:
            avg_recall = np.mean(recalls_t2i[f'recall@{k}'])
            metrics_t2i[f'recall@{k}'] = avg_recall
        else:
            metrics_t2i[f'recall@{k}'] = 0.0
    
    return metrics_t2i, predictions_t2i


def compute_reverse_retrieval_metrics(text_features, image_features, text_ids, image_ids, ground_truth):
    """
    Compute reverse retrieval metrics (Image-to-Text, Recall@K) in shared feature space
    
    Args:
        text_features: Dictionary of {text_id: feature}
        image_features: Dictionary of {image_id: feature}
        text_ids: List of text IDs
        image_ids: List of image IDs
        ground_truth: Ground truth dictionary {text_id: [image_ids]}
    
    Returns:
        metrics: Dictionary of metrics
        predictions: Dictionary of predictions
    """
    # Convert to numpy arrays
    text_vectors = np.array([text_features[tid] for tid in text_ids])
    image_vectors = np.array([image_features[iid] for iid in image_ids])
    
    # Create mapping
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Compute similarity matrix (transpose for image-to-text)
    similarity_matrix = cosine_similarity(image_vectors, text_vectors)
    
    # Build reverse ground truth: image_id -> [text_ids]
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
            
            if img_id_normalized not in reverse_gt:
                reverse_gt[img_id_normalized] = []
            if text_id not in reverse_gt[img_id_normalized]:
                reverse_gt[img_id_normalized].append(text_id)
    
    # Get top-k predictions for each image
    k_values = [1, 5, 10]
    predictions_i2t = {}
    recalls_i2t = {f'recall@{k}': [] for k in k_values}
    
    for image_id in image_ids:
        if image_id not in image_id_to_idx:
            continue
        
        image_idx = image_id_to_idx[image_id]
        similarities = similarity_matrix[image_idx]
        
        # Get top-k text indices
        topk_indices = np.argsort(similarities)[::-1]
        
        # Get top-k text IDs with deduplication
        seen_ids = set()
        formatted_text_ids = []
        for idx in topk_indices:
            if len(formatted_text_ids) >= max(k_values):
                break
            text_id = text_ids[idx]
            
            if text_id not in seen_ids:
                formatted_text_ids.append(text_id)
                seen_ids.add(text_id)
        
        predictions_i2t[image_id] = formatted_text_ids
        
        # Compute recall@K
        if image_id in reverse_gt:
            gt_text_ids = reverse_gt[image_id]
            gt_text_id = gt_text_ids[0] if gt_text_ids else None  # One-to-one
            
            for k in k_values:
                pred_k = set(formatted_text_ids[:k])
                hit = gt_text_id in pred_k if gt_text_id else False
                recall_k = 1.0 if hit else 0.0
                recalls_i2t[f'recall@{k}'].append(recall_k)
    
    # Average recalls
    metrics_i2t = {}
    for k in k_values:
        if recalls_i2t[f'recall@{k}']:
            avg_recall = np.mean(recalls_i2t[f'recall@{k}'])
            metrics_i2t[f'recall@{k}'] = avg_recall
        else:
            metrics_i2t[f'recall@{k}'] = 0.0
    
    return metrics_i2t, predictions_i2t


def generate_detailed_report_t2i(predictions, ground_truth, k_values=[1, 5, 10], output_file='t2i_detailed_report.txt'):
    """Generate detailed report with checkmarks for text-to-image retrieval"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('Text-to-Image Retrieval Detailed Report (Shared Feature Space)\n')
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
        f.write('Image-to-Text Retrieval Detailed Report (Shared Feature Space)\n')
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


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate shared feature space mapping')
    
    # Data arguments
    parser.add_argument('--test-texts', type=str, required=True,
                        help='Path to test_texts.jsonl')
    parser.add_argument('--create-jsonl', type=str, required=True,
                        help='Path to create.jsonl')
    parser.add_argument('--text-features-dir', type=str, required=True,
                        help='Directory containing text feature files')
    parser.add_argument('--image-features-dir', type=str, required=True,
                        help='Directory containing image feature files')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
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
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading checkpoint...')
    mapping_module = load_checkpoint(args.checkpoint, device, args)
    
    print('Loading ground truth...')
    ground_truth = load_ground_truth(args.test_texts)
    print(f'Loaded ground truth for {len(ground_truth)} texts')
    
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
    
    print('Extracting text features in shared space...')
    text_features, text_ids = extract_shared_features(mapping_module, test_loader, device, mode='text')
    print(f'Extracted {len(text_features)} text features')
    
    # Filter text features to test set
    test_text_ids = set(ground_truth.keys())
    text_features = {tid: text_features[tid] for tid in text_ids if tid in test_text_ids}
    text_ids = [tid for tid in text_ids if tid in test_text_ids]
    print(f'Filtered to {len(text_ids)} test set text features')
    
    # Get test set image IDs
    test_image_ids_set = set()
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
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
    
    # Load all image features and map to shared space
    print('Loading all image features and mapping to shared space...')
    all_image_features = load_all_image_features(
        args.image_features_dir,
        args.component_types,
        mapping_module,
        device
    )
    
    # Normalize image feature IDs to 6-digit strings for reliable matching
    normalized_image_features = {}
    for raw_id, feat in all_image_features.items():
        try:
            norm_id = f"{int(raw_id):06d}"
        except Exception:
            norm_id = str(raw_id).zfill(6)
        normalized_image_features[norm_id] = feat
    
    # Filter to test set images
    image_features = {
        img_id: normalized_image_features[img_id]
        for img_id in test_image_ids_set
        if img_id in normalized_image_features
    }
    image_ids = sorted(list(image_features.keys()))
    
    missing = len(test_image_ids_set) - len(image_features)
    if missing > 0:
        print(f'Warning: {missing} test images not found in features')
    
    print(f'Filtered to {len(image_features)} unique image features from test set')
    print(f'Test set: {len(text_ids)} texts, {len(image_ids)} images')
    
    print('Computing text-to-image retrieval metrics...')
    metrics_t2i, predictions_t2i = compute_retrieval_metrics(
        text_features, image_features, text_ids, image_ids, ground_truth
    )
    
    print('\nText-to-Image Retrieval Metrics:')
    for k, v in metrics_t2i.items():
        print(f'  {k}: {v:.4f}')
    
    print('\nComputing image-to-text retrieval metrics...')
    metrics_i2t, predictions_i2t = compute_reverse_retrieval_metrics(
        text_features, image_features, text_ids, image_ids, ground_truth
    )
    
    print('\nImage-to-Text Retrieval Metrics:')
    for k, v in metrics_i2t.items():
        print(f'  {k}: {v:.4f}')
    
    print('\nSaving results...')
    
    all_metrics = {
        'text_to_image': metrics_t2i,
        'image_to_text': metrics_i2t
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f'Saved metrics to {metrics_path}')
    
    predictions_path = os.path.join(args.output_dir, 'test_predictions.jsonl')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for text_id in sorted(predictions_t2i.keys()):
            pred = {
                'text_id': text_id,
                'image_ids': predictions_t2i[text_id][:10]
            }
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f'Saved text-to-image predictions to {predictions_path}')
    
    predictions_i2t_path = os.path.join(args.output_dir, 'image_to_text_predictions.jsonl')
    with open(predictions_i2t_path, 'w', encoding='utf-8') as f:
        for image_id in sorted(predictions_i2t.keys()):
            pred = {
                'image_id': image_id,
                'text_ids': predictions_i2t[image_id][:10]
            }
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f'Saved image-to-text predictions to {predictions_i2t_path}')
    
    print('\nGenerating detailed reports...')
    
    # Build reverse ground truth for report
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
    
    # Generate text-to-image detailed report
    t2i_report_path = os.path.join(args.output_dir, 't2i_detailed_report.txt')
    generate_detailed_report_t2i(predictions_t2i, ground_truth, k_values=[1, 5, 10], output_file=t2i_report_path)
    
    # Generate image-to-text detailed report
    i2t_report_path = os.path.join(args.output_dir, 'i2t_detailed_report.txt')
    generate_detailed_report_i2t(predictions_i2t, reverse_gt_for_report, k_values=[1, 5, 10], output_file=i2t_report_path)
    
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()

