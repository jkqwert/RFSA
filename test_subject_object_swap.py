# -*- coding: utf-8 -*-
'''
Test script for subject-object swap evaluation
Tests retrieval performance when subject and object are swapped
'''

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity

from mapping_model_shared import SharedMappingModule
from eval_utils import compute_retrieval_metrics_t2i, compute_retrieval_metrics_i2t
from eval_utils import generate_detailed_report_t2i, generate_detailed_report_i2t


def load_checkpoint(checkpoint_path, device):
    """Load trained mapping module from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get args from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        if isinstance(args, dict):
            embed_dim = args.get('embed_dim', 512)
            prompt_length = args.get('prompt_length', 4)
            hidden_dim = args.get('hidden_dim', 512)
            num_layers = args.get('num_layers', 2)
            dropout = args.get('dropout', 0.1)
            fusion_method = args.get('fusion_method', 'weighted_sum')
            component_types = args.get('component_types', ['subject', 'object', 'second', 'relation'])
        else:
            embed_dim = getattr(args, 'embed_dim', 512)
            prompt_length = getattr(args, 'prompt_length', 4)
            hidden_dim = getattr(args, 'hidden_dim', 512)
            num_layers = getattr(args, 'num_layers', 2)
            dropout = getattr(args, 'dropout', 0.1)
            fusion_method = getattr(args, 'fusion_method', 'weighted_sum')
            component_types = getattr(args, 'component_types', ['subject', 'object', 'second', 'relation'])
    else:
        # Default values
        embed_dim = 512
        prompt_length = 4
        hidden_dim = 512
        num_layers = 2
        dropout = 0.1
        fusion_method = 'weighted_sum'
        component_types = ['subject', 'object', 'second', 'relation']
    
    mapping_module = SharedMappingModule(
        embed_dim=embed_dim,
        prompt_length=prompt_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        fusion_method=fusion_method,
        component_types=component_types
    ).to(device)
    
    mapping_module.load_state_dict(checkpoint['model_state_dict'])
    mapping_module.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, val_loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    return mapping_module, component_types


def _load_features_file(file_path):
    """Load features from JSON or JSONL file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                # Try JSONL format
                f.seek(0)
                data = {}
                for line in f:
                    obj = json.loads(line.strip())
                    if 'text_id' in obj and 'feature' in obj:
                        data[obj['text_id']] = obj['feature']
                    elif len(obj) == 2:
                        # Assume first key is ID, second is feature
                        key = list(obj.keys())[0]
                        data[key] = obj[key]
                return data
    except json.JSONDecodeError:
        # Try JSONL format
        with open(file_path, 'r', encoding='utf-8') as f:
            data = {}
            for line in f:
                obj = json.loads(line.strip())
                if 'text_id' in obj and 'feature' in obj:
                    data[obj['text_id']] = obj['feature']
                elif len(obj) == 2:
                    key = list(obj.keys())[0]
                    data[key] = obj[key]
            return data


def load_text_features(text_features_dir, component_types, text_ids, swap_subject_object=False):
    """
    Load text features for specified text IDs
    
    Args:
        text_features_dir: Directory containing text feature files
        component_types: List of component types
        text_ids: Set of text IDs to load
        swap_subject_object: If True, swap subject and object features
    
    Returns:
        text_features_dict: Dictionary of {text_id: {component: feature}}
    """
    text_features_dict = {}
    
    for comp_type in component_types:
        feature_file = os.path.join(text_features_dir, f"{comp_type}_text_features.json")
        
        if not os.path.exists(feature_file):
            print(f"Warning: {feature_file} not found, skipping {comp_type}")
            continue
        
        features = _load_features_file(feature_file)
        
        for text_id in text_ids:
            if text_id not in text_features_dict:
                text_features_dict[text_id] = {}
            
            if text_id in features:
                text_features_dict[text_id][comp_type] = np.array(features[text_id])
            else:
                # Use zero vector if missing
                text_features_dict[text_id][comp_type] = np.zeros(512)
    
    # Swap subject and object if requested
    if swap_subject_object and 'subject' in component_types and 'object' in component_types:
        print("Swapping subject and object features...")
        for text_id in text_features_dict:
            if 'subject' in text_features_dict[text_id] and 'object' in text_features_dict[text_id]:
                # Swap features
                subject_feat = text_features_dict[text_id]['subject'].copy()
                object_feat = text_features_dict[text_id]['object'].copy()
                text_features_dict[text_id]['subject'] = object_feat
                text_features_dict[text_id]['object'] = subject_feat
    
    return text_features_dict


def load_all_image_features(image_features_dir, component_types, mapping_module, device, swap_subject_object=False):
    """
    Load all image features and map to shared space
    
    Args:
        image_features_dir: Directory containing image feature files
        component_types: List of component types
        mapping_module: Trained mapping module
        device: Device
        swap_subject_object: If True, swap subject and object features
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
            features = _load_features_file(feature_file)
            for img_id, feat in features.items():
                if img_id not in all_image_features:
                    all_image_features[img_id] = {}
                all_image_features[img_id][comp_type] = np.array(feat)
                all_image_ids.add(img_id)
    
    # Swap subject and object if requested
    if swap_subject_object and 'subject' in component_types and 'object' in component_types:
        print("Swapping subject and object features for images...")
        for img_id in all_image_features:
            if 'subject' in all_image_features[img_id] and 'object' in all_image_features[img_id]:
                # Swap features
                subject_feat = all_image_features[img_id]['subject'].copy()
                object_feat = all_image_features[img_id]['object'].copy()
                all_image_features[img_id]['subject'] = object_feat
                all_image_features[img_id]['object'] = subject_feat
    
    # Map all images to shared space
    print(f'Mapping {len(all_image_ids)} images to shared space...')
    batch_size = 64
    image_ids_list = sorted(list(all_image_ids))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_ids_list), batch_size), desc='Mapping images'):
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


def map_text_features_to_shared_space(text_features_dict, text_ids, component_types, mapping_module, device):
    """
    Map text features to shared space
    
    Args:
        text_features_dict: Dictionary of {text_id: {component: feature}}
        text_ids: List of text IDs
        component_types: List of component types
        mapping_module: Trained mapping module
        device: Device
    
    Returns:
        shared_features_dict: Dictionary of {text_id: shared_feature}
    """
    mapping_module.eval()
    shared_features_dict = {}
    
    batch_size = 32
    text_ids_list = list(text_ids)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(text_ids_list), batch_size), desc='Mapping texts'):
            batch_ids = text_ids_list[i:i+batch_size]
            
            # Prepare batch
            batch_text_features = {}
            for comp_type in component_types:
                batch_feats = []
                for text_id in batch_ids:
                    if text_id in text_features_dict and comp_type in text_features_dict[text_id]:
                        batch_feats.append(text_features_dict[text_id][comp_type])
                    else:
                        # Use zero vector if missing
                        batch_feats.append(np.zeros(512))
                
                batch_text_features[comp_type] = torch.FloatTensor(np.array(batch_feats)).to(device)
            
            # Map to shared space
            text_shared = mapping_module.forward_text(batch_text_features)
            
            # Store features
            for text_id, feat in zip(batch_ids, text_shared.cpu().numpy()):
                shared_features_dict[text_id] = feat
    
    return shared_features_dict


def load_ground_truth(texts_jsonl):
    """Load ground truth text-image pairs"""
    ground_truth = {}
    with open(texts_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            image_ids = obj.get('image_ids', [])
            # Normalize image IDs to 6-digit string format
            normalized_ids = []
            for img_id in image_ids:
                try:
                    if isinstance(img_id, str):
                        num = int(img_id)
                    else:
                        num = int(img_id)
                    normalized_ids.append(f"{num:06d}")
                except:
                    normalized_ids.append(str(img_id))
            ground_truth[text_id] = normalized_ids
    return ground_truth


def normalize_image_id(img_id):
    """Normalize image ID to 6-digit string format"""
    try:
        if isinstance(img_id, str):
            num = int(img_id)
        else:
            num = int(img_id)
        return f"{num:06d}"
    except:
        return str(img_id)


def main():
    parser = argparse.ArgumentParser(description='Test subject-object swap evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-texts', type=str, required=True,
                        help='Path to test texts JSONL file')
    parser.add_argument('--create-jsonl', type=str, required=True,
                        help='Path to create.jsonl file with component annotations')
    parser.add_argument('--text-features-dir', type=str, required=True,
                        help='Directory containing text feature files')
    parser.add_argument('--image-features-dir', type=str, required=True,
                        help='Directory containing image feature files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature mapping')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print("Loading checkpoint...")
    mapping_module, component_types = load_checkpoint(args.checkpoint, device)
    
    # Load ground truth
    print("Loading ground truth...")
    ground_truth = load_ground_truth(args.test_texts)
    
    # Get test text IDs
    test_text_ids = set(ground_truth.keys())
    print(f"Loaded {len(test_text_ids)} test samples")
    
    # Filter image IDs to test set
    test_image_ids = set()
    for text_id, img_ids in ground_truth.items():
        test_image_ids.update(img_ids)
    
    print(f"Test set contains {len(test_image_ids)} unique image IDs")
    
    # Test 1: Original (no swap)
    print("\n" + "="*80)
    print("Test 1: Original (No Swap)")
    print("="*80)
    
    # Load original image features
    print("Loading original image features...")
    all_image_features_original = load_all_image_features(
        args.image_features_dir,
        component_types,
        mapping_module,
        device,
        swap_subject_object=False
    )
    
    # Normalize image IDs
    normalized_image_features_original = {}
    for img_id, feat in all_image_features_original.items():
        normalized_id = normalize_image_id(img_id)
        normalized_image_features_original[normalized_id] = feat
    
    # Only keep images that are in test set
    test_image_features_original = {img_id: normalized_image_features_original[img_id] 
                                    for img_id in test_image_ids 
                                    if img_id in normalized_image_features_original}
    print(f"Loaded {len(test_image_features_original)} image features for test set")
    
    original_text_features = load_text_features(
        args.text_features_dir,
        component_types,
        test_text_ids,
        swap_subject_object=False
    )
    
    original_shared_features = map_text_features_to_shared_space(
        original_text_features,
        test_text_ids,
        component_types,
        mapping_module,
        device
    )
    
    # Compute metrics
    original_metrics_t2i, original_predictions_t2i = compute_retrieval_metrics_t2i(
        original_shared_features,
        test_image_features_original,
        list(test_text_ids),
        list(test_image_features_original.keys()),
        ground_truth
    )
    
    original_metrics_i2t, original_predictions_i2t, original_reverse_gt = compute_retrieval_metrics_i2t(
        original_shared_features,
        test_image_features_original,
        list(test_text_ids),
        list(test_image_features_original.keys()),
        ground_truth
    )
    
    print("\nOriginal Results (Text-to-Image):")
    for k, v in original_metrics_t2i.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nOriginal Results (Image-to-Text):")
    for k, v in original_metrics_i2t.items():
        print(f"  {k}: {v:.4f}")
    
    # Test 2: Text Swapped (only text subject and object swapped)
    print("\n" + "="*80)
    print("Test 2: Text Swapped (Only Text Subject and Object Swapped)")
    print("="*80)
    
    text_swapped_text_features = load_text_features(
        args.text_features_dir,
        component_types,
        test_text_ids,
        swap_subject_object=True
    )
    
    text_swapped_shared_features = map_text_features_to_shared_space(
        text_swapped_text_features,
        test_text_ids,
        component_types,
        mapping_module,
        device
    )
    
    # Compute metrics (using original image features)
    text_swapped_metrics_t2i, text_swapped_predictions_t2i = compute_retrieval_metrics_t2i(
        text_swapped_shared_features,
        test_image_features_original,
        list(test_text_ids),
        list(test_image_features_original.keys()),
        ground_truth
    )
    
    text_swapped_metrics_i2t, text_swapped_predictions_i2t, text_swapped_reverse_gt = compute_retrieval_metrics_i2t(
        text_swapped_shared_features,
        test_image_features_original,
        list(test_text_ids),
        list(test_image_features_original.keys()),
        ground_truth
    )
    
    print("\nText Swapped Results (Text-to-Image):")
    for k, v in text_swapped_metrics_t2i.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nText Swapped Results (Image-to-Text):")
    for k, v in text_swapped_metrics_i2t.items():
        print(f"  {k}: {v:.4f}")
    
    # Test 3: Image Swapped (only image subject and object swapped)
    print("\n" + "="*80)
    print("Test 3: Image Swapped (Only Image Subject and Object Swapped)")
    print("="*80)
    
    # Load swapped image features
    print("Loading swapped image features...")
    all_image_features_swapped = load_all_image_features(
        args.image_features_dir,
        component_types,
        mapping_module,
        device,
        swap_subject_object=True
    )
    
    # Normalize image IDs
    normalized_image_features_swapped = {}
    for img_id, feat in all_image_features_swapped.items():
        normalized_id = normalize_image_id(img_id)
        normalized_image_features_swapped[normalized_id] = feat
    
    # Only keep images that are in test set
    test_image_features_swapped = {img_id: normalized_image_features_swapped[img_id] 
                                   for img_id in test_image_ids 
                                   if img_id in normalized_image_features_swapped}
    print(f"Loaded {len(test_image_features_swapped)} swapped image features for test set")
    
    # Use original text features
    image_swapped_text_features = load_text_features(
        args.text_features_dir,
        component_types,
        test_text_ids,
        swap_subject_object=False
    )
    
    image_swapped_shared_features = map_text_features_to_shared_space(
        image_swapped_text_features,
        test_text_ids,
        component_types,
        mapping_module,
        device
    )
    
    # Compute metrics (using swapped image features)
    image_swapped_metrics_t2i, image_swapped_predictions_t2i = compute_retrieval_metrics_t2i(
        image_swapped_shared_features,
        test_image_features_swapped,
        list(test_text_ids),
        list(test_image_features_swapped.keys()),
        ground_truth
    )
    
    image_swapped_metrics_i2t, image_swapped_predictions_i2t, image_swapped_reverse_gt = compute_retrieval_metrics_i2t(
        image_swapped_shared_features,
        test_image_features_swapped,
        list(test_text_ids),
        list(test_image_features_swapped.keys()),
        ground_truth
    )
    
    print("\nImage Swapped Results (Text-to-Image):")
    for k, v in image_swapped_metrics_t2i.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImage Swapped Results (Image-to-Text):")
    for k, v in image_swapped_metrics_i2t.items():
        print(f"  {k}: {v:.4f}")
    
    # Test 4: Both Swapped (both text and image subject and object swapped)
    print("\n" + "="*80)
    print("Test 4: Both Swapped (Both Text and Image Subject and Object Swapped)")
    print("="*80)
    
    both_swapped_text_features = load_text_features(
        args.text_features_dir,
        component_types,
        test_text_ids,
        swap_subject_object=True
    )
    
    both_swapped_shared_features = map_text_features_to_shared_space(
        both_swapped_text_features,
        test_text_ids,
        component_types,
        mapping_module,
        device
    )
    
    # Compute metrics (using swapped image features)
    both_swapped_metrics_t2i, both_swapped_predictions_t2i = compute_retrieval_metrics_t2i(
        both_swapped_shared_features,
        test_image_features_swapped,
        list(test_text_ids),
        list(test_image_features_swapped.keys()),
        ground_truth
    )
    
    both_swapped_metrics_i2t, both_swapped_predictions_i2t, both_swapped_reverse_gt = compute_retrieval_metrics_i2t(
        both_swapped_shared_features,
        test_image_features_swapped,
        list(test_text_ids),
        list(test_image_features_swapped.keys()),
        ground_truth
    )
    
    print("\nBoth Swapped Results (Text-to-Image):")
    for k, v in both_swapped_metrics_t2i.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nBoth Swapped Results (Image-to-Text):")
    for k, v in both_swapped_metrics_i2t.items():
        print(f"  {k}: {v:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison: All Configurations")
    print("="*80)
    
    print("\nText-to-Image Retrieval:")
    print(f"{'Metric':<20} {'Original':<15} {'Text Swap':<15} {'Image Swap':<15} {'Both Swap':<15}")
    print("-" * 80)
    for k in original_metrics_t2i.keys():
        orig_val = original_metrics_t2i[k]
        text_swap_val = text_swapped_metrics_t2i[k]
        img_swap_val = image_swapped_metrics_t2i[k]
        both_swap_val = both_swapped_metrics_t2i[k]
        print(f"{k:<20} {orig_val:<15.4f} {text_swap_val:<15.4f} {img_swap_val:<15.4f} {both_swap_val:<15.4f}")
    
    print("\nImage-to-Text Retrieval:")
    print(f"{'Metric':<20} {'Original':<15} {'Text Swap':<15} {'Image Swap':<15} {'Both Swap':<15}")
    print("-" * 80)
    for k in original_metrics_i2t.keys():
        orig_val = original_metrics_i2t[k]
        text_swap_val = text_swapped_metrics_i2t[k]
        img_swap_val = image_swapped_metrics_i2t[k]
        both_swap_val = both_swapped_metrics_i2t[k]
        print(f"{k:<20} {orig_val:<15.4f} {text_swap_val:<15.4f} {img_swap_val:<15.4f} {both_swap_val:<15.4f}")
    
    # Save results
    results = {
        'original': {
            't2i': original_metrics_t2i,
            'i2t': original_metrics_i2t
        },
        'text_swapped': {
            't2i': text_swapped_metrics_t2i,
            'i2t': text_swapped_metrics_i2t
        },
        'image_swapped': {
            't2i': image_swapped_metrics_t2i,
            'i2t': image_swapped_metrics_i2t
        },
        'both_swapped': {
            't2i': both_swapped_metrics_t2i,
            'i2t': both_swapped_metrics_i2t
        },
        'comparison': {
            't2i': {
                'text_swap_diff': {k: text_swapped_metrics_t2i[k] - original_metrics_t2i[k] 
                                  for k in original_metrics_t2i.keys()},
                'image_swap_diff': {k: image_swapped_metrics_t2i[k] - original_metrics_t2i[k] 
                                   for k in original_metrics_t2i.keys()},
                'both_swap_diff': {k: both_swapped_metrics_t2i[k] - original_metrics_t2i[k] 
                                  for k in original_metrics_t2i.keys()}
            },
            'i2t': {
                'text_swap_diff': {k: text_swapped_metrics_i2t[k] - original_metrics_i2t[k] 
                                  for k in original_metrics_i2t.keys()},
                'image_swap_diff': {k: image_swapped_metrics_i2t[k] - original_metrics_i2t[k] 
                                   for k in original_metrics_i2t.keys()},
                'both_swap_diff': {k: both_swapped_metrics_i2t[k] - original_metrics_i2t[k] 
                                  for k in original_metrics_i2t.keys()}
            }
        }
    }
    
    results_file = os.path.join(args.output_dir, 'swap_test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")
    
    # Generate detailed reports
    print("\nGenerating detailed reports...")
    
    # Original reports
    original_t2i_report = os.path.join(args.output_dir, 'original_t2i_detailed_report.txt')
    generate_detailed_report_t2i(
        original_predictions_t2i,
        ground_truth,
        k_values=[1, 5, 10],
        output_file=original_t2i_report
    )
    print(f"Original T2I report saved to {original_t2i_report}")
    
    original_i2t_report = os.path.join(args.output_dir, 'original_i2t_detailed_report.txt')
    generate_detailed_report_i2t(
        original_predictions_i2t,
        original_reverse_gt,
        k_values=[1, 5, 10],
        output_file=original_i2t_report
    )
    print(f"Original I2T report saved to {original_i2t_report}")
    
    # Text Swapped reports
    text_swapped_t2i_report = os.path.join(args.output_dir, 'text_swapped_t2i_detailed_report.txt')
    generate_detailed_report_t2i(
        text_swapped_predictions_t2i,
        ground_truth,
        k_values=[1, 5, 10],
        output_file=text_swapped_t2i_report
    )
    print(f"Text Swapped T2I report saved to {text_swapped_t2i_report}")
    
    text_swapped_i2t_report = os.path.join(args.output_dir, 'text_swapped_i2t_detailed_report.txt')
    generate_detailed_report_i2t(
        text_swapped_predictions_i2t,
        text_swapped_reverse_gt,
        k_values=[1, 5, 10],
        output_file=text_swapped_i2t_report
    )
    print(f"Text Swapped I2T report saved to {text_swapped_i2t_report}")
    
    # Image Swapped reports
    image_swapped_t2i_report = os.path.join(args.output_dir, 'image_swapped_t2i_detailed_report.txt')
    generate_detailed_report_t2i(
        image_swapped_predictions_t2i,
        ground_truth,
        k_values=[1, 5, 10],
        output_file=image_swapped_t2i_report
    )
    print(f"Image Swapped T2I report saved to {image_swapped_t2i_report}")
    
    image_swapped_i2t_report = os.path.join(args.output_dir, 'image_swapped_i2t_detailed_report.txt')
    generate_detailed_report_i2t(
        image_swapped_predictions_i2t,
        image_swapped_reverse_gt,
        k_values=[1, 5, 10],
        output_file=image_swapped_i2t_report
    )
    print(f"Image Swapped I2T report saved to {image_swapped_i2t_report}")
    
    # Both Swapped reports
    both_swapped_t2i_report = os.path.join(args.output_dir, 'both_swapped_t2i_detailed_report.txt')
    generate_detailed_report_t2i(
        both_swapped_predictions_t2i,
        ground_truth,
        k_values=[1, 5, 10],
        output_file=both_swapped_t2i_report
    )
    print(f"Both Swapped T2I report saved to {both_swapped_t2i_report}")
    
    both_swapped_i2t_report = os.path.join(args.output_dir, 'both_swapped_i2t_detailed_report.txt')
    generate_detailed_report_i2t(
        both_swapped_predictions_i2t,
        both_swapped_reverse_gt,
        k_values=[1, 5, 10],
        output_file=both_swapped_i2t_report
    )
    print(f"Both Swapped I2T report saved to {both_swapped_i2t_report}")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == '__main__':
    main()

