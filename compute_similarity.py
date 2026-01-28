# -*- coding: utf-8 -*-
'''
This script computes similarity scores between text and four types of image features,
then fuses them with different weights for composite retrieval.
'''

import os
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity


def load_features(feature_path):
    """Load features from JSON file"""
    with open(feature_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
    return features


def compute_similarity_matrix(text_features, image_features):
    """Compute cosine similarity matrix between text and image features"""
    text_ids = list(text_features.keys())
    image_ids = list(image_features.keys())
    
    # Convert to numpy arrays
    text_vectors = np.array([text_features[tid] for tid in text_ids])
    image_vectors = np.array([image_features[iid] for iid in image_ids])
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(text_vectors, image_vectors)
    
    return similarity_matrix, text_ids, image_ids


def fuse_similarities(similarities, weights, text_ids_list, image_ids_list):
    """Fuse multiple similarity matrices with weights"""
    # Get all unique text and image IDs
    all_text_ids = set()
    all_image_ids = set()
    
    for text_ids, image_ids in zip(text_ids_list, image_ids_list):
        all_text_ids.update(text_ids)
        all_image_ids.update(image_ids)
    
    all_text_ids = sorted(list(all_text_ids))
    all_image_ids = sorted(list(all_image_ids))
    
    # Create unified similarity matrix
    fused_similarity = np.zeros((len(all_text_ids), len(all_image_ids)))
    
    # Create mapping from ID to index
    text_id_to_idx = {tid: i for i, tid in enumerate(all_text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(all_image_ids)}
    
    # Add weighted similarities
    for sim, weight, text_ids, image_ids in zip(similarities, weights, text_ids_list, image_ids_list):
        if sim is not None:
            for i, text_id in enumerate(text_ids):
                for j, image_id in enumerate(image_ids):
                    if text_id in text_id_to_idx and image_id in image_id_to_idx:
                        text_idx = text_id_to_idx[text_id]
                        image_idx = image_id_to_idx[image_id]
                        fused_similarity[text_idx, image_idx] += weight * sim[i, j]
    
    return fused_similarity, all_text_ids, all_image_ids


def get_topk_predictions(similarity_matrix, text_ids, image_ids, k=10):
    """Get top-k predictions for each text"""
    predictions = {}
    
    for i, text_id in enumerate(text_ids):
        # Get top-k image indices
        topk_indices = np.argsort(similarity_matrix[i])[::-1][:k]
        topk_image_ids = [image_ids[idx] for idx in topk_indices]
        
        # Convert image IDs to proper format (6-digit string with leading zeros)
        formatted_image_ids = []
        for img_id in topk_image_ids:
            if isinstance(img_id, str) and img_id.startswith('tensor('):
                # Extract number from tensor string
                num = int(img_id[7:-1])  # Remove 'tensor(' and ')'
                formatted_id = f"{num:06d}"
            else:
                # Convert to int then format
                try:
                    num = int(img_id)
                    formatted_id = f"{num:06d}"
                except:
                    formatted_id = str(img_id)
            formatted_image_ids.append(formatted_id)
        
        predictions[text_id] = formatted_image_ids
    
    return predictions


def get_topk_predictions_reverse(similarity_matrix, text_ids, image_ids, k=10):
    """Get top-k predictions for each image (image-to-text)"""
    predictions = {}
    
    for j, image_id in enumerate(image_ids):
        # Get top-k text indices
        topk_indices = np.argsort(similarity_matrix[:, j])[::-1][:k]
        topk_text_ids = [text_ids[idx] for idx in topk_indices]
        
        # Convert image ID to proper format
        if isinstance(image_id, str) and image_id.startswith('tensor('):
            num = int(image_id[7:-1])
            formatted_image_id = f"{num:06d}"
        else:
            try:
                num = int(image_id)
                formatted_image_id = f"{num:06d}"
            except:
                formatted_image_id = str(image_id)
        
        predictions[formatted_image_id] = topk_text_ids
    
    return predictions


def compute_retrieval_metrics(text_to_image_preds, image_to_text_preds, ground_truth, k_values=[1, 5, 10]):
    """Compute retrieval metrics (Recall@K) for both directions"""
    metrics = {}
    
    # Text-to-Image retrieval
    for k in k_values:
        correct = 0
        total = len(ground_truth)
        
        for text_id, pred_image_ids in text_to_image_preds.items():
            if text_id in ground_truth:
                gt_image_ids = set(ground_truth[text_id])
                pred_topk = set(pred_image_ids[:k])
                
                if gt_image_ids.intersection(pred_topk):
                    correct += 1
        
        recall = correct / total if total > 0 else 0
        metrics[f'text_to_image_recall@{k}'] = recall
    
    # Image-to-Text retrieval
    # Create reverse ground truth mapping
    image_to_text_gt = {}
    for text_id, image_ids in ground_truth.items():
        for image_id in image_ids:
            if image_id not in image_to_text_gt:
                image_to_text_gt[image_id] = []
            image_to_text_gt[image_id].append(text_id)
    
    for k in k_values:
        correct = 0
        total = len(image_to_text_gt)
        
        for image_id, pred_text_ids in image_to_text_preds.items():
            if image_id in image_to_text_gt:
                gt_text_ids = set(image_to_text_gt[image_id])
                pred_topk = set(pred_text_ids[:k])
                
                if gt_text_ids.intersection(pred_topk):
                    correct += 1
        
        recall = correct / total if total > 0 else 0
        metrics[f'image_to_text_recall@{k}'] = recall
    
    return metrics


def load_ground_truth(gt_file):
    """Load ground truth from test_texts.jsonl"""
    ground_truth = {}
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            image_ids = obj['image_ids']
            ground_truth[text_id] = image_ids
    
    return ground_truth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', type=str, default='features',
                        help='Directory containing feature files')
    parser.add_argument('--subject-features', type=str, default='features/subject_features.json',
                        help='Path to subject image features file')
    parser.add_argument('--object-features', type=str, default='features/object_features.json',
                        help='Path to object image features file')
    parser.add_argument('--second-object-features', type=str, default='features/second_object_features.json',
                        help='Path to second object image features file')
    parser.add_argument('--relation-features', type=str, default='features/relation_features.json',
                        help='Path to relation image features file')
    parser.add_argument('--ground-truth', type=str, default='Dataloader/datasets/new/test_texts.jsonl',
                        help='Path to ground truth file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--weights', type=float, nargs=4, default=[0.4, 0.3, 0.2, 0.1],
                        help='Weights for [subject, object, second_object, relation] similarities')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top predictions to return')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 5, 10],
                        help='K values for recall computation')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Loading features...")
    
    # Load text and image features for each type
    feature_types = ['subject', 'object', 'second_object', 'relation']
    text_feature_paths = [
        f"{args.features_dir}/subject_text_features.json",
        f"{args.features_dir}/object_text_features.json", 
        f"{args.features_dir}/second_text_features.json",
        f"{args.features_dir}/relation_text_features.json"
    ]
    image_feature_paths = [
        args.subject_features,
        args.object_features,
        args.second_object_features,
        args.relation_features
    ]
    
    similarities = []
    text_ids_list = []
    image_ids_list = []
    
    for feat_type, text_path, img_path in zip(feature_types, text_feature_paths, image_feature_paths):
        if os.path.exists(text_path) and os.path.exists(img_path):
            print(f"Loading {feat_type} text features from {text_path}")
            text_features = load_features(text_path)
            print(f"Loaded {len(text_features)} {feat_type} text features")
            
            print(f"Loading {feat_type} image features from {img_path}")
            image_features = load_features(img_path)
            print(f"Loaded {len(image_features)} {feat_type} image features")
            
            # Compute similarity matrix
            print(f"Computing {feat_type} similarities...")
            sim_matrix, text_ids, image_ids = compute_similarity_matrix(text_features, image_features)
            similarities.append(sim_matrix)
            text_ids_list.append(text_ids)
            image_ids_list.append(image_ids)
        else:
            print(f"Warning: Missing features for {feat_type}, skipping...")
            similarities.append(None)
            text_ids_list.append([])
            image_ids_list.append([])
    
    # Normalize weights
    weights = np.array(args.weights)
    weights = weights / weights.sum()
    print(f"Using weights: {weights}")
    
    # Fuse similarities
    print("Fusing similarities...")
    valid_similarities = [sim for sim in similarities if sim is not None]
    valid_text_ids = [text_ids for text_ids in text_ids_list if text_ids]
    valid_image_ids = [image_ids for image_ids in image_ids_list if image_ids]
    valid_weights = weights[:len(valid_similarities)]
    valid_weights = valid_weights / valid_weights.sum()
    
    fused_similarity, all_text_ids, all_image_ids = fuse_similarities(valid_similarities, valid_weights, valid_text_ids, valid_image_ids)
    
    # Get predictions for both directions
    print("Generating text-to-image predictions...")
    text_to_image_predictions = get_topk_predictions(fused_similarity, all_text_ids, all_image_ids, args.top_k)
    
    print("Generating image-to-text predictions...")
    image_to_text_predictions = get_topk_predictions_reverse(fused_similarity, all_text_ids, all_image_ids, args.top_k)
    
    # Load ground truth
    if os.path.exists(args.ground_truth):
        print(f"Loading ground truth from {args.ground_truth}")
        ground_truth = load_ground_truth(args.ground_truth)
        
        # Debug information
        print(f"Ground truth has {len(ground_truth)} entries")
        print(f"Text-to-image predictions has {len(text_to_image_predictions)} entries")
        print(f"Image-to-text predictions has {len(image_to_text_predictions)} entries")
        
        # Check sample predictions vs ground truth
        sample_text_id = list(ground_truth.keys())[0]
        print(f"\nSample comparison for text_id {sample_text_id}:")
        print(f"Ground truth image_ids: {ground_truth[sample_text_id]}")
        print(f"Predicted image_ids: {text_to_image_predictions[sample_text_id]}")
        
        # Compute metrics
        print("Computing retrieval metrics...")
        metrics = compute_retrieval_metrics(text_to_image_predictions, image_to_text_predictions, ground_truth, args.k_values)
        
        print("\nRetrieval Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save metrics
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
    
    # Save predictions in standard format
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save text-to-image predictions (like test_predictions.jsonl)
    text_to_image_path = os.path.join(args.output_dir, 'test_predictions.jsonl')
    with open(text_to_image_path, 'w', encoding='utf-8') as f:
        for text_id, image_ids in text_to_image_predictions.items():
            pred_obj = {
                "text_id": text_id,
                "image_ids": image_ids
            }
            f.write(json.dumps(pred_obj, ensure_ascii=False) + '\n')
    print(f"Saved text-to-image predictions to {text_to_image_path}")
    
    # Save image-to-text predictions
    image_to_text_path = os.path.join(args.output_dir, 'image_to_text_predictions.jsonl')
    with open(image_to_text_path, 'w', encoding='utf-8') as f:
        for image_id, text_ids in image_to_text_predictions.items():
            pred_obj = {
                "image_id": image_id,
                "text_ids": text_ids
            }
            f.write(json.dumps(pred_obj, ensure_ascii=False) + '\n')
    print(f"Saved image-to-text predictions to {image_to_text_path}")
    
    # Save detailed predictions (original format)
    predictions_path = os.path.join(args.output_dir, 'predictions.jsonl')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for text_id, image_ids in text_to_image_predictions.items():
            pred_obj = {
                "text_id": text_id,
                "image_ids": image_ids
            }
            f.write(json.dumps(pred_obj, ensure_ascii=False) + '\n')
    print(f"Saved detailed predictions to {predictions_path}")
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, 'detailed_results.json')
    detailed_results = {
        'weights': weights.tolist(),
        'num_text_features': len(all_text_ids),
        'num_image_features': len(all_image_ids),
        'text_to_image_predictions': text_to_image_predictions,
        'image_to_text_predictions': image_to_text_predictions
    }
    
    if os.path.exists(args.ground_truth):
        detailed_results['metrics'] = metrics
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved detailed results to {results_path}")
    print("Similarity computation and fusion completed!")


if __name__ == "__main__":
    main()
