# -*- coding: utf-8 -*-
'''
Alignment Metrics for Feature Space Analysis
Computes various metrics to measure alignment between text and image features
'''

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import torch


def compute_cosine_similarity_matrix(text_features, image_features):
    """
    Compute cosine similarity matrix between text and image features
    
    Args:
        text_features: Dictionary of {text_id: feature_vector}
        image_features: Dictionary of {image_id: feature_vector}
    
    Returns:
        similarity_matrix: [num_texts, num_images] similarity matrix
        text_ids: List of text IDs
        image_ids: List of image IDs
    """
    text_ids = sorted(text_features.keys())
    image_ids = sorted(image_features.keys())
    
    text_vectors = np.array([text_features[tid] for tid in text_ids])
    image_vectors = np.array([image_features[iid] for iid in image_ids])
    
    similarity_matrix = cosine_similarity(text_vectors, image_vectors)
    
    return similarity_matrix, text_ids, image_ids


def compute_alignment_metrics(text_features, image_features, ground_truth):
    """
    Compute alignment metrics between text and image features
    
    Args:
        text_features: Dictionary of {text_id: feature_vector}
        image_features: Dictionary of {image_id: feature_vector}
        ground_truth: Dictionary of {text_id: [image_ids]} (positive pairs)
    
    Returns:
        metrics: Dictionary of alignment metrics
    """
    similarity_matrix, text_ids, image_ids = compute_cosine_similarity_matrix(
        text_features, image_features
    )
    
    # Create mapping from ID to index
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Compute positive and negative similarities
    positive_similarities = []
    negative_similarities = []
    
    for text_id, gt_image_ids in ground_truth.items():
        if text_id not in text_id_to_idx:
            continue
        
        text_idx = text_id_to_idx[text_id]
        
        # Positive pairs (ground truth)
        for img_id in gt_image_ids:
            # Try different ID formats
            img_id_str = str(img_id).zfill(6)
            if img_id_str in image_id_to_idx:
                img_idx = image_id_to_idx[img_id_str]
                positive_similarities.append(similarity_matrix[text_idx, img_idx])
            else:
                # Try integer matching
                try:
                    img_id_num = int(img_id)
                    for iid, idx in image_id_to_idx.items():
                        try:
                            if int(iid) == img_id_num:
                                positive_similarities.append(similarity_matrix[text_idx, idx])
                                break
                        except:
                            pass
                except:
                    pass
        
        # Negative pairs (all other images)
        for img_id in image_ids:
            is_positive = False
            for gt_img_id in gt_image_ids:
                try:
                    if int(str(img_id).zfill(6)) == int(str(gt_img_id).zfill(6)):
                        is_positive = True
                        break
                except:
                    if str(img_id).zfill(6) == str(gt_img_id).zfill(6):
                        is_positive = True
                        break
            
            if not is_positive:
                img_idx = image_id_to_idx[img_id]
                negative_similarities.append(similarity_matrix[text_idx, img_idx])
    
    positive_similarities = np.array(positive_similarities)
    negative_similarities = np.array(negative_similarities)
    
    # Debug output for T2I
    print(f'    [T2I] Positive samples: {len(positive_similarities)}, Negative samples: {len(negative_similarities)}')
    if len(positive_similarities) > 0:
        print(f'    [T2I] Positive mean: {np.mean(positive_similarities):.4f}')
    if len(negative_similarities) > 0:
        print(f'    [T2I] Negative mean: {np.mean(negative_similarities):.4f}')
    
    # Compute metrics
    metrics = {
        'positive_mean': float(np.mean(positive_similarities)) if len(positive_similarities) > 0 else 0.0,
        'positive_std': float(np.std(positive_similarities)) if len(positive_similarities) > 0 else 0.0,
        'negative_mean': float(np.mean(negative_similarities)) if len(negative_similarities) > 0 else 0.0,
        'negative_std': float(np.std(negative_similarities)) if len(negative_similarities) > 0 else 0.0,
        'separation': float(np.mean(positive_similarities) - np.mean(negative_similarities)) if len(positive_similarities) > 0 and len(negative_similarities) > 0 else 0.0,
        'positive_count': len(positive_similarities),
        'negative_count': len(negative_similarities)
    }
    
    return metrics, similarity_matrix, text_ids, image_ids


def compute_alignment_metrics_i2t(text_features, image_features, ground_truth):
    """
    Compute alignment metrics for Image-to-Text retrieval
    
    Args:
        text_features: Dictionary of {text_id: feature_vector}
        image_features: Dictionary of {image_id: feature_vector}
        ground_truth: Dictionary of {text_id: [image_ids]} (positive pairs)
    
    Returns:
        metrics: Dictionary of alignment metrics for I2T
    """
    similarity_matrix, text_ids, image_ids = compute_cosine_similarity_matrix(
        text_features, image_features
    )
    
    # Create mapping from ID to index
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Build reverse ground truth: image_id -> [text_ids]
    reverse_gt = {}
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
            img_id_normalized = str(img_id).zfill(6)
            if img_id_normalized not in reverse_gt:
                reverse_gt[img_id_normalized] = []
            if text_id not in reverse_gt[img_id_normalized]:
                reverse_gt[img_id_normalized].append(text_id)
    
    print(f'    [I2T] Built reverse GT with {len(reverse_gt)} images')
    print(f'    [I2T] Sample reverse GT keys: {list(reverse_gt.keys())[:5]}')
    print(f'    [I2T] Available image_ids: {image_ids[:5]}')
    
    # Compute positive and negative similarities for I2T
    positive_similarities = []
    negative_similarities = []
    
    for image_id in image_ids:
        if image_id not in image_id_to_idx:
            continue
        
        img_idx = image_id_to_idx[image_id]
        
        # Positive pairs (ground truth texts for this image)
        if image_id in reverse_gt:
            for text_id in reverse_gt[image_id]:
                if text_id in text_id_to_idx:
                    text_idx = text_id_to_idx[text_id]
                    positive_similarities.append(similarity_matrix[text_idx, img_idx])
        
        # Negative pairs (all other texts)
        for text_id in text_ids:
            is_positive = image_id in reverse_gt and text_id in reverse_gt[image_id]
            
            if not is_positive:
                text_idx = text_id_to_idx[text_id]
                negative_similarities.append(similarity_matrix[text_idx, img_idx])
    
    positive_similarities = np.array(positive_similarities)
    negative_similarities = np.array(negative_similarities)
    
    # Debug output for I2T
    print(f'    [I2T] Positive samples: {len(positive_similarities)}, Negative samples: {len(negative_similarities)}')
    if len(positive_similarities) > 0:
        print(f'    [I2T] Positive mean: {np.mean(positive_similarities):.4f}')
    if len(negative_similarities) > 0:
        print(f'    [I2T] Negative mean: {np.mean(negative_similarities):.4f}')
    
    # Compute metrics
    metrics = {
        'positive_mean': float(np.mean(positive_similarities)) if len(positive_similarities) > 0 else 0.0,
        'positive_std': float(np.std(positive_similarities)) if len(positive_similarities) > 0 else 0.0,
        'negative_mean': float(np.mean(negative_similarities)) if len(negative_similarities) > 0 else 0.0,
        'negative_std': float(np.std(negative_similarities)) if len(negative_similarities) > 0 else 0.0,
        'separation': float(np.mean(positive_similarities) - np.mean(negative_similarities)) if len(positive_similarities) > 0 and len(negative_similarities) > 0 else 0.0,
        'positive_count': len(positive_similarities),
        'negative_count': len(negative_similarities)
    }
    
    return metrics


def compute_feature_diversity(features_dict):
    """
    Compute feature diversity metrics
    
    Args:
        features_dict: Dictionary of {id: feature_vector}
    
    Returns:
        metrics: Dictionary of diversity metrics
    """
    features = np.array(list(features_dict.values()))
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(features)
    
    # Exclude diagonal
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    pairwise_similarities = similarity_matrix[mask]
    
    metrics = {
        'mean_pairwise_similarity': float(np.mean(pairwise_similarities)),
        'std_pairwise_similarity': float(np.std(pairwise_similarities)),
        'min_pairwise_similarity': float(np.min(pairwise_similarities)),
        'max_pairwise_similarity': float(np.max(pairwise_similarities)),
        'feature_norm_mean': float(np.mean([np.linalg.norm(f) for f in features])),
        'feature_norm_std': float(np.std([np.linalg.norm(f) for f in features]))
    }
    
    return metrics


def compute_alignment_quality(similarity_matrix, ground_truth, text_ids, image_ids):
    """
    Compute alignment quality metrics
    
    Args:
        similarity_matrix: [num_texts, num_images] similarity matrix
        ground_truth: Dictionary of {text_id: [image_ids]}
        text_ids: List of text IDs
        image_ids: List of image IDs
    
    Returns:
        metrics: Dictionary of quality metrics
    """
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    # Compute rank statistics for positive pairs
    ranks = []
    positive_similarities = []
    
    for text_id, gt_image_ids in ground_truth.items():
        if text_id not in text_id_to_idx:
            continue
        
        text_idx = text_id_to_idx[text_id]
        similarities = similarity_matrix[text_idx]
        
        # Get rank for each positive image
        for img_id in gt_image_ids:
            img_id_str = str(img_id).zfill(6)
            if img_id_str in image_id_to_idx:
                img_idx = image_id_to_idx[img_id_str]
                sim = similarities[img_idx]
                positive_similarities.append(sim)
                
                # Compute rank (how many images have higher similarity)
                rank = np.sum(similarities > sim) + 1
                ranks.append(rank)
            else:
                # Try integer matching
                try:
                    img_id_num = int(img_id)
                    for iid, idx in image_id_to_idx.items():
                        try:
                            if int(iid) == img_id_num:
                                sim = similarities[idx]
                                positive_similarities.append(sim)
                                rank = np.sum(similarities > sim) + 1
                                ranks.append(rank)
                                break
                        except:
                            pass
                except:
                    pass
    
    ranks = np.array(ranks)
    positive_similarities = np.array(positive_similarities)
    
    metrics = {
        'mean_rank': float(np.mean(ranks)) if len(ranks) > 0 else 0.0,
        'median_rank': float(np.median(ranks)) if len(ranks) > 0 else 0.0,
        'mean_reciprocal_rank': float(np.mean(1.0 / ranks)) if len(ranks) > 0 else 0.0,
        'top1_accuracy': float(np.sum(ranks == 1) / len(ranks)) if len(ranks) > 0 else 0.0,
        'top5_accuracy': float(np.sum(ranks <= 5) / len(ranks)) if len(ranks) > 0 else 0.0,
        'top10_accuracy': float(np.sum(ranks <= 10) / len(ranks)) if len(ranks) > 0 else 0.0,
        'positive_similarity_mean': float(np.mean(positive_similarities)) if len(positive_similarities) > 0 else 0.0
    }
    
    return metrics



