# -*- coding: utf-8 -*-
'''
Evaluation utilities for ablation study
'''

import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def generate_detailed_report_t2i(predictions, ground_truth, k_values=[1, 5, 10], output_file='t2i_detailed_report.txt', title_suffix=''):
    """Generate detailed report with checkmarks for text-to-image retrieval"""
    title = f'Text-to-Image Retrieval Detailed Report{title_suffix}'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write(title + '\n')
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


def generate_detailed_report_i2t(predictions, reverse_gt, k_values=[1, 5, 10], output_file='i2t_detailed_report.txt', title_suffix=''):
    """Generate detailed report with checkmarks for image-to-text retrieval"""
    title = f'Image-to-Text Retrieval Detailed Report{title_suffix}'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write(title + '\n')
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


def compute_retrieval_metrics_t2i(text_features, image_features, text_ids, image_ids, ground_truth):
    """Compute text-to-image retrieval metrics"""
    text_vectors = np.array([text_features[tid] for tid in text_ids])
    image_vectors = np.array([image_features[iid] for iid in image_ids])
    
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    similarity_matrix = cosine_similarity(text_vectors, image_vectors)
    
    k_values = [1, 5, 10]
    predictions = {}
    recalls = {f'recall@{k}': [] for k in k_values}
    
    for text_id in text_ids:
        if text_id not in text_id_to_idx:
            continue
        
        text_idx = text_id_to_idx[text_id]
        similarities = similarity_matrix[text_idx]
        topk_indices = np.argsort(similarities)[::-1]
        
        seen_ids = set()
        formatted_image_ids = []
        for idx in topk_indices:
            if len(formatted_image_ids) >= max(k_values):
                break
            img_id = image_ids[idx]
            
            try:
                if isinstance(img_id, str):
                    num = int(img_id)
                else:
                    num = int(img_id)
                formatted_id = f"{num:06d}"
            except:
                formatted_id = str(img_id)
            
            if formatted_id not in seen_ids:
                formatted_image_ids.append(formatted_id)
                seen_ids.add(formatted_id)
        
        predictions[text_id] = formatted_image_ids
        
        if text_id in ground_truth:
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
                if len(intersection) > 0:
                    recall_k = 1.0
                else:
                    recall_k = 0.0
                recalls[f'recall@{k}'].append(recall_k)
    
    metrics = {}
    for k in k_values:
        if recalls[f'recall@{k}']:
            avg_recall = np.mean(recalls[f'recall@{k}'])
            metrics[f'recall@{k}'] = avg_recall
        else:
            metrics[f'recall@{k}'] = 0.0
    
    return metrics, predictions


def compute_retrieval_metrics_i2t(text_features, image_features, text_ids, image_ids, ground_truth):
    """Compute image-to-text retrieval metrics"""
    text_vectors = np.array([text_features[tid] for tid in text_ids])
    image_vectors = np.array([image_features[iid] for iid in image_ids])
    
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    
    similarity_matrix = cosine_similarity(image_vectors, text_vectors)
    
    reverse_gt = {}
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
            
            if img_id_normalized not in reverse_gt:
                reverse_gt[img_id_normalized] = []
            if text_id not in reverse_gt[img_id_normalized]:
                reverse_gt[img_id_normalized].append(text_id)
    
    k_values = [1, 5, 10]
    predictions = {}
    recalls = {f'recall@{k}': [] for k in k_values}
    
    for image_id in image_ids:
        if image_id not in image_id_to_idx:
            continue
        
        image_idx = image_id_to_idx[image_id]
        similarities = similarity_matrix[image_idx]
        topk_indices = np.argsort(similarities)[::-1]
        
        seen_ids = set()
        formatted_text_ids = []
        for idx in topk_indices:
            if len(formatted_text_ids) >= max(k_values):
                break
            text_id = text_ids[idx]
            
            if text_id not in seen_ids:
                formatted_text_ids.append(text_id)
                seen_ids.add(text_id)
        
        predictions[image_id] = formatted_text_ids
        
        if image_id in reverse_gt:
            gt_text_ids = reverse_gt[image_id]
            gt_text_id = gt_text_ids[0] if gt_text_ids else None
            
            for k in k_values:
                pred_k = set(formatted_text_ids[:k])
                hit = gt_text_id in pred_k if gt_text_id else False
                recall_k = 1.0 if hit else 0.0
                recalls[f'recall@{k}'].append(recall_k)
    
    metrics = {}
    for k in k_values:
        if recalls[f'recall@{k}']:
            avg_recall = np.mean(recalls[f'recall@{k}'])
            metrics[f'recall@{k}'] = avg_recall
        else:
            metrics[f'recall@{k}'] = 0.0
    
    return metrics, predictions, reverse_gt



