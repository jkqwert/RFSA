# -*- coding: utf-8 -*-
'''
Ablation Study Evaluation Script
'''

import os
import json
import argparse
import numpy as np
import torch
from data_loader_ablation import get_data_loader
from mapping_model_ablation import SingleDirectionMappingModule, SharedMappingModule
from feature_loader_ablation import load_all_image_features_ablation, extract_mapped_features_ablation
from eval_utils import (
    compute_retrieval_metrics_t2i, 
    compute_retrieval_metrics_i2t,
    generate_detailed_report_t2i,
    generate_detailed_report_i2t
)


def load_checkpoint(checkpoint_path, device, args):
    """Load trained mapping module from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration from checkpoint if available
    checkpoint_args = checkpoint.get('args', {})
    use_prompt = checkpoint_args.get('use_prompt', args.use_prompt)
    use_component = checkpoint_args.get('use_component', args.use_component)
    use_shared_space = checkpoint_args.get('use_shared_space', args.use_shared_space)
    embed_dim = checkpoint_args.get('embed_dim', args.embed_dim)
    prompt_length = checkpoint_args.get('prompt_length', args.prompt_length)
    hidden_dim = checkpoint_args.get('hidden_dim', args.hidden_dim)
    num_layers = checkpoint_args.get('num_layers', args.num_layers)
    dropout = checkpoint_args.get('dropout', args.dropout)
    fusion_method = checkpoint_args.get('fusion_method', args.fusion_method)
    component_types = checkpoint_args.get('component_types', args.component_types)
    
    print(f"Loading checkpoint with configuration:")
    print(f"  Use Prompt Learning: {use_prompt}")
    print(f"  Use Component Fusion: {use_component}")
    print(f"  Use Shared Space: {use_shared_space}")
    
    if use_shared_space:
        mapping_module = SharedMappingModule(
            embed_dim=embed_dim,
            prompt_length=prompt_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            fusion_method=fusion_method,
            component_types=component_types,
            use_prompt=use_prompt,
            use_component=use_component
        ).to(device)
    else:
        mapping_module = SingleDirectionMappingModule(
            embed_dim=embed_dim,
            prompt_length=prompt_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            fusion_method=fusion_method,
            component_types=component_types,
            use_prompt=use_prompt,
            use_component=use_component
        ).to(device)
    
    mapping_module.load_state_dict(checkpoint['model_state_dict'])
    mapping_module.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, val_loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Update args to match checkpoint configuration
    args.use_prompt = use_prompt
    args.use_component = use_component
    args.use_shared_space = use_shared_space
    
    return mapping_module


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


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation study evaluation')
    
    # Ablation switches (must match training)
    parser.add_argument('--use-prompt', action='store_true', default=False,
                        help='Use prompt learning mechanism')
    parser.add_argument('--use-component', action='store_true', default=False,
                        help='Use component fusion')
    parser.add_argument('--use-shared-space', action='store_true', default=False,
                        help='Use shared space mapping')
    
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
    
    print('=' * 80)
    print('Ablation Study Evaluation Configuration (from command line)')
    print('=' * 80)
    print(f'  Use Prompt Learning: {args.use_prompt}')
    print(f'  Use Component Fusion: {args.use_component}')
    print(f'  Use Shared Space: {args.use_shared_space}')
    print(f'  Checkpoint: {args.checkpoint}')
    print('=' * 80)
    
    print('Loading checkpoint...')
    mapping_module = load_checkpoint(args.checkpoint, device, args)
    
    print('=' * 80)
    print('Actual Configuration (from checkpoint)')
    print('=' * 80)
    print(f'  Use Prompt Learning: {args.use_prompt}')
    print(f'  Use Component Fusion: {args.use_component}')
    print(f'  Use Shared Space: {args.use_shared_space}')
    print('=' * 80)
    
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
        component_types=args.component_types,
        use_component=args.use_component
    )
    
    print('Extracting text features...')
    text_features, text_ids = extract_mapped_features_ablation(
        mapping_module, test_loader, device, args.use_shared_space
    )
    print(f'Extracted {len(text_features)} text features')
    
    # Debug: check first feature norm
    if text_features:
        first_id = list(text_features.keys())[0]
        first_feat = text_features[first_id]
        feat_norm = np.linalg.norm(first_feat)
        feat_max = np.abs(first_feat).max()
        print(f'Debug - First text feature (ID: {first_id}): norm={feat_norm:.4f}, max={feat_max:.4f}')
    
    # Filter to test set
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
    
    # Load all image features
    print('Loading all image features...')
    all_image_features = load_all_image_features_ablation(
        args.image_features_dir,
        args.component_types,
        mapping_module,
        device,
        args.use_component,
        args.use_shared_space
    )
    
    # Filter to test set images
    image_features = {}
    image_ids = []
    for normalized_id in test_image_ids_set:
        found = False
        for img_id in all_image_features.keys():
            try:
                if int(img_id) == int(normalized_id):
                    image_features[normalized_id] = all_image_features[img_id]
                    image_ids.append(normalized_id)
                    found = True
                    break
            except:
                if str(img_id).zfill(6) == normalized_id:
                    image_features[normalized_id] = all_image_features[img_id]
                    image_ids.append(normalized_id)
                    found = True
                    break
        
        if not found and normalized_id in all_image_features:
            image_features[normalized_id] = all_image_features[normalized_id]
            image_ids.append(normalized_id)
    
    print(f'Filtered to {len(image_features)} unique image features from test set')
    
    # Debug: check first image feature norm
    if image_features:
        first_img_id = list(image_features.keys())[0]
        first_img_feat = image_features[first_img_id]
        img_feat_norm = np.linalg.norm(first_img_feat)
        img_feat_max = np.abs(first_img_feat).max()
        print(f'Debug - First image feature (ID: {first_img_id}): norm={img_feat_norm:.4f}, max={img_feat_max:.4f}')
    
    # Debug: check checkpoint path
    print(f'Debug - Checkpoint path: {args.checkpoint}')
    print(f'Debug - Configuration: Prompt={args.use_prompt}, Component={args.use_component}, Shared={args.use_shared_space}')
    
    # Compute metrics
    print('Computing text-to-image retrieval metrics...')
    metrics_t2i, predictions_t2i = compute_retrieval_metrics_t2i(
        text_features, image_features, text_ids, image_ids, ground_truth
    )
    
    print('\nText-to-Image Retrieval Metrics:')
    for k, v in metrics_t2i.items():
        print(f'  {k}: {v:.4f}')
    
    print('\nComputing image-to-text retrieval metrics...')
    metrics_i2t, predictions_i2t, reverse_gt = compute_retrieval_metrics_i2t(
        text_features, image_features, text_ids, image_ids, ground_truth
    )
    
    print('\nImage-to-Text Retrieval Metrics:')
    for k, v in metrics_i2t.items():
        print(f'  {k}: {v:.4f}')
    
    # Save results
    print('\nSaving results...')
    
    all_metrics = {
        'text_to_image': metrics_t2i,
        'image_to_text': metrics_i2t
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f'Saved metrics to {metrics_path}')
    
    # Save predictions
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
    
    # Generate detailed reports
    print('\nGenerating detailed reports...')
    
    config_str = f" (Prompt:{args.use_prompt}, Component:{args.use_component}, Shared:{args.use_shared_space})"
    
    t2i_report_path = os.path.join(args.output_dir, 't2i_detailed_report.txt')
    generate_detailed_report_t2i(
        predictions_t2i, ground_truth, 
        k_values=[1, 5, 10], 
        output_file=t2i_report_path,
        title_suffix=config_str
    )
    
    i2t_report_path = os.path.join(args.output_dir, 'i2t_detailed_report.txt')
    generate_detailed_report_i2t(
        predictions_i2t, reverse_gt,
        k_values=[1, 5, 10],
        output_file=i2t_report_path,
        title_suffix=config_str
    )
    
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()

