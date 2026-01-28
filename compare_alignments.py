# -*- coding: utf-8 -*-
'''
Compare Alignment Effects between Shared Space and Single Direction Mapping
'''

import os
import json
import argparse
import numpy as np
import torch
from data_loader_ablation import get_data_loader
from mapping_model_ablation import SingleDirectionMappingModule, SharedMappingModule as AblationSharedMappingModule
from mapping_model_shared import SharedMappingModule as OriginalSharedMappingModule
from feature_loader_ablation import load_all_image_features_ablation, extract_mapped_features_ablation
from alignment_metrics import (
    compute_alignment_metrics,
    compute_feature_diversity,
    compute_alignment_quality
)
from visualization_utils import (
    plot_similarity_heatmap,
    plot_feature_distribution_2d,
    plot_similarity_distribution,
    plot_similarity_distribution_combined,
    plot_alignment_comparison
)


def load_checkpoint(checkpoint_path, device, args):
    """Load trained mapping module from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    checkpoint_args = checkpoint.get('args', {})
    # Use getattr for Namespace objects, or .get() for dict
    if isinstance(checkpoint_args, dict):
        use_prompt = checkpoint_args.get('use_prompt', getattr(args, 'use_prompt', False))
        use_component = checkpoint_args.get('use_component', getattr(args, 'use_component', False))
        use_shared_space = checkpoint_args.get('use_shared_space', getattr(args, 'use_shared_space', False))
        embed_dim = checkpoint_args.get('embed_dim', getattr(args, 'embed_dim', 512))
        prompt_length = checkpoint_args.get('prompt_length', getattr(args, 'prompt_length', 4))
        hidden_dim = checkpoint_args.get('hidden_dim', getattr(args, 'hidden_dim', 512))
        num_layers = checkpoint_args.get('num_layers', getattr(args, 'num_layers', 2))
        dropout = checkpoint_args.get('dropout', getattr(args, 'dropout', 0.1))
        fusion_method = checkpoint_args.get('fusion_method', getattr(args, 'fusion_method', 'weighted_sum'))
        component_types = checkpoint_args.get('component_types', getattr(args, 'component_types', ['subject', 'object', 'second', 'relation']))
    else:
        # If checkpoint_args is a Namespace object
        use_prompt = getattr(checkpoint_args, 'use_prompt', getattr(args, 'use_prompt', False))
        use_component = getattr(checkpoint_args, 'use_component', getattr(args, 'use_component', False))
        use_shared_space = getattr(checkpoint_args, 'use_shared_space', getattr(args, 'use_shared_space', False))
        embed_dim = getattr(checkpoint_args, 'embed_dim', getattr(args, 'embed_dim', 512))
        prompt_length = getattr(checkpoint_args, 'prompt_length', getattr(args, 'prompt_length', 4))
        hidden_dim = getattr(checkpoint_args, 'hidden_dim', getattr(args, 'hidden_dim', 512))
        num_layers = getattr(checkpoint_args, 'num_layers', getattr(args, 'num_layers', 2))
        dropout = getattr(checkpoint_args, 'dropout', getattr(args, 'dropout', 0.1))
        fusion_method = getattr(checkpoint_args, 'fusion_method', getattr(args, 'fusion_method', 'weighted_sum'))
        component_types = getattr(checkpoint_args, 'component_types', getattr(args, 'component_types', ['subject', 'object', 'second', 'relation']))
    
    # Check model state_dict keys to determine model type if use_shared_space is not set
    # First, check if use_shared_space is explicitly in checkpoint_args
    checkpoint_args_dict = checkpoint_args if isinstance(checkpoint_args, dict) else vars(checkpoint_args) if hasattr(checkpoint_args, '__dict__') else {}
    
    # Check state_dict keys to determine model architecture
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    has_text_mappings_plural = any('text_mappings.' in k for k in state_dict_keys)  # OriginalSharedMappingModule
    has_image_mappings_plural = any('image_mappings.' in k for k in state_dict_keys)  # OriginalSharedMappingModule
    has_text_mapping_singular = any('text_mapping.mlp' in k for k in state_dict_keys)  # AblationSharedMappingModule
    has_image_mapping_singular = any('image_mapping.mlp' in k for k in state_dict_keys)  # AblationSharedMappingModule
    has_single_mapping = any('mapping.mlp' in k for k in state_dict_keys) and not has_text_mappings_plural
    
    # Determine if this is original SharedMappingModule or ablation SharedMappingModule
    use_original_shared = has_text_mappings_plural and has_image_mappings_plural
    use_ablation_shared = has_text_mapping_singular and has_image_mapping_singular
    
    if 'use_shared_space' not in checkpoint_args_dict or checkpoint_args_dict.get('use_shared_space') is None:
        if use_original_shared:
            use_shared_space = True
            print(f'  Inferred: OriginalSharedMappingModule (from state_dict keys)')
        elif use_ablation_shared:
            use_shared_space = True
            print(f'  Inferred: AblationSharedMappingModule (from state_dict keys)')
        elif has_single_mapping:
            use_shared_space = False
            print(f'  Inferred: SingleDirectionMappingModule (from state_dict keys)')
        else:
            # Default: assume shared space if checkpoint is from train_shared_mapping.py
            # Check checkpoint path or other indicators
            if 'shared' in checkpoint_path.lower():
                use_shared_space = True
                print(f'  Inferred: SharedMappingModule (from checkpoint path)')
            else:
                use_shared_space = False
                print(f'  Inferred: SingleDirectionMappingModule (default)')
    else:
        # use_shared_space is explicitly set in checkpoint
        if isinstance(checkpoint_args, dict):
            use_shared_space = checkpoint_args.get('use_shared_space', False)
        else:
            use_shared_space = getattr(checkpoint_args, 'use_shared_space', False)
    
    if use_shared_space:
        # Determine which SharedMappingModule to use
        if use_original_shared:
            # Use original SharedMappingModule (from mapping_model_shared.py)
            # OriginalSharedMappingModule always uses component features
            use_component = True
            use_prompt = True  # OriginalSharedMappingModule always has prompts
            mapping_module = OriginalSharedMappingModule(
                embed_dim=embed_dim,
                prompt_length=prompt_length,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                fusion_method=fusion_method,
                component_types=component_types
            ).to(device)
        else:
            # Use ablation SharedMappingModule (from mapping_model_ablation.py)
            mapping_module = AblationSharedMappingModule(
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
    
    return mapping_module, use_prompt, use_component, use_shared_space


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


def analyze_configuration(checkpoint_path, test_texts, create_jsonl, 
                         text_features_dir, image_features_dir, 
                         output_dir, device, args):
    """Analyze alignment for a single configuration"""
    print(f'\n{"="*80}')
    print(f'Analyzing configuration: {checkpoint_path}')
    print(f'{"="*80}')
    
    # Load checkpoint
    mapping_module, use_prompt, use_component, use_shared_space = load_checkpoint(
        checkpoint_path, device, args
    )
    
    # Check if this is OriginalSharedMappingModule (always uses components)
    state_dict_keys = list(torch.load(checkpoint_path, map_location=device)['model_state_dict'].keys())
    is_original_shared = any('text_mappings.' in k for k in state_dict_keys) and any('image_mappings.' in k for k in state_dict_keys)
    
    if is_original_shared:
        # OriginalSharedMappingModule always uses component features
        use_component = True
        use_prompt = True
        print(f'  Note: OriginalSharedMappingModule always uses component features and prompts')
    
    print(f'Configuration: Prompt={use_prompt}, Component={use_component}, Shared={use_shared_space}')
    
    # Load ground truth
    ground_truth = load_ground_truth(test_texts)
    
    # Create data loader
    test_loader = get_data_loader(
        texts_jsonl=test_texts,
        create_jsonl=create_jsonl,
        text_features_dir=text_features_dir,
        image_features_dir=image_features_dir,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        component_types=args.component_types,
        use_component=use_component
    )
    
    # Extract text features
    print('Extracting text features...')
    print(f'  use_shared_space={use_shared_space}')
    text_features, text_ids = extract_mapped_features_ablation(
        mapping_module, test_loader, device, use_shared_space
    )
    
    # Debug: check first text feature
    if text_features:
        first_text_id = list(text_features.keys())[0]
        first_text_feat = text_features[first_text_id]
        text_norm = np.linalg.norm(first_text_feat)
        text_max = np.abs(first_text_feat).max()
        print(f'  Debug - First text feature (ID: {first_text_id}): norm={text_norm:.4f}, max={text_max:.4f}')
    
    # Filter to test set
    test_text_ids = set(ground_truth.keys())
    text_features = {tid: text_features[tid] for tid in text_ids if tid in test_text_ids}
    text_ids = [tid for tid in text_ids if tid in test_text_ids]
    
    # Load image features
    print('Loading image features...')
    print(f'  use_component={use_component}, use_shared_space={use_shared_space}')
    all_image_features = load_all_image_features_ablation(
        image_features_dir,
        args.component_types,
        mapping_module,
        device,
        use_component,
        use_shared_space
    )
    
    # Debug: check first image feature
    if all_image_features:
        first_img_id = list(all_image_features.keys())[0]
        first_img_feat = all_image_features[first_img_id]
        img_norm = np.linalg.norm(first_img_feat)
        img_max = np.abs(first_img_feat).max()
        print(f'  Debug - First image feature (ID: {first_img_id}): norm={img_norm:.4f}, max={img_max:.4f}')
    
    # Filter to test set images
    test_image_ids_set = set()
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
            try:
                img_id_normalized = f"{int(img_id):06d}"
                test_image_ids_set.add(img_id_normalized)
            except:
                test_image_ids_set.add(str(img_id).zfill(6))
    
    image_features = {}
    for normalized_id in test_image_ids_set:
        for img_id in all_image_features.keys():
            try:
                if int(img_id) == int(normalized_id):
                    image_features[normalized_id] = all_image_features[img_id]
                    break
            except:
                if str(img_id).zfill(6) == normalized_id:
                    image_features[normalized_id] = all_image_features[img_id]
                    break
    
    # Debug: compare features between configurations
    print(f'\nDebug - Feature Statistics:')
    print(f'  Text features: {len(text_features)} samples')
    print(f'  Image features: {len(image_features)} samples')
    if text_features and image_features:
        # Sample a few features to check
        sample_text_id = list(text_features.keys())[0]
        sample_img_id = list(image_features.keys())[0]
        sample_text_feat = text_features[sample_text_id]
        sample_img_feat = image_features[sample_img_id]
        
        # Check if features are identical (which would be wrong)
        text_feat_hash = hash(sample_text_feat.tobytes())
        img_feat_hash = hash(sample_img_feat.tobytes())
        print(f'  Sample text feature hash: {text_feat_hash}')
        print(f'  Sample image feature hash: {img_feat_hash}')
        
        # Check similarity between text and image features
        text_img_sim = np.dot(sample_text_feat, sample_img_feat) / (
            np.linalg.norm(sample_text_feat) * np.linalg.norm(sample_img_feat)
        )
        print(f'  Sample text-image similarity: {text_img_sim:.4f}')
    
    # Compute alignment metrics
    print('Computing alignment metrics...')
    alignment_metrics, similarity_matrix, text_ids_sorted, image_ids_sorted = compute_alignment_metrics(
        text_features, image_features, ground_truth
    )
    
    # Compute feature diversity
    print('Computing feature diversity...')
    text_diversity = compute_feature_diversity(text_features)
    image_diversity = compute_feature_diversity(image_features)
    
    # Compute alignment quality
    print('Computing alignment quality...')
    quality_metrics = compute_alignment_quality(
        similarity_matrix, ground_truth, text_ids_sorted, image_ids_sorted
    )
    
    # Combine all metrics
    all_metrics = {
        'alignment': alignment_metrics,
        'text_diversity': text_diversity,
        'image_diversity': image_diversity,
        'quality': quality_metrics
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'alignment_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f'Saved metrics to {metrics_path}')
    
    # Generate visualizations
    print('Generating visualizations...')
    
    # Similarity heatmap
    heatmap_path = os.path.join(output_dir, 'similarity_heatmap.png')
    config_name = f"Prompt:{use_prompt}, Component:{use_component}, Shared:{use_shared_space}"
    plot_similarity_heatmap(
        similarity_matrix, text_ids_sorted, image_ids_sorted,
        heatmap_path, title=f'Similarity Matrix - {config_name}'
    )
    
    # Feature distribution (t-SNE)
    tsne_path = os.path.join(output_dir, 'feature_distribution_tsne.png')
    plot_feature_distribution_2d(
        text_features, image_features, ground_truth,
        tsne_path, method='tsne', title=f'Feature Distribution (t-SNE) - {config_name}'
    )
    
    # Similarity distribution
    # Extract positive and negative similarities for both T2I and I2T
    positive_sims_t2i = []
    negative_sims_t2i = []
    positive_sims_i2t = []
    negative_sims_i2t = []
    
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids_sorted)}
    image_id_to_idx = {iid: i for i, iid in enumerate(image_ids_sorted)}
    
    # Build reverse ground truth for I2T
    reverse_gt = {}
    for text_id, image_id_list in ground_truth.items():
        for img_id in image_id_list:
            img_id_str = str(img_id).zfill(6)
            if img_id_str not in reverse_gt:
                reverse_gt[img_id_str] = []
            reverse_gt[img_id_str].append(text_id)
    
    # Text-to-Image similarities
    for text_id, gt_image_ids in ground_truth.items():
        if text_id not in text_id_to_idx:
            continue
        text_idx = text_id_to_idx[text_id]
        
        for img_id in gt_image_ids:
            img_id_str = str(img_id).zfill(6)
            if img_id_str in image_id_to_idx:
                img_idx = image_id_to_idx[img_id_str]
                positive_sims_t2i.append(similarity_matrix[text_idx, img_idx])
        
        for img_id in image_ids_sorted:
            is_positive = any(int(str(img_id).zfill(6)) == int(str(gt_img_id).zfill(6)) 
                            for gt_img_id in gt_image_ids)
            if not is_positive:
                img_idx = image_id_to_idx[img_id]
                negative_sims_t2i.append(similarity_matrix[text_idx, img_idx])
    
    # Image-to-Text similarities
    for img_id in image_ids_sorted:
        if img_id not in image_id_to_idx:
            continue
        img_idx = image_id_to_idx[img_id]
        
        # Positive pairs
        if img_id in reverse_gt:
            for text_id in reverse_gt[img_id]:
                if text_id in text_id_to_idx:
                    text_idx = text_id_to_idx[text_id]
                    positive_sims_i2t.append(similarity_matrix[text_idx, img_idx])
        
        # Negative pairs
        for text_id in text_ids_sorted:
            is_positive = img_id in reverse_gt and text_id in reverse_gt[img_id]
            if not is_positive:
                text_idx = text_id_to_idx[text_id]
                negative_sims_i2t.append(similarity_matrix[text_idx, img_idx])
    
    # Plot combined similarity distribution (merge T2I and I2T)
    if positive_sims_t2i and negative_sims_t2i:
        # Merge T2I and I2T samples
        positive_sims_combined = np.concatenate([positive_sims_t2i, positive_sims_i2t])
        negative_sims_combined = np.concatenate([negative_sims_t2i, negative_sims_i2t])
        
        sim_dist_path = os.path.join(output_dir, 'similarity_distribution_combined.png')
        plot_similarity_distribution_combined(
            positive_sims_combined, negative_sims_combined,
            sim_dist_path, title=f'Similarity Distribution (Combined) - {config_name}'
        )
        
        # Also save separate T2I and I2T for reference
        sim_dist_path_separate = os.path.join(output_dir, 'similarity_distribution_separate.png')
        plot_similarity_distribution(
            np.array(positive_sims_t2i), np.array(negative_sims_t2i),
            np.array(positive_sims_i2t), np.array(negative_sims_i2t),
            sim_dist_path_separate, title=f'Similarity Distribution - {config_name}'
        )
    
    return all_metrics, config_name


def parse_args():
    parser = argparse.ArgumentParser(description='Compare alignment effects')
    
    # Configuration checkpoints
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to checkpoint files to compare')
    parser.add_argument('--config-names', type=str, nargs='+',
                        help='Names for each configuration (optional)')
    
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
    
    # Analyze each configuration
    all_metrics_dict = {}
    config_names_list = []
    
    for i, checkpoint_path in enumerate(args.checkpoints):
        config_name = args.config_names[i] if args.config_names and i < len(args.config_names) else f'Config_{i}'
        config_names_list.append(config_name)
        
        config_output_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)
        
        metrics, actual_config_name = analyze_configuration(
            checkpoint_path, args.test_texts, args.create_jsonl,
            args.text_features_dir, args.image_features_dir,
            config_output_dir, device, args
        )
        
        all_metrics_dict[config_name] = metrics['alignment']
    
    # Generate comparison plots
    print(f'\n{"="*80}')
    print('Generating comparison plots...')
    print(f'{"="*80}')
    
    # Alignment comparison (use same metrics for both T2I and I2T since they're combined)
    comparison_path = os.path.join(args.output_dir, 'alignment_comparison.png')
    plot_alignment_comparison(
        all_metrics_dict, all_metrics_dict, comparison_path,
        title='Alignment Comparison: Ablation Study'
    )
    
    # Save summary
    summary = {
        'configurations': config_names_list,
        'metrics': all_metrics_dict
    }
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Saved comparison summary to {summary_path}')
    
    # Print summary
    print(f'\n{"="*80}')
    print('Alignment Comparison Summary')
    print(f'{"="*80}')
    for config_name, metrics in all_metrics_dict.items():
        print(f'\n{config_name}:')
        print(f'  Positive Mean: {metrics["positive_mean"]:.4f}')
        print(f'  Negative Mean: {metrics["negative_mean"]:.4f}')
        print(f'  Separation: {metrics["separation"]:.4f}')
    
    print('\nComparison completed!')


if __name__ == '__main__':
    main()

