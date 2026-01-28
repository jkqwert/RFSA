# -*- coding: utf-8 -*-
'''
Main script for visualizing feature alignment
Compares shared space mapping vs single direction mapping
'''

import os
import json
import argparse
import numpy as np
import torch
from compare_alignments import analyze_configuration, load_checkpoint, load_ground_truth
from data_loader_ablation import get_data_loader
from feature_loader_ablation import load_all_image_features_ablation, extract_mapped_features_ablation
from alignment_metrics import compute_alignment_metrics
from visualization_utils import (
    plot_similarity_heatmap,
    plot_feature_distribution_2d,
    plot_similarity_distribution
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature alignment')
    
    # Single configuration analysis
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint file')
    parser.add_argument('--config-name', type=str, default='Current',
                        help='Name for this configuration')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple configurations')
    parser.add_argument('--checkpoints', type=str, nargs='+',
                        help='Paths to checkpoint files to compare')
    parser.add_argument('--config-names', type=str, nargs='+',
                        help='Names for each configuration')
    
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
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--prompt-length', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fusion-method', type=str, default='weighted_sum')
    parser.add_argument('--component-types', type=str, nargs='+',
                        default=['subject', 'object', 'second', 'relation'])
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for visualizations')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.compare and args.checkpoints:
        # Comparison mode
        from compare_alignments import main as compare_main
        import sys
        # Temporarily modify args for compare_main
        original_checkpoints = args.checkpoints
        args.checkpoints = original_checkpoints
        compare_main()
    else:
        # Single configuration analysis
        if not args.checkpoint:
            print("Error: --checkpoint is required for single configuration analysis")
            return
        
        print(f'Analyzing single configuration: {args.config_name}')
        
        # Load checkpoint
        mapping_module, use_prompt, use_component, use_shared_space = load_checkpoint(
            args.checkpoint, device, args
        )
        
        print(f'Configuration: Prompt={use_prompt}, Component={use_component}, Shared={use_shared_space}')
        
        # Load ground truth
        ground_truth = load_ground_truth(args.test_texts)
        
        # Create data loader
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
            use_component=use_component
        )
        
        # Extract features
        print('Extracting text features...')
        text_features, text_ids = extract_mapped_features_ablation(
            mapping_module, test_loader, device, use_shared_space
        )
        
        test_text_ids = set(ground_truth.keys())
        text_features = {tid: text_features[tid] for tid in text_ids if tid in test_text_ids}
        text_ids = [tid for tid in text_ids if tid in test_text_ids]
        
        print('Loading image features...')
        all_image_features = load_all_image_features_ablation(
            args.image_features_dir,
            args.component_types,
            mapping_module,
            device,
            use_component,
            use_shared_space
        )
        
        # Filter to test set
        test_image_ids_set = set()
        for text_id, image_id_list in ground_truth.items():
            for img_id in image_id_list:
                try:
                    test_image_ids_set.add(f"{int(img_id):06d}")
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
        
        # Compute metrics
        print('Computing alignment metrics...')
        alignment_metrics, similarity_matrix, text_ids_sorted, image_ids_sorted = compute_alignment_metrics(
            text_features, image_features, ground_truth
        )
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'alignment_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(alignment_metrics, f, indent=2, ensure_ascii=False)
        
        # Generate visualizations
        config_name = f"Prompt:{use_prompt}, Component:{use_component}, Shared:{use_shared_space}"
        
        print('Generating visualizations...')
        
        # Similarity heatmap
        heatmap_path = os.path.join(args.output_dir, 'similarity_heatmap.png')
        plot_similarity_heatmap(
            similarity_matrix, text_ids_sorted, image_ids_sorted,
            heatmap_path, title=f'Similarity Matrix - {config_name}'
        )
        
        # Feature distribution
        tsne_path = os.path.join(args.output_dir, 'feature_distribution_tsne.png')
        plot_feature_distribution_2d(
            text_features, image_features, ground_truth,
            tsne_path, method='tsne', title=f'Feature Distribution (t-SNE) - {config_name}'
        )
        
        pca_path = os.path.join(args.output_dir, 'feature_distribution_pca.png')
        plot_feature_distribution_2d(
            text_features, image_features, ground_truth,
            pca_path, method='pca', title=f'Feature Distribution (PCA) - {config_name}'
        )
        
        print(f'\nVisualization completed! Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()



