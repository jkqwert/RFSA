# -*- coding: utf-8 -*-
'''
Visualization Utilities for Feature Alignment
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def plot_similarity_heatmap(similarity_matrix, text_ids, image_ids, output_path, 
                           title='Text-Image Similarity Matrix', max_samples=100):
    """
    Plot similarity heatmap between text and image features
    Shows both T2I (Text-to-Image) and I2T (Image-to-Text) views
    
    Args:
        similarity_matrix: [num_texts, num_images] similarity matrix
        text_ids: List of text IDs
        image_ids: List of image IDs
        output_path: Path to save the plot
        title: Plot title
        max_samples: Maximum number of samples to plot (for visualization)
    """
    # Subsample if too large
    if similarity_matrix.shape[0] > max_samples or similarity_matrix.shape[1] > max_samples:
        step_t = max(1, similarity_matrix.shape[0] // max_samples)
        step_i = max(1, similarity_matrix.shape[1] // max_samples)
        similarity_matrix_sub = similarity_matrix[::step_t, ::step_i]
        text_ids_sub = text_ids[::step_t]
        image_ids_sub = image_ids[::step_i]
    else:
        similarity_matrix_sub = similarity_matrix
        text_ids_sub = text_ids
        image_ids_sub = image_ids
    
    # Create 1x2 subplots for T2I and I2T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Left: Text-to-Image (T2I)
    sns.heatmap(similarity_matrix_sub, cmap='Blues', cbar=True, ax=ax1,
                xticklabels=False, yticklabels=False, vmin=-1, vmax=1)
    ax1.set_title('Text → Image Similarity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Image Features', fontsize=11)
    ax1.set_ylabel('Text Features', fontsize=11)
    
    # Right: Image-to-Text (I2T) - transpose the matrix
    sns.heatmap(similarity_matrix_sub.T, cmap='Greens', cbar=True, ax=ax2,
                xticklabels=False, yticklabels=False, vmin=-1, vmax=1)
    ax2.set_title('Image → Text Similarity', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Text Features', fontsize=11)
    ax2.set_ylabel('Image Features', fontsize=11)
    
    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved bidirectional similarity heatmap to {output_path}')


def plot_feature_distribution_2d(text_features, image_features, ground_truth, 
                                output_path, method='tsne', title='Feature Distribution'):
    """
    Plot 2D visualization of text and image features using t-SNE or PCA
    
    Args:
        text_features: Dictionary of {text_id: feature_vector}
        image_features: Dictionary of {image_id: feature_vector}
        ground_truth: Dictionary of {text_id: [image_ids]}
        output_path: Path to save the plot
        method: 'tsne' or 'pca'
        title: Plot title
    """
    # Collect all features and labels
    all_features = []
    labels = []  # 0 for text, 1 for image
    text_indices = []
    image_indices = []
    
    text_ids = sorted(text_features.keys())
    image_ids = sorted(image_features.keys())
    
    for tid in text_ids:
        all_features.append(text_features[tid])
        labels.append(0)
        text_indices.append(len(all_features) - 1)
    
    for iid in image_ids:
        all_features.append(image_features[iid])
        labels.append(1)
        image_indices.append(len(all_features) - 1)
    
    all_features = np.array(all_features)
    
    # Data validation and cleaning
    # Check for NaN and Inf values
    if np.any(np.isnan(all_features)) or np.any(np.isinf(all_features)):
        print('Warning: Found NaN or Inf values in features, cleaning...')
        # Replace NaN with 0
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Check for zero variance features (constant features)
    feature_vars = np.var(all_features, axis=0)
    zero_var_mask = feature_vars == 0
    if np.any(zero_var_mask):
        print(f'Warning: Found {np.sum(zero_var_mask)} features with zero variance, removing...')
        # Remove zero variance features
        all_features = all_features[:, ~zero_var_mask]
    
    # Check if we have enough features after cleaning
    if all_features.shape[1] < 2:
        print('Error: Not enough valid features for dimensionality reduction')
        return
    
    # Normalize features to prevent numerical issues
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    # Avoid division by zero
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    all_features = (all_features - feature_mean) / feature_std
    
    # Dimensionality reduction
    if method == 'tsne':
        # Adjust perplexity based on sample size
        n_samples = all_features.shape[0]
        perplexity = min(30, max(5, n_samples // 4))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       n_iter=1000, verbose=0)
    else:
        # For PCA, ensure we don't have more components than features
        n_components = min(2, all_features.shape[1])
        reducer = PCA(n_components=n_components, random_state=42)
    
    print(f'Reducing dimensions using {method.upper()}...')
    try:
        reduced_features = reducer.fit_transform(all_features)
        
        # Check for NaN in reduced features
        if np.any(np.isnan(reduced_features)) or np.any(np.isinf(reduced_features)):
            print('Warning: Reduced features contain NaN or Inf, using PCA fallback...')
            # Fallback to PCA
            reducer = PCA(n_components=min(2, all_features.shape[1]), random_state=42)
            reduced_features = reducer.fit_transform(all_features)
            # Clean again
            reduced_features = np.nan_to_num(reduced_features, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception as e:
        print(f'Error in dimensionality reduction: {e}, using PCA fallback...')
        # Fallback to PCA
        reducer = PCA(n_components=min(2, all_features.shape[1]), random_state=42)
        reduced_features = reducer.fit_transform(all_features)
        # Clean
        reduced_features = np.nan_to_num(reduced_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot text features
    text_2d = reduced_features[text_indices]
    plt.scatter(text_2d[:, 0], text_2d[:, 1], c='blue', alpha=0.6, 
               label='Text Features', s=30)
    
    # Plot image features
    image_2d = reduced_features[image_indices]
    plt.scatter(image_2d[:, 0], image_2d[:, 1], c='red', alpha=0.6, 
               label='Image Features', s=30)
    
    # Highlight positive pairs
    positive_count = 0
    for text_id, gt_image_ids in ground_truth.items():
        if text_id not in text_features:
            continue
        
        try:
            text_idx_in_all = text_indices[text_ids.index(text_id)]
            text_pos = reduced_features[text_idx_in_all]
            
            for img_id in gt_image_ids:
                img_id_str = str(img_id).zfill(6)
                if img_id_str in image_ids:
                    img_idx_in_all = image_indices[image_ids.index(img_id_str)]
                    img_pos = reduced_features[img_idx_in_all]
                    
                    # Draw line connecting positive pairs
                    plt.plot([text_pos[0], img_pos[0]], [text_pos[1], img_pos[1]], 
                            'g-', alpha=0.3, linewidth=0.5)
                    positive_count += 1
                    if positive_count >= 50:  # Limit connections for clarity
                        break
        except:
            pass
        
        if positive_count >= 50:
            break
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=20)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved feature distribution plot to {output_path}')


def plot_similarity_distribution(positive_similarities_t2i, negative_similarities_t2i,
                                positive_similarities_i2t, negative_similarities_i2t,
                                output_path, title='Similarity Distribution'):
    """
    Plot distribution of positive and negative similarities for both directions
    
    Args:
        positive_similarities_t2i: Array of T2I positive pair similarities
        negative_similarities_t2i: Array of T2I negative pair similarities
        positive_similarities_i2t: Array of I2T positive pair similarities
        negative_similarities_i2t: Array of I2T negative pair similarities
        output_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Text-to-Image
    ax1.hist(positive_similarities_t2i, bins=50, alpha=0.7, label='Positive Pairs', 
             color='green', density=True)
    ax1.hist(negative_similarities_t2i, bins=50, alpha=0.7, label='Negative Pairs', 
             color='red', density=True)
    
    ax1.axvline(np.mean(positive_similarities_t2i), color='green', linestyle='--', 
                linewidth=2, label=f'Pos Mean: {np.mean(positive_similarities_t2i):.3f}')
    ax1.axvline(np.mean(negative_similarities_t2i), color='red', linestyle='--', 
                linewidth=2, label=f'Neg Mean: {np.mean(negative_similarities_t2i):.3f}')
    
    ax1.set_xlabel('Cosine Similarity', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Text → Image', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Image-to-Text
    ax2.hist(positive_similarities_i2t, bins=50, alpha=0.7, label='Positive Pairs', 
             color='green', density=True)
    ax2.hist(negative_similarities_i2t, bins=50, alpha=0.7, label='Negative Pairs', 
             color='red', density=True)
    
    ax2.axvline(np.mean(positive_similarities_i2t), color='green', linestyle='--', 
                linewidth=2, label=f'Pos Mean: {np.mean(positive_similarities_i2t):.3f}')
    ax2.axvline(np.mean(negative_similarities_i2t), color='red', linestyle='--', 
                linewidth=2, label=f'Neg Mean: {np.mean(negative_similarities_i2t):.3f}')
    
    ax2.set_xlabel('Cosine Similarity', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Image → Text', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved bidirectional similarity distribution to {output_path}')


def plot_similarity_distribution_combined(positive_similarities, negative_similarities,
                                          output_path, title='Similarity Distribution (Combined)'):
    """
    Plot distribution of positive and negative similarities (combined T2I and I2T)
    
    Args:
        positive_similarities: Array of positive pair similarities (T2I + I2T combined)
        negative_similarities: Array of negative pair similarities (T2I + I2T combined)
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot histograms
    ax.hist(positive_similarities, bins=50, alpha=0.7, label='Positive Pairs', 
            color='green', density=True, edgecolor='darkgreen', linewidth=1.2)
    ax.hist(negative_similarities, bins=50, alpha=0.7, label='Negative Pairs', 
            color='red', density=True, edgecolor='darkred', linewidth=1.2)
    
    # Add mean lines
    pos_mean = np.mean(positive_similarities)
    neg_mean = np.mean(negative_similarities)
    separation = pos_mean - neg_mean
    
    ax.axvline(pos_mean, color='green', linestyle='--', 
               linewidth=2.5, label=f'Pos Mean: {pos_mean:.3f}')
    ax.axvline(neg_mean, color='red', linestyle='--', 
               linewidth=2.5, label=f'Neg Mean: {neg_mean:.3f}')
    
    # Add separation annotation
    ax.annotate('', xy=(pos_mean, ax.get_ylim()[1]*0.9), xytext=(neg_mean, ax.get_ylim()[1]*0.9),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text((pos_mean + neg_mean)/2, ax.get_ylim()[1]*0.92, 
            f'Separation: {separation:.3f}',
            ha='center', fontsize=11, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.8))
    
    ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved combined similarity distribution to {output_path}')


def plot_alignment_comparison(metrics_dict_t2i, metrics_dict_i2t, output_path, title='Alignment Comparison'):
    """
    Plot comparison of alignment metrics across different configurations for both directions
    
    Args:
        metrics_dict_t2i: Dictionary of {config_name: metrics_dict} for T2I
        metrics_dict_i2t: Dictionary of {config_name: metrics_dict} for I2T
        output_path: Path to save the plot
        title: Plot title
    """
    configs = list(metrics_dict_t2i.keys())
    
    # Extract T2I metrics
    positive_means_t2i = [metrics_dict_t2i[cfg]['positive_mean'] for cfg in configs]
    negative_means_t2i = [metrics_dict_t2i[cfg]['negative_mean'] for cfg in configs]
    separations_t2i = [metrics_dict_t2i[cfg]['separation'] for cfg in configs]
    
    # Extract I2T metrics
    positive_means_i2t = [metrics_dict_i2t[cfg]['positive_mean'] for cfg in configs]
    negative_means_i2t = [metrics_dict_i2t[cfg]['negative_mean'] for cfg in configs]
    separations_i2t = [metrics_dict_i2t[cfg]['separation'] for cfg in configs]
    
    x = np.arange(len(configs))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Text-to-Image
    bars1_t2i = ax1.bar(x - width, positive_means_t2i, width, label='Positive Mean', color='green', alpha=0.7)
    bars2_t2i = ax1.bar(x, negative_means_t2i, width, label='Negative Mean', color='red', alpha=0.7)
    bars3_t2i = ax1.bar(x + width, separations_t2i, width, label='Separation', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Similarity', fontsize=11)
    ax1.set_title('Text → Image', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Image-to-Text
    bars1_i2t = ax2.bar(x - width, positive_means_i2t, width, label='Positive Mean', color='green', alpha=0.7)
    bars2_i2t = ax2.bar(x, negative_means_i2t, width, label='Negative Mean', color='red', alpha=0.7)
    bars3_i2t = ax2.bar(x + width, separations_i2t, width, label='Separation', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Similarity', fontsize=11)
    ax2.set_title('Image → Text', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved bidirectional alignment comparison to {output_path}')

