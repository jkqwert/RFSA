# -*- coding: utf-8 -*-
'''
Feature loading utilities for ablation study
'''

import os
import json
import numpy as np
import torch
from tqdm import tqdm


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


def load_all_image_features_ablation(image_features_dir, component_types, mapping_module, device, use_component, use_shared_space):
    """
    Load all image features and map to target space
    
    Args:
        image_features_dir: Directory containing image feature files
        component_types: List of component types
        mapping_module: Trained mapping module
        device: Device
        use_component: If True, use component features; If False, use full features
        use_shared_space: If True, map to shared space; If False, return original features
    
    Returns:
        image_features_dict: Dictionary of {image_id: feature}
    """
    mapping_module.eval()
    image_features_dict = {}
    
    def _load_features_file(file_path):
        """Load features from JSON or JSONL file"""
        features = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    features = data
                else:
                    raise ValueError("JSON file is not a dictionary")
        except (json.JSONDecodeError, ValueError):
            # Try JSONL format
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            if 'text_id' in obj or 'image_id' in obj or 'id' in obj:
                                feat_id = obj.get('text_id') or obj.get('image_id') or obj.get('id')
                                feat = None
                                for key in ['feature', 'feat', 'embedding', 'emb']:
                                    if key in obj:
                                        feat = obj[key]
                                        break
                                if feat is None:
                                    for k, v in obj.items():
                                        if k not in ['text_id', 'image_id', 'id'] and isinstance(v, (list, tuple)):
                                            feat = v
                                            break
                                if feat is not None:
                                    features[str(feat_id)] = feat
                            else:
                                features.update(obj)
                    except json.JSONDecodeError:
                        continue
        return features
    
    if use_component:
        # Load component features
        image_type_map = {
            'subject': 'subject',
            'object': 'object',
            'second': 'second_object',
            'relation': 'relation'
        }
        
        all_image_features = {}
        all_image_ids = set()
        
        for comp_type in component_types:
            image_type = image_type_map[comp_type]
            feature_file = os.path.join(image_features_dir, f"{image_type}_features.json")
            
            if os.path.exists(feature_file):
                features = _load_features_file(feature_file)
                print(f'  Loaded {len(features)} features from {image_type}_features.json')
                if features:
                    sample_key = list(features.keys())[0]
                    print(f'    Sample key: {sample_key} -> normalized: {normalize_feature_key(sample_key)}')
                
                for img_id, feat in features.items():
                    # Normalize the image ID
                    normalized_img_id = normalize_feature_key(img_id)
                    
                    if normalized_img_id not in all_image_features:
                        all_image_features[normalized_img_id] = {}
                    all_image_features[normalized_img_id][comp_type] = np.array(feat)
                    all_image_ids.add(normalized_img_id)
            else:
                print(f'  Warning: {feature_file} not found')
        
        if use_shared_space:
            # Map to shared space
            print(f'Mapping {len(all_image_ids)} images to shared space...')
            batch_size = 64
            image_ids_list = sorted(list(all_image_ids))
            
            with torch.no_grad():
                for i in range(0, len(image_ids_list), batch_size):
                    batch_ids = image_ids_list[i:i+batch_size]
                    
                    batch_image_features = {}
                    for comp_type in component_types:
                        batch_feats = []
                        for img_id in batch_ids:
                            if img_id in all_image_features and comp_type in all_image_features[img_id]:
                                batch_feats.append(all_image_features[img_id][comp_type])
                            else:
                                batch_feats.append(np.zeros(512))
                        
                        batch_image_features[comp_type] = torch.FloatTensor(np.array(batch_feats)).to(device)
                    
                    image_shared = mapping_module.forward_image(batch_image_features)
                    
                    for img_id, feat in zip(batch_ids, image_shared.cpu().numpy()):
                        image_features_dict[img_id] = feat
        else:
            # Fuse component features (mean)
            for img_id in all_image_ids:
                component_feats = []
                for comp_type in component_types:
                    if img_id in all_image_features and comp_type in all_image_features[img_id]:
                        component_feats.append(all_image_features[img_id][comp_type])
                
                if component_feats:
                    fused_feat = np.mean(component_feats, axis=0)
                    image_features_dict[img_id] = fused_feat
    else:
        # Load full features
        full_image_feature_file = os.path.join(image_features_dir, 'full_image_features.json')
        
        if os.path.exists(full_image_feature_file):
            features = _load_features_file(full_image_feature_file)
            print(f'  Loaded {len(features)} full image features')
            if features:
                sample_key = list(features.keys())[0]
                print(f'    Sample key: {sample_key} -> normalized: {normalize_feature_key(sample_key)}')
            
            # Normalize feature keys
            normalized_features = {
                normalize_feature_key(img_id): np.array(feat)
                for img_id, feat in features.items()
            }
            
            if use_shared_space:
                # Map to shared space
                print(f'Mapping {len(normalized_features)} images to shared space...')
                batch_size = 64
                image_ids_list = sorted(list(normalized_features.keys()))
                
                with torch.no_grad():
                    for i in range(0, len(image_ids_list), batch_size):
                        batch_ids = image_ids_list[i:i+batch_size]
                        
                        batch_feats = []
                        for img_id in batch_ids:
                            batch_feats.append(normalized_features[img_id])
                        
                        batch_image_features = {'full': torch.FloatTensor(np.array(batch_feats)).to(device)}
                        image_shared = mapping_module.forward_image(batch_image_features)
                        
                        for img_id, feat in zip(batch_ids, image_shared.cpu().numpy()):
                            image_features_dict[img_id] = feat
            else:
                # Return original features
                image_features_dict = normalized_features
    
    print(f'Loaded {len(image_features_dict)} image features')
    return image_features_dict


def extract_mapped_features_ablation(mapping_module, data_loader, device, use_shared_space):
    """Extract mapped features for all samples"""
    mapping_module.eval()
    features_dict = {}
    ids_list = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Extracting features')
        for batch in pbar:
            text_ids = batch['text_ids']
            text_features = batch['text_features']
            
            text_features = {k: v.to(device) for k, v in text_features.items()}
            
            if use_shared_space:
                mapped_feat = mapping_module.forward_text(text_features)
            else:
                mapped_feat = mapping_module(text_features)
            
            for text_id, feat in zip(text_ids, mapped_feat.cpu().numpy()):
                features_dict[text_id] = feat
                ids_list.append(text_id)
    
    return features_dict, ids_list

