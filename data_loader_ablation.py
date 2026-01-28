# -*- coding: utf-8 -*-
'''
Data Loader for Ablation Study
Supports loading component features or full features
'''

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np


class AblationDataset(Dataset):
    """Dataset for ablation study - supports component or full features"""
    
    def __init__(
        self,
        texts_jsonl: str,
        create_jsonl: str,
        text_features_dir: str,
        image_features_dir: str,
        split: str = 'train',
        component_types: List[str] = ['subject', 'object', 'second', 'relation'],
        use_component: bool = True
    ):
        """
        Args:
            texts_jsonl: Path to texts.jsonl file
            create_jsonl: Path to create.jsonl file with component text annotations
            text_features_dir: Directory containing text feature files
            image_features_dir: Directory containing image feature files
            split: Dataset split ('train', 'valid', 'test')
            component_types: List of component types
            use_component: If True, use component features; If False, use full features
        """
        self.split = split
        self.component_types = component_types
        self.use_component = use_component
        
        # Load text-image pairs
        self.text_image_pairs = []
        with open(texts_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                text_id = obj['text_id']
                image_ids = obj.get('image_ids', [])
                self.text_image_pairs.append({
                    'text_id': text_id,
                    'image_ids': image_ids
                })
        
        if use_component:
            # Load component text annotations
            self.component_texts = {}
            with open(create_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line.strip())
                    text_id = obj['text_id']
                    self.component_texts[text_id] = {
                        'subject': obj.get('subject', ''),
                        'object': obj.get('object', ''),
                        'second': obj.get('second', ''),
                        'relation': obj.get('relation', '')
                    }
            
            # Load text features for each component
            self.text_features = {}
            for comp_type in component_types:
                feature_file = os.path.join(text_features_dir, f"{comp_type}_text_features.json")
                if os.path.exists(feature_file):
                    features = self._load_features_file(feature_file)
                    self.text_features[comp_type] = {
                        text_id: np.array(feat) for text_id, feat in features.items()
                    }
                else:
                    print(f"Warning: {feature_file} not found")
                    self.text_features[comp_type] = {}
            
            # Load image features for each component
            self.image_features = {}
            image_type_map = {
                'subject': 'subject',
                'object': 'object',
                'second': 'second_object',
                'relation': 'relation'
            }
            
            for comp_type in component_types:
                image_type = image_type_map[comp_type]
                feature_file = os.path.join(image_features_dir, f"{image_type}_features.json")
                if os.path.exists(feature_file):
                    features = self._load_features_file(feature_file)
                    self.image_features[comp_type] = {
                        img_id: np.array(feat) for img_id, feat in features.items()
                    }
                else:
                    print(f"Warning: {feature_file} not found")
                    self.image_features[comp_type] = {}
        else:
            # Load full text features
            full_text_feature_file = os.path.join(text_features_dir, 'full_text_features.json')
            if os.path.exists(full_text_feature_file):
                features = self._load_features_file(full_text_feature_file)
                self.text_features = {'full': {
                    text_id: np.array(feat) for text_id, feat in features.items()
                }}
            else:
                print(f"Warning: {full_text_feature_file} not found")
                self.text_features = {'full': {}}
            
            # Load full image features
            full_image_feature_file = os.path.join(image_features_dir, 'full_image_features.json')
            if os.path.exists(full_image_feature_file):
                features = self._load_features_file(full_image_feature_file)
                self.image_features = {'full': {
                    img_id: np.array(feat) for img_id, feat in features.items()
                }}
            else:
                print(f"Warning: {full_image_feature_file} not found")
                self.image_features = {'full': {}}
        
        print(f"Loaded {len(self.text_image_pairs)} {split} samples")
        print(f"Use component: {use_component}")
        if use_component:
            print(f"Text features: {[f'{k}: {len(v)}' for k, v in self.text_features.items()]}")
            print(f"Image features: {[f'{k}: {len(v)}' for k, v in self.image_features.items()]}")
        else:
            print(f"Full text features: {len(self.text_features.get('full', {}))}")
            print(f"Full image features: {len(self.image_features.get('full', {}))}")
    
    def _load_features_file(self, file_path):
        """
        Load features from JSON or JSONL file
        
        Args:
            file_path: Path to feature file
        
        Returns:
            Dictionary of {id: feature}
        """
        features = {}
        try:
            # Try JSON format first (single JSON object)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    features = data
                else:
                    raise ValueError("JSON file is not a dictionary")
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON fails, try JSONL format (one JSON object per line)
            print(f"Warning: Failed to load {file_path} as JSON, trying JSONL format...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # JSONL format: each line is a JSON object with 'id' and 'feature' keys
                        # Or it could be a dict with id as key
                        if isinstance(obj, dict):
                            # Check if it's a single feature object
                            if 'text_id' in obj or 'image_id' in obj or 'id' in obj:
                                feat_id = obj.get('text_id') or obj.get('image_id') or obj.get('id')
                                # Try to find feature vector
                                feat = None
                                for key in ['feature', 'feat', 'embedding', 'emb']:
                                    if key in obj:
                                        feat = obj[key]
                                        break
                                if feat is None:
                                    # Get the first list/array value that's not an ID
                                    for k, v in obj.items():
                                        if k not in ['text_id', 'image_id', 'id'] and isinstance(v, (list, tuple)):
                                            feat = v
                                            break
                                if feat is not None:
                                    features[str(feat_id)] = feat
                            else:
                                # It's already a dict with id as key
                                features.update(obj)
                    except json.JSONDecodeError:
                        if line_num <= 3:  # Only print first few errors
                            print(f"Warning: Failed to parse line {line_num} in {file_path}")
                        continue
        
        if len(features) == 0:
            print(f"Warning: No features loaded from {file_path}")
        else:
            # Check first feature to see if it's valid
            first_id = list(features.keys())[0]
            first_feat = features[first_id]
            if isinstance(first_feat, list):
                feat_norm = sum(x*x for x in first_feat[:10])  # Check first 10 elements
                if feat_norm < 1e-10:
                    print(f"Warning: Features in {file_path} appear to be all zeros!")
            print(f"Loaded {len(features)} features from {file_path}")
        
        return features
    
    def __len__(self):
        return len(self.text_image_pairs)
    
    def __getitem__(self, idx):
        pair = self.text_image_pairs[idx]
        text_id = pair['text_id']
        image_ids = pair['image_ids']
        
        # Load text features
        text_features_dict = {}
        if self.use_component:
            for comp_type in self.component_types:
                if comp_type in self.text_features and text_id in self.text_features[comp_type]:
                    text_features_dict[comp_type] = self.text_features[comp_type][text_id]
                else:
                    # Use zero vector if not found
                    if text_features_dict:
                        text_features_dict[comp_type] = np.zeros_like(list(text_features_dict.values())[0])
                    else:
                        text_features_dict[comp_type] = np.zeros(512)
        else:
            if 'full' in self.text_features and text_id in self.text_features['full']:
                text_features_dict['full'] = self.text_features['full'][text_id]
            else:
                text_features_dict['full'] = np.zeros(512)
        
        # Load image features
        image_features_dict = {}
        if image_ids:
            first_image_id = image_ids[0]
            img_id_formats = [
                str(first_image_id).zfill(6),
                str(first_image_id),
                str(int(first_image_id)) if isinstance(first_image_id, str) else str(first_image_id),
            ]
            
            if self.use_component:
                for comp_type in self.component_types:
                    found = False
                    for img_id_str in img_id_formats:
                        if comp_type in self.image_features and img_id_str in self.image_features[comp_type]:
                            image_features_dict[comp_type] = self.image_features[comp_type][img_id_str]
                            found = True
                            break
                        try:
                            img_id_num = int(img_id_str)
                            tensor_id = f'tensor({img_id_num})'
                            if comp_type in self.image_features and tensor_id in self.image_features[comp_type]:
                                image_features_dict[comp_type] = self.image_features[comp_type][tensor_id]
                                found = True
                                break
                        except (ValueError, TypeError):
                            pass
                    
                    if not found:
                        # Use zero vector if not found
                        if image_features_dict:
                            image_features_dict[comp_type] = np.zeros_like(list(image_features_dict.values())[0])
                        else:
                            image_features_dict[comp_type] = np.zeros(512)
            else:
                # Full image features
                found = False
                for img_id_str in img_id_formats:
                    if 'full' in self.image_features and img_id_str in self.image_features['full']:
                        image_features_dict['full'] = self.image_features['full'][img_id_str]
                        found = True
                        break
                    try:
                        img_id_num = int(img_id_str)
                        tensor_id = f'tensor({img_id_num})'
                        if 'full' in self.image_features and tensor_id in self.image_features['full']:
                            image_features_dict['full'] = self.image_features['full'][tensor_id]
                            found = True
                            break
                    except (ValueError, TypeError):
                        pass
                
                if not found:
                    image_features_dict['full'] = np.zeros(512)
        else:
            # No matching images
            if self.use_component:
                for comp_type in self.component_types:
                    if image_features_dict:
                        image_features_dict[comp_type] = np.zeros_like(list(image_features_dict.values())[0])
                    else:
                        image_features_dict[comp_type] = np.zeros(512)
            else:
                image_features_dict['full'] = np.zeros(512)
        
        # Convert to tensors
        text_features_tensor = {
            k: torch.FloatTensor(v) for k, v in text_features_dict.items()
        }
        
        image_features_tensor = {
            k: torch.FloatTensor(v) for k, v in image_features_dict.items()
        }
        
        return {
            'text_id': text_id,
            'image_ids': image_ids,
            'text_features': text_features_tensor,
            'image_features': image_features_tensor
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    text_ids = [item['text_id'] for item in batch]
    image_ids_list = [item['image_ids'] for item in batch]
    
    # Stack text features
    text_features = {}
    for comp_type in batch[0]['text_features'].keys():
        text_features[comp_type] = torch.stack([
            item['text_features'][comp_type] for item in batch
        ])
    
    # Stack image features
    image_features = {}
    for comp_type in batch[0]['image_features'].keys():
        image_features[comp_type] = torch.stack([
            item['image_features'][comp_type] for item in batch
        ])
    
    return {
        'text_ids': text_ids,
        'image_ids': image_ids_list,
        'text_features': text_features,
        'image_features': image_features
    }


def get_data_loader(
    texts_jsonl: str,
    create_jsonl: str,
    text_features_dir: str,
    image_features_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    component_types: List[str] = ['subject', 'object', 'second', 'relation'],
    use_component: bool = True
) -> DataLoader:
    """
    Create a DataLoader for ablation study
    
    Args:
        texts_jsonl: Path to texts.jsonl file
        create_jsonl: Path to create.jsonl file
        text_features_dir: Directory containing text feature files
        image_features_dir: Directory containing image feature files
        split: Dataset split
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        component_types: List of component types
        use_component: If True, use component features; If False, use full features
    
    Returns:
        DataLoader instance
    """
    dataset = AblationDataset(
        texts_jsonl=texts_jsonl,
        create_jsonl=create_jsonl,
        text_features_dir=text_features_dir,
        image_features_dir=image_features_dir,
        split=split,
        component_types=component_types,
        use_component=use_component
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader

