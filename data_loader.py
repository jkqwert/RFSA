# -*- coding: utf-8 -*-
'''
Data Loader for Mapping Training
Loads training/validation/test sets with aligned text and image features
'''

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np


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


class MappingDataset(Dataset):
    """Dataset for text-to-image mapping training"""
    
    def __init__(
        self,
        texts_jsonl: str,
        create_jsonl: str,
        text_features_dir: str,
        image_features_dir: str,
        split: str = 'train',
        component_types: List[str] = ['subject', 'object', 'second', 'relation']
    ):
        """
        Args:
            texts_jsonl: Path to texts.jsonl file (train_texts.jsonl, valid_texts.jsonl, test_texts.jsonl)
            create_jsonl: Path to create.jsonl file with component text annotations
            text_features_dir: Directory containing text feature files
            image_features_dir: Directory containing image feature files
            split: Dataset split ('train', 'valid', 'test')
            component_types: List of component types
        """
        self.split = split
        self.component_types = component_types
        
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
                with open(feature_file, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                    # Convert to numpy arrays and normalize keys
                    self.text_features[comp_type] = {
                        normalize_feature_key(text_id): np.array(feat) 
                        for text_id, feat in features.items()
                    }
            else:
                print(f"Warning: {feature_file} not found, using empty features for {comp_type}")
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
                with open(feature_file, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                    # Convert to numpy arrays and normalize keys
                    self.image_features[comp_type] = {
                        normalize_feature_key(img_id): np.array(feat) 
                        for img_id, feat in features.items()
                    }
            else:
                print(f"Warning: {feature_file} not found, using empty features for {comp_type}")
                self.image_features[comp_type] = {}
        
        print(f"Loaded {len(self.text_image_pairs)} {split} samples")
        print(f"Text features: {[f'{k}: {len(v)}' for k, v in self.text_features.items()]}")
        print(f"Image features: {[f'{k}: {len(v)}' for k, v in self.image_features.items()]}")
        
        # Debug: Check a sample to see if image features can be found
        if len(self.text_image_pairs) > 0:
            sample_pair = self.text_image_pairs[0]
            sample_text_id = sample_pair['text_id']
            sample_image_ids = sample_pair.get('image_ids', [])
            print(f"\nSample check - text_id: {sample_text_id}, image_ids: {sample_image_ids}")
            if sample_image_ids:
                sample_img_id = str(sample_image_ids[0]).zfill(6)
                sample_img_id_no_zeros = str(int(sample_image_ids[0]))
                print(f"  Trying to find image_id: {sample_img_id} or {sample_img_id_no_zeros}")
                for comp_type in self.component_types:
                    found = False
                    # Try normalized formats
                    if sample_img_id in self.image_features[comp_type]:
                        feat_norm = np.linalg.norm(self.image_features[comp_type][sample_img_id])
                        print(f"  {comp_type}: Found image feature (ID: {sample_img_id}, norm: {feat_norm:.4f})")
                        found = True
                    elif sample_img_id_no_zeros in self.image_features[comp_type]:
                        feat_norm = np.linalg.norm(self.image_features[comp_type][sample_img_id_no_zeros])
                        print(f"  {comp_type}: Found image feature (ID: {sample_img_id_no_zeros}, norm: {feat_norm:.4f})")
                        found = True
                    
                    if not found:
                        print(f"  {comp_type}: Image feature NOT FOUND")
                        # Show available IDs for this component (first 5)
                        avail_ids = list(self.image_features[comp_type].keys())[:5]
                        print(f"    Available IDs (first 5): {avail_ids}")
    
    def __len__(self):
        return len(self.text_image_pairs)
    
    def __getitem__(self, idx):
        pair = self.text_image_pairs[idx]
        text_id = pair['text_id']
        image_ids = pair['image_ids']
        
        # Get text features for each component
        text_features_dict = {}
        for comp_type in self.component_types:
            if text_id in self.text_features[comp_type]:
                text_features_dict[comp_type] = self.text_features[comp_type][text_id]
            else:
                # Use zero vector if feature not found
                if text_features_dict:
                    text_features_dict[comp_type] = np.zeros_like(list(text_features_dict.values())[0])
                else:
                    # Default to 512-dim zero vector
                    text_features_dict[comp_type] = np.zeros(512)
        
        # Get image features for each component
        # Each component should load its corresponding component image feature
        # The image_id from train_texts.jsonl is the full image ID
        # We need to find the corresponding component image features
        image_features_dict = {}
        
        if image_ids:
            # Use first image ID (full image ID)
            first_image_id = image_ids[0]
            
            # Normalize the image ID
            normalized_img_id = str(first_image_id).zfill(6)
            
            # For each component, try to find matching image feature
            for comp_type in self.component_types:
                found = False
                
                # Try normalized ID directly
                if normalized_img_id in self.image_features[comp_type]:
                    image_features_dict[comp_type] = self.image_features[comp_type][normalized_img_id]
                    found = True
                
                # Try without leading zeros
                if not found:
                    try:
                        img_id_no_zeros = str(int(first_image_id))
                        if img_id_no_zeros in self.image_features[comp_type]:
                            image_features_dict[comp_type] = self.image_features[comp_type][img_id_no_zeros]
                            found = True
                    except (ValueError, TypeError):
                        pass
                
                if not found:
                    # If not found, check if we have any features for this component
                    # Maybe the component image doesn't exist for this image_id
                    # In that case, we should use zero vector only if the text component also doesn't exist
                    # But if text component exists, we need to handle this differently
                    
                    # Check if text component exists for this text_id
                    text_comp_exists = (text_id in self.text_features[comp_type] and 
                                       self.text_features[comp_type][text_id] is not None and
                                       np.linalg.norm(self.text_features[comp_type][text_id]) > 0)
                    
                    if text_comp_exists:
                        # Text exists but image doesn't - this is a problem
                        # Use a small random vector instead of zero to allow learning
                        # Or use zero vector (will be handled by loss function)
                        if image_features_dict:
                            image_features_dict[comp_type] = np.zeros_like(list(image_features_dict.values())[0])
                        else:
                            image_features_dict[comp_type] = np.zeros(512)
                    else:
                        # Both text and image don't exist - use zero vector
                        if image_features_dict:
                            image_features_dict[comp_type] = np.zeros_like(list(image_features_dict.values())[0])
                        else:
                            image_features_dict[comp_type] = np.zeros(512)
        else:
            # No matching images (for test set without ground truth)
            for comp_type in self.component_types:
                if image_features_dict:
                    image_features_dict[comp_type] = np.zeros_like(list(image_features_dict.values())[0])
                else:
                    image_features_dict[comp_type] = np.zeros(512)
        
        # Convert to tensors
        text_features_tensor = {
            comp_type: torch.FloatTensor(text_features_dict[comp_type])
            for comp_type in self.component_types
        }
        
        image_features_tensor = {
            comp_type: torch.FloatTensor(image_features_dict[comp_type])
            for comp_type in self.component_types
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
    component_types: List[str] = ['subject', 'object', 'second', 'relation']
) -> DataLoader:
    """
    Create a DataLoader for mapping training
    
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
    
    Returns:
        DataLoader instance
    """
    dataset = MappingDataset(
        texts_jsonl=texts_jsonl,
        create_jsonl=create_jsonl,
        text_features_dir=text_features_dir,
        image_features_dir=image_features_dir,
        split=split,
        component_types=component_types
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

