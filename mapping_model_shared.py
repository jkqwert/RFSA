# -*- coding: utf-8 -*-
'''
Shared Feature Space Mapping Module
Maps both text and image features to a common feature space
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PromptLearner(nn.Module):
    """Learnable prompt embeddings for four components"""
    
    def __init__(self, embed_dim=512, prompt_length=4, component_types=['subject', 'object', 'second', 'relation']):
        super(PromptLearner, self).__init__()
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.component_types = component_types
        
        # Create learnable prompt embeddings for each component type
        self.prompts = nn.ParameterDict()
        for comp_type in component_types:
            prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)
            self.prompts[comp_type] = prompt
        
        self.alpha = nn.Parameter(torch.ones(len(component_types)))
    
    def forward(self, feature, component_type):
        """Enhance feature with learnable prompt"""
        if component_type not in self.prompts:
            return feature
        
        prompt = self.prompts[component_type]
        prompt_enhancement = prompt.mean(dim=0)  # [embed_dim]
        
        batch_size = feature.size(0)
        prompt_enhancement = prompt_enhancement.unsqueeze(0).expand(batch_size, -1)
        
        enhanced_feature = feature + prompt_enhancement * 0.1
        return enhanced_feature


class ComponentMapping(nn.Module):
    """MLP mapping module for a single component"""
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512, num_layers=2, dropout=0.1):
        super(ComponentMapping, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        output_linear = nn.Linear(in_dim, output_dim)
        nn.init.xavier_uniform_(output_linear.weight, gain=0.1)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class SharedMappingModule(nn.Module):
    """
    Shared feature space mapping module
    Maps both text and image features to a common feature space
    """
    
    def __init__(
        self,
        embed_dim=512,
        prompt_length=4,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        fusion_method='weighted_sum',
        component_types=['subject', 'object', 'second', 'relation']
    ):
        """
        Args:
            embed_dim: Feature embedding dimension (default: 512)
            prompt_length: Length of learnable prompt tokens
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of layers in MLP
            dropout: Dropout rate
            fusion_method: Method to fuse component features ('weighted_sum', 'concat', 'attention')
            component_types: List of component types
        """
        super(SharedMappingModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.component_types = component_types
        self.fusion_method = fusion_method
        
        # Shared prompt learner (used for both text and image)
        self.prompt_learner = PromptLearner(embed_dim, prompt_length, component_types)
        
        # Text mapping modules (text -> shared space)
        self.text_mappings = nn.ModuleDict()
        for comp_type in component_types:
            self.text_mappings[comp_type] = ComponentMapping(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Image mapping modules (image -> shared space)
        self.image_mappings = nn.ModuleDict()
        for comp_type in component_types:
            self.image_mappings[comp_type] = ComponentMapping(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Fusion weights (learnable)
        if fusion_method == 'weighted_sum':
            self.fusion_weights = nn.Parameter(torch.ones(len(component_types)) / len(component_types))
        elif fusion_method == 'concat':
            self.fusion_proj = nn.Linear(embed_dim * len(component_types), embed_dim)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.fusion_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward_text(self, text_features_dict):
        """
        Map text features to shared feature space
        
        Args:
            text_features_dict: Dictionary of {component_type: text_feature}
                text_feature: [batch_size, embed_dim]
        
        Returns:
            Mapped feature in shared space [batch_size, embed_dim]
        """
        mapped_features = []
        
        for comp_type in self.component_types:
            if comp_type not in text_features_dict:
                if text_features_dict:
                    zero_feat = torch.zeros_like(list(text_features_dict.values())[0])
                else:
                    zero_feat = torch.zeros(1, self.embed_dim, 
                                          device=next(iter(text_features_dict.values())).device if text_features_dict else 'cpu')
                mapped_features.append(zero_feat)
                continue
            
            text_feat = text_features_dict[comp_type]
            
            # Enhance with learnable prompt
            enhanced_feat = self.prompt_learner(text_feat, comp_type)
            
            # Map to shared space
            mapped_feat = self.text_mappings[comp_type](enhanced_feat)
            
            # Normalize
            norm = mapped_feat.norm(p=2, dim=-1, keepdim=True)
            mapped_feat = mapped_feat / (norm + 1e-8)
            
            mapped_features.append(mapped_feat)
        
        # Fuse component features
        fused_feature = self._fuse_features(mapped_features)
        
        # Final normalization
        norm = fused_feature.norm(p=2, dim=-1, keepdim=True)
        fused_feature = fused_feature / (norm + 1e-8)
        
        return fused_feature
    
    def forward_image(self, image_features_dict):
        """
        Map image features to shared feature space
        
        Args:
            image_features_dict: Dictionary of {component_type: image_feature}
                image_feature: [batch_size, embed_dim]
        
        Returns:
            Mapped feature in shared space [batch_size, embed_dim]
        """
        mapped_features = []
        
        for comp_type in self.component_types:
            if comp_type not in image_features_dict:
                if image_features_dict:
                    zero_feat = torch.zeros_like(list(image_features_dict.values())[0])
                else:
                    zero_feat = torch.zeros(1, self.embed_dim,
                                          device=next(iter(image_features_dict.values())).device if image_features_dict else 'cpu')
                mapped_features.append(zero_feat)
                continue
            
            image_feat = image_features_dict[comp_type]
            
            # Enhance with learnable prompt (shared prompts)
            enhanced_feat = self.prompt_learner(image_feat, comp_type)
            
            # Map to shared space
            mapped_feat = self.image_mappings[comp_type](enhanced_feat)
            
            # Normalize
            norm = mapped_feat.norm(p=2, dim=-1, keepdim=True)
            mapped_feat = mapped_feat / (norm + 1e-8)
            
            mapped_features.append(mapped_feat)
        
        # Fuse component features
        fused_feature = self._fuse_features(mapped_features)
        
        # Final normalization
        norm = fused_feature.norm(p=2, dim=-1, keepdim=True)
        fused_feature = fused_feature / (norm + 1e-8)
        
        return fused_feature
    
    def _fuse_features(self, mapped_features):
        """Fuse component features"""
        if self.fusion_method == 'weighted_sum':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_feature = sum(w * feat for w, feat in zip(weights, mapped_features))
        elif self.fusion_method == 'concat':
            concat_feat = torch.cat(mapped_features, dim=-1)
            fused_feature = self.fusion_proj(concat_feat)
        elif self.fusion_method == 'attention':
            stacked = torch.stack(mapped_features, dim=1)
            query = stacked.mean(dim=1, keepdim=True)
            attended, _ = self.attention(query, stacked, stacked)
            fused_feature = self.fusion_proj(attended.squeeze(1))
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_feature



