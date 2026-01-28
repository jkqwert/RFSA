# -*- coding: utf-8 -*-
'''
Ablation Study Mapping Module
Supports three modules: prompt learning, component fusion, shared space mapping
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PromptLearner(nn.Module):
    """Learnable prompt embeddings for four components"""
    
    def __init__(self, embed_dim=512, prompt_length=4, component_types=['subject', 'object', 'second', 'relation'], use_prompt=True):
        super(PromptLearner, self).__init__()
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.component_types = component_types
        self.use_prompt = use_prompt
        
        if use_prompt:
            self.prompts = nn.ParameterDict()
            for comp_type in component_types:
                prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)
                self.prompts[comp_type] = prompt
            self.alpha = nn.Parameter(torch.ones(len(component_types)))
        else:
            # No prompt learning, just return original features
            self.prompts = nn.ParameterDict()
            for comp_type in component_types:
                # Dummy parameter to maintain structure
                self.prompts[comp_type] = nn.Parameter(torch.zeros(prompt_length, embed_dim))
    
    def forward(self, feature, component_type):
        if not self.use_prompt or component_type not in self.prompts:
            return feature
        
        prompt = self.prompts[component_type]
        prompt_enhancement = prompt.mean(dim=0)
        
        batch_size = feature.size(0)
        prompt_enhancement = prompt_enhancement.unsqueeze(0).expand(batch_size, -1)
        
        # Use learnable alpha weight if available, otherwise use fixed weight
        if hasattr(self, 'alpha') and self.alpha is not None:
            # Get alpha for this component (or use first one if only one component)
            if len(self.component_types) > 0:
                comp_idx = self.component_types.index(component_type) if component_type in self.component_types else 0
                alpha_weight = torch.sigmoid(self.alpha[comp_idx]) if len(self.alpha) > comp_idx else 0.1
            else:
                alpha_weight = torch.sigmoid(self.alpha[0]) if len(self.alpha) > 0 else 0.1
            # Convert to scalar if it's a tensor
            if isinstance(alpha_weight, torch.Tensor):
                alpha_weight = alpha_weight.item()
        else:
            alpha_weight = 0.1
        
        enhanced_feature = feature + prompt_enhancement * alpha_weight
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


class SingleDirectionMappingModule(nn.Module):
    """Single direction mapping: text -> image feature space"""
    
    def __init__(
        self,
        embed_dim=512,
        prompt_length=4,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        fusion_method='weighted_sum',
        component_types=['subject', 'object', 'second', 'relation'],
        use_prompt=True,
        use_component=True
    ):
        super(SingleDirectionMappingModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.component_types = component_types
        self.fusion_method = fusion_method
        self.use_component = use_component
        
        if use_component:
            # Component-based mapping
            self.prompt_learner = PromptLearner(embed_dim, prompt_length, component_types, use_prompt)
            self.mappings = nn.ModuleDict()
            for comp_type in component_types:
                self.mappings[comp_type] = ComponentMapping(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=embed_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
            
            if fusion_method == 'weighted_sum':
                self.fusion_weights = nn.Parameter(torch.ones(len(component_types)) / len(component_types))
            elif fusion_method == 'concat':
                self.fusion_proj = nn.Linear(embed_dim * len(component_types), embed_dim)
            elif fusion_method == 'attention':
                self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
                self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        else:
            # Single full feature mapping (no components)
            # But still need prompt learner if use_prompt is True
            if use_prompt:
                # Create a single prompt for full features (use 'full' as component type)
                self.prompt_learner = PromptLearner(embed_dim, prompt_length, ['full'], use_prompt)
            self.mapping = ComponentMapping(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout
            )
    
    def forward(self, text_features_dict):
        if self.use_component:
            # Component-based mapping
            mapped_features = []
            
            for comp_type in self.component_types:
                if comp_type not in text_features_dict:
                    if text_features_dict:
                        zero_feat = torch.zeros_like(list(text_features_dict.values())[0])
                    else:
                        zero_feat = torch.zeros(1, self.embed_dim, device='cpu')
                    mapped_features.append(zero_feat)
                    continue
                
                text_feat = text_features_dict[comp_type]
                enhanced_feat = self.prompt_learner(text_feat, comp_type)
                mapped_feat = self.mappings[comp_type](enhanced_feat)
                
                norm = mapped_feat.norm(p=2, dim=-1, keepdim=True)
                mapped_feat = mapped_feat / (norm + 1e-8)
                
                mapped_features.append(mapped_feat)
            
            # Fuse component features
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
            # Single full feature mapping
            if 'full' in text_features_dict:
                text_feat = text_features_dict['full']
            else:
                # Fallback: use first available feature
                text_feat = list(text_features_dict.values())[0]
            
            # Apply prompt learning if enabled
            if hasattr(self, 'prompt_learner') and self.prompt_learner.use_prompt:
                enhanced_feat = self.prompt_learner(text_feat, 'full')
            else:
                enhanced_feat = text_feat
            
            fused_feature = self.mapping(enhanced_feat)
        
        # Final normalization
        norm = fused_feature.norm(p=2, dim=-1, keepdim=True)
        fused_feature = fused_feature / (norm + 1e-8)
        
        return fused_feature


class SharedMappingModule(nn.Module):
    """Shared feature space mapping: both text and image -> shared space"""
    
    def __init__(
        self,
        embed_dim=512,
        prompt_length=4,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        fusion_method='weighted_sum',
        component_types=['subject', 'object', 'second', 'relation'],
        use_prompt=True,
        use_component=True
    ):
        super(SharedMappingModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.component_types = component_types
        self.fusion_method = fusion_method
        self.use_component = use_component
        
        if use_component:
            # Component-based mapping
            self.prompt_learner = PromptLearner(embed_dim, prompt_length, component_types, use_prompt)
            
            # Text mapping modules
            self.text_mappings = nn.ModuleDict()
            for comp_type in component_types:
                self.text_mappings[comp_type] = ComponentMapping(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=embed_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
            
            # Image mapping modules
            self.image_mappings = nn.ModuleDict()
            for comp_type in component_types:
                self.image_mappings[comp_type] = ComponentMapping(
                    input_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_dim=embed_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
            
            if fusion_method == 'weighted_sum':
                self.fusion_weights = nn.Parameter(torch.ones(len(component_types)) / len(component_types))
            elif fusion_method == 'concat':
                self.fusion_proj = nn.Linear(embed_dim * len(component_types), embed_dim)
            elif fusion_method == 'attention':
                self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
                self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        else:
            # Single full feature mapping
            # But still need prompt learner if use_prompt is True
            if use_prompt:
                # Create a single prompt for full features (use 'full' as component type)
                self.prompt_learner = PromptLearner(embed_dim, prompt_length, ['full'], use_prompt)
            self.text_mapping = ComponentMapping(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            self.image_mapping = ComponentMapping(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout
            )
    
    def forward_text(self, text_features_dict):
        if self.use_component:
            mapped_features = []
            
            for comp_type in self.component_types:
                if comp_type not in text_features_dict:
                    if text_features_dict:
                        zero_feat = torch.zeros_like(list(text_features_dict.values())[0])
                    else:
                        zero_feat = torch.zeros(1, self.embed_dim, device='cpu')
                    mapped_features.append(zero_feat)
                    continue
                
                text_feat = text_features_dict[comp_type]
                enhanced_feat = self.prompt_learner(text_feat, comp_type)
                mapped_feat = self.text_mappings[comp_type](enhanced_feat)
                
                norm = mapped_feat.norm(p=2, dim=-1, keepdim=True)
                mapped_feat = mapped_feat / (norm + 1e-8)
                
                mapped_features.append(mapped_feat)
            
            fused_feature = self._fuse_features(mapped_features)
        else:
            if 'full' in text_features_dict:
                text_feat = text_features_dict['full']
            else:
                text_feat = list(text_features_dict.values())[0]
            
            # Apply prompt learning if enabled
            if hasattr(self, 'prompt_learner') and self.prompt_learner.use_prompt:
                enhanced_feat = self.prompt_learner(text_feat, 'full')
            else:
                enhanced_feat = text_feat
            
            fused_feature = self.text_mapping(enhanced_feat)
        
        norm = fused_feature.norm(p=2, dim=-1, keepdim=True)
        fused_feature = fused_feature / (norm + 1e-8)
        
        return fused_feature
    
    def forward_image(self, image_features_dict):
        if self.use_component:
            mapped_features = []
            
            for comp_type in self.component_types:
                if comp_type not in image_features_dict:
                    if image_features_dict:
                        zero_feat = torch.zeros_like(list(image_features_dict.values())[0])
                    else:
                        zero_feat = torch.zeros(1, self.embed_dim, device='cpu')
                    mapped_features.append(zero_feat)
                    continue
                
                image_feat = image_features_dict[comp_type]
                enhanced_feat = self.prompt_learner(image_feat, comp_type)
                mapped_feat = self.image_mappings[comp_type](enhanced_feat)
                
                norm = mapped_feat.norm(p=2, dim=-1, keepdim=True)
                mapped_feat = mapped_feat / (norm + 1e-8)
                
                mapped_features.append(mapped_feat)
            
            fused_feature = self._fuse_features(mapped_features)
        else:
            if 'full' in image_features_dict:
                image_feat = image_features_dict['full']
            else:
                image_feat = list(image_features_dict.values())[0]
            
            # Apply prompt learning if enabled
            if hasattr(self, 'prompt_learner') and self.prompt_learner.use_prompt:
                enhanced_feat = self.prompt_learner(image_feat, 'full')
            else:
                enhanced_feat = image_feat
            
            fused_feature = self.image_mapping(enhanced_feat)
        
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

