# -*- coding: utf-8 -*-
'''
Mapping Module for Text-to-Image Feature Mapping
Includes learnable prompt learning mechanism for four components (subject, object, second, relation)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PromptLearner(nn.Module):
    """Learnable prompt embeddings for four components"""
    
    def __init__(self, embed_dim=512, prompt_length=4, component_types=['subject', 'object', 'second', 'relation']):
        """
        Args:
            embed_dim: Feature embedding dimension (default: 512)
            prompt_length: Length of learnable prompt tokens (default: 4)
            component_types: List of component types to create prompts for
        """
        super(PromptLearner, self).__init__()
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.component_types = component_types
        
        # Create learnable prompt embeddings for each component type
        # Use ParameterDict instead of ModuleDict to store Parameters
        self.prompts = nn.ParameterDict()
        for comp_type in component_types:
            # Initialize prompts with small random values
            prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)
            self.prompts[comp_type] = prompt
        
        # Learnable weight for combining original feature with prompt-enhanced feature
        self.alpha = nn.Parameter(torch.ones(len(component_types)))
    
    def forward(self, text_feature, component_type):
        """
        Enhance text feature with learnable prompt
        
        Args:
            text_feature: Text feature tensor [batch_size, embed_dim]
            component_type: Component type string (subject/object/second/relation)
        
        Returns:
            Enhanced text feature [batch_size, embed_dim]
        """
        if component_type not in self.prompts:
            return text_feature
        
        # Get prompt for this component type [prompt_length, embed_dim]
        prompt = self.prompts[component_type]
        
        # Project prompts to get enhancement vector
        # Average pooling over prompt tokens: [prompt_length, embed_dim] -> [embed_dim]
        prompt_enhancement = prompt.mean(dim=0)  # [embed_dim]
        
        # Expand to match batch dimension: [embed_dim] -> [1, embed_dim] -> [batch_size, embed_dim]
        batch_size = text_feature.size(0)
        prompt_enhancement = prompt_enhancement.unsqueeze(0).expand(batch_size, -1)
        
        # Add prompt enhancement to text feature
        # Using residual connection with learnable scaling
        enhanced_feature = text_feature + prompt_enhancement * 0.1
        
        return enhanced_feature


class ComponentMapping(nn.Module):
    """MLP mapping module for a single component"""
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super(ComponentMapping, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            linear = nn.Linear(in_dim, hidden_dim)
            # Xavier initialization for better gradient flow
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer without activation (for feature space mapping)
        output_linear = nn.Linear(in_dim, output_dim)
        # Small initialization for output layer to start near identity
        nn.init.xavier_uniform_(output_linear.weight, gain=0.1)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input feature [batch_size, input_dim]
        Returns:
            Mapped feature [batch_size, output_dim]
        """
        return self.mlp(x)


class CompositeMappingModule(nn.Module):
    """Complete mapping module with prompt learning and component-wise mapping"""
    
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
        super(CompositeMappingModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.component_types = component_types
        self.fusion_method = fusion_method
        
        # Prompt learner for each component
        self.prompt_learner = PromptLearner(embed_dim, prompt_length, component_types)
        
        # Mapping module for each component
        self.mappings = nn.ModuleDict()
        for comp_type in component_types:
            self.mappings[comp_type] = ComponentMapping(
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
            # If concatenating, need a projection layer
            self.fusion_proj = nn.Linear(embed_dim * len(component_types), embed_dim)
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.fusion_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, text_features_dict):
        """
        Map text features to image feature space
        
        Args:
            text_features_dict: Dictionary of {component_type: text_feature}
                text_feature: [batch_size, embed_dim]
        
        Returns:
            Mapped image feature [batch_size, embed_dim]
        """
        mapped_features = []
        
        # Process each component
        for comp_type in self.component_types:
            if comp_type not in text_features_dict:
                # If component is missing, use zero vector
                if text_features_dict:
                    zero_feat = torch.zeros_like(list(text_features_dict.values())[0])
                else:
                    # Fallback: create zero vector with embed_dim
                    zero_feat = torch.zeros(text_features_dict.get('batch_size', 1), self.embed_dim, 
                                          device=next(iter(text_features_dict.values())).device if text_features_dict else 'cpu')
                mapped_features.append(zero_feat)
                continue
            
            text_feat = text_features_dict[comp_type]
            
            # Enhance with learnable prompt
            enhanced_feat = self.prompt_learner(text_feat, comp_type)
            
            # Map to image feature space
            mapped_feat = self.mappings[comp_type](enhanced_feat)
            
            # Normalize (avoid zero division)
            norm = mapped_feat.norm(p=2, dim=-1, keepdim=True)
            mapped_feat = mapped_feat / (norm + 1e-8)
            
            mapped_features.append(mapped_feat)
        
        # Fuse component features
        if self.fusion_method == 'weighted_sum':
            # Weighted sum
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_feature = sum(w * feat for w, feat in zip(weights, mapped_features))
        
        elif self.fusion_method == 'concat':
            # Concatenate and project
            concat_feat = torch.cat(mapped_features, dim=-1)
            fused_feature = self.fusion_proj(concat_feat)
        
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            # Stack features: [batch_size, num_components, embed_dim]
            stacked = torch.stack(mapped_features, dim=1)
            # Use mean as query
            query = stacked.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            attended, _ = self.attention(query, stacked, stacked)
            fused_feature = self.fusion_proj(attended.squeeze(1))
        
        # Final normalization (avoid zero division)
        norm = fused_feature.norm(p=2, dim=-1, keepdim=True)
        fused_feature = fused_feature / (norm + 1e-8)
        
        return fused_feature
    
    def get_component_mappings(self, text_features_dict):
        """Get individual component mappings (for analysis/debugging)"""
        component_outputs = {}
        for comp_type in self.component_types:
            if comp_type in text_features_dict:
                text_feat = text_features_dict[comp_type]
                enhanced_feat = self.prompt_learner(text_feat, comp_type)
                mapped_feat = self.mappings[comp_type](enhanced_feat)
                component_outputs[comp_type] = F.normalize(mapped_feat, p=2, dim=-1)
        return component_outputs

