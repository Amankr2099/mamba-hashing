import torch
import torch.nn as nn
import numpy as np

import os
import sys
VIM_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vim', 'vim'))

if VIM_SRC_DIR not in sys.path:
    sys.path.insert(0, VIM_SRC_DIR)

from models_mamba import VisionMamba

# Configuration mapping for different Vision Mamba variants
VIM_CONFIGS = {
    "ViM-T_16": {"embed_dim": 192, "depth": 24}, 
    "ViM-S_16": {"embed_dim": 384, "depth": 24}, 
    "ViM-B_16": {"embed_dim": 768, "depth": 24}, 
}

class VisionMambaHashing(nn.Module):
    """
    Wrapper for VisionMamba to integrate it as a backbone in the hashing framework.
    Includes proper hash layer architecture similar to VisionTransformer implementation.
    """
    def __init__(self, vim_config, crop_size, zero_head=True, num_classes=1000, hash_bit=64):
        super().__init__()
        
        embed_dim = vim_config.get("embed_dim", 192)
        depth = vim_config.get("depth", 24)

        # Initialize VisionMamba with intermediate feature dimension
        # Use a dummy num_classes first, we'll replace the head
        self.model = VisionMamba(
            img_size=crop_size,
            patch_size=16, 
            embed_dim=embed_dim,
            depth=depth,
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            final_pool_type='mean', 
            if_abs_pos_embed=True, 
            bimamba_type="v2", 
            if_cls_token=True, 
            use_middle_cls_token=True,
            num_classes=0  # We'll replace the head
        )
        
        # Calculate number of patches for feature dimension
        num_patches = (crop_size // 16) ** 2
        
        # Create intermediate projection layer (similar to ViT)
        self.head = nn.Linear(embed_dim, 1024)
        self._init_weights()
        
        # Create hash layer matching ViT architecture
        self.hash_layer = nn.Sequential(
            nn.Dropout(0.1),
            self.head,
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit),
        )
        
        self.embed_dim = embed_dim
        self.zero_head = zero_head

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.head.weight, mode='fan_out')
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # Get features from ViM backbone (before final classifier)
        # This depends on ViM's architecture - you may need to modify based on actual implementation
        features = self.model.forward_features(x) if hasattr(self.model, 'forward_features') else self.model(x)
        
        # If features are pooled (batch_size, embed_dim), use directly
        # If features include sequence (batch_size, num_patches, embed_dim), pool them
        if len(features.shape) == 3:
            features = features.mean(dim=1)  # Global average pooling
        
        # Apply hash layer
        logits = self.hash_layer(features)
        return logits

    def load_from(self, weights):
        """Load pretrained weights from .npz file"""
        print("Loading pretrained weights...")
        print(f"Weight file keys: {list(weights.keys())[:10]}")  # Debug
        
        with torch.no_grad():
            # Reset head weights if zero_head is True
            if self.zero_head:
                nn.init.kaiming_uniform_(self.head.weight, mode='fan_out')
                nn.init.zeros_(self.head.bias)
                print("Initialized hash layer head with random weights")
            
            # Load ViM backbone weights
            try:
                # Create state dict from numpy weights
                state_dict = {}
                for key in weights.keys():
                    if key.startswith('head'):
                        continue  # Skip original classifier head
                    
                    weight = weights[key]
                    if isinstance(weight, np.ndarray):
                        state_dict[key] = torch.from_numpy(weight)
                    else:
                        state_dict[key] = weight
                
                # Load into ViM model
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                
                if len(missing_keys) > 0:
                    print(f"Missing keys sample: {missing_keys[:5]}")
                    
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                import traceback
                traceback.print_exc()
                print("Proceeding with random initialization...")
