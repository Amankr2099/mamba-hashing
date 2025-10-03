import torch.nn as nn
import numpy as np

import os
import sys
VIM_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vim', 'vim'))

# 2. Add this path to the Python system path (sys.path).
if VIM_SRC_DIR not in sys.path:
    sys.path.insert(0, VIM_SRC_DIR)

from models_mamba import VisionMamba

# from Vim.vim.models_mamba import VisionMamba

# Configuration mapping for different Vision Mamba variants
# (Similar to how your repo uses VIT_CONFIGS)
VIM_CONFIGS = {
    # ViM-Tiny (192 embedding dimension, 24 layers)
    "ViM-T_16": {"embed_dim": 192, "depth": 24}, 
    # ViM-Small (384 embedding dimension, 24 layers)
    "ViM-S_16": {"embed_dim": 384, "depth": 24}, 
    # ViM-Base (768 embedding dimension, 24 layers)
    "ViM-B_16": {"embed_dim": 768, "depth": 24}, 
}

class VisionMambaHashing(nn.Module):
    """
    Wrapper for VisionMamba to integrate it as a backbone in the hashing framework.
    It passes 'hash_bit' to the ViM model's classifier head as 'num_classes'.
    The constructor signature is designed to match the expected ViT constructor.
    """
    def __init__(self, vim_config, crop_size, zero_head=True, num_classes=1000, hash_bit=64):
        super().__init__()
        
        embed_dim = vim_config.get("embed_dim", 192)
        depth = vim_config.get("depth", 24)

        # Initialize VisionMamba
        self.model = VisionMamba(
            img_size=crop_size, # typically 224
            patch_size=16, 
            embed_dim=embed_dim,
            depth=depth,
            
            # --- Fixed ViM Model Parameters (from vim_tiny_patch16_224_bimambav2_...) ---
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            final_pool_type='mean', 
            if_abs_pos_embed=True, 
            bimamba_type="v2", 
            if_cls_token=True, 
            use_middle_cls_token=True,
            # --------------------------------------------------------------------------
            
            # This is the key change: setting the final classification head to the hash bit length
            num_classes=hash_bit 
        )
        
        # Expose the load_from method for loading pretrained NumPy weights (.npz)
        # to maintain compatibility with the original HashNet loading logic.
        if hasattr(self.model, 'load_from'):
            self.load_from = self.model.load_from
        else:
            # Fallback in case load_from is not available
            self.load_from = lambda x: print("Warning: load_from method not found on ViM model.")


    def forward(self, x):
        # The forward method runs the data through the ViM backbone, 
        # producing the final hash code vector.
        return self.model(x)