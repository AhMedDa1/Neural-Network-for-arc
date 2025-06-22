import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Encoder_ViT(nn.Module):
    def __init__(self, grid_shape: Tuple[int,int], patch_size: int, num_colors: int, embed_dim: int, depth: int, heads: int):
        super().__init__()
        self.grid_h, self.grid_w = grid_shape
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.input_channels = num_colors * 2 # X0 and X1 concatenated

        self.pad_h = (patch_size - (self.grid_h % patch_size)) % patch_size
        self.pad_w = (patch_size - (self.grid_w % patch_size)) % patch_size
        self.padded_h = self.grid_h + self.pad_h
        self.padded_w = self.grid_w + self.pad_w

        self.num_patches_h = self.padded_h // patch_size
        self.num_patches_w = self.padded_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        if self.padded_h < self.patch_size or self.padded_w < self.patch_size:
            print(f"Warning: Encoder padded grid ({self.padded_h}x{self.padded_w}) smaller than patch size ({self.patch_size}). Num patches will be 0.")
            self.num_patches = 0

        patch_dim = self.input_channels * patch_size * patch_size
        self.patch_proj = nn.Conv2d(self.input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        num_effective_patches = max(0, self.num_patches)
        self.pos_embed = nn.Parameter(torch.randn(1, num_effective_patches + 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation=F.gelu, batch_first=True, norm_first=True
        )
        eff_depth = max(1, depth) if depth > 0 else 0
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=eff_depth) if eff_depth > 0 else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.last_attn_weights = [] 
        self._feature_maps = {}


        if hasattr(self, 'transformer_encoder') and not isinstance(self.transformer_encoder, nn.Identity):
            for i, layer in enumerate(self.transformer_encoder.layers):
                def save_attn_hook(module, input, output, idx=i):
                    attn_weights = None
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attn_weights = output[1].detach().cpu()
                    if attn_weights is not None:
                        self.last_attn_weights.append(attn_weights)
                layer.self_attn.register_forward_hook(save_attn_hook)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        self.last_attn_weights = []  
        B, H, W = x0.shape
        if self.num_patches == 0:
            print(f"Warning: Encoder returning CLS token embedding directly due to 0 image patches for grid {H}x{W} (padded {self.padded_h}x{self.padded_w}).")
            cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
            pos_embed_cls = self.pos_embed[:, :1, :]
            return self.norm(cls_tokens + pos_embed_cls).squeeze(1) # Return (B, D)
        x0_onehot = F.one_hot(x0.clamp(0, self.num_colors-1), num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        x1_onehot = F.one_hot(x1.clamp(0, self.num_colors-1), num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        if self.pad_h > 0 or self.pad_w > 0:
            x0_onehot = F.pad(x0_onehot, (0, self.pad_w, 0, self.pad_h), "constant", 0)
            x1_onehot = F.pad(x1_onehot, (0, self.pad_w, 0, self.pad_h), "constant", 0)
        x_pair = torch.cat((x0_onehot, x1_onehot), dim=1)
        x_patched = self.patch_proj(x_pair) # (B, D, num_patches_h, num_patches_w)
        x_patched = x_patched.flatten(2).transpose(1, 2) # (B, num_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x_embed = torch.cat((cls_tokens, x_patched), dim=1) # (B, num_patches + 1, D)
        if x_embed.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(f"Positional embedding size mismatch: Input tokens {x_embed.shape[1]}, Pos Embed size {self.pos_embed.shape[1]}")
        x_embed = x_embed + self.pos_embed
        z = self.transformer_encoder(x_embed)
        z = self.norm(z)
        e = z[:, 0] # (B, D)
        self._feature_maps['patch_proj'] = x_patched.detach().cpu() 
        self._feature_maps['transformer_out'] = z.detach().cpu() 
        return e

    def get_last_attention_weights(self):
        return self.last_attn_weights

    def get_intermediate_feature_maps(self):
        return self._feature_maps 