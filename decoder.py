import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        def get_groups(channels):
            if channels <= 0: return 1
            g = min(32, channels)
            while channels % g != 0 and g > 1: g -= 1
            if channels == 1: g = 1
            g = max(1, g)
            return g
        groups1 = get_groups(in_channels)
        groups2 = get_groups(out_channels)
        if out_channels <= 0: raise ValueError(f"ConvBlock out_channels must be positive, got {out_channels}")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups2, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        residual = x
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.norm2(x)
        if self.residual_conv is not None: residual = self.residual_conv(residual)
        if residual.shape[-2:] != x.shape[-2:]:
            residual = F.interpolate(residual, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + residual
        x = self.relu2(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        num_groups = min(32, feature_dim) if feature_dim > 0 else 1
        while feature_dim > 0 and feature_dim % num_groups != 0 and num_groups > 1: num_groups -= 1
        num_groups = max(1, num_groups)
        self.norm_feature = nn.GroupNorm(num_groups, feature_dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.kv_proj = nn.Linear(context_dim, feature_dim * 2)
        attn_heads = num_heads
        if feature_dim % num_heads != 0:
            print(f"Warning: feature_dim {feature_dim} not divisible by num_heads {num_heads}. Adjusting heads.")
            while feature_dim % attn_heads != 0 and attn_heads > 1:
                attn_heads -= 1
            attn_heads = max(1, attn_heads)
            print(f"Using {attn_heads} heads for CrossAttentionBlock.")
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=attn_heads, batch_first=True)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.last_attn_weights = None
    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape; _, K, D = context.shape
        features_norm = self.norm_feature(features)
        features_flat = features_norm.view(B, C, H * W).transpose(1, 2) # (B, HW, C)
        query = self.query_proj(features_flat)
        context_norm = self.norm_context(context)
        kv = self.kv_proj(context_norm); k, v = kv.chunk(2, dim=-1)
        attn_output, attn_weights = self.attention(query, k, v)
        self.last_attn_weights = attn_weights.detach().cpu() 
        attn_output_proj = self.output_proj(attn_output)
        attn_output_spatial = attn_output_proj.transpose(1, 2).view(B, C, H, W)
        return features + attn_output_spatial
    def get_last_attention_weights(self):
        return self.last_attn_weights

class DirectDecoder_UNet(nn.Module):
    def __init__(self,
                 target_grid_shape: Tuple[int, int],
                 num_colors: int,
                 init_channels: int,
                 channel_mults: Tuple[int, ...],
                 encoder_embed_dim: int,
                 cross_attention_heads: int = 4):
        super().__init__()
        self.target_h, self.target_w = target_grid_shape
        self.num_colors = num_colors
        self.init_channels = init_channels
        self.encoder_embed_dim = encoder_embed_dim
        self.num_levels = len(channel_mults)
        min_dim = min(self.target_h, self.target_w); max_levels = 0
        if min_dim > 0 : max_levels = int(math.log2(min_dim / 1)) if min_dim >= 1 else 0
        if self.num_levels > max_levels:
            self.num_levels = max_levels; channel_mults = channel_mults[:self.num_levels]
            print(f"Warning: Decoder levels adjusted to {self.num_levels} due to grid size.")
        self.init_conv = nn.Conv2d(num_colors, init_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList(); down_path_channels = [init_channels]
        current_channels = init_channels
        for i in range(self.num_levels):
            out_channels = init_channels * channel_mults[i]
            self.downs.append(nn.ModuleList([ConvBlock(current_channels, out_channels), nn.MaxPool2d(2)]))
            current_channels = out_channels; down_path_channels.append(current_channels)
        bottleneck_mult = channel_mults[-1] if self.num_levels > 0 else 1
        mid_channels = init_channels * bottleneck_mult
        self.mid_block1 = ConvBlock(mid_channels, mid_channels * 2)
        self.mid_attn = CrossAttentionBlock(mid_channels * 2, encoder_embed_dim, cross_attention_heads)
        self.mid_block2 = ConvBlock(mid_channels * 2, mid_channels)
        self.ups = nn.ModuleList(); current_up_channels = mid_channels
        for i in reversed(range(self.num_levels)):
            target_out_channels = init_channels * channel_mults[i]
            skip_channels = down_path_channels[i+1]
            block_input_channels = skip_channels + target_out_channels
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(current_up_channels, target_out_channels, kernel_size=2, stride=2),
                CrossAttentionBlock(block_input_channels, encoder_embed_dim, cross_attention_heads),
                ConvBlock(block_input_channels, target_out_channels),
            ]))
            current_up_channels = target_out_channels
        self.final_conv = nn.Conv2d(current_up_channels, num_colors, kernel_size=1)
        
    def forward(self, x_in: torch.Tensor, training_pairs_embeddings: torch.Tensor) -> torch.Tensor:
        self._feature_maps = {}
        B_in, H_in, W_in = x_in.shape; B_e, K, D = training_pairs_embeddings.shape
        if B_e == 1 and B_in > 1: training_pairs_embeddings = training_pairs_embeddings.expand(B_in, -1, -1)
        elif B_e != B_in: raise ValueError(f"Batch size mismatch: input x ({B_in}), training embeddings ({B_e})")
        x = F.one_hot(x_in.clamp(0, self.num_colors - 1), num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        pad_h = self.target_h - H_in; pad_w = self.target_w - W_in
        if pad_h < 0 or pad_w < 0: x = x[:, :, :self.target_h, :self.target_w]; pad_h, pad_w = 0, 0
        if pad_h > 0 or pad_w > 0: x = F.pad(x, (0, pad_w, 0, pad_h), "constant", 0)
        x = self.init_conv(x)
        self._feature_maps['init_conv'] = x.detach().cpu()
        skip_connections = []
        for i, (block1, downsample) in enumerate(self.downs):
            x = block1(x)
            self._feature_maps[f'down_{i}'] = x.detach().cpu()
            skip_connections.append(x)
            x = downsample(x)
        x = self.mid_block1(x)
        self._feature_maps['mid_block1'] = x.detach().cpu()
        x = self.mid_attn(x, training_pairs_embeddings)
        self._feature_maps['mid_attn'] = x.detach().cpu()
        x = self.mid_block2(x)
        self._feature_maps['mid_block2'] = x.detach().cpu()
        for i, (upsample, attention_block, conv_block) in enumerate(self.ups):
            x = upsample(x)
            skip = skip_connections.pop()
            if x.shape[-2:] != skip.shape[-2:]: x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = attention_block(x, training_pairs_embeddings)
            self._feature_maps[f'up_attn_{i}'] = x.detach().cpu()
            x = conv_block(x)
            self._feature_maps[f'up_conv_{i}'] = x.detach().cpu()
        logits = self.final_conv(x)
        self._feature_maps['final_conv'] = logits.detach().cpu()
        return logits
    
    def get_all_attention_weights(self):
        attn_weights = []
        if hasattr(self, 'mid_attn') and hasattr(self.mid_attn, 'get_last_attention_weights'):
            attn_weights.append(self.mid_attn.get_last_attention_weights())
        if hasattr(self, 'ups'):
            for up in self.ups:
                attn_block = up[1]
                if hasattr(attn_block, 'get_last_attention_weights'):
                    attn_weights.append(attn_block.get_last_attention_weights())
        return attn_weights
    
    def get_intermediate_feature_maps(self):
        return getattr(self, '_feature_maps', {}) 