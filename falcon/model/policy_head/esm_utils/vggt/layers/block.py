import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x

class PoseEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(9, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # 初始化（可选）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = (B, N, 9)
        Returns:
            Tensor, shape = (B, N, 1024)
        """
        # 把最后一维当作特征向量送入 MLP
        x = F.relu(self.fc1(x))   # (B, N, hidden_dim)
        x = self.fc2(x)           # (B, N, 1024)
        x = self.norm(x)          # (B, N, 1024)
        return x

class DepthEmbedding(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, out_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # 初始化（可选）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = (B, N, 9)
        Returns:
            Tensor, shape = (B, N, 1024)
        """
        x = F.relu(self.fc1(x))   # (B, N, hidden_dim)
        x = self.fc2(x)           # (B, N, 1024)
        x = self.norm(x)          # (B, N, 1024)
        return x


class BlockWithInjection(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        pose_hidden_dim = 512,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,)
            
        self.pose_embedding = PoseEmbedding(hidden_dim=pose_hidden_dim, out_dim=dim)
        self.depth_embedding = DepthEmbedding(input_dim=dim, hidden_dim=dim, out_dim=dim)
        
    def forward(self, x, pos=None, pose_encoding=None, depth_encoding=None,
                B = None, S = None, P = None, C = None) -> Tensor:
        if pose_encoding is not None:
            pose_tokens = self.pose_embedding(pose_encoding)  # (B, N, 1024)     # (2, 2, 1024)
            pose_tokens_flat = pose_tokens.reshape(-1, C)  # (4, 1024)
        else:
            pose_tokens_flat = torch.zeros((B * S, C), device=x.device)  
        
        if depth_encoding is not None:  
            depth_tokens = self.depth_embedding(depth_encoding)
        else:
            depth_tokens = torch.zeros((B * S, P, C), device=x.device)
        
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)
                x[:, 0, :] += pose_tokens_flat     
                x[:, -depth_tokens.shape[1]:, :] += depth_tokens
                x = x.view(B, S, P, C).view(B, S * P, C)
            else:
                x[:, 0, :] += pose_tokens_flat
                x[:, -depth_tokens.shape[1]:, :] += depth_tokens
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)
                x[:, 0, :] += pose_tokens_flat     
                x[:, -depth_tokens.shape[1]:, :] += depth_tokens
                x = x.view(B, S, P, C).view(B, S * P, C)
            else:
                x[:, 0, :] += pose_tokens_flat
                x[:, -depth_tokens.shape[1]:, :] += depth_tokens
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)          # T=782
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)
                x[:, 0, :] += pose_tokens_flat     
                x[:, -depth_tokens.shape[1]:, :] += depth_tokens
                x = x.view(B, S, P, C).view(B, S * P, C)
            else:
                x[:, 0, :] += pose_tokens_flat
                x[:, -depth_tokens.shape[1]:, :] += depth_tokens
            x = x + ffn_residual_func(x)
        return x
    
class MixBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        pose_hidden_dim = 512,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,)
            
        self.pose_embedding = PoseEmbedding(hidden_dim=pose_hidden_dim, out_dim=dim)
        self.depth_embedding = DepthEmbedding(input_dim=dim, hidden_dim=dim, out_dim=dim)
        
    def forward(self, x, pos=None, pose_encoding=None, domain = None, 
                block_type = None, B = None, S = None, P = None, C = None) -> Tensor:
        
        if pose_encoding is not None:
            pose_tokens = self.pose_embedding(pose_encoding)  # (B, N, 1024)     # (2, 2, 1024)
            pose_tokens_flat = pose_tokens.reshape(-1, C)  # (4, 1024)
        else:
            pose_tokens_flat = torch.zeros((B * S, C), device=x.device)  
        
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        pose_tokens = pose_tokens_flat.unsqueeze(1)  
        
        x_new, pos_new = self.replace_tokens(
            x = x, pose_tokens = pose_tokens,
            domain = domain, block_type = block_type,
            pos =pos, B = B, S = S, P = P, C = C)
        
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x_new = drop_add_residual_stochastic_depth(
                x_new,
                pos=pos_new,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x_new = drop_add_residual_stochastic_depth(
                x_new,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x_new = x_new + self.drop_path1(attn_residual_func(x_new, pos=pos_new))
            x_new = x_new + self.drop_path1(ffn_residual_func(x_new))  # FIXME: drop_path2
        else:
            x_new = x_new + attn_residual_func(x_new, pos=pos_new)
            x_new = x_new + ffn_residual_func(x_new)
            
        x = self.fetch_tokens(x_new = x_new, domain = domain, 
                              B = B, S = S, P = P, C = C, block_type = block_type)  
        return x

    def replace_tokens(self, x: Tensor, pose_tokens: Tensor,
                       domain: str, pos: Tensor, block_type: str,
                       B: int, S: int, P: int, C: int) -> Tensor:
        """
        替换 x 中的 pose 和 depth tokens
        """
        if domain == "image":
            x_new = x
            pos_new = pos
            
            
        elif domain == "camera":
            if block_type == "global":
                if x.shape != (B * S, P, C):
                    x = x.view(B, S, P, C).view(B * S, P, C)

                if pos is not None and pos.shape != (B * S, P, 2):
                    pos = pos.view(B, S, P, 2).view(B * S, P, 2)
            
            x_new = torch.cat([  
                x[:, :2, :],                  
                pose_tokens,                   
                x[:, 2:, :],                               
            ], dim=1)  
            
            # 原始 pos 拆分
            zeros = torch.zeros((B * S, 1, 2), device=pos.device, dtype = pos.dtype)  # [32, 1, 2]
            pos_prefix = pos[:, :2, :]  # 前两个
            pos_main = pos[:, 2:, :]    # 后 410
            
            # 重新拼接
            pos_new = torch.cat([
                pos_prefix,           # x[:, :1]
                zeros,                # pose_token_flat
                pos_main,             # x[:, 1:]
            ], dim=1)  # 最终 shape: [32, 821, 2]
            
            if block_type == "global":
                if x.shape != (B, S * P, C):
                    x = x.view(B, S, P, C).view(B, S * P, C)

                if pos is not None and pos.shape != (B, S * P, 2):
                    pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            
        return x_new, pos_new
    
    def fetch_tokens(self, x_new: Tensor, domain: str, B: int, S: int, P: int, C: int, block_type:str) -> Tuple[Tensor, Tensor]:
        
        if domain == "image":
            x = x_new
        elif domain == "camera":
            if block_type == "global":
                if x_new.shape != (B * S, P+1, C):
                    x_new = x_new.view(B, S, P+1, C).view(B * S, P+1, C)
                
            x = torch.cat([
                x_new[:, :2, :],   # 原 x[:, 0:1, :]
                x_new[:, 3:, :]  # 原 x[:, 1:, :]
            ], dim=1)  # → [32, 412, 1024]
            
            
            if block_type == "global":
                if x.shape != (B, S * P, C):
                    x = x.view(B, S, P, C).view(B, S * P, C)
        
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
