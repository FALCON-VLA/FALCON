import logging
import torch
import torch.nn as nn
from typing import Tuple, List
import random
import numpy as np

from falcon.model.policy_head.esm_utils.vggt.layers import PatchEmbed
from falcon.model.policy_head.esm_utils.vggt.layers.block import Block
from falcon.model.policy_head.esm_utils.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from falcon.model.policy_head.esm_utils.vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from falcon.model.policy_head.esm_utils.vggt.utils.pose_enc import extri_intri_to_pose_encoding
from torch.utils.checkpoint import checkpoint
from falcon.model.policy_head.esm_utils.vggt.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_checkpoint=True
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.use_checkpoint = enable_checkpoint
        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
            
            if patch_embed == "dinov2_vitl14_reg":
                dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
                self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
            elif patch_embed == "dinov2_vitb14_reg":
                dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
                self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
            elif patch_embed == "dinov2_vits14_reg":
                dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
                self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
            elif patch_embed == "dinov2_vitg2_reg":
                dinov2_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg2_reg')
                self.patch_embed.load_state_dict(dinov2_pretrained.state_dict(), strict=True)
                

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            blk = self.frame_blocks[frame_idx]
            if self.use_checkpoint and self.training:
                # 只把 Tensor 当参数传入 checkpoint，其余用闭包捕获
                tokens = checkpoint(
                    lambda inp, p: blk(inp, pos=p,),
                    tokens,
                    pos,                       # 额外 tensor 参数
                    use_reentrant=False
                )
            else:
                tokens = blk(tokens, pos=pos,)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx,
                                  pos=None, pose_encoding=None, depth_encoding=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            blk = self.global_blocks[global_idx]   
            if self.use_checkpoint and self.training:
                tokens = checkpoint(
                    lambda inp, p: blk(inp, pos=p, ),        
                    tokens,
                    pos,
                    use_reentrant=False
                )
            else:
                tokens = blk(tokens, pos=pos, )

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


class CasualAggregatorStage1(Aggregator):
    def __init__(self, img_size=518, 
                 patch_size=14, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 num_register_tokens=4, 
                 block_fn=Block, 
                 pose_hidden_dim=512,
                 qkv_bias=True, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 patch_embed="dinov2_vitl14_reg", 
                 aa_order=["frame", "global"], 
                 aa_block_size=1, 
                 qk_norm=True, 
                 rope_freq=100, 
                 init_values=0.01,
                 enable_checkpoint=True):
        super().__init__(img_size, 
                         patch_size, 
                         embed_dim, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         num_register_tokens, 
                         block_fn,
                         qkv_bias, 
                         proj_bias, 
                         ffn_bias, 
                         patch_embed, 
                         aa_order, 
                         aa_block_size, 
                         qk_norm, 
                         rope_freq, 
                         init_values)
        
        
        self.patch_start_idx = 1 + num_register_tokens
        self.camera_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.depth_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.use_checkpoint = enable_checkpoint
        self.pose_embedding = nn.Linear(pose_hidden_dim, embed_dim)
        self.depth_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim)
        
    def _match_dtype(self, x, reference):
        return x.to(dtype=reference.dtype, device=reference.device)
    
    def forward(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,
                camera_gt_pt: float = None,
                depth_gt_pt: float = None) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        
        # pass generate depth_encoding, follow pow3r
            
        # pass generate pose_encoding
            
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        K, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        
        gt_camera_token = self.camera_placeholder.expand(K, 1, C)
        depth_token = self.depth_placeholder.expand(K, P, C)
        camera_token = camera_token + gt_camera_token
        patch_tokens = patch_tokens + depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for index in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx   

class CasualAggregatorStage2(Aggregator):
    def __init__(self, img_size=518, 
                 patch_size=14, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 num_register_tokens=4, 
                 block_fn=Block, 
                 pose_hidden_dim=512,
                 qkv_bias=True, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 patch_embed="dinov2_vitl14_reg", 
                 aa_order=["frame", "global"], 
                 aa_block_size=1, 
                 qk_norm=True, 
                 rope_freq=100, 
                 init_values=0.01,
                 enable_checkpoint=True):
        super().__init__(img_size, 
                         patch_size, 
                         embed_dim, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         num_register_tokens, 
                         block_fn,
                         qkv_bias, 
                         proj_bias, 
                         ffn_bias, 
                         patch_embed, 
                         aa_order, 
                         aa_block_size, 
                         qk_norm, 
                         rope_freq, 
                         init_values)
        
        
        self.patch_start_idx = 1 + num_register_tokens
        self.camera_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.depth_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.use_checkpoint = enable_checkpoint
        self.pose_embedding = nn.Linear(pose_hidden_dim, embed_dim)
        self.depth_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim)
        
    def _match_dtype(self, x, reference):
        return x.to(dtype=reference.dtype, device=reference.device)
    
    def select_depth_gt(self, S, rng=None):
        rng = rng or np.random.default_rng()

        # 33% 的概率直接返回空集
        if rng.random() < 0.33:
            return []

        k = rng.integers(0, S + 1)
        if k == 0:
            return []

        # 其余元素从 [1, S-1] 里抽取
        others = rng.choice(np.arange(1, S), size=k - 1, replace=False) if S > 1 else []
        idx = np.concatenate(([0], others))

        return sorted(map(int, idx))
    
    def select_camera_gt(self, S, rng=None):
        rng = rng or np.random.default_rng()

        # 33% 的概率直接返回空集
        if rng.random() < 0.33:
            return []
        # 从 1..S 之间随机选取要保留的个数
        k = rng.integers(0, S + 1)
        if k == 0:
            return []

        # 按顺序从 0 开始选取 k 个
        idx = list(range(k))

        return idx
    
    def select_depth_gt_infer(self, S, percent=0.5, rng=None):
        """
        根据概率选取索引：
        - percent ∈ [0,1]
        - num = round(S * percent)
        - num = 0 → 返回 []
        - num > 0 → 至少包含 index 0，其余随机选
        """
        rng = rng or np.random.default_rng()

        # 计算要选几个
        num = int(round(S * percent))
        num = max(0, min(num, S))  # 保证范围在 0..S

        if num == 0:
            return []

        if num == 1:
            return [0]

        # 额外随机选 num-1 个 [1..S-1]
        others = rng.choice(np.arange(1, S), size=num - 1, replace=False)
        idx = np.concatenate(([0], others))

        return sorted(map(int, idx))

    def select_camera_gt_infer(self, S, percent=0.5, rng=None):
        rng = rng or np.random.default_rng()

        # 计算要选几个
        num = int(round(S * percent))
        num = max(0, min(num, S))  # 保证范围在 0..S

        if num == 0:
            return []

        if num == 1:
            return [0]

        # 额外随机选 num-1 个 [1..S-1]
        idx = list(range(num))

        return idx
    
    def normalize_extrinsics(self, extrinsics):
        B, S, _, _ = extrinsics.shape
        device = extrinsics.device
        extrinsics_homog = torch.cat(
            [
                extrinsics,
                torch.zeros((B, S, 1, 4), device=device),
            ],
            dim=-2,
        )
        extrinsics_homog[:, :, -1, -1] = 1.0
        first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
        new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
        cam_centers = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
        ref_cam = cam_centers[:, 0:1, :]  # (B,1,3)
        rel_distances = torch.norm(cam_centers - ref_cam, dim=-1)  # (B, S)
        scale = rel_distances.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        new_extrinsics[:, :, :3, 3] /= scale.unsqueeze(-1)
        return new_extrinsics[:, :, :3]
    
    def normalize_depth(self, depth, mask, eps=1e-8):
        """
        depth: [B, V, H, W, 1]
        mask:  [B, V, H, W]
        """
        assert depth.shape[:4] == mask.shape, "mask 和 depth 前四维必须一致"

        B, V, H, W, _ = depth.shape
        depth_squeezed = depth.squeeze(-1)  # [B,V,H,W]
        norm = torch.zeros_like(depth_squeezed)

        for b in range(B):
            valid = depth_squeezed[b][mask[b] > 0]
            if valid.numel() == 0:
                # 如果这一批没有有效像素，保持为 0
                continue

            mean = valid.mean()
            norm_b = depth_squeezed[b] / (mean + eps)

            # 只保留 mask 内的值
            norm[b] = norm_b * mask[b]

        return norm.unsqueeze(-1)  # [B,V,H,W,1]
    
    def forward(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
            
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        K, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # 获取随机保留的index
        camera_gt_index = self.select_camera_gt(S)
        depth_gt_index  = self.select_depth_gt(S)
        
        if len(camera_gt_index) != 0:

            camera_gt_length = len(camera_gt_index)
            idx_tensor = torch.tensor(camera_gt_index, device=depth.device)

            extrinsics_selected = torch.index_select(extrinsics, dim=1, index=idx_tensor)
            intrinsics_selected = torch.index_select(intrinsics, dim=1, index=idx_tensor)
            
            extrinsics_gt_normalized = self.normalize_extrinsics(extrinsics_selected)    
            pose_encoding = extri_intri_to_pose_encoding(
                        extrinsics=extrinsics_gt_normalized,
                        intrinsics=intrinsics_selected,
                        image_size_hw=(H, W),
                        pose_encoding_type="absT_quaR_FoV",
            )
            gt_camera_token = self.pose_embedding(pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)
            
            device = depth.device

            camera_full = self.camera_placeholder.expand(K, 1, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[rows] = gt_camera_token.to(dtype = camera_token.dtype)            # [B*gt_len, 1, C] → 指定行
            gt_camera_token = camera_full                     # [B*S, 1, C]
        else:
            gt_camera_token = self.camera_placeholder.expand(K, 1, C)
            pose_encoding = None
            
        if len(depth_gt_index) != 0:
            depth_gt_length = len(depth_gt_index)
            idx_tensor = torch.tensor(depth_gt_index, device=depth.device)

            depth_selected = torch.index_select(depth, dim=1, index=idx_tensor)
            mask_selected = torch.index_select(mask, dim=1, index=idx_tensor)
            
            depth_gt_normalized = self.normalize_depth(depth_selected, mask_selected)
            
            depth_gt_normalized = depth_gt_normalized.view(B * depth_gt_length, 1, H, W)
            mask_selected = mask_selected.view(B * depth_gt_length, 1, H, W)
            
            depthmaps = torch.cat([depth_gt_normalized, mask_selected], dim=1)
            depthmaps = self._match_dtype(depthmaps, self.depth_patch_embed.proj.weight)
            gt_depth_token = self.depth_patch_embed(depthmaps)
            
            
            device = depth.device
            
            depth_full  = self.depth_placeholder.expand(K, P, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            depth_full[rows]  = gt_depth_token.to(dtype = patch_tokens.dtype)                # [B*gt_len, P, C] → 指定行
            gt_depth_token  = depth_full                      # [B*S, P, C]          
              
        else:
            gt_depth_token = self.depth_placeholder.expand(K, P, C)

        camera_token = camera_token + gt_camera_token
        patch_tokens = patch_tokens + gt_depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for index in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, camera_gt_index, pose_encoding
    
    def inference(self, images: torch.Tensor, 
                    extrinsics: torch.Tensor, 
                    intrinsics: torch.Tensor,
                    depth: torch.Tensor,
                    mask: torch.Tensor,
                    camera_gt_pt: float,
                    depth_gt_pt: float) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
            
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        K, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # 获取随机保留的index, prob 为不带任何真值的概率
        camera_gt_index = self.select_camera_gt_infer(S, percent = camera_gt_pt)
        depth_gt_index  = self.select_depth_gt_infer(S, percent = depth_gt_pt)
        
        if len(camera_gt_index) != 0:

            camera_gt_length = len(camera_gt_index)
            idx_tensor = torch.tensor(camera_gt_index, device=depth.device)

            extrinsics_selected = torch.index_select(extrinsics, dim=1, index=idx_tensor)
            intrinsics_selected = torch.index_select(intrinsics, dim=1, index=idx_tensor)
            
            extrinsics_gt_normalized = self.normalize_extrinsics(extrinsics_selected)    
            pose_encoding = extri_intri_to_pose_encoding(
                        extrinsics=extrinsics_gt_normalized,
                        intrinsics=intrinsics_selected,
                        image_size_hw=(H, W),
                        pose_encoding_type="absT_quaR_FoV",
            )
            gt_camera_token = self.pose_embedding(pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)
            
            device = depth.device

            camera_full = self.camera_placeholder.expand(K, 1, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[rows] = gt_camera_token.to(dtype = camera_token.dtype)            # [B*gt_len, 1, C] → 指定行
            gt_camera_token = camera_full                     # [B*S, 1, C]
            
        else:
            gt_camera_token = self.camera_placeholder.expand(K, 1, C)
            pose_encoding = None
            
        if len(depth_gt_index) != 0:
            depth_gt_length = len(depth_gt_index)
            idx_tensor = torch.tensor(depth_gt_index, device=depth.device)

            depth_selected = torch.index_select(depth, dim=1, index=idx_tensor)
            mask_selected = torch.index_select(mask, dim=1, index=idx_tensor)
            
            depth_gt_normalized = self.normalize_depth(depth_selected, mask_selected)
            
            depth_gt_normalized = depth_gt_normalized.view(B * depth_gt_length, 1, H, W)
            mask_selected = mask_selected.view(B * depth_gt_length, 1, H, W)
            
            depthmaps = torch.cat([depth_gt_normalized, mask_selected], dim=1)
            depthmaps = self._match_dtype(depthmaps, self.depth_patch_embed.proj.weight)
            gt_depth_token = self.depth_patch_embed(depthmaps)
            
            
            device = depth.device
            
            depth_full  = self.depth_placeholder.expand(K, P, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            depth_full[rows]  = gt_depth_token.to(dtype = patch_tokens.dtype)                # [B*gt_len, P, C] → 指定行
            gt_depth_token  = depth_full                      # [B*S, P, C]          
              
        else:
            gt_depth_token = self.depth_placeholder.expand(K, P, C)

        camera_token = camera_token + gt_camera_token
        patch_tokens = patch_tokens + gt_depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for index in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, camera_gt_index, pose_encoding 
    
def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
