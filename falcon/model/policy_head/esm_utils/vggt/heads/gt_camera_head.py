import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from falcon.model.policy_head.esm_utils.vggt.layers import Mlp
from falcon.model.policy_head.esm_utils.vggt.layers.block import Block
from falcon.model.policy_head.esm_utils.vggt.heads.head_act import activate_pose
from falcon.model.policy_head.esm_utils.vggt.heads.camera_head import CameraHead, modulate


class GT_CameraHead(CameraHead):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()
        self.empty_gt_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_gt_pose = nn.Linear(self.target_dim, dim_in)
        nn.init.xavier_uniform_(self.embed_gt_pose.weight)
        if self.embed_gt_pose.bias is not None:
            nn.init.constant_(self.embed_gt_pose.bias, 0.0)
        
    def forward(self, aggregated_tokens_list: list, camera_gt_index: list, 
                pose_encoding: torch.Tensor, num_iterations: int = 4) -> list:
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)
        B, S, C = pose_tokens.shape 

        empty_camera_pose = self.embed_gt_pose(self.empty_gt_pose_tokens.expand(B, S, -1))
        full_empty_pose_tokens = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
        if len(camera_gt_index) != 0:
            idx_tensor = torch.tensor(camera_gt_index, device=pose_tokens.device)
            
            gt_camera_pose = self.embed_gt_pose(pose_encoding)
            rows = (torch.arange(B, device=pose_tokens.device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            
            gt_camera_pose = gt_camera_pose.view(B * len(camera_gt_index), C)
            empty_camera_pose = empty_camera_pose.view(B * S, C)
            empty_camera_pose[rows] = gt_camera_pose.to(dtype = full_empty_pose_tokens.dtype)
            empty_camera_pose = empty_camera_pose.view(B, S, C)
            
        module_input = full_empty_pose_tokens + empty_camera_pose        
        pred_pose_enc_list = self.trunk_fn(pose_tokens, module_input, num_iterations)
        return pred_pose_enc_list
    
    def trunk_fn(self, pose_tokens: torch.Tensor, module_input: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is not None:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list