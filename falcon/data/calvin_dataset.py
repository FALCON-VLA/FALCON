import io
import json
import logging
import os
import random
import tarfile
from dataclasses import dataclass
from multiprocessing import Value
import numpy as np
from PIL import Image
import copy
import torchvision.transforms as T

import falcon
from falcon.utils.model_utils import build_tokenizer
from falcon.data.data_utils import (
    generate_chunck_data,
    get_text_function,
    mu_law_companding,
    normalize_action,
    regularize_action,
    euler2rotm,
    rotm2euler,
    compute_rel_robot_state,
    accumulate_tcp_rel_actions,
    save_tensor_images,
)

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from calvin_agent.datasets.utils.episode_utils import (
        get_state_info_dict,
        process_actions,
        process_depth,
        process_language,
        process_rgb,
        process_state,
    )
    from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern

    import pyhash
    import torch
    from torch.utils.data import Dataset
    from falcon.data.data_utils import get_prompt_builder, world_to_tcp_frame, rand_sample_pcd, rand_sample_pcd_filter, jitter_point_cloud, random_point_dropout
    from falcon.model.policy_head.action_tokenizer import ActionTokenizer

    hasher = pyhash.fnv1_32()
    logger = logging.getLogger(__name__)
except:
    pass

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

MIN_KB = 10
MAX_NUM_IMAGES = 5

import logging
from pathlib import Path
from typing import Dict, Tuple, Union
from omegaconf import DictConfig

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        # "actions": ["actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

from typing import Any, Dict, List, Tuple, Callable
from itertools import chain

import pickle
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n * t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n * t, *base_grid.shape[2:])
        shift = torch.randint(
            1, 2 * self.pad + 1, size=(n * t, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x


class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
        # TODO act_step actually is fwd_pred_next_n but not be rightly forward
    """

    def __init__(
        self,
        data_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=True,
        text_aug=False,
        dif_ws=False,
        fwd_pred_next_n=1,
        norm_action=False,
        norm_min=-1,
        norm_max=1,
        regular_action=False,
        x_mean=0,
        x_std=1,
        **kwargs,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size

        # you need to add one at act step for geting one more image than action
        self.act_step = fwd_pred_next_n + 1
        self.fwd_pred_next_n = fwd_pred_next_n

        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.regular_action = regular_action
        self.x_mean = x_mean
        self.x_std = x_std
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        # print(data_dir)
        self.abs_datasets_dir = data_dir
        if "calvin_data_copy" in str(self.abs_datasets_dir):
            lang_folder = "lang_annotations_test"
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons

        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        assert (
            "validation" in self.abs_datasets_dir.as_posix()
            or "training" in self.abs_datasets_dir.as_posix()
        )
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()

            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        if with_lang:
            return {"lang": episode["language"]}
        else:
            return {"lang": "execute random action."}

    def discretize_action_bins(self, action, action_bin=256):
        action_min = -1.001
        action_max = 1.001
        action_len = (action_max - action_min) / action_bin
        action = torch.FloatTensor(action)
        pose_action = (pose_action - action_min) / action_len
        pose_action = torch.floor(pose_action).long().view(-1).tolist()
        pose_action[-1] = int(action[-1])
        return pose_action

    def process_rt2_ag_text(self, text, action):
        action_id = self.discretize_action_bins(action)
        action_text = ["<Action_{}>".format(i) for i in action_id]
        action_text.append("<Gripper_{}>".format(action[-1]))

        return action_text

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        head = False
        sequence = self._get_sequences(idx, self.window_size, head=head)
        import copy

        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))

        image_tensors = self.image_fn(new_list) # [ws+fwd, 3, 224, 224]
        if self.rgb_pad != -1:
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(
                    image_tensors.unsqueeze(0)
                ).squeeze(0)
            else:
                image_tensors = self.rgb_shift(image_tensors)
        sequence["rgb_obs"]["rgb_static"] = image_tensors

        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))

        gripper_tensors = self.image_fn(new_list)
        if self.gripper_pad != -1:
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(
                    gripper_tensors.unsqueeze(0)
                ).squeeze(0)
            else:
                gripper_tensors = self.gripper_shift(gripper_tensors)
        sequence["rgb_obs"]["rgb_gripper"] = gripper_tensors

        # preprocess img, depth, cam params to fit esm input
        if self.act_head_configs["type"] in {"FCDecoder_ESM", "LSTMDecoder_ESM"}:
            from falcon.data.data_utils import process_sequence_data
            # TODO: hardcode for param setup
            vggt_target_res = (224, 224) # (W, H)
            np_rgb_vggt = copy.deepcopy(sequence["vggt"]["rgb_static"].numpy()) # (ws+fwd, 200, 200, 3), dtype: uint8 
            np_depth_vggt = copy.deepcopy(sequence['depth_obs']['depth_static'].numpy()) # (ws+fwd, 200, 200)
            invalid_mask = ~np.isfinite(np_depth_vggt)
            np_depth_vggt[invalid_mask] = 0
            assert np.all(np.isfinite(np_depth_vggt)), "still got nan in depth!"
            assert np.all(np_depth_vggt >= 0), "depth val cannot be negetive!"
            static_cam_intr_mat = sequence['static_cam_intr_mat'] # (ws+fwd, 3, 3)
            static_cam_ex_mat = sequence['static_cam_ex_mat'] # (ws+fwd, 4, 4)
            # Call the processing function, the np_rgb_vggt and np_depth_vggt should be H, W format
            processed_rgb, processed_depth, processed_intrinsics, point_masks = process_sequence_data(
                np_rgb_vggt=np_rgb_vggt,
                np_depth_vggt=np_depth_vggt,
                static_cam_intr_mat=static_cam_intr_mat,
                target_resolution=vggt_target_res
            )
            processed_extrinsics = torch.from_numpy(static_cam_ex_mat)[:, :3, :] # (ws+fwd, 3, 4)

            sequence["vggt"]["rgb_static"] = processed_rgb # (ws+fwd, 3, 224, 224)
            sequence["vggt"]["depth_static"] = processed_depth # (ws+fwd, 224, 224, 1)
            sequence["vggt"]["point_masks"] = point_masks # (ws+fwd, 224, 224)
            sequence["vggt"]["static_cam_intr_mat"] = processed_intrinsics # (ws+fwd, 3, 3)
            sequence["vggt"]["static_cam_ex_mat"] = processed_extrinsics # (ws+fwd, 3, 4)

        # compute robot relative state
        if self.act_head_configs.get("state_mode", "ee_abs_state") == "ee_rel_state":
            # process state
            tlen = self.window_size + self.fwd_pred_next_n
            states = np.array(sequence['robot_obs']) # (ws+fwd, 15)
            arm_states = states[:, :6]
            gripper_states = (states[:, -1:] + 1.0) / 2.0 # (-1, 1) -> (0, 1)

            rel_states = compute_rel_robot_state(arm_states, gripper_states, tlen, self.act_head_configs.get("state_dim", 7))
            rel_state_data= torch.zeros(tlen, self.act_head_configs.get("state_dim", 7)).float() # (ws+fwd, state_dim)
            rel_state_data[:tlen] = torch.from_numpy(rel_states)

            sequence["rel_robot_obs"] = []
            sequence["rel_robot_obs"] = rel_state_data # (ws+fwd, state_dim)

        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool = False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  # type:ignore

        # process and aug pcds
        if episode.get("static_pcd", None) is not None:
            seq_pcd_obs = rand_sample_pcd_filter(episode, self.act_head_configs["pcd_n_points"])
            # pcd augmentation
            if self.pcd_augment:
                static_pcd_seq = seq_pcd_obs['static_pcd'].numpy()
                static_aug = jitter_point_cloud(static_pcd_seq, self.jitter_sigma, self.jitter_clip)
                static_aug = random_point_dropout(static_aug, max_dropout_ratio=self.dropout_ratio)
                seq_pcd_obs['static_pcd'] = torch.from_numpy(static_aug)

                gripper_pcd_seq = seq_pcd_obs['gripper_pcd'].numpy()
                gripper_aug = jitter_point_cloud(gripper_pcd_seq, self.jitter_sigma, self.jitter_clip)
                gripper_aug = random_point_dropout(gripper_aug, max_dropout_ratio=self.dropout_ratio)                
                seq_pcd_obs['gripper_pcd'] = torch.from_numpy(gripper_aug)
            seq_dict.update(seq_pcd_obs)

        # process static cam params for esm!!
        if episode.get("static_cam_intr_mat", None) is not None:
            seq_cam_param_obs = {}
            seq_cam_param_obs['static_cam_intr_mat'] = episode['static_cam_intr_mat']
            seq_cam_param_obs['static_cam_ex_mat'] = episode['static_cam_ex_mat']
            seq_dict.update(seq_cam_param_obs)

        seq_dict["idx"] = idx  # type:ignore
        seq_dict["action_mask"] = episode["action_mask"]
        seq_dict["image_mask"] = episode["image_mask"]
        
        if self.act_head_configs["type"] in {"FCDecoder_ESM", "LSTMDecoder_ESM"}:
            seq_dict["vggt"] = {}
            seq_dict["vggt"]["rgb_static"] = copy.deepcopy(seq_rgb_obs["rgb_obs"]["rgb_static"])
            seq_dict["vggt"]["rgb_gripper"] = copy.deepcopy(seq_rgb_obs["rgb_obs"]["rgb_gripper"])

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool = False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update(
                    {
                        "actions": self._pad_with_repetition(
                            seq["actions"], pad_size, head
                        )
                    }
                )
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(
                            seq["actions"][..., -1:], pad_size, head
                        ),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(
        input_tensor: torch.Tensor, pad_size: int, head: bool = False
    ) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(
        input_tensor: torch.Tensor, pad_size: int, head: bool = False
    ) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info


class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        image_fn_policy: Callable,
        tokenizer: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        decoder_type="lstm",
        discrete_action=False,
        action_tokenizer=None,
        model_name="vicuna",
        predict_stop_token=True,
        use_mu_law=False,
        mu_val=255,
        n_bin=256,
        min_action=-1,
        max_action=1,
        task_type="calvin_action",
        tcp_rel=False,
        act_head_configs=None,
        few_shot=False,
        exclude_tasks=[],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_type = decoder_type
        self.save_format = save_format
        self.image_fn = image_fn
        self.image_fn_policy = image_fn_policy

        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.use_mu_law = use_mu_law
        self.mu_val = mu_val
        self.task_type = task_type
        self.tcp_rel = tcp_rel
        self.act_head_configs = act_head_configs
        self.few_shot = few_shot
        self.exclude_tasks = exclude_tasks
        print(self.task_type)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )
        (
            self.episode_lookup,
            self.lang_lookup,
            self.right_pad_lookup,
            self.lang_ann,
            self.lang_task,
        ) = self._build_file_indices_lang(self.abs_datasets_dir)

        self.model_name = model_name
        self.discrete_action = discrete_action
        self.predict_stop_token = predict_stop_token
        if self.discrete_action:
            if action_tokenizer is None:
                action_tokenizer = ActionTokenizer(
                    self.tokenizer,
                    bins=n_bin,
                    min_action=min_action,
                    max_action=max_action,
                )
            self.action_tokenizer = action_tokenizer

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]

        end_idx = start_idx + window_size + self.act_step - 1
        right_pad = self.right_pad_lookup[idx]
        idx_range = np.arange(start_idx, end_idx)
        action_mask = np.ones_like(idx_range)
        image_mask = np.ones_like(idx_range)
        if right_pad != 0:
            idx_range[right_pad:] = idx_range[right_pad]
            action_mask[right_pad:] = 0
            image_mask[right_pad:] = 0

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in idx_range
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(
                    self.enrich_lang[task] + [episode["language"]]
                )
                episode["language"] = enrich_lang
        episode["action_mask"] = action_mask
        episode["image_mask"] = image_mask

        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        # add support for partial calvin data
        # partial_st_ed_list = []
        partial_st_ed_list = load_partial_traj_data()
        few_shot_st_ed_list = load_few_shot_traj_data()
        # import pdb; pdb.set_trace()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if (start_idx, end_idx) not in partial_st_ed_list:
                    continue
            if self.few_shot:
                if (start_idx, end_idx) not in few_shot_st_ed_list:
                    continue
            if lang_task[i] in self.exclude_tasks:
                continue
            cnt = 0
            right_pad = end_idx - start_idx - self.act_step - self.window_size + 1
            for idx in range(start_idx, end_idx + 1 - self.window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                    right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
                cnt += 1

        return (
            np.array(episode_lookup),
            lang_lookup,
            right_pad_lookup,
            lang_ann,
            lang_task,
        )

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            right_pad = end_idx - start_idx - self.act_step - self.window_size
            for idx in range(start_idx, end_idx + 2 - self.window_size):
                episode_lookup.append(idx)
                right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
        return np.array(episode_lookup), right_pad_lookup

    def wrap_instruction_and_action(self, lang, action, action_mask):
        # modified from OpenVLA
        IGNORE_INDEX = -100
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token
        )
        # if pass in multi-step actions, we concat them
        action_mask = action_mask.astype(bool)
        # action = action[:-1]
        action_dim = action.shape[1]
        action = action.flatten()
        conversation = [
            {
                "from": "human",
                "value": (
                    f"What action should the robot take to {lang}?"
                    if self.act_step == 1
                    else f"What {self.act_step} step actions should the robot take to {lang}?"
                ),
            },
            {"from": "gpt", "value": ""},
        ]
        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        action_ids = self.action_tokenizer.encode_actions_to_token_ids(action)

        all_input_ids = []
        all_labels = []

        for i in range(self.window_size):
            start_action_index = i
            end_action_index = (i + self.act_step) - 1
            tmp_action_ids = action_ids[
                start_action_index * action_dim : end_action_index * action_dim
            ]
            tmp_action_mask = action_mask[start_action_index:end_action_index]
            right_pad_len = (~tmp_action_mask).sum() * action_dim

            if self.tokenizer.eos_token is None:
                tmp_input_ids = input_ids + tmp_action_ids
            else:
                tmp_input_ids = input_ids[:-1] + tmp_action_ids + [input_ids[-1]]
            tmp_labels = list(tmp_input_ids)
            tmp_input_ids, tmp_labels = torch.tensor(tmp_input_ids), torch.tensor(
                tmp_labels
            )

            if self.tokenizer.eos_token is None:
                tmp_labels[: -len(tmp_action_ids)] = IGNORE_INDEX
                if right_pad_len != 0:
                    tmp_labels[-right_pad_len:] = IGNORE_INDEX
            else:
                tmp_labels[: -(len(tmp_action_ids) + 1)] = IGNORE_INDEX
                if right_pad_len != 0:
                    tmp_labels[-right_pad_len - 1 : -1] = IGNORE_INDEX
            if not self.predict_stop_token and self.tokenizer.eos_token:
                tmp_labels[-1] = IGNORE_INDEX
            all_input_ids.append(tmp_input_ids)
            all_labels.append(tmp_labels)

        all_input_ids = torch.stack(all_input_ids)
        all_labels = torch.stack(all_labels)
        return all_input_ids, all_labels

    # NOTE
    def collater(self, sample):
        # forward transformation of mu law companding
        if self.norm_action:
            new_sample = []
            for s in sample:
                s["actions"] = normalize_action(
                    s["actions"], self.norm_min, self.norm_max, maintain_last=True
                )
                new_sample.append(s)
            sample = new_sample

        if self.regular_action:
            new_sample = []
            for s in sample:
                s["actions"] = regularize_action(s["actions"], self.x_mean, self.x_std)
                new_sample.append(s)
            sample = new_sample
            pass

        if self.use_mu_law: # 对continusous action没有用
            new_sample = []
            for s in sample:
                s["actions"] = mu_law_companding(s["actions"], self.mu_val)
                new_sample.append(s)
            sample = new_sample

        action_tensors = torch.from_numpy(
            np.array([np.stack(s["actions"]) for s in sample])
        )[:, :-1]
        action_mask = torch.from_numpy(
            np.array([np.stack(s["action_mask"]) for s in sample])
        )[:, :-1]
        robot_obs = torch.from_numpy(
            np.array([np.stack(s["robot_obs"]) for s in sample])
        )[:, :-1]

        if self.tcp_rel:
            action_tensors = world_to_tcp_frame(action_tensors, robot_obs)
        # Converting world coord rel_actions in a chunk to accum actions relative to the init frames in tcp coord 
        if self.act_head_configs.get("act_chunk_mode", "rel_pre_state") == "rel_init_state_tcp":
            action_tensors = accumulate_tcp_rel_actions(action_tensors, robot_obs)

        image_mask = torch.from_numpy(
            np.array([np.stack(s["image_mask"]) for s in sample])
        )
        image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
        gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])

        stacked_language = [s["lang"] for s in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)
        action_tensors[..., -1] = ((action_tensors[..., -1] + 1) // 2).float()
        robot_obs[..., -1] = ((robot_obs[..., -1] + 1) // 2).float()
        state_tensors = robot_obs[:, : self.window_size]

        image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]
        image_tensors = image_tensors[:, : self.window_size]
        if gripper_tensors is not None:
            gripper_chunk = gripper_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
                0, 1, 5, 2, 3, 4
            )[:, 1:]
            gripper_tensors = gripper_tensors[:, : self.window_size]
        else:
            gripper_chunk = None

        fwd_mask = image_mask.unfold(1, self.fwd_pred_next_n, 1)[:, 1:]

        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 3, 2
        )
        action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)      

        rel_state_tensors = None
        if self.act_head_configs.get("state_mode", "ee_abs_state") == "ee_rel_state":
            rel_robot_obs = torch.from_numpy(
                np.array([np.stack(s["rel_robot_obs"]) for s in sample])
            )[:, :-1]
            rel_state_tensors = rel_robot_obs[:, : self.window_size] # [bs, ws, 7]

        # robot_obs_chunk = robot_obs.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        # if self.tcp_rel:
        #     action_chunck = world_to_tcp_frame(action_chunck, robot_obs_chunk)

        bs = len(sample)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None

        if self.discrete_action:
            res = [
                self.wrap_instruction_and_action(
                    s["lang"], s["actions"], s["action_mask"]
                )
                for s in sample
            ]
            tmp_input_ids = [_[0] for _ in res]
            tmp_labels = [_[1] for _ in res]
            max_len = max([_.shape[-1] for _ in tmp_input_ids])
            instr_and_action_ids = torch.zeros(
                (bs, self.window_size, max_len), dtype=torch.long
            )
            instr_and_action_labels = (
                torch.ones((bs, self.window_size, max_len), dtype=torch.long) * -100
            )
            instr_and_action_mask = torch.zeros(
                (bs, self.window_size, max_len), dtype=torch.bool
            )

            for i in range(bs):
                instr_and_action_ids[
                    i, :, : tmp_input_ids[i].shape[-1]
                ] = tmp_input_ids[i]
                instr_and_action_labels[i, :, : tmp_labels[i].shape[-1]] = tmp_labels[i]
                instr_and_action_mask[i, :, : tmp_input_ids[i].shape[-1]] = 1

        res = {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "state": state_tensors,
            "rel_state": rel_state_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": attention_mask,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_chunk,
            "fwd_mask": fwd_mask,
            "action_chunck": action_chunck,
            "chunck_mask": action_mask,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "raw_text": stacked_language,
            "data_source": self.task_type,
        }
        return res


class DiskCalvinDataset3D(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        image_fn_policy: Callable,
        tokenizer: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        decoder_type="lstm",
        discrete_action=False,
        action_tokenizer=None,
        model_name="vicuna",
        predict_stop_token=True,
        use_mu_law=False,
        mu_val=255,
        n_bin=256,
        min_action=-1,
        max_action=1,
        task_type="calvin_action",
        tcp_rel=False,
        act_head_configs=None,
        pcd_data_dir=None,
        pcd_augment=False,
        jitter_sigma=0.01,
        jitter_clip=0.03,
        dropout_ratio=0.3,
        few_shot=False,
        exclude_tasks=[],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_type = decoder_type
        self.save_format = save_format
        self.image_fn = image_fn
        self.image_fn_policy = image_fn_policy

        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.use_mu_law = use_mu_law
        self.mu_val = mu_val
        self.task_type = task_type
        self.tcp_rel = tcp_rel
        self.act_head_configs = act_head_configs
        if isinstance(pcd_data_dir, str):
            pcd_data_dir = Path(pcd_data_dir)
        self.pcd_data_dir = pcd_data_dir
        self.pcd_augment = pcd_augment
        print("use pcd augment: ", self.pcd_augment)
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.dropout_ratio = dropout_ratio
        self.few_shot = few_shot
        self.exclude_tasks = exclude_tasks
        print(self.task_type)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )
        self.pcd_naming_pattern, self.pcd_n_digits = lookup_naming_pattern(
            self.pcd_data_dir, self.save_format
        )
        (
            self.episode_lookup,
            self.lang_lookup,
            self.right_pad_lookup,
            self.lang_ann,
            self.lang_task,
        ) = self._build_file_indices_lang(self.abs_datasets_dir)

        self.model_name = model_name
        self.discrete_action = discrete_action
        self.predict_stop_token = predict_stop_token
        if self.discrete_action:
            if action_tokenizer is None:
                action_tokenizer = ActionTokenizer(
                    self.tokenizer,
                    bins=n_bin,
                    min_action=min_action,
                    max_action=max_action,
                )
            self.action_tokenizer = action_tokenizer

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_pcd_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.pcd_naming_pattern[0]}{file_idx:0{self.pcd_n_digits}d}{self.pcd_naming_pattern[1]}"
        )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]

        end_idx = start_idx + window_size + self.act_step - 1
        right_pad = self.right_pad_lookup[idx]
        idx_range = np.arange(start_idx, end_idx)
        action_mask = np.ones_like(idx_range)
        image_mask = np.ones_like(idx_range)
        if right_pad != 0:
            idx_range[right_pad:] = idx_range[right_pad]
            action_mask[right_pad:] = 0
            image_mask[right_pad:] = 0

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in idx_range
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

        # read pre-processed pcds data
        pcd_keys = ["static_pcd", "static_cam_ex_mat", "gripper_pcd", "gripper_cam_ex_mat"]
        pcd_episodes = [
            self.load_file(self._get_pcd_episode_name(pcd_file_idx)) for pcd_file_idx in idx_range
        ]
        pcd_episode = {pcd_key: np.stack([pcd_ep[pcd_key] for pcd_ep in pcd_episodes]) for pcd_key in pcd_keys}

        # add pcd data
        episode["static_pcd"] = pcd_episode["static_pcd"]
        episode["static_cam_ex_mat"] = pcd_episode["static_cam_ex_mat"]
        episode["gripper_pcd"] = pcd_episode["gripper_pcd"]
        episode["gripper_cam_ex_mat"] = pcd_episode["gripper_cam_ex_mat"]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(
                    self.enrich_lang[task] + [episode["language"]]
                )
                episode["language"] = enrich_lang
        episode["action_mask"] = action_mask
        episode["image_mask"] = image_mask

        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        # add support for partial calvin data
        # partial_st_ed_list = []
        partial_st_ed_list = load_partial_traj_data()
        few_shot_st_ed_list = load_few_shot_traj_data()
        # import pdb; pdb.set_trace()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if (start_idx, end_idx) not in partial_st_ed_list:
                    continue
            if self.few_shot:
                if (start_idx, end_idx) not in few_shot_st_ed_list:
                    continue
            if lang_task[i] in self.exclude_tasks:
                continue
            cnt = 0
            right_pad = end_idx - start_idx - self.act_step - self.window_size + 1
            for idx in range(start_idx, end_idx + 1 - self.window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                    right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
                cnt += 1

        return (
            np.array(episode_lookup),
            lang_lookup,
            right_pad_lookup,
            lang_ann,
            lang_task,
        )

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            right_pad = end_idx - start_idx - self.act_step - self.window_size
            for idx in range(start_idx, end_idx + 2 - self.window_size):
                episode_lookup.append(idx)
                right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
        return np.array(episode_lookup), right_pad_lookup

    def wrap_instruction_and_action(self, lang, action, action_mask):
        # modified from OpenVLA
        IGNORE_INDEX = -100
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token
        )
        # if pass in multi-step actions, we concat them
        action_mask = action_mask.astype(bool)
        # action = action[:-1]
        action_dim = action.shape[1]
        action = action.flatten()
        conversation = [
            {
                "from": "human",
                "value": (
                    f"What action should the robot take to {lang}?"
                    if self.act_step == 1
                    else f"What {self.act_step} step actions should the robot take to {lang}?"
                ),
            },
            {"from": "gpt", "value": ""},
        ]
        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        action_ids = self.action_tokenizer.encode_actions_to_token_ids(action)

        all_input_ids = []
        all_labels = []

        for i in range(self.window_size):
            start_action_index = i
            end_action_index = (i + self.act_step) - 1
            tmp_action_ids = action_ids[
                start_action_index * action_dim : end_action_index * action_dim
            ]
            tmp_action_mask = action_mask[start_action_index:end_action_index]
            right_pad_len = (~tmp_action_mask).sum() * action_dim

            if self.tokenizer.eos_token is None:
                tmp_input_ids = input_ids + tmp_action_ids
            else:
                tmp_input_ids = input_ids[:-1] + tmp_action_ids + [input_ids[-1]]
            tmp_labels = list(tmp_input_ids)
            tmp_input_ids, tmp_labels = torch.tensor(tmp_input_ids), torch.tensor(
                tmp_labels
            )

            if self.tokenizer.eos_token is None:
                tmp_labels[: -len(tmp_action_ids)] = IGNORE_INDEX
                if right_pad_len != 0:
                    tmp_labels[-right_pad_len:] = IGNORE_INDEX
            else:
                tmp_labels[: -(len(tmp_action_ids) + 1)] = IGNORE_INDEX
                if right_pad_len != 0:
                    tmp_labels[-right_pad_len - 1 : -1] = IGNORE_INDEX
            if not self.predict_stop_token and self.tokenizer.eos_token:
                tmp_labels[-1] = IGNORE_INDEX
            all_input_ids.append(tmp_input_ids)
            all_labels.append(tmp_labels)

        all_input_ids = torch.stack(all_input_ids)
        all_labels = torch.stack(all_labels)
        return all_input_ids, all_labels

    # NOTE
    def collater(self, sample):
        # forward transformation of mu law companding
        if self.norm_action:
            new_sample = []
            for s in sample:
                s["actions"] = normalize_action(
                    s["actions"], self.norm_min, self.norm_max, maintain_last=True
                )
                new_sample.append(s)
            sample = new_sample

        if self.regular_action:
            new_sample = []
            for s in sample:
                s["actions"] = regularize_action(s["actions"], self.x_mean, self.x_std)
                new_sample.append(s)
            sample = new_sample
            pass

        if self.use_mu_law: # 对continusous action没有用
            new_sample = []
            for s in sample:
                s["actions"] = mu_law_companding(s["actions"], self.mu_val)
                new_sample.append(s)
            sample = new_sample

        action_tensors = torch.from_numpy(
            np.array([np.stack(s["actions"]) for s in sample])
        )[:, :-1]
        action_mask = torch.from_numpy(
            np.array([np.stack(s["action_mask"]) for s in sample])
        )[:, :-1]
        robot_obs = torch.from_numpy(
            np.array([np.stack(s["robot_obs"]) for s in sample])
        )[:, :-1]

        if self.tcp_rel:
            action_tensors = world_to_tcp_frame(action_tensors, robot_obs)

        image_mask = torch.from_numpy(
            np.array([np.stack(s["image_mask"]) for s in sample])
        )
        image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
        gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])

        stacked_language = [s["lang"] for s in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)
        action_tensors[..., -1] = ((action_tensors[..., -1] + 1) // 2).float()
        robot_obs[..., -1] = ((robot_obs[..., -1] + 1) // 2).float()
        state_tensors = robot_obs[:, : self.window_size]

        image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]
        image_tensors = image_tensors[:, : self.window_size]
        if gripper_tensors is not None:
            gripper_chunk = gripper_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
                0, 1, 5, 2, 3, 4
            )[:, 1:]
            gripper_tensors = gripper_tensors[:, : self.window_size]
        else:
            gripper_chunk = None

        fwd_mask = image_mask.unfold(1, self.fwd_pred_next_n, 1)[:, 1:]

        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 3, 2
        )
        action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)

        # prepare pcd input
        static_pcd_tensors = None
        static_cam_ex_mat_tensors = None
        gripper_pcd_tensors = None
        gripper_cam_ex_mat_tensors = None
        static_pcd_tensors = torch.stack([s["static_pcd"] for s in sample])
        static_cam_ex_mat_tensors = torch.stack([s["static_cam_ex_mat"] for s in sample])
        static_pcd_tensors = static_pcd_tensors[:, : self.window_size]
        static_cam_ex_mat_tensors = static_cam_ex_mat_tensors[:, : self.window_size]
        if self.act_head_configs.get("use_hand_pcd", False):
            gripper_pcd_tensors = torch.stack([s["gripper_pcd"] for s in sample])
            gripper_cam_ex_mat_tensors = torch.stack([s["gripper_cam_ex_mat"] for s in sample])
            gripper_pcd_tensors = gripper_pcd_tensors[:, : self.window_size]
            gripper_cam_ex_mat_tensors = gripper_cam_ex_mat_tensors[:, : self.window_size]

        rel_state_tensors = None
        if self.act_head_configs.get("state_mode", "ee_abs_state") == "ee_rel_state":
            rel_robot_obs = torch.from_numpy(
                np.array([np.stack(s["rel_robot_obs"]) for s in sample])
            )[:, :-1]
            rel_state_tensors = rel_robot_obs[:, : self.window_size] # [bs, ws, 7]

        # robot_obs_chunk = robot_obs.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        # if self.tcp_rel:
        #     action_chunck = world_to_tcp_frame(action_chunck, robot_obs_chunk)

        bs = len(sample)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None

        if self.discrete_action:
            res = [
                self.wrap_instruction_and_action(
                    s["lang"], s["actions"], s["action_mask"]
                )
                for s in sample
            ]
            tmp_input_ids = [_[0] for _ in res]
            tmp_labels = [_[1] for _ in res]
            max_len = max([_.shape[-1] for _ in tmp_input_ids])
            instr_and_action_ids = torch.zeros(
                (bs, self.window_size, max_len), dtype=torch.long
            )
            instr_and_action_labels = (
                torch.ones((bs, self.window_size, max_len), dtype=torch.long) * -100
            )
            instr_and_action_mask = torch.zeros(
                (bs, self.window_size, max_len), dtype=torch.bool
            )

            for i in range(bs):
                instr_and_action_ids[
                    i, :, : tmp_input_ids[i].shape[-1]
                ] = tmp_input_ids[i]
                instr_and_action_labels[i, :, : tmp_labels[i].shape[-1]] = tmp_labels[i]
                instr_and_action_mask[i, :, : tmp_input_ids[i].shape[-1]] = 1

        res = {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "state": state_tensors,
            "rel_state": rel_state_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": attention_mask,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_chunk,
            "fwd_mask": fwd_mask,
            "action_chunck": action_chunck,
            "chunck_mask": action_mask,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "raw_text": stacked_language,
            "data_source": self.task_type,
            # pcd inputs
            "static_pcd": static_pcd_tensors,
            "static_cam_ex_mat": static_cam_ex_mat_tensors,
            "gripper_pcd": gripper_pcd_tensors,
            "gripper_cam_ex_mat": gripper_cam_ex_mat_tensors,
        }
        return res


class DiskCalvinDataset_ESM(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        image_fn_policy: Callable,
        tokenizer: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        decoder_type="lstm",
        discrete_action=False,
        action_tokenizer=None,
        model_name="vicuna",
        predict_stop_token=True,
        use_mu_law=False,
        mu_val=255,
        n_bin=256,
        min_action=-1,
        max_action=1,
        task_type="calvin_action",
        tcp_rel=False,
        act_head_configs=None,
        cam_params_dir=None,
        few_shot=False,
        exclude_tasks=[],
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_type = decoder_type
        self.save_format = save_format
        self.image_fn = image_fn
        self.image_fn_policy = image_fn_policy

        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.use_mu_law = use_mu_law
        self.mu_val = mu_val
        self.task_type = task_type
        self.tcp_rel = tcp_rel
        self.act_head_configs = act_head_configs
        if isinstance(cam_params_dir, str):
            cam_params_dir = Path(cam_params_dir)
        self.pcd_data_dir = cam_params_dir
        self.few_shot = few_shot
        self.exclude_tasks = exclude_tasks
        print(self.task_type)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )
        self.pcd_naming_pattern, self.pcd_n_digits = lookup_naming_pattern(
            self.pcd_data_dir, self.save_format
        )
        (
            self.episode_lookup,
            self.lang_lookup,
            self.right_pad_lookup,
            self.lang_ann,
            self.lang_task,
        ) = self._build_file_indices_lang(self.abs_datasets_dir)

        self.model_name = model_name
        self.discrete_action = discrete_action
        self.predict_stop_token = predict_stop_token
        if self.discrete_action:
            if action_tokenizer is None:
                action_tokenizer = ActionTokenizer(
                    self.tokenizer,
                    bins=n_bin,
                    min_action=min_action,
                    max_action=max_action,
                )
            self.action_tokenizer = action_tokenizer

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _get_pcd_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.pcd_naming_pattern[0]}{file_idx:0{self.pcd_n_digits}d}{self.pcd_naming_pattern[1]}"
        )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]

        end_idx = start_idx + window_size + self.act_step - 1
        right_pad = self.right_pad_lookup[idx]
        idx_range = np.arange(start_idx, end_idx)
        action_mask = np.ones_like(idx_range)
        image_mask = np.ones_like(idx_range)
        if right_pad != 0:
            idx_range[right_pad:] = idx_range[right_pad]
            action_mask[right_pad:] = 0
            image_mask[right_pad:] = 0

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in idx_range
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

        # read pre-processed static cam params data
        pcd_keys = ["static_cam_intr_mat", "static_cam_ex_mat"]
        pcd_episodes = [
            self.load_file(self._get_pcd_episode_name(pcd_file_idx)) for pcd_file_idx in idx_range
        ]
        pcd_episode = {pcd_key: np.stack([pcd_ep[pcd_key] for pcd_ep in pcd_episodes]) for pcd_key in pcd_keys}

        # add static cam params data
        episode["static_cam_intr_mat"] = pcd_episode["static_cam_intr_mat"]
        episode["static_cam_ex_mat"] = pcd_episode["static_cam_ex_mat"]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(
                    self.enrich_lang[task] + [episode["language"]]
                )
                episode["language"] = enrich_lang
        episode["action_mask"] = action_mask
        episode["image_mask"] = image_mask

        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        # add support for partial calvin data
        # partial_st_ed_list = []
        partial_st_ed_list = load_partial_traj_data()
        few_shot_st_ed_list = load_few_shot_traj_data()
        # import pdb; pdb.set_trace()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if (start_idx, end_idx) not in partial_st_ed_list:
                    continue
            if self.few_shot:
                if (start_idx, end_idx) not in few_shot_st_ed_list:
                    continue
            if lang_task[i] in self.exclude_tasks:
                continue
            cnt = 0
            right_pad = end_idx - start_idx - self.act_step - self.window_size + 1
            for idx in range(start_idx, end_idx + 1 - self.window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                    right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
                cnt += 1

        return (
            np.array(episode_lookup),
            lang_lookup,
            right_pad_lookup,
            lang_ann,
            lang_task,
        )

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        right_pad_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            right_pad = end_idx - start_idx - self.act_step - self.window_size
            for idx in range(start_idx, end_idx + 2 - self.window_size):
                episode_lookup.append(idx)
                right_pad_lookup.append(min(0, right_pad))
                right_pad -= 1
        return np.array(episode_lookup), right_pad_lookup

    def wrap_instruction_and_action(self, lang, action, action_mask):
        # modified from OpenVLA
        IGNORE_INDEX = -100
        prompt_builder = get_prompt_builder(
            self.model_name, eos=self.tokenizer.eos_token, bos=self.tokenizer.bos_token
        )
        # if pass in multi-step actions, we concat them
        action_mask = action_mask.astype(bool)
        # action = action[:-1]
        action_dim = action.shape[1]
        action = action.flatten()
        conversation = [
            {
                "from": "human",
                "value": (
                    f"What action should the robot take to {lang}?"
                    if self.act_step == 1
                    else f"What {self.act_step} step actions should the robot take to {lang}?"
                ),
            },
            {"from": "gpt", "value": ""},
        ]
        input_ids = []
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        action_ids = self.action_tokenizer.encode_actions_to_token_ids(action)

        all_input_ids = []
        all_labels = []

        for i in range(self.window_size):
            start_action_index = i
            end_action_index = (i + self.act_step) - 1
            tmp_action_ids = action_ids[
                start_action_index * action_dim : end_action_index * action_dim
            ]
            tmp_action_mask = action_mask[start_action_index:end_action_index]
            right_pad_len = (~tmp_action_mask).sum() * action_dim

            if self.tokenizer.eos_token is None:
                tmp_input_ids = input_ids + tmp_action_ids
            else:
                tmp_input_ids = input_ids[:-1] + tmp_action_ids + [input_ids[-1]]
            tmp_labels = list(tmp_input_ids)
            tmp_input_ids, tmp_labels = torch.tensor(tmp_input_ids), torch.tensor(
                tmp_labels
            )

            if self.tokenizer.eos_token is None:
                tmp_labels[: -len(tmp_action_ids)] = IGNORE_INDEX
                if right_pad_len != 0:
                    tmp_labels[-right_pad_len:] = IGNORE_INDEX
            else:
                tmp_labels[: -(len(tmp_action_ids) + 1)] = IGNORE_INDEX
                if right_pad_len != 0:
                    tmp_labels[-right_pad_len - 1 : -1] = IGNORE_INDEX
            if not self.predict_stop_token and self.tokenizer.eos_token:
                tmp_labels[-1] = IGNORE_INDEX
            all_input_ids.append(tmp_input_ids)
            all_labels.append(tmp_labels)

        all_input_ids = torch.stack(all_input_ids)
        all_labels = torch.stack(all_labels)
        return all_input_ids, all_labels

    # NOTE
    def collater(self, sample):
        # forward transformation of mu law companding
        if self.norm_action:
            new_sample = []
            for s in sample:
                s["actions"] = normalize_action(
                    s["actions"], self.norm_min, self.norm_max, maintain_last=True
                )
                new_sample.append(s)
            sample = new_sample

        if self.regular_action:
            new_sample = []
            for s in sample:
                s["actions"] = regularize_action(s["actions"], self.x_mean, self.x_std)
                new_sample.append(s)
            sample = new_sample
            pass

        if self.use_mu_law: # 对continusous action没有用
            new_sample = []
            for s in sample:
                s["actions"] = mu_law_companding(s["actions"], self.mu_val)
                new_sample.append(s)
            sample = new_sample

        action_tensors = torch.from_numpy(
            np.array([np.stack(s["actions"]) for s in sample])
        )[:, :-1]
        action_mask = torch.from_numpy(
            np.array([np.stack(s["action_mask"]) for s in sample])
        )[:, :-1]
        robot_obs = torch.from_numpy(
            np.array([np.stack(s["robot_obs"]) for s in sample])
        )[:, :-1]

        if self.tcp_rel:
            action_tensors = world_to_tcp_frame(action_tensors, robot_obs)

        image_mask = torch.from_numpy(
            np.array([np.stack(s["image_mask"]) for s in sample])
        )
        image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
        gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])

        stacked_language = [s["lang"] for s in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)
        action_tensors[..., -1] = ((action_tensors[..., -1] + 1) // 2).float()
        robot_obs[..., -1] = ((robot_obs[..., -1] + 1) // 2).float()
        state_tensors = robot_obs[:, : self.window_size]

        image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]
        image_tensors = image_tensors[:, : self.window_size]
        if gripper_tensors is not None:
            gripper_chunk = gripper_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
                0, 1, 5, 2, 3, 4
            )[:, 1:]
            gripper_tensors = gripper_tensors[:, : self.window_size]
        else:
            gripper_chunk = None

        fwd_mask = image_mask.unfold(1, self.fwd_pred_next_n, 1)[:, 1:]

        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 3, 2
        )
        action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)

        # prepare inputs for esm
        # img
        image_tensors_vggt = None
        gripper_tensors_vggt = None
        image_tensors_vggt = torch.stack([s["vggt"]["rgb_static"] for s in sample])
        image_tensors_vggt = image_tensors_vggt[:, : self.window_size] # (bs, ws, 3, 224, 224)
        # depth and point masks
        static_depth_tensors_vggt = None
        point_mask_tensors_vggt = None
        static_depth_tensors_vggt = torch.stack([s["vggt"]["depth_static"] for s in sample])
        point_mask_tensors_vggt = torch.stack([s["vggt"]["point_masks"] for s in sample])
        static_depth_tensors_vggt = static_depth_tensors_vggt[:, : self.window_size] # (bs, ws, 224, 224, 1)
        point_mask_tensors_vggt = point_mask_tensors_vggt[:, : self.window_size] # (bs, ws, 224, 224)
        # cam params
        static_cam_intr_mat_tensors = None
        static_cam_ex_mat_tensors = None
        static_cam_intr_mat_tensors = torch.stack([s["vggt"]["static_cam_intr_mat"] for s in sample])
        static_cam_ex_mat_tensors = torch.stack([s["vggt"]["static_cam_ex_mat"] for s in sample])
        static_cam_intr_mat_tensors = static_cam_intr_mat_tensors[:, : self.window_size] # (bs, ws, 3, 3)
        static_cam_ex_mat_tensors = static_cam_ex_mat_tensors[:, : self.window_size] # (bs, ws, 3, 4)

        # robot_obs_chunk = robot_obs.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        # if self.tcp_rel:
        #     action_chunck = world_to_tcp_frame(action_chunck, robot_obs_chunk)

        bs = len(sample)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None

        if self.discrete_action:
            res = [
                self.wrap_instruction_and_action(
                    s["lang"], s["actions"], s["action_mask"]
                )
                for s in sample
            ]
            tmp_input_ids = [_[0] for _ in res]
            tmp_labels = [_[1] for _ in res]
            max_len = max([_.shape[-1] for _ in tmp_input_ids])
            instr_and_action_ids = torch.zeros(
                (bs, self.window_size, max_len), dtype=torch.long
            )
            instr_and_action_labels = (
                torch.ones((bs, self.window_size, max_len), dtype=torch.long) * -100
            )
            instr_and_action_mask = torch.zeros(
                (bs, self.window_size, max_len), dtype=torch.bool
            )

            for i in range(bs):
                instr_and_action_ids[
                    i, :, : tmp_input_ids[i].shape[-1]
                ] = tmp_input_ids[i]
                instr_and_action_labels[i, :, : tmp_labels[i].shape[-1]] = tmp_labels[i]
                instr_and_action_mask[i, :, : tmp_input_ids[i].shape[-1]] = 1

        res = {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "state": state_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": attention_mask,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_chunk,
            "fwd_mask": fwd_mask,
            "action_chunck": action_chunck,
            "chunck_mask": action_mask,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "raw_text": stacked_language,
            "data_source": self.task_type,
            # esm inputs
            "rgb_vggt": image_tensors_vggt,
            "hand_rgb_vggt": gripper_tensors_vggt,
            "static_depth_vggt": static_depth_tensors_vggt,
            "point_masks_vggt": point_mask_tensors_vggt,
            "static_cam_intr_mat_vggt": static_cam_intr_mat_tensors,
            "static_cam_ex_mat_vggt": static_cam_ex_mat_tensors,
        }
        return res


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


def preprocess_text_calvin(sample, tokenizer, decoder_type="lstm"):
    tokenizer.padding_side = "right"
    max_length = 48 if decoder_type == "rt2_enc" else 32
    if decoder_type == "rt2_enc":
        action_str = "".join([f"<Action_{i}>" for i in range(7)])
        sample = [
            (f"<image>{s.strip()}{action_str}<|endofchunk|>{tokenizer.eos_token}")
            for s in sample
        ]

    else:
        sample = [
            (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
    text = tokenizer(
        sample,
        max_length=max_length,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def preprocess_interleaved(sample, tokenizer, clip_processor, sim_threshold):
    info = json.loads(sample[0])
    tar_file_obj = io.BytesIO(sample[1])
    image_tar = tarfile.open(fileobj=tar_file_obj)
    sentences = info["text_list"]

    images, image_idxs = [], []
    for image_path, sim in zip(info["image_info"], info["similarity_matrix"]):
        # pick one image per sentence
        if info["image_info"][image_path]["matched_text_index"] in image_idxs:
            continue
        rawbytes = image_tar.extractfile(
            os.path.join(image_tar.getnames()[0], image_path)
        ).read()

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sim[info["image_info"][image_path]["matched_text_index"]] < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        image_idxs.append(info["image_info"][image_path]["matched_text_index"])

    if len(images) == 0:
        raise ValueError("No images in sample")

    # filter out images that are exact duplicates
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), MAX_NUM_IMAGES))
    images_tensors = images_tensors[keep_ixs]
    image_idxs = [image_idxs[ix] for ix in keep_ixs]

    # pad to 5 images
    if len(images_tensors) < MAX_NUM_IMAGES:
        zero_padding = torch.zeros(
            (MAX_NUM_IMAGES - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # add in <image> and <eoc> tokens
    # eoc after sentence = "sentence loss"
    for ix in image_idxs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"

    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )

    if num_images == 0:
        raise ValueError("No images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def load_partial_traj_data():
    file = open(
        f"{Path(os.path.abspath(falcon.__path__[0])).parent.as_posix()}/configs/data/calvin/data_name_list.txt",
        "r",
    )
    lines = file.readlines()
    lines = [tuple([int(_) for _ in l.split()[1:]]) for l in lines]
    return lines


def load_few_shot_traj_data():
    file = json.load(
        open(
            f"{Path(os.path.abspath(falcon.__path__[0])).parent.as_posix()}/configs/data/calvin/10_shot_task_data.json",
            "r",
        )
    )
    res = []
    for task in file:
        res.extend([tuple(_) for _ in file[task]])
    return res
