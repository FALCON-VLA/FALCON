import json
import os.path
from copy import deepcopy
import torch
from PIL import Image
from typing import Literal
import numpy as np
import functools

from falcon.train.base_trainer import BaseTrainer
from falcon.utils.model_utils import build_tokenizer
from falcon.data.data_utils import get_text_function
from falcon.data.data_utils import (
    preprocess_image,
    get_prompt_builder,
    tcp_to_world_frame,
)
from queue import Queue
from falcon.model.policy_head.action_tokenizer import ActionTokenizer
from eval.calvin.eval_utils import (
    deproject,
    get_gripper_camera_view_matrix,
    eval_filtered_sample_pcd,
)

fwd_decay_ratio = 1
# ---------------------------------------------
# Global configuration for pcd filtering: maximum distance from centroid (meters)
# Points farther than this from their frame centroid are considered noise.
# ---------------------------------------------
MAX_DISTANCE = 1.0

class CustomModel:
    # model option
    def __init__(
        self,
        ckpt_path,
        configs,
        device,
        save_dir=None,
        raw_calvin=True,
        debug=False,
        use_act_chunk=False,
        action_ensemble=False,
    ):
        self.model = BaseTrainer(configs=configs)
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug, use_act_chunk)
        # self.model.model.lm_head.window_size = 1

    def init_config(
        self, ckpt_path, configs, device, save_dir=None, raw_calvin=False, debug=False, use_act_chunk=False
    ):
        ### load and convert checkpoint
        self.debug = debug
        self.use_act_chunk = use_act_chunk
        if configs["model"] == "kosmos":
            import transformers

            package_dir = transformers.__path__[0]
            os.system(
                "cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(
                    package_dir
                )
            )

        if not self.debug:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                new_state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                new_state_dict = ckpt["model_state_dict"]
            else:
                raise KeyError("no checkpoint dict in the loaded pretrain parameters")

            new_state_dict = self.convert_old_state_dict(new_state_dict)
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"FALCON CKPT Loaded \n {msg}")
            del new_state_dict

            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_name = os.path.basename(ckpt_path)
            save_dir = ckpt_dir if save_dir is None else save_dir
            load_info_path = os.path.join(save_dir, f"{ckpt_name}_loading_msg.json")
            if os.path.exists(load_info_path):
                os.system(f"sudo rm {load_info_path}")
            with open(load_info_path, "w") as f:
                _info = {
                    "missing_keys": msg.missing_keys,
                    "unexpected_keys": msg.unexpected_keys,
                }
                json.dump(_info, f, indent=2)
                print(f"Model loading msg is updated to: {load_info_path}")

        self.configs = configs

        dtype = torch.float32
        if self.configs["trainer"]["precision"] == "bf16":
            dtype = torch.bfloat16
        elif self.configs["trainer"]["precision"] == "fp16":
            dtype = torch.float16
        self.dtype = dtype
        self.act_head_configs = self.configs["act_head"]
        self.raw_calvin = raw_calvin
        self.tcp_rel = self.configs.get("tcp_rel", False)

        print(f"raw action: {self.raw_calvin}")

        self.device = device
        self.policy = self.model
        self.policy = self.policy.to(self.dtype)
        self.policy.to(self.device)
        self.policy.eval()

        if not hasattr(self.policy.model, "lm_head"):
            self.policy.model.lm_head = self.policy.model.act_head

        self.tokenizer = build_tokenizer(self.configs["tokenizer"])

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.act_step = self.fwd_pred_next_n + 1
        self.use_hand_rgb = self.configs["use_hand_rgb"]

        if hasattr(self, "policy_setup"):
            data_mix = "bridge" if self.policy_setup == "widowx_bridge" else "rt_1"
            configs["train_dataset"]["data_mix"] = data_mix
            configs["val_dataset"]["data_mix"] = data_mix

        image_preprocess = self.model.model.image_processor
        self.image_preprocess = functools.partial(
            preprocess_image,
            image_processor=image_preprocess,
            model_type=configs["model"],
        )

        self.text_preprocess = get_text_function(
            self.model.model.tokenizer, configs["model"]
        )

        self.action_space = self.configs["act_head"].get("action_space", "continuous")
        if self.action_space == "discrete":
            self.action_tokenizer = ActionTokenizer(
                self.tokenizer,
                bins=self.act_head_configs["n_bin"],
                min_action=self.act_head_configs["min_action"],
                max_action=self.act_head_configs["max_action"],
            )

        print(f"Evaluating checkpoint {ckpt_path}")

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def ensemble_action(self, action):
        if action.ndim >= 3:
            action = action.squeeze()

        if action.ndim == 1:
            action = action.unsqueeze(0)

        self.action_hist_list.append(action)

        act_cache = []
        # max_len = self.fwd_pred_next_n
        max_len = 1
        # max_len = 5
        while len(self.action_hist_list) > max_len:
            self.action_hist_list.pop(0)

        idx = 0
        for act in self.action_hist_list[::-1]:
            # print(act.shape)
            act_cache.append(act[idx])
            idx += 1

        act_cache = torch.stack(act_cache, dim=0)

        weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])
        weights = weights / weights.sum()

        weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)

        return weighted_act

    @staticmethod
    def convert_old_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k.replace("module.", "")
            else:
                new_k = k

            if not new_k.startswith("model."):
                new_k = "model." + new_k

            new_state_dict[new_k] = state_dict[k]
        return new_state_dict

    def _get_default_calvin_config(self):
        return {
            "type": "DiskCalvinDataset",
            "data_dir": "CALVIN/task_ABCD_D/val",
            "c_act_scaler": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }

    def add_element_to_queue(self, q: Queue, element):
        while q.qsize() >= q.maxsize:
            q.get()
        q.put(element)

    def get_history(self, q: Queue, pad: Literal["zero", "first"] = "zero"):
        queue_list = list(q.queue)
        if len(queue_list) == 0:
            return queue_list, None
        history_type = self.configs["act_head"].get("history_type", "pre")
        if history_type == "pre":
            pad_len = 0
        else:
            raise ValueError(f"Unsupported history type {history_type}")
        element = queue_list[0]
        if pad == "zero":
            if isinstance(element, torch.Tensor):
                element = torch.zeros_like(element)
            elif isinstance(element, np.ndarray):
                element = np.zeros_like(element)
            else:
                raise ValueError("This type is not supported")
            queue_list = [element for _ in range(pad_len)] + queue_list
        else:
            if isinstance(element, torch.Tensor):
                pad_list = [element.clone() for _ in range(pad_len)]
            elif isinstance(element, np.ndarray):
                pad_list = [deepcopy(element) for _ in range(pad_len)]
            queue_list = pad_list + queue_list
        pad_mask = np.ones(q.maxsize, dtype=bool)
        pad_mask[:pad_len] = False
        return queue_list, pad_mask

    def preprocess(self, obs, lang, mode="continuous"):
        # preprocess image
        image = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0) # (1, 1, 3, 224, 224)

        gripper_x = None
        if "rgb_gripper" in obs["rgb_obs"]:
            gripper = obs["rgb_obs"]["rgb_gripper"]
            gripper = Image.fromarray(gripper)
            gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
            gripper_x = gripper_x.to(self.device).to(self.dtype)

        text_x, mask = self.text_preprocess([lang])

        return (
            image_x.to(self.device).to(self.dtype),
            gripper_x,
            text_x.to(self.device),
            mask.to(self.device),
        )

    def preprocess_esm(self, obs, lang, env, mode="continuous"):
        # get inputs for esm
        image_vggt = deepcopy(obs["rgb_obs"]["rgb_static"])
        # preprocess image
        image = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0) # (1, 1, 3, 224, 224)

        gripper_x = None
        if "rgb_gripper" in obs["rgb_obs"]:
            gripper = obs["rgb_obs"]["rgb_gripper"]
            gripper = Image.fromarray(gripper)
            gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
            gripper_x = gripper_x.to(self.device).to(self.dtype)

        # prepare inputs for esm
        depth_static = obs["depth_obs"]["depth_static"]  # (200, 200)
        static_cam = env.cameras[0]
        from eval.calvin.eval_utils import extract_camera_parameters
        width = 200
        height = 200
        _, static_intrinsic_matrix = extract_camera_parameters(
            static_cam.viewMatrix, static_cam.projectionMatrix, width, height
        )

        from falcon.data.data_utils import process_sequence_data
        # TODO: hardcode for param setup
        vggt_target_res = (224, 224)
        np_rgb_vggt = image_vggt.reshape(1, 200, 200, 3) # (1, 200, 200, 3), dtype: uint8 
        np_depth_vggt = depth_static.reshape(1, 200, 200) # (1, 200, 200)
        invalid_mask = ~np.isfinite(np_depth_vggt)
        np_depth_vggt[invalid_mask] = 0
        assert np.all(np.isfinite(np_depth_vggt)), "still got nan in depth!"
        assert np.all(np_depth_vggt >= 0), "depth val cannot be negetive!"
        static_cam_intr_mat = static_intrinsic_matrix.reshape(1, 3, 3) # (1, 3, 3)

        # Call the processing function
        processed_rgb, processed_depth, _, point_masks = process_sequence_data(
            np_rgb_vggt=np_rgb_vggt,
            np_depth_vggt=np_depth_vggt,
            static_cam_intr_mat=static_cam_intr_mat,
            target_resolution=vggt_target_res
        )

        image_vggt_x = processed_rgb.unsqueeze(0).to(self.device).to(self.dtype) # (1, 1, 3, 224, 224)
        static_depth_vggt_x = processed_depth.unsqueeze(0).to(self.device).to(self.dtype) # (1, 1, 224, 224, 1)
        point_masks_vggt_x = point_masks.unsqueeze(0).to(self.device).to(self.dtype) # (1, 1, 224, 224)

        text_x, mask = self.text_preprocess([lang])

        return (
            image_x.to(self.device).to(self.dtype),
            gripper_x,
            text_x.to(self.device),
            mask.to(self.device),
            image_vggt_x,
            static_depth_vggt_x,
            point_masks_vggt_x,
        )

    def preprocess_3D(self, obs, lang, env, mode="continuous"):
        # preprocess image
        image = obs["rgb_obs"]["rgb_static"]
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0)

        gripper_x = None
        if "rgb_gripper" in obs["rgb_obs"]:
            gripper = obs["rgb_obs"]["rgb_gripper"]
            gripper = Image.fromarray(gripper)
            gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
            gripper_x = gripper_x.to(self.device).to(self.dtype)

        # preprocess pcd
        depth_static = obs["depth_obs"]["depth_static"]  # (200, 200)
        depth_gripper = obs["depth_obs"]["depth_gripper"]  # (84, 84)

        static_cam = env.cameras[0]
        gripper_cam = env.cameras[1]
        gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)
        n_points = self.act_head_configs.get("pcd_n_points", 2048)

        static_pcd_x = None
        static_pcd = deproject(
            static_cam, depth_static,
            homogeneous=False, sanity_check=False
        ).transpose(1, 0)
        static_pcd = np.reshape(
            static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
        )
        static_pcd_x = eval_filtered_sample_pcd(static_pcd, n_points, MAX_DISTANCE).unsqueeze(0)
        static_pcd_x = static_pcd_x.unsqueeze(0).to(self.device).to(self.dtype)

        gripper_pcd_x = None
        if self.act_head_configs.get("use_hand_pcd", False):
            gripper_pcd = deproject(
                gripper_cam, depth_gripper,
                homogeneous=False, sanity_check=False
            ).transpose(1, 0)
            gripper_pcd = np.reshape(
                gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
            )
            gripper_pcd_x = eval_filtered_sample_pcd(gripper_pcd, n_points, MAX_DISTANCE).unsqueeze(0)
            gripper_pcd_x = gripper_pcd_x.unsqueeze(0).to(self.device).to(self.dtype)

        text_x, mask = self.text_preprocess([lang])

        return (
            image_x.to(self.device).to(self.dtype),
            gripper_x,
            text_x.to(self.device),
            mask.to(self.device),
            static_pcd_x,
            gripper_pcd_x,
        )

    def step(self, obs, goal, env):
        """Step function."""
        input_dict = dict()
        if self.act_head_configs["type"] in {"FCDecoder_3D"}:
            image_x, gripper_x, text_x, mask, static_pcd_x, gripper_pcd_x = self.preprocess_3D(obs, goal, env, self.action_space)

            input_dict["rgb"] = image_x
            input_dict["hand_rgb"] = gripper_x
            input_dict["text"] = text_x
            input_dict["text_mask"] = mask
            input_dict["static_pcd"] = static_pcd_x
            input_dict["gripper_pcd"] = gripper_pcd_x

        elif self.act_head_configs["type"] in {"FCDecoder_ESM", "LSTMDecoder_ESM"}:
            image_x, gripper_x, text_x, mask, image_vggt_x, static_depth_vggt_x, point_masks_vggt_x = self.preprocess_esm(obs, goal, env, self.action_space)

            input_dict["rgb"] = image_x
            input_dict["hand_rgb"] = gripper_x
            input_dict["text"] = text_x
            input_dict["text_mask"] = mask
            input_dict["rgb_vggt"] = image_vggt_x
            input_dict["static_depth_vggt"] = static_depth_vggt_x
            input_dict["point_masks_vggt"] = point_masks_vggt_x

        else:
            image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)

            input_dict["rgb"] = image_x
            input_dict["hand_rgb"] = gripper_x
            input_dict["text"] = text_x
            input_dict["text_mask"] = mask

        with torch.no_grad():
            action = self.policy.inference_step(input_dict)["action"]

        if self.action_space != "discrete":
            if action[0].ndim == action[1].ndim + 1:
                action = (action[0], action[1].unsqueeze(2))
            action = torch.cat(
                [action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()],
                dim=-1,
            )

        if isinstance(action, tuple):
            action = torch.cat([action[0], action[1]], dim=-1)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        if action.ndim == 2:
            action = action.unsqueeze(1)

        if action.ndim == 3:
            action = action.unsqueeze(1)

        action = action.detach().cpu()

        if self.tcp_rel:
            robot_obs = (
                torch.from_numpy(obs["robot_obs"])
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, self.window_size, self.fwd_pred_next_n, 1)
            )
            action = tcp_to_world_frame(action, robot_obs)

        if not self.use_act_chunk:
            action = self.ensemble_action(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze()
            if not self.use_act_chunk:           
                if action.ndim == 2:
                    action = action[0]

        if self.configs.get("use_mu_law", False):
            from falcon.data.data_utils import inverse_mu_law_companding

            action = inverse_mu_law_companding(
                action, self.configs.get("mu_val", 255), maintain_last=True
            )

        if self.configs.get("norm_action", False):
            from falcon.data.data_utils import unnoramalize_action

            if isinstance(action, tuple):
                action = (
                    unnoramalize_action(
                        action[0], self.configs["norm_min"], self.configs["norm_max"]
                    ),
                    action[1],
                )
            else:
                action = unnoramalize_action(
                    action, self.configs["norm_min"], self.configs["norm_max"]
                )

        if action.dim() == 2:
            # use the whole act_chunk for rollout
            action[:, -1] = (action[:, -1] - 0.5) * 2
        elif action.dim() == 1:
            action[-1] = (action[-1] - 0.5) * 2
        else:
            raise ValueError("ERROR: action dim is not 2 or 1")

        self.rollout_step_counter += 1

        if action.dim() == 2:
            for i in range(action.shape[0]):
                action[i, -1] = 1 if action[i, -1] > 0 else -1
        else:
            action[-1] = 1 if action[-1] > 0 else -1
        # print(f"step {self.rollout_step_counter} action {action}")
        return action

    def reset(self):
        if hasattr(self.model.model, "lm_head"):
            self.model.model.lm_head.hidden_state = None
            self.model.model.lm_head.history_memory = []
        if hasattr(self.model.model, "act_head"):
            self.model.model.act_head.hidden_state = None
            self.model.model.act_head.history_memory = []
            self.model.model.act_head.esm_history_memory_rgb = []

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()
