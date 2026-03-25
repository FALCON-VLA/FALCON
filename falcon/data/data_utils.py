import os
from typing import List, Literal, Tuple
import sys
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
import json
import csv
from einops import rearrange, repeat
import numpy as np
import logging
import logging.handlers
import requests
import random
import math
import copy

import torch
import torch.nn as nn
from torch.utils.data import default_collate
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from pathlib import Path

logger = logging.getLogger(__name__)


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
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
            0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    @torch.no_grad()
    def forward(self, x):
        assert isinstance(x, torch.Tensor) and len(x.size()) == 4
        x = x.float()
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


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
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
            0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def collate_with_none(batch):
    assert isinstance(batch[0], dict)

    delete_keys = set()
    data_type = None
    for k in batch[0]:
        if batch[0][k] is None:
            delete_keys.add(k)
        elif "data_type" in batch[0]:
            data_type = batch[0]["data_type"]

    delete_keys.add("data_type")
    for k in delete_keys:
        for d in batch:
            d.pop(k, None)

    collated = default_collate(batch)
    for k in delete_keys:
        collated[k] = None
    collated["data_type"] = data_type

    return collated


def list_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if os.path.isdir(folder):
            files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
        elif os.path.isfile(folder):
            files.append(folder)
        else:
            print("Path {} is invalid".format(folder))
            sys.stdout.flush()
    return files


def list_all_files(dirs, verbose=False):
    sub_dirs = list_files(dirs)
    all_files = []
    all_dirs = []

    if verbose:
        _iter = tqdm(sub_dirs)
    else:
        _iter = sub_dirs

    for d in _iter:
        if os.path.isdir(d):
            all_dirs.append(d)
        else:
            all_files.append(d)

    if all_dirs:
        all_files.extend(list_all_files(all_dirs))
    return all_files


def list_dir_with_cache(data_dir, cache_dir=None, verbose=True):
    from falcon.utils.dist_train import get_rank

    data_dir = data_dir.rstrip("/")

    if cache_dir is None:
        _parent_dir = os.path.dirname(data_dir)
        _base_name = os.path.basename(data_dir)
        _cache_file = os.path.join(_parent_dir, _base_name + f"_filelist.json")
    else:
        max_name_length = os.pathconf("/", "PC_NAME_MAX")
        _cache_name = data_dir.strip("/").replace("/", "_") + ".json"
        _cache_name = _cache_name[-max_name_length:]
        os.makedirs(cache_dir, exist_ok=True)
        _cache_file = os.path.join(cache_dir, _cache_name)

    if os.path.exists(_cache_file):
        if get_rank() == 0 and verbose:
            print(f"Loading data list from {_cache_file}...")

        with open(_cache_file) as f:
            return json.load(f)

    else:
        verbose = get_rank() == 0 and verbose
        data_list = list_all_files([data_dir], verbose=verbose)
        _temp_cache = _cache_file + f".rank{str(get_rank())}"
        max_name_length = os.pathconf("/", "PC_NAME_MAX")
        _temp_cache = _temp_cache[-max_name_length:]
        with open(_temp_cache, "w") as f:
            json.dump(data_list, f)
        if not os.path.exists(_cache_file):
            import shutil

            shutil.move(_temp_cache, _cache_file)

    return data_list


def grouping(data_list, num_group):
    groups = [[] for _ in range(num_group)]
    for i, d in enumerate(data_list):
        groups[i % num_group].append(d)
    return groups


def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def read_csv(rpath, encoding=None, **kwargs):
    if rpath.startswith("hdfs"):
        raise NotImplementedError
    cfg_args = dict(delimiter=",")
    cfg_args.update(kwargs)
    try:
        data = []
        with open(rpath, encoding=encoding) as csv_file:
            csv_reader = csv.reader(csv_file, **cfg_args)
            columns = next(csv_reader)
            for row in csv_reader:
                data.append(dict(zip(columns, row)))
        return data
    except:
        return []


def claw_matrix(n, k, device="cpu"):
    upper_triangle_matrix = torch.triu(torch.ones(n, n), diagonal=0).to(device)
    lower_triangle_matrix = torch.tril(torch.ones(n, n), diagonal=k).to(device)

    claw = upper_triangle_matrix * lower_triangle_matrix

    return claw


def generate_chunck_data(data, window_size, chunk_size):
    if data is None:
        return None
    bs, seq_len = data.shape[:2]
    raw_data_shape = data.shape[2:]
    data_flatten = data.flatten().view(bs, seq_len, -1)
    assert (
        seq_len == window_size + chunk_size
    ), f"The sequence length should be {window_size + chunk_size}"
    data_flatten = repeat(data_flatten, "b s d -> b w s d", w=window_size)

    mask = claw_matrix(seq_len, chunk_size - 1, data_flatten.device)
    # mask = mask - torch.diag_embed(mask.diag()) # set current obs mask to 0
    mask = mask[:window_size].bool()

    mask = repeat(mask, "w s -> b w s d", b=bs, d=data_flatten.shape[-1])
    data_flatten = torch.masked_select(data_flatten, mask)

    # data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)
    data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)

    return data_flatten


def get_text_function(tokenizer, tokenizer_type, max_length=256):
    import functools

    if tokenizer_type == "flamingo":

        def preprocess_text_flamingo(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [
                (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}")
                for s in sample
            ]
            text = tokenizer(
                sample,
                max_length=max_length,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_flamingo, tokenizer=tokenizer)
    elif tokenizer_type == "llava":
        DEFAULT_IMAGE_TOKEN = "<image>"

        def preprocess_text_llava(sample, tokenizer):
            # tokenizer.padding_side = "right"
            # sample = [
            #     (f"{tokenizer.bos_token}{DEFAULT_IMAGE_TOKEN}{s.strip()}{tokenizer.eos_token}") for s in sample
            # ]
            sample = [(f"{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                max_length=2048,
                padding="longest",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_llava, tokenizer=tokenizer)
    elif tokenizer_type == "llava_orig":
        DEFAULT_IMAGE_TOKEN = "<image>"

        def preprocess_text_llava(sample, tokenizer):
            system_prompt = "You are a helpful language and vision assistant. \
            You are able to understand the visual content that the user provides, \
            and assist the user with a variety of tasks using natural language."
            rgb_prompt = "This is the static camera observation: "
            gripper_prompt = "This is the egocentric girpper camera observation: "
            tokenizer.padding_side = "right"
            sample = [
                (
                    f"{tokenizer.bos_token}{system_prompt}{rgb_prompt}{DEFAULT_IMAGE_TOKEN}{gripper_prompt}{DEFAULT_IMAGE_TOKEN}{s.strip()}{tokenizer.eos_token}"
                )
                for s in sample
            ]
            text = tokenizer(
                sample,
                max_length=max_length,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )

            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_llava, tokenizer=tokenizer)
    elif tokenizer_type == "qwen":

        def preprocess_text_qwen(sample, tokenizer):
            tokenizer.padding_side = "right"
            tokenizer.pad_token_id = tokenizer.eod_id
            sample = [(f"{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_qwen, tokenizer=tokenizer)
    elif tokenizer_type == "kosmos":

        def preprocess_text_kosmos(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<grounding>An image of a robot {s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_kosmos, tokenizer=tokenizer)
    elif tokenizer_type == "moondream":

        def preprocess_text_moondream(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<|endoftext|>{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=True,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_moondream, tokenizer=tokenizer)
    elif tokenizer_type == "uform":

        def preprocess_text_uform(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<image> {s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=True,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_uform, tokenizer=tokenizer)
    elif tokenizer_type == "paligemma":

        def preprocess_text_paligemma(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"{tokenizer.eos_token}{s.strip()}\n") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=False,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_paligemma, tokenizer=tokenizer)
    else:

        def preprocess_text_default(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [(f"<|endoftext|>{s.strip()}") for s in sample]
            text = tokenizer(
                sample,
                truncation="only_first",
                return_tensors="pt",
                padding="longest",
                max_length=512,
                add_special_tokens=True,
            )
            return text["input_ids"], text["attention_mask"]

        return functools.partial(preprocess_text_default, tokenizer=tokenizer)


def preprocess_image(sample, image_processor, model_type):
    # if model_type.lower() == 'flamingo':
    #     image = [image_processor(s).unsqueeze(0) for s in sample]
    #     image = torch.cat(image, dim=0)

    # elif model_type == 'kosmos':
    #     image = [image_processor(s, return_tensors="pt")['pixel_values'] for s in sample]
    #     image = torch.cat(image, dim=0)
    if model_type.lower() in ["paligemma"]:
        image = [
            image_processor(images=s, return_tensors="pt")["pixel_values"]
            for s in sample
        ]
        image = torch.cat(image, dim=0)
    # elif model_type.lower() in ['llava']:
    #     image = [torch.stack(image_processor(s)['pixel_values'], dim=0) for s in sample]
    #     image = torch.cat(image, dim=0)
    else:
        # default clip preprocess
        image = [image_processor(s).unsqueeze(0) for s in sample]
        image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image


# modified from @property image_processor(self) in falcon/model/backbone/base_backbone.py. using for low level policy image data preprocess
def policy_image_processor(image_size=224):
    clip_mean = (0.485, 0.456, 0.406)
    clip_std = (0.229, 0.224, 0.225)
    
    image_preprocess = T.Compose(
        [
            T.Resize(
                (image_size, image_size),
                interpolation=Image.BICUBIC,
            ),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ]
    )
    return image_preprocess


def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    print(f"WARNING: total file: {len(lst)}, random pick: {k}." f" (ignored)")
    return new_lst


def build_logger(logger_name, logger_filename):
    global handler
    from falcon.data.vid_llava_constants import LOGDIR

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="UTF-8"
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def get_llava_image_processor(name):
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(name)


def get_llava_video_processor(tokenizer, video_decode_backend="decord", num_frames=8):
    from falcon.data.llava_vid_processor import LanguageBindVideoProcessor

    return LanguageBindVideoProcessor(video_decode_backend, num_frames, tokenizer)


def get_prompt_builder(model_name, eos=None, bos=None):
    model_family = "openvla"
    from falcon.data import prompting

    if "vicuna" in model_name.lower():
        return prompting.VicunaV15ChatPromptBuilder(model_family, eos=eos, bos=bos)
    elif "mistral" in model_name.lower():
        return prompting.MistralInstructPromptBuilder(
            model_family, eos=eos, bos=bos
        )
    elif "llama" in model_name.lower():
        return prompting.LLaMa2ChatPromptBuilder(model_family, eos=eos, bos=bos)
    elif "mpt" in model_name.lower():
        return prompting.PhiPromptBuilder(model_family, eos=eos, bos=bos)
    elif "qwen" in model_name.lower():
        return prompting.QwenPromptBuilder(model_family, eos=eos, bos=bos)
    else:
        return prompting.PhiPromptBuilder(model_family, eos=eos, bos=bos)


def mu_law_companding(x, mu=255, maintain_last=True):
    """Applies μ-law companding to the input array."""
    last_val = x[-1]
    res = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    if maintain_last:
        res[-1] = last_val
    return res


def inverse_mu_law_companding(y, mu=255, maintain_last=True):
    """Applies the inverse of μ-law companding to the input array."""
    last_val = y[-1]
    res = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)
    if maintain_last:
        res[-1] = last_val
    return res


def regularize_action(x, x_mean, x_std, eps=1e-6, maintain_last=True):
    # return a value ~ N(0, 1)
    last_val = x[-1]
    res = (x - x_mean) / (x_std + eps)
    if maintain_last:
        res[-1] = last_val
    return res


def unregularize_action(x, x_mean, x_std, eps=1e-6, maintain_last=True):
    last_val = x[-1]
    res = x * (x_std + eps) + x_mean
    if maintain_last:
        res[-1] = last_val
    return res


class PatchMask(nn.Module):
    def __init__(self, patch_size=16, mask_ratio=0.35):
        super(PatchMask, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Generate random mask coordinates.
        mask_coords = []
        for i in range(batch_size):
            for j in range(0, height, self.patch_size):
                for k in range(0, width, self.patch_size):
                    if random.random() < self.mask_ratio:
                        mask_coords.append((i, j, k))

        # Mask out the patches.
        masked_x = x.clone()
        for i, j, k in mask_coords:
            masked_x[i, :, j : j + self.patch_size, k : k + self.patch_size] = 0.0

        return masked_x


def normalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = action[..., -1]
    action = np.clip(action, a_min=float(action_min), a_max=float(action_max))
    res = 2 * (action - action_min) / (action_max - action_min) - 1
    if maintain_last:
        res[..., -1] = last_val
    return res


def normalize_action_perdim(action, action_min, action_max, maintain_last=False):
    """
    Normalize each dimension of `action` to [-1, 1], 
    setting dimensions with zero range (max==min) to 0.
    """
    # Input validation
    action = np.asarray(action)
    action_min = np.asarray(action_min)
    action_max = np.asarray(action_max)
    assert action.ndim >= 1, "action must have at least one dimension"
    D = action.shape[-1]
    assert action_min.ndim == 1 and action_max.ndim == 1, \
        "action_min and action_max must be 1D arrays"
    assert action_min.shape[0] == D and action_max.shape[0] == D, \
        f"length of action_min/action_max ({action_min.shape[0]}/{action_max.shape[0]}) must match action last dim ({D})"

    # Extract last-dimension original values for optional preservation
    last_vals = copy.deepcopy(action[..., -1])

    # Compute per-dimension range and identify zero-range dims
    diff = action_max - action_min
    if np.any(diff < 0):
        raise ValueError("Each action_max must be >= corresponding action_min")
    zero_range = diff == 0
    diff_safe = diff.copy()
    diff_safe[zero_range] = 1.0  # avoid division by zero

    # Clip action into [min, max] per dimension
    clipped = np.clip(action, a_min=action_min, a_max=action_max)

    # Linear mapping to [-1, 1]
    res = 2 * (clipped - action_min) / diff_safe - 1

    # For zero-range dims, set output to 0 (midpoint)
    if np.any(zero_range):
        idx = np.nonzero(zero_range)[0]
        # idx is array of dimension indices; broadcast assignment to all samples
        res[..., idx] = 0.0

    # Optionally restore the last dimension to its original value
    if maintain_last:
        res[..., -1] = last_vals

    return res


def unnoramalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = action[..., -1]
    res = 0.5 * (action + 1) * (action_max - action_min) + action_min
    if maintain_last:
        res[..., -1] = last_val
    return res


def unnoramalize_action_perdim(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = copy.deepcopy(action[..., -1])
    res = 0.5 * (action + 1) * (action_max - action_min) + action_min
    if maintain_last:
        res[..., -1] = last_val
    return res


def get_chunked_episode(
    window_sample: Literal["sliding", "range"],
    left_pad: bool,
    window_size: int,
    fwd_pred_next_n: int,
    episode_idx_range: np.ndarray,
):
    if window_sample == "range":
        window_range = np.arange(window_size)
        chunk_range = np.arange(window_size + fwd_pred_next_n)
        left_pad_mask = window_range[:, None] <= chunk_range[None, :]
    else:
        left_pad_mask = np.ones((window_size, window_size + fwd_pred_next_n))

    traj_len = len(episode_idx_range)
    chunk_indices = np.broadcast_to(
        np.arange(-window_size + 1, fwd_pred_next_n + 1),
        [traj_len, window_size + fwd_pred_next_n],
    ) + np.broadcast_to(
        np.arange(traj_len)[:, None],
        [traj_len, window_size + fwd_pred_next_n],
    )
    chunk_mask = (chunk_indices >= 0) & (chunk_indices < traj_len)
    chunk_indices = np.clip(chunk_indices, 0, traj_len - 1)
    left_index = 0 if left_pad else window_size - 1
    chunk_indices = chunk_indices[left_index:]
    chunk_mask = chunk_mask[left_index:]
    if window_sample == "range":
        tile_times = chunk_indices.shape[0]
        chunk_indices = np.repeat(chunk_indices, repeats=window_size, axis=0)
        chunk_mask = np.repeat(chunk_mask, repeats=window_size, axis=0)
        chunk_mask = chunk_mask & np.tile(left_pad_mask, (tile_times, 1))

    return episode_idx_range[chunk_indices], chunk_mask


def permute_tensor_last_dim(x: torch.Tensor, insert_dim: int):
    old_permutation = list(range(x.ndim))
    new_permutation = (
        old_permutation[:insert_dim]
        + [old_permutation[-1]]
        + old_permutation[insert_dim:-1]
    )
    return x.permute(new_permutation).contiguous()


def get_tensor_chunk(x: torch.Tensor, fwd_pred_next_n: int):
    chunk_x = x.unfold(0, fwd_pred_next_n, 1)
    chunk_x = permute_tensor_last_dim(chunk_x, 1)
    return chunk_x


def pad_sequences(sequences: List[torch.Tensor], padding_value):
    # 找出最后一维的最大长度
    max_len = max(tensor.shape[-1] for tensor in sequences)

    # 对每个 tensor 在最后一维进行 padding
    padded_tensors = [
        F.pad(
            tensor,
            (0, max_len - tensor.shape[-1]),
            mode="constant",
            value=padding_value,
        )
        for tensor in sequences
    ]

    # 将 list of tensor 堆叠为一个 tensor
    return torch.stack(padded_tensors)


def world_to_tcp_frame(action, robot_obs):
    # from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
    from falcon.data.pose_transforms import euler_angles_to_matrix, matrix_to_euler_angles

    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.reshape(b, s * f, -1)
            robot_obs = robot_obs.reshape(b, s * f, -1)
        b, s, _ = action.shape
        world_T_tcp = (
            euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        tcp_T_world = torch.inverse(world_T_tcp)
        pos_w_rel = action[..., :3].reshape(-1, 3, 1)
        pos_tcp_rel = tcp_T_world @ pos_w_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_w_rel = action[..., 3:6] * 0.01
        world_T_tcp_new = (
            euler_angles_to_matrix(robot_obs[..., 3:6] + orn_w_rel, convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
        orn_tcp_rel = matrix_to_euler_angles(
            tcp_new_T_tcp_old, convention="XYZ"
        ).float()
        orn_tcp_rel = torch.where(
            orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel
        )
        orn_tcp_rel = torch.where(
            orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel
        )
        # upscaling again
        orn_tcp_rel *= 100
        action_tcp = torch.cat(
            [
                pos_tcp_rel.reshape(b, s, -1),
                orn_tcp_rel.reshape(b, s, -1),
                action[..., -1:],
            ],
            dim=-1,
        )
        if flag:
            action_tcp = action_tcp.reshape(b, s, -1, action_tcp.shape[-1])
        assert not torch.any(action_tcp.isnan())
    return action_tcp


def accumulate_tcp_rel_actions(rel_actions, robot_obs):
    """
    1) world->tcp frame relative deltas
    2) position: prefix-sum along time dimension
       orientation: prefix-product of rotation matrices + euler normalization
    3) gripper: unchanged
    """
    from falcon.data.pose_transforms import euler_angles_to_matrix, matrix_to_euler_angles
    with autocast(dtype=torch.float32):
        # Step 1: transform to TCP-frame deltas
        tcp_rel = world_to_tcp_frame(rel_actions, robot_obs)
        B, T, _ = tcp_rel.shape

        # Step 2a: position cumulative sum (prefix-sum)
        pos_cum = torch.cumsum(tcp_rel[..., :3], dim=1)

        # Step 2b: orientation cumulative via rotation matrices
        orn_delta = tcp_rel[..., 3:6] * 0.01  # recover true small-angle deltas
        # orn_delta = tcp_rel[..., 3:6]  # not use true small-angle deltas
        R_delta = euler_angles_to_matrix(orn_delta.contiguous().view(-1, 3), convention="XYZ")
        R_delta = R_delta.float().view(B, T, 3, 3)

        # prefix-product along time
        R_cum = torch.zeros_like(R_delta)
        R_cum[:, 0] = R_delta[:, 0]
        for t in range(1, T):
            R_cum[:, t] = R_delta[:, t].matmul(R_cum[:, t-1])

        # extract cumulative Euler angles
        orn_cum = matrix_to_euler_angles(R_cum.view(-1, 3, 3), convention="XYZ")
        orn_cum = orn_cum.view(B, T, 3)
        # normalize angles into [-pi, pi]
        orn_cum = torch.remainder(orn_cum + np.pi, 2 * np.pi) - np.pi

        # scale back to original dyn range
        orn_cum = orn_cum * 100.0

        # Step 3: gripper remains unchanged
        gripper = tcp_rel[..., 6:].clone()

        # Step 4: assemble final chunk
        new_actions = torch.cat([pos_cum, orn_cum, gripper], dim=-1)

        # Final check: no NaNs
        assert not torch.isnan(new_actions).any(), "NaN in accumulated TCP-frame actions"

        return new_actions


def tcp_to_world_frame(action, robot_obs):
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    from falcon.data.pose_transforms import euler_angles_to_matrix, matrix_to_euler_angles

    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.reshape(b, s * f, -1)
            robot_obs = robot_obs.reshape(b, s * f, -1)
        # import pdb; pdb.set_trace()
        b, s, _ = action.shape
        world_T_tcp = (
            euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        pos_tcp_rel = action[..., :3].reshape(-1, 3, 1)
        pos_w_rel = world_T_tcp @ pos_tcp_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_tcp_rel = action[..., 3:6] * 0.01
        tcp_new_T_tcp_old = (
            euler_angles_to_matrix(orn_tcp_rel, convention="XYZ")
            .float()
            .reshape(-1, 3, 3)
        )
        world_T_tcp_new = world_T_tcp @ torch.inverse(tcp_new_T_tcp_old)

        orn_w_new = matrix_to_euler_angles(world_T_tcp_new, convention="XYZ").float()
        if torch.any(orn_w_new.isnan()):
            logger.warning("NaN value in euler angles.")
            orn_w_new = matrix_to_euler_angles(
                quaternion_to_matrix(matrix_to_quaternion(world_T_tcp_new)),
                convention="XYZ",
            ).float()
        orn_w_rel = orn_w_new - robot_obs[..., 3:6].reshape(-1, 3)
        orn_w_rel = torch.where(orn_w_rel < -np.pi, orn_w_rel + 2 * np.pi, orn_w_rel)
        orn_w_rel = torch.where(orn_w_rel > np.pi, orn_w_rel - 2 * np.pi, orn_w_rel)
        # upscaling again
        orn_w_rel *= 100
        action_w = torch.cat(
            [
                pos_w_rel.reshape(b, s, -1),
                orn_w_rel.reshape(b, s, -1),
                action[..., -1:],
            ],
            dim=-1,
        )
        if flag:
            action_w = action_w.reshape(b, s, -1, action_w.shape[-1])
        assert not torch.any(action_w.isnan())
    return action_w


def invert_accumulate_tcp_rel_actions(cum_actions, robot_obs):
    """
    invert of accumulate_tcp_rel_actions
      pos: backward diff,
      rot: R_delta = R_cum(t) @ inv(R_cum(t-1)), then extract Euler + scale,
      gripper: copy
      then tcp_to_world_frame -> world rel_actions
    """
    from falcon.data.pose_transforms import euler_angles_to_matrix, matrix_to_euler_angles
    with autocast(dtype=torch.float32):
        flag = False
        if len(cum_actions.shape) == 4:
            flag = True
            b, s, f, _ = cum_actions.shape
            cum_actions = cum_actions.reshape(b, s * f, -1)
        B, T, _ = cum_actions.shape

        # 1) position diff
        pos_diff = torch.zeros_like(cum_actions[..., :3])
        pos_diff[:, 0] = cum_actions[:, 0, :3]
        pos_diff[:, 1:] = cum_actions[:, 1:, :3] - cum_actions[:, :-1, :3]

        # 2a) reconstruct R_cum from cum_actions' Euler angles
        orn_cum = (cum_actions[..., 3:6] * 0.01).contiguous().view(-1,3)  # true rad
        # orn_cum = cum_actions[..., 3:6].contiguous().view(-1,3)  # true rad
        R_cum = euler_angles_to_matrix(orn_cum, convention="XYZ").view(B, T, 3, 3)

        # 2b) compute per-step R_delta by R_cum[t] @ inv(R_cum[t-1])
        R_delta = torch.zeros_like(R_cum)
        R_delta[:,0] = R_cum[:,0]
        for t in range(1, T):
            # R_delta[t] = R_cum[t] * inv(R_cum[t-1])
            R_delta[:,t] = R_cum[:,t].matmul(torch.inverse(R_cum[:,t-1]))

        # 2c) extract Euler deltas, normalize, scale
        orn_diff = matrix_to_euler_angles(R_delta.view(-1,3,3), convention="XYZ")
        orn_diff = orn_diff.view(B, T, 3)
        orn_diff = torch.remainder(orn_diff + np.pi, 2*np.pi) - np.pi
        orn_diff = orn_diff * 100.0

        # 3) gripper
        grip = cum_actions[..., 6:].clone()

        # 4) reassemble TCP-frame per-step rel_actions
        tcp_rel = torch.cat([pos_diff, orn_diff, grip], dim=-1)
        if flag:
            tcp_rel = tcp_rel.reshape(b, s, -1, tcp_rel.shape[-1])

        # 5) finally map back to world-frame rel_actions
        world_rel = tcp_to_world_frame(tcp_rel, robot_obs)
        assert not torch.isnan(world_rel).any(), "NaN in world rel_actions"

        return world_rel


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert(isRotm(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    # (-pi , pi]
    while x > np.pi:
        x -= (2 * np.pi)
    while x <= -np.pi:
        x += (2 * np.pi)
    while y > np.pi:
        y -= (2 * np.pi)
    while y <= -np.pi:
        y += (2 * np.pi)
    while z > np.pi:
        z -= (2 * np.pi)
    while z <= -np.pi:
        z += (2 * np.pi)
    return np.array([x, y, z])


def compute_rel_robot_state(arm_states, gripper_states, seq_len, state_dim):
    """
    Compute the relative robot state.

    Args:
        arm_states (np.ndarray): An array of shape (seq_len, 6) where each row contains the position and Euler angles (x, y, z, roll, pitch, yaw).
        gripper_states (np.ndarray): An array of shape (seq_len,) containing the gripper state at each timestep.
        seq_len (int): The sequence length (number of timesteps).
        state_dim (int): The dimension of the state, which should be at least 7.

    Returns:
        np.ndarray: The computed relative states with shape (seq_len, state_dim) containing relative position, relative Euler angles, and gripper state.
    """
    # Initialize the relative state array
    rel_states = np.zeros((seq_len, state_dim), dtype=np.float32)
    
    # Get the initial state
    first_xyz = arm_states[0, 0:3]
    first_rpy = arm_states[0, 3:6]
    first_rotm = euler2rotm(first_rpy)
    first_gripper = gripper_states[0]
    rel_states[0, -1] = first_gripper

    # Compute the relative state for each timestep
    for i in range(1, seq_len):
        curr_xyz = arm_states[i, 0:3]
        curr_rpy = arm_states[i, 3:6]
        curr_rotm = euler2rotm(curr_rpy)
        curr_gripper = gripper_states[i]
        # Compute the relative rotation matrix
        rel_rotm = first_rotm.T @ curr_rotm
        # Convert the relative rotation matrix to Euler angles
        rel_rpy = rotm2euler(rel_rotm)
        # Compute the relative position
        rel_xyz = first_rotm.T @ (curr_xyz - first_xyz)
        # Store the computed results
        rel_states[i, 0:3] = rel_xyz
        rel_states[i, 3:6] = rel_rpy
        rel_states[i, 6] = curr_gripper

    return rel_states


def make_env(dataset_path):
    from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    env = get_env(dataset_path, show_gui=False)

    return env


def get_gripper_camera_view_matrix(cam):
    import pybullet as pb

    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid,
        linkIndex=cam.gripper_cam_link,
        physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = pb.computeViewMatrix(
        camera_pos, camera_pos + cam_rot_y, -cam_rot_z
    )
    return view_matrix


def process_pcd(episode, abs_datasets_dir):
    """Process episode data to generate RGB point cloud sequences.
    
    Args:
        episode: Dict containing sequential sensor data
        abs_datasets_dir: Path for initializing simulation environment
        
    Returns:
        seq_pcd_obs: Dict with shape {
            'static_pcds': np.ndarray (seq_len, 200, 200, 6),
            'gripper_pcds': np.ndarray (seq_len, 84, 84, 6)
        }
    """
    # Initialize calvin environment
    env = make_env(abs_datasets_dir)
    
    # Validate input shapes
    seq_len = episode['depth_static'].shape[0]
    assert episode['rgb_static'].shape[0] == seq_len, "RGB static sequence length mismatch"
    assert episode['depth_gripper'].shape[0] == seq_len, "Depth gripper sequence length mismatch"
    assert episode['rgb_gripper'].shape[0] == seq_len, "RGB gripper sequence length mismatch"
    assert episode['robot_obs'].shape[0] == seq_len, "Robot obs sequence length mismatch"
    assert episode['scene_obs'].shape[0] == seq_len, "Scene obs sequence length mismatch"

    # Initialize output containers
    seq_pcd_obs = {
        'static_pcds': np.zeros((seq_len, 200, 200, 6), dtype=np.float32),
        'gripper_pcds': np.zeros((seq_len, 84, 84, 6), dtype=np.float32)
    }
    
    for t in range(seq_len):
        # Get current timestep data
        robot_obs = episode['robot_obs'][t]        # (15,)
        scene_obs = episode['scene_obs'][t]        # (24,)
        depth_static = episode['depth_static'][t]  # (200, 200)
        rgb_static = episode['rgb_static'][t]      # (200, 200, 3)
        depth_gripper = episode['depth_gripper'][t] # (84, 84)
        rgb_gripper = episode['rgb_gripper'][t]     # (84, 84, 3)

        # Reset environment state
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        static_cam = env.cameras[0]
        gripper_cam = env.cameras[1]
        # Update gripper camera matrix
        gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam) 

        # Process static camera
        static_pcd = deproject(
            static_cam, 
            depth_static,
            homogeneous=False,
            sanity_check=False  # Disable sanity check for performance
        ).transpose(1, 0)
        static_pcd = np.reshape(
            static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
        )
        # Combine with RGB (normalize to [0,1]) #TODO: check if this is correct 
        seq_pcd_obs['static_pcds'][t] = np.concatenate([
            static_pcd,
            rgb_static.astype(np.float32)/255.0  # Assume uint8 input
        ], axis=-1)

        # Process gripper camera
        gripper_pcd = deproject(
            gripper_cam,
            depth_gripper,
            homogeneous=False,
            sanity_check=False
        ).transpose(1, 0)
        gripper_pcd = np.reshape(
            gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
        )
        
        seq_pcd_obs['gripper_pcds'][t] = np.concatenate([
            gripper_pcd,
            rgb_gripper.astype(np.float32)/255.0
        ], axis=-1)

    # Final validation before return
    assert not np.isnan(seq_pcd_obs['static_pcds']).any(), "NaN values in static pcds"
    assert not np.isnan(seq_pcd_obs['gripper_pcds']).any(), "NaN values in gripper pcds"
    
    return seq_pcd_obs


def vis_pcd(pcd, save_dir):
    """
    Save point cloud to a specified directory with filenames numbered sequentially.

    Args:
        pcd (numpy.ndarray): Point cloud data with shape (N, 3) or (N, 6).
        save_dir (str): Directory path to save the point cloud.

    Raises:
        ValueError: If the point cloud shape is not (N, 3) or (N, 6).
    """
    import open3d as o3d
    if pcd.shape[1] not in [3, 6]:
        raise ValueError(f"Invalid point cloud shape {pcd.shape}. Expected (N, 3) or (N, 6).")

    # Convert to Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])

    if pcd.shape[1] == 6:  # If color information is included
        point_cloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6] / 255.0)

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Determine the next file number
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(".ply")]
    next_index = len(existing_files)

    # Generate the filename
    file_path = os.path.join(save_dir, f"point_cloud_{next_index}.ply")

    # Save the point cloud
    o3d.io.write_point_cloud(file_path, point_cloud)
    print(f"Point cloud saved to {file_path}")


def rand_sample_pcd(episode, n_points):
    """
    Process point cloud data with random sampling and convert to tensors.
    
    Args:
        episode: Dictionary containing numpy arrays
        n_points: Target number of points to sample per frame
        
    Returns:
        Dictionary containing processed torch tensors with same keys
    """
    processed = {}
    
    # Process static point cloud
    static_pcd = episode["static_pcd"] # (seq_len, 200, 200, 3)
    # Reshape to (seq_len, H*W, 3)
    seq_len = static_pcd.shape[0]
    static_pcd = static_pcd.reshape(seq_len, -1, 3)  # (seq_len, n_pts, 3)
    # vis original pcd
    # vis_pcd(static_pcd[2], "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/original")
    
    # Random sampling if needed
    if static_pcd.shape[1] > n_points:
        random_indices = np.random.choice(
            static_pcd.shape[1],  # Total available points
            n_points,            # Target points
            replace=False        # Ensure unique sampling
        )
        static_pcd = static_pcd[:, random_indices]
    # vis downsampled pcd
    # vis_pcd(static_pcd[2], "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/downsampled")
    processed["static_pcd"] = torch.from_numpy(static_pcd)
    
    # Process gripper point cloud (same logic)
    gripper_pcd = episode["gripper_pcd"].reshape(seq_len, -1, 3)
    if gripper_pcd.shape[1] > n_points:
        random_indices = np.random.choice(
            gripper_pcd.shape[1],
            n_points,
            replace=False
        )
        gripper_pcd = gripper_pcd[:, random_indices]
    processed["gripper_pcd"] = torch.from_numpy(gripper_pcd)
    
    # Convert camera matrices to tensors
    processed["static_cam_ex_mat"] = torch.from_numpy(episode["static_cam_ex_mat"])
    processed["gripper_cam_ex_mat"] = torch.from_numpy(episode["gripper_cam_ex_mat"])
    
    return processed


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    """
    Normalize a single point cloud to fit in [-1, 1] along its longest radius,
    safely handling empty input.
    Args:
        pc: np.ndarray of shape (N, 3)
    Returns:
        np.ndarray of shape (N, 3), centered at zero and scaled so that
        max ||pc[i]|| == 1; if input is empty, returns empty array.
    """
    # 1) empty guard
    if pc.shape[0] == 0:
        return pc.astype(np.float32)
    # 2) center
    pc = pc - np.mean(pc, axis=0)
    # 3) compute max radius
    max_r = np.max(np.linalg.norm(pc, axis=1))
    if max_r < 1e-6:
        return np.zeros_like(pc, dtype=np.float32)
    # 4) scale
    return (pc / max_r).astype(np.float32)


def _filter_and_sample_batch(pcd_batch, n_samples, max_dist):
        """
        Args:
            pcd_batch: np.ndarray of shape (seq_len, N, 3)  (float32)
            n_samples: int, number of points to sample per frame
            max_dist: float, threshold distance from centroid

        Returns:
            sampled_batch: np.ndarray of shape (seq_len, n_samples, 3)
        """
        seq_len, N, _ = pcd_batch.shape

        # 1. Compute centroids of all frames at once, shape = (seq_len, 3)
        #    centroid[i] = mean of pcd_batch[i,:,:]
        centroids = np.mean(pcd_batch, axis=1)  # (seq_len, 3)

        # 2. Compute distances from each point to its frame centroid, shape = (seq_len, N)
        #    We broadcast centroids[:, None, :] to subtract from pcd_batch
        diffs = pcd_batch - centroids[:, None, :]        # (seq_len, N, 3)
        dists = np.linalg.norm(diffs, axis=2)            # (seq_len, N)

        # 3. Create a container for the sampled points
        sampled = np.zeros((seq_len, n_samples, 3), dtype=np.float32)

        # 4. For each frame, apply mask + random sampling
        for i in range(seq_len):
            # 4.1 Obtain boolean mask of points within max_dist
            mask = (dists[i] < max_dist)                  # (N,)
            valid_pts = pcd_batch[i][mask]              # shape = (M, 3), M <= N
            # vis pcd after filtering
            # vis_pcd(valid_pts, "/home/bytedance/zhengshen_ws/falcon/pcd_vis/filtered")
            # 4.2 Normalize valid points
            valid_pts = normalize_pc(valid_pts)
            M = valid_pts.shape[0]
            if M >= n_samples:
                # 4.3a If there are at least n_samples valid points, sample without replacement
                idx = np.random.choice(M, n_samples, replace=False)
                sampled[i] = valid_pts[idx]
            elif M > 0:
                # 4.3b: 0 < M < n_samples -> preserve all then sample extra
                sampled[i, :M] = valid_pts
                extra = n_samples - M
                # sample 'extra' indices with replacement from [0, M-1]
                extra_idx = np.random.choice(M, extra, replace=True)
                sampled[i, M:] = valid_pts[extra_idx]
            else:
                # 4.3c If zero valid points, leave this frame as zeros
                sampled[i] = np.zeros((n_samples, 3), dtype=np.float32)

        return sampled  # shape = (seq_len, n_samples, 3)


def rand_sample_pcd_filter(episode, n_points):
    """
    Process point cloud data with centroid-based noise removal and random sampling, then convert to tensors.

    Args:
        episode: Dictionary containing numpy arrays
                 - "static_pcd": shape (seq_len, H, W, 3)  (float32)
                 - "gripper_pcd": shape (seq_len, H2, W2, 3) (float32)
                 - "static_cam_ex_mat": shape (...)
                 - "gripper_cam_ex_mat": shape (...)
        n_points: Target number of points to sample per frame (int)

    Returns:
        processed: Dictionary containing processed torch tensors with same keys:
                   - "static_pcd": shape (seq_len, n_points, 3)
                   - "gripper_pcd": shape (seq_len, n_points, 3)
                   - "static_cam_ex_mat": torch tensor
                   - "gripper_cam_ex_mat": torch tensor
    """
    processed = {}

    # ---------------------------------------------
    # Global configuration: maximum distance from centroid (meters)
    # Points farther than this from their frame centroid are considered noise.
    # ---------------------------------------------
    # TODO: may need a separate factor for gripper pcd and put them all in the configs
    MAX_DISTANCE = 1.0

    static_pcd = episode["static_pcd"]  # (seq_len, H, W, 3)
    seq_len = static_pcd.shape[0]
    static_pcd = static_pcd.reshape(seq_len, -1, 3)  # (seq_len, N, 3)
    # vis pcd
    # vis_pcd(static_pcd[2], "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/original")
    # Apply filtering + random sampling in batch
    static_filtered = _filter_and_sample_batch(static_pcd, n_points, MAX_DISTANCE)
    # vis_pcd(static_filtered[2], "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/downsampled")
    # Convert to torch.Tensor
    processed["static_pcd"] = torch.from_numpy(static_filtered)  # (seq_len, n_points, 3)

    # ----------------------------
    # 2. Process gripper_pcd (same logic)
    # ----------------------------
    gripper_pcd = episode["gripper_pcd"]  # (seq_len, H2, W2, 3)
    gripper_pcd = gripper_pcd.reshape(seq_len, -1, 3)      # (seq_len, N2, 3)
    gripper_filtered = _filter_and_sample_batch(gripper_pcd, n_points, MAX_DISTANCE)
    processed["gripper_pcd"] = torch.from_numpy(gripper_filtered)  # (seq_len, n_points, 3)

    # ----------------------------
    # 3. Pass through other matrices unchanged
    # ----------------------------
    processed["static_cam_ex_mat"] = torch.from_numpy(episode["static_cam_ex_mat"])
    processed["gripper_cam_ex_mat"] = torch.from_numpy(episode["gripper_cam_ex_mat"])

    return processed


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.03):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array
        Return:
          BxNx3 array, jittered
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    noise = np.clip(sigma * np.random.randn(B, N, C), -clip, clip)
    return batch_data + noise


def random_point_dropout(batch_pc, max_dropout_ratio=0.3):
    """ Randomly drop points in each pcd by replacing them with the first point.
        Input:
          BxNx3 array
        Return:
          BxNx3 array, dropped out
    """
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def save_tensor_images(img_tensor, output_dir):
    """
    专门处理形状为 (seq_len, C, H, W) 的 torch.Tensor
    
    参数:
    img_tensor: 形状为 (seq_len, C, H, W) 的 torch.Tensor
    output_dir: 输出目录路径
    """
    # 确保输入是 torch.Tensor
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError(f"输入应为 torch.Tensor，实际类型为 {type(img_tensor)}")
    
    # 检查张量维度
    if len(img_tensor.shape) != 4:
        raise ValueError(f"输入张量应有4个维度 (seq_len, C, H, W)，实际维度为 {len(img_tensor.shape)}")
    
    # 分离维度信息
    seq_len, C, H, W = img_tensor.shape
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 将张量移动到CPU并转换为numpy
    tensor_cpu = img_tensor.detach().cpu()
    
    # 转换通道顺序: (seq_len, C, H, W) -> (seq_len, H, W, C)
    tensor_permuted = tensor_cpu.permute(0, 2, 3, 1).numpy()
    
    # 保存每一帧图像
    for i in range(seq_len):
        frame = tensor_permuted[i]
        
        # 检查并处理单通道图像
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)  # 转换为3通道灰度图
        
        # 处理浮点数输入
        if np.issubdtype(frame.dtype, np.floating):
            # 归一化到0-255范围
            if frame.min() < 0 or frame.max() > 1.0:
                # 自动归一化到0-1范围
                frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = (frame * 255).astype(np.uint8)
        elif np.issubdtype(frame.dtype, np.integer):
            # 处理整数输入
            frame = frame.astype(np.uint8)
        else:
            # 未知类型，尝试直接转换
            frame = frame.astype(np.uint8)
        
        # 创建并保存图像
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
    
    print(f"成功保存 {seq_len} 张图像到 {output_dir}")


def crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng=None, info=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    import PIL
    import falcon.model.policy_head.esm_utils.vggt.datasets.utils.cropping as cropping
    aug_focal = False
    aug_crop = 0
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    assert min_margin_x > W/5, f'Bad principal point in view={info}'
    assert min_margin_y > H/5, f'Bad principal point in view={info}'
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # transpose the resolution if necessary
    W, H = image.size  # new size
    assert resolution[0] >= resolution[1]
    if H > 1.1*W:
        # image is portrait mode
        # resolution = resolution[::-1]
        pass
        
    elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        # image is square, so we chose (portrait, landscape) randomly
        if rng.integers(2):
            # resolution = resolution[::-1]
            pass

    # high-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    # augmentation
    if aug_focal:
        crop_scale = aug_focal + (1.0 - aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
        image, depthmap, intrinsics = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale)

    if aug_crop > 1:
        target_resolution += rng.integers(0, aug_crop)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution) # slightly scale the image a bit larger than the target resolution

    # actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2


def process_sequence_data(
    np_rgb_vggt: np.ndarray, 
    np_depth_vggt: np.ndarray, 
    static_cam_intr_mat: np.ndarray,
    target_resolution: Tuple[int, int] = (518, 378)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process RGB images, depth maps, and camera intrinsic matrices in sequence data
    
    Args:
        np_rgb_vggt: RGB image sequence with shape (seq_len, H, W, 3), dtype uint8
        np_depth_vggt: Depth map sequence with shape (seq_len, H, W), dtype float32
        static_cam_intr_mat: Camera intrinsic matrix sequence with shape (seq_len, 3, 3)
        target_resolution: Target resolution (W, H), default is (518, 378)
    
    Returns:
        Processed RGB images, depth maps, and camera intrinsic matrices as torch tensors
    """
    # Validate input data shapes and types
    seq_len = np_rgb_vggt.shape[0]
    assert static_cam_intr_mat.shape == (seq_len, 3, 3), f"Camera intrinsics should have shape (seq_len, 3, 3), got {static_cam_intr_mat.shape}"
    
    # Initialize output tensors
    processed_rgb = torch.zeros((seq_len, 3, target_resolution[1], target_resolution[0]))
    processed_depth = torch.zeros((seq_len, target_resolution[1], target_resolution[0], 1))
    processed_intrinsics = torch.zeros((seq_len, 3, 3))
    point_masks = torch.ones(
        size=(seq_len, target_resolution[1], target_resolution[0]),  # (seq_len, H, W)
        dtype=torch.bool
    )
    
    # Process each frame
    for i in range(seq_len):
        # Get current frame data
        rgb_frame = np_rgb_vggt[i]
        depth_frame = np_depth_vggt[i]
        intrinsics_frame = static_cam_intr_mat[i].copy()  # Create a copy to avoid modifying original data
        
        pil_image = Image.fromarray(rgb_frame)
        # Convert to RGB
        pil_image = pil_image.convert("RGB") # (W, H)
        
        # Process with crop_resize_if_necessary function
        processed_pil, processed_depth_frame, processed_intrinsics_frame = crop_resize_if_necessary(
            image=pil_image,
            depthmap=depth_frame,
            intrinsics=intrinsics_frame,
            resolution=target_resolution,
            rng=None,  # No random augmentation
            info=f"frame_{i}"  # Frame info for error reporting
        )
        
        # Convert processed data to torch tensors
        # Convert PIL Image to numpy array and then to torch tensor
        rgb_numpy = np.array(processed_pil)  # Shape: (H, W, 3)
        rgb_tensor = torch.from_numpy(rgb_numpy.transpose(2, 0, 1))
        # Convert depth map to torch tensor
        depth_tensor = torch.from_numpy(processed_depth_frame).unsqueeze(-1)
        # Convert intrinsics to torch tensor
        intrinsics_tensor = torch.from_numpy(processed_intrinsics_frame)
        
        # Store in output tensors
        processed_rgb[i] = rgb_tensor # (3, 224, 224)
        processed_depth[i] = depth_tensor # (224, 224, 1)
        processed_intrinsics[i] = intrinsics_tensor # (3, 3)
    
    return processed_rgb, processed_depth, processed_intrinsics, point_masks
