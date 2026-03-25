import os

ckpt_paths = [
    (
        "path/to/VLA-Checkpoint-{epoch}-{steps}.ckpt/or/VLA-Checkpoint.pt",
        "path/to/VLA-Checkpoint-config.json",
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/run_eval_raw_ddp_torchrun.sh {} {}".format(ckpt, config))
