import os

ckpt_paths = [
    (
        "path/to/VLA-Checkpoint-{epoch}-{steps}.ckpt/or/VLA-Checkpoint.pt",
        "path/to/VLA-Checkpoint-config.json",
        "path/to/SimplerEnv-eval-logs"
    )    
]

for i, (ckpt, config, log_path) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.makedirs(log_path, exist_ok=True)
    os.system("bash scripts/bridge.bash {} {} > {}".format(ckpt, config, f"{log_path}/bridge.log"))
