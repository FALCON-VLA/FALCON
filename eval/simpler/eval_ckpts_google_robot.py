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
    os.system("bash scripts/openvla_pick_coke_can_visual_matching.sh {} {} > {}".format(ckpt, config, f"{log_path}/openvla_pick_coke_can_visual_matching.log"))
    os.system("bash scripts/openvla_move_near_visual_matching.sh {} {} > {}".format(ckpt, config, f"{log_path}/openvla_move_near_visual_matching.log"))
    os.system("bash scripts/openvla_put_in_drawer_visual_matching.sh {} {} > {}".format(ckpt, config, f"{log_path}/openvla_put_in_drawer_visual_matching.log"))
    os.system("bash scripts/openvla_drawer_visual_matching.sh {} {} > {}".format(ckpt, config, f"{log_path}/openvla_drawer_visual_matching.log"))
