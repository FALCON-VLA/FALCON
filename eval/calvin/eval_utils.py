from collections import Counter
import json
import math
import copy
import numpy as np
from pathlib import Path
import os
import torch

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def init_trainer_config(configs):
    # TODO: currently for other strategy we directly use the default settings.
    trainer_config = copy.deepcopy(configs["trainer"])
    trainer_config["devices"] = configs.get("gpus", "auto")
    trainer_config["num_nodes"] = configs.get("num_nodes", 1)
    trainer_config["gradient_clip_val"] = configs.get("gradient_clip_val", 0.0)
    exp_name = configs.get("exp_name", "default")

    if "strategy" not in trainer_config or trainer_config["strategy"] == "ddp":
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)

    # init loggers
    loggers = None
    log_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["log_dir"] = configs["log_root"] / log_dir
    if isinstance(trainer_config.get("logger"), list):
        loggers = []
        for logger in trainer_config.get("logger"):
            if logger == "tensorboard":
                loggers.append(
                    TensorBoardLogger(configs["log_dir"].as_posix(), name=exp_name)
                )
            elif logger == "csv":
                loggers.append(CSVLogger(configs["log_dir"].as_posix(), name=exp_name))
            else:
                raise NotImplementedError

    trainer_config["logger"] = loggers

    ckpt_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["output_dir"] = configs["output_root"] / ckpt_dir

    configs["log_dir"].mkdir(parents=True, exist_ok=True)
    configs["output_dir"].mkdir(parents=True, exist_ok=True)
    configs["cache_root"].mkdir(parents=True, exist_ok=True)
    # os.system(f"sudo chmod 777 -R runs/")

    configs["log_dir"] = configs["log_dir"].as_posix()
    configs["output_dir"] = configs["output_dir"].as_posix()
    configs.pop("output_root")
    configs.pop("log_root")
    configs["cache_root"] = configs["cache_root"].as_posix()

    trainer_config["callbacks"] = [
        init_setup_callback(configs),
        init_lr_monitor_callback(),
        ModelCheckpoint(dirpath=configs["output_dir"], save_top_k=-1, every_n_epochs=1),
    ]

    return trainer_config


def count_success(results):
    try:
        count = Counter(results)
        step_success = []
        for i in range(1, 6):
            n_success = sum(count[j] for j in reversed(range(i, 6)))
            sr = n_success / len(results)
            step_success.append(sr)
    except:
        import pdb

        pdb.set_trace()
    return step_success


def print_and_save(results, sequences, eval_result_path, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(
            f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%"
        )

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
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
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # (-pi , pi]
    while x > np.pi:
        x -= 2 * np.pi
    while x <= -np.pi:
        x += 2 * np.pi
    while y > np.pi:
        y -= 2 * np.pi
    while y <= -np.pi:
        y += 2 * np.pi
    while z > np.pi:
        z -= 2 * np.pi
    while z <= -np.pi:
        z += 2 * np.pi
    return np.array([x, y, z])


def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a pixel point to 3D coordinates
    Args
        point: tuple (u, v); pixel coordinates of point to deproject
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output
        (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix).reshape((4, 4)).T)
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.
    if sanity_check:
        sample_inds = np.random.permutation(u.shape[0])[:2000]
        for ind in sample_inds:
            cam_world_pos = cam.deproject((u[ind], v[ind]), depth_img, homogeneous=True)
            assert np.abs(cam_world_pos-world_pos[:, ind]).max() <= 1e-3

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos


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


def eval_filtered_sample_pcd(pcd: np.ndarray, n_points: int, max_dist: float) -> torch.Tensor:
    """
    Process a single-frame point cloud with centroid-based noise removal and random sampling,
    then convert to a torch tensor of shape (n_points, 3).

    Args:
        pcd: np.ndarray of shape (H, W, 3) or (N, 3); float32 coordinates in world frame
        n_points: int, target number of points to sample (must be > 0)
        max_dist: float, threshold distance from the frame centroid (in same units as pcd)

    Returns:
        torch.Tensor of shape (n_points, 3), dtype=torch.float32.
        - Points farther from the centroid than max_dist are removed.
        - If M = number of remaining points:
            • If M >= n_points, sample without replacement.
            • If 0 < M < n_points, preserve all M and randomly duplicate to fill to n_points.
            • If M == 0, return a zero tensor of shape (n_points, 3).
    """
    assert isinstance(pcd, np.ndarray), f"Expected numpy array, got {type(pcd)}"
    assert n_points > 0, f"n_points must be positive, got {n_points}"
    # Flatten to (N, 3) if necessary
    pcd_flat = pcd.reshape(-1, 3)  # shape = (N, 3)
    # vis original pcd
    # vis_pcd(pcd_flat, "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/eval/dis_0.8/original")
    
    # If the flat array is empty, return zeros immediately
    if pcd_flat.shape[0] == 0:
        return torch.zeros((n_points, 3), dtype=torch.float32)
    
    # 1. Compute centroid of the entire point set
    centroid = np.mean(pcd_flat, axis=0)  # shape = (3,)
    # 2. Compute distances from each point to centroid
    diffs = pcd_flat - centroid[None, :]      # shape = (N, 3)
    dists = np.linalg.norm(diffs, axis=1)     # shape = (N,)
    # 3. Filter out points with distance >= max_dist
    mask = (dists < max_dist)                 # shape = (N,)
    valid_pts = pcd_flat[mask]                # shape = (M, 3), M <= N
    # vis filtered pcd
    # vis_pcd(valid_pts, "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/eval/dis_0.8/filtered")
    # 4. Normalize valid points
    valid_pts = normalize_pc(valid_pts)
    M = valid_pts.shape[0]
    # 5. Allocate array to hold exactly n_points samples
    sampled = np.zeros((n_points, 3), dtype=np.float32)
    
    if M >= n_points:
        # A: At least n_points valid points -> sample without replacement
        idx = np.random.choice(M, n_points, replace=False)
        sampled[:] = valid_pts[idx]
    elif M > 0:
        # B: 0 < M < n_points -> preserve all M points, then randomly duplicate to fill
        sampled[:M] = valid_pts
        extra = n_points - M
        extra_idx = np.random.choice(M, extra, replace=True)
        sampled[M:] = valid_pts[extra_idx]
    else:
        # C: M == 0 -> no valid points, leave sampled as all zeros
        # sampled already initialized to zeros
        pass
    # vis downsampled pcd
    # vis_pcd(sampled, "/mnt/bn/robotics-zzs-lq2/RoboVLMs_pointvla/pcd_vis/eval/dis_0.8/downsampled")
    
    return torch.as_tensor(sampled, dtype=torch.float32)


def extract_camera_parameters(view_matrix_tuple, projection_matrix_tuple, width, height):
    """
    根据PyBullet官方deproject代码修正的相机参数提取函数
    
    参数:
        view_matrix_tuple: PyBullet返回的viewMatrix元组
        projection_matrix_tuple: PyBullet返回的projectionMatrix元组
        width: 图像宽度
        height: 图像高度
        
    返回:
        camera_to_world: 4x4相机到世界坐标系的变换矩阵
        intrinsic_matrix: 3x3相机内参矩阵
    """
    # 将viewMatrix转换为4x4矩阵并转置（根据官方代码）
    view_matrix = np.array(view_matrix_tuple).reshape(4, 4).T
    
    # 相机到世界坐标系的变换矩阵（官方代码中的T_world_cam）
    camera_to_world = np.linalg.inv(view_matrix)
    
    # 将projectionMatrix转换为4x4矩阵
    projection_matrix = np.array(projection_matrix_tuple).reshape(4, 4)
    
    # 从投影矩阵提取内参
    # 根据官方代码，焦距计算方式为：foc = height / (2 * tan(fov/2))
    # 但我们可以从投影矩阵中提取更精确的值
    fx = projection_matrix[0, 0] * width / 2.0
    fy = projection_matrix[1, 1] * height / 2.0
    cx = width / 2.0
    cy = height / 2.0
    
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return camera_to_world, intrinsic_matrix