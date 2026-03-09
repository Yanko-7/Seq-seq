import json
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

# import zstandard as zstd
from litdata import optimize

sys.path.insert(0, "..")

from src.tokenizer import ORTHOGONAL_ROTATIONS, _bfs_reorder_faces

# --- 新增：用于子进程存储自己专属的 Tokenizer 和 GPU ID ---
_LOCAL_TOKENIZER = None
_LOCAL_DEVICE = "cuda"


def tokenize_fn(file_path, checkpoint_dir, augment=True):
    global _LOCAL_TOKENIZER, _LOCAL_DEVICE

    if _LOCAL_TOKENIZER is None:
        # 1. 获取 Worker ID
        process_name = multiprocessing.current_process().name
        try:
            worker_id = int(process_name.split("-")[-1])
        except ValueError:
            worker_id = os.getpid()

        # 2. 手动指定你的物理显卡数量 (你有4张卡)
        num_gpus = 3
        gpu_id = worker_id % num_gpus

        # 3. 杀手锏：在 PyTorch 初始化该进程的 CUDA 前，强制屏蔽其他显卡！
        # 这样当前进程看到的唯一一张显卡 (cuda:0) 就是物理上的 gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 因为已经被隔离，所以直接用 "cuda" 即可，它会自动映射到分配好的卡
        _LOCAL_DEVICE = "cuda"

        # 4. 现在再加载模型，它怎么硬编码都不会漏到其他卡上了
        from src.tokenizer import (
            Tokenizer,
        )  # 最好把 import 也放在函数内，避免主进程提前加载

        _LOCAL_TOKENIZER = Tokenizer(checkpoint_dir)

    tokenizer = _LOCAL_TOKENIZER

    with np.load(file_path) as npz_data:
        face_points = npz_data["face_points"]
        edge_points = npz_data["edge_points"]
        outer_edge_indices = npz_data["outer_edge_indices"]
        face_outer_offsets = npz_data["face_outer_offsets"]
        inner_edge_indices = npz_data["inner_edge_indices"]
        inner_loop_offsets = npz_data["inner_loop_offsets"]
        face_inner_offsets = npz_data["face_inner_offsets"]

    if (
        len(face_points) == 0
        or len(edge_points) == 0
        or len(outer_edge_indices) == 0
        or len(face_outer_offsets) == 0
    ):
        print(f"Warning: Empty data in file {file_path}, skipping.")
        return

    num_augments = 8

    if augment:
        # 随机抽取 num_augments 个不重复的旋转矩阵索引
        # 这样可以保证每次存下来的几份数据朝向绝对不会重复
        chosen_indices = np.random.choice(24, size=num_augments, replace=False)
        for i, rot_idx in enumerate(chosen_indices):
            R_total = ORTHOGONAL_ROTATIONS[rot_idx]
            # 矩阵乘法应用旋转
            aug_face_points = face_points @ R_total.T
            aug_edge_points = edge_points @ R_total.T
            max_face = np.max(np.abs(aug_face_points)) if face_points.size > 0 else 0
            max_edge = np.max(np.abs(aug_edge_points)) if edge_points.size > 0 else 0
            current_max = max(max_face, max_edge)
            if current_max > 1e-8:
                # random_scale = np.random.uniform(0.95, 1.0)
                random_scale = 1.0
                final_scale = (1.0 / current_max) * random_scale
                aug_face_points *= final_scale
                aug_edge_points *= final_scale

            (
                aug_face_points,
                new_outer_edge_indices,
                new_face_outer_offsets,
                new_inner_edge_indices,
                new_inner_loop_offsets,
                new_face_inner_offsets,
            ) = _bfs_reorder_faces(
                aug_face_points,
                outer_edge_indices,
                face_outer_offsets,
                inner_edge_indices,
                inner_loop_offsets,
                face_inner_offsets,
            )
            try:
                input_ids = tokenizer.encode(
                    face_points=aug_face_points,
                    edge_points=aug_edge_points,
                    outer_edge_indices=new_outer_edge_indices,
                    face_outer_offsets=new_face_outer_offsets,
                    inner_edge_indices=new_inner_edge_indices,
                    inner_loop_offsets=new_inner_loop_offsets,
                    face_inner_offsets=new_face_inner_offsets,
                    device=_LOCAL_DEVICE,  # <--- 2. 这里修改为使用分配好的设备
                )
            except Exception as e:
                print(
                    f"Error encoding file {file_path} with rotation index {rot_idx}: {e}"
                )
                continue
            yield input_ids
    else:
        (
            face_points,
            new_outer_edge_indices,
            new_face_outer_offsets,
            new_inner_edge_indices,
            new_inner_loop_offsets,
            new_face_inner_offsets,
        ) = _bfs_reorder_faces(
            face_points,
            outer_edge_indices,
            face_outer_offsets,
            inner_edge_indices,
            inner_loop_offsets,
            face_inner_offsets,
        )
        try:
            input_ids = tokenizer.encode(
                face_points=face_points,
                edge_points=edge_points,
                outer_edge_indices=new_outer_edge_indices,
                face_outer_offsets=new_face_outer_offsets,
                inner_edge_indices=new_inner_edge_indices,
                inner_loop_offsets=new_inner_loop_offsets,
                face_inner_offsets=new_face_inner_offsets,
                device=_LOCAL_DEVICE,  # <--- 2. 这里修改为使用分配好的设备
            )
        except Exception as e:
            print(f"Error encoding file {file_path}: {e}")
            return
        yield input_ids


def load_dataset_fast(json_path, root_dir, ext=".npz"):
    file_map = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(ext):
                file_map[f[:8]] = os.path.join(root, f)

    with open(json_path, "r") as f:
        data = json.load(f)

    return {
        split: [
            file_map[Path(item).stem[:8]]
            for item in items
            if Path(item).stem[:8] in file_map
        ]
        for split, items in data.items()
    }


if __name__ == "__main__":
    dataset = load_dataset_fast(
        json_path="dataset_split.json",
        root_dir="/cache/yanko/dataset/abc-reorder-p32/organized/",
    )
    inputs = [str(file) for file in dataset["val"]]
    checkpoint_dir = "../weights"

    outputs = optimize(
        # 4. 把 checkpoint_dir 传进去，让 tokenize_fn 在子进程自己去加载模型
        fn=partial(tokenize_fn, checkpoint_dir=checkpoint_dir, augment=False),
        inputs=inputs,
        output_dir="./abc-optimized-sep-val",
        chunk_size=4096 * 16,
        num_workers=4,
        keep_data_ordered=False,
    )
