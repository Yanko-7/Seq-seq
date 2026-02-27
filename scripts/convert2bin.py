import json
from pathlib import Path

# import zstandard as zstd
from litdata import optimize, TokensLoader
from functools import partial
import numpy as np
import sys
import os

sys.path.insert(0, "..")

from src.tokenizer import ORTHOGONAL_ROTATIONS, Tokenizer, _bfs_reorder_faces


def tokenize_fn(file_path, tokenizer, augment=True):
    with np.load(file_path) as npz_data:
        face_points = npz_data["face_points"]
        edge_points = npz_data["edge_points"]
        edge_adjacency = npz_data["edge_adjacency"]
        outer_edge_indices = npz_data["outer_edge_indices"]
        face_outer_offsets = npz_data["face_outer_offsets"]
        inner_edge_indices = npz_data["inner_edge_indices"]
        inner_loop_offsets = npz_data["inner_loop_offsets"]
        face_inner_offsets = npz_data["face_inner_offsets"]

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
                current_max_after_scale = current_max * final_scale
                margin = 1.0 - current_max_after_scale
                margin = max(0, margin - 1e-5)
                shift = np.random.uniform(-margin, margin, size=(1, 3))
                aug_face_points += shift
                aug_edge_points += shift
            (
                aug_face_points,
                new_outer_edge_indices,
                new_face_outer_offsets,
                new_inner_edge_indices,
                new_inner_loop_offsets,
                new_face_inner_offsets,
            ) = _bfs_reorder_faces(
                aug_face_points,
                edge_adjacency,
                outer_edge_indices,
                face_outer_offsets,
                inner_edge_indices,
                inner_loop_offsets,
                face_inner_offsets,
            )
            input_ids = tokenizer.encode(
                face_points=aug_face_points,
                edge_points=aug_edge_points,
                outer_edge_indices=new_outer_edge_indices,
                face_outer_offsets=new_face_outer_offsets,
                inner_edge_indices=new_inner_edge_indices,
                inner_loop_offsets=new_inner_loop_offsets,
                face_inner_offsets=new_face_inner_offsets,
                device="cuda",
            )
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
            edge_adjacency,
            outer_edge_indices,
            face_outer_offsets,
            inner_edge_indices,
            inner_loop_offsets,
            face_inner_offsets,
        )
        input_ids = tokenizer.encode(
            face_points=face_points,
            edge_points=edge_points,
            outer_edge_indices=new_outer_edge_indices,
            face_outer_offsets=new_face_outer_offsets,
            inner_edge_indices=new_inner_edge_indices,
            inner_loop_offsets=new_inner_loop_offsets,
            face_inner_offsets=new_face_inner_offsets,
            device="cuda",
        )
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
    # 2. Generate the inputs (we are going to optimize all the compressed json files from SlimPajama dataset )
    dataset = load_dataset_fast(
        json_path="abc_filtered_final.json",
        root_dir="/cache/yanko/datasets/npz_files/organized_by_face_count/",
    )
    inputs = [str(file) for file in dataset["val"]]
    checkpoint_dir = "../weights"
    _tokenizer = Tokenizer(checkpoint_dir)
    # 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"

    outputs = optimize(
        fn=partial(tokenize_fn, tokenizer=_tokenizer, augment=True),
        inputs=inputs,
        output_dir="./abc-optimized-sep-val",
        chunk_size=2048,
        # item_loader=TokensLoader(),
        num_workers=16,
    )
