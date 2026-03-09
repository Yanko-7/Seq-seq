from concurrent.futures import ThreadPoolExecutor
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
from typing import List, Union, Iterable

sys.path.insert(0, "..")
from src.tokenizer import (
    Tokenizer,
    ORTHOGONAL_ROTATIONS,
    _bfs_reorder_faces,
    normalize_points_with_bbox,
    bbox_to_tokens,
    edge_id_to_token,
    EdgeRegistry,
    BRepTokenType,
    SPATIAL_RESOLUTION,
    EDGE_VQ_TOKEN_OFFSET,
    FACE_VQ_TOKEN_OFFSET,
)

VAE_MAX_BATCH = 6400


def build_input_ids(
    face_bboxes,
    face_vq_indices,
    edge_bboxes,
    edge_vq_indices,
    outer_edge_indices,
    face_outer_offsets,
    inner_edge_indices,
    inner_loop_offsets,
    face_inner_offsets,
    max_length=-1,
) -> List[int]:
    input_ids: List[int] = []
    registry = EdgeRegistry()
    num_faces = len(face_bboxes)

    def emit(token_id: Union[int, BRepTokenType]):
        input_ids.append(int(token_id))

    def emit_bbox(bbox: np.ndarray):
        for tok in bbox_to_tokens(bbox, SPATIAL_RESOLUTION):
            emit(tok)

    def emit_loop(edge_ids: Iterable[int]):
        if len(edge_ids) == 0:
            return
        emit(BRepTokenType.LOOP_START)
        for eid in edge_ids:
            is_new, tid = registry.register(int(eid))
            if is_new:
                emit(BRepTokenType.EDGE_NEW)
                emit(edge_id_to_token(tid))
                emit_bbox(edge_bboxes[eid])
                for vq_idx in edge_vq_indices[eid]:
                    emit(vq_idx + EDGE_VQ_TOKEN_OFFSET)
            else:
                emit(BRepTokenType.EDGE_REF)
                emit(edge_id_to_token(tid))
        emit(BRepTokenType.LOOP_END)

    emit(BRepTokenType.BOS)
    for fi in range(num_faces):
        emit(BRepTokenType.FACE_START)
        emit_bbox(face_bboxes[fi])
        for vq_idx in face_vq_indices[fi]:
            emit(vq_idx + FACE_VQ_TOKEN_OFFSET)

        outer = outer_edge_indices[face_outer_offsets[fi] : face_outer_offsets[fi + 1]]
        emit_loop(outer)

        for j in range(face_inner_offsets[fi], face_inner_offsets[fi + 1]):
            inner = inner_edge_indices[
                inner_loop_offsets[j] : inner_loop_offsets[j + 1]
            ]
            emit_loop(inner)

        emit(BRepTokenType.FACE_END)
    emit(BRepTokenType.EOS)

    if max_length > 0:
        input_ids = input_ids[:max_length]
    return input_ids


# ---------------------------------------------------------
# 2. Dataset: CPU 负责繁重的数据增强、BFS和 Normalize
# ---------------------------------------------------------
class BatchedCADDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with np.load(file_path) as npz_data:
                face_points = npz_data["face_points"]
                edge_points = npz_data["edge_points"]
                outer_edge_indices = npz_data["outer_edge_indices"]
                face_outer_offsets = npz_data["face_outer_offsets"]
                inner_edge_indices = npz_data["inner_edge_indices"]
                inner_loop_offsets = npz_data["inner_loop_offsets"]
                face_inner_offsets = npz_data["face_inner_offsets"]

            if len(face_points) == 0 or len(edge_points) == 0:
                return file_path, None

            variants = []
            for rot_idx in range(24):
                R_total = ORTHOGONAL_ROTATIONS[rot_idx]
                aug_face_points = face_points @ R_total.T
                aug_edge_points = edge_points @ R_total.T

                max_face = (
                    np.max(np.abs(aug_face_points)) if face_points.size > 0 else 0
                )
                max_edge = (
                    np.max(np.abs(aug_edge_points)) if edge_points.size > 0 else 0
                )
                current_max = max(max_face, max_edge)
                if current_max > 1e-8:
                    final_scale = 1.0 / current_max
                    aug_face_points *= final_scale
                    aug_edge_points *= final_scale

                # BFS 排序
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

                # CPU 上提前做好 normalize 和 bbox 提取！
                face_bboxes, face_norms = [], []
                for pts in aug_face_points:
                    norm_pts, bbox = normalize_points_with_bbox(pts)
                    face_norms.append(norm_pts)
                    face_bboxes.append(bbox)

                edge_bboxes, edge_norms = [], []
                for pts in aug_edge_points:
                    norm_pts, bbox = normalize_points_with_bbox(pts)
                    edge_norms.append(norm_pts)
                    edge_bboxes.append(bbox)

                variants.append(
                    {
                        "face_norms": np.stack(face_norms)
                        if face_norms
                        else np.empty((0, face_points.shape[1], 3)),
                        "face_bboxes": face_bboxes,
                        "edge_norms": np.stack(edge_norms)
                        if edge_norms
                        else np.empty((0, edge_points.shape[1], 3)),
                        "edge_bboxes": edge_bboxes,
                        "outer_edge_indices": new_outer_edge_indices,
                        "face_outer_offsets": new_face_outer_offsets,
                        "inner_edge_indices": new_inner_edge_indices,
                        "inner_loop_offsets": new_inner_loop_offsets,
                        "face_inner_offsets": new_face_inner_offsets,
                    }
                )
            return file_path, variants
        except Exception:
            return file_path, None


def collate_fn(batch):
    # 将一个 batch 内的文件打平
    valid_batch = [b for b in batch if b[1] is not None]
    return valid_batch


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


# 2. 定义一个独立的后台任务函数
def process_and_save_file(save_path, variants_subset, face_vq_subset, edge_vq_subset):
    """后台线程负责：拼接 Token 并写入磁盘"""
    file_input_ids = []
    for i in range(24):
        variant = variants_subset[i]
        input_ids = build_input_ids(
            variant["face_bboxes"],
            face_vq_subset[i],
            variant["edge_bboxes"],
            edge_vq_subset[i],
            variant["outer_edge_indices"],
            variant["face_outer_offsets"],
            variant["inner_edge_indices"],
            variant["inner_loop_offsets"],
            variant["face_inner_offsets"],
        )
        file_input_ids.append(input_ids)

    # 写盘 (这里会自动释放 Python 的 GIL 锁，不会阻塞其他代码)
    # torch.save(file_input_ids, save_path)


if __name__ == "__main__":
    PRECOMPUTE_DIR = "../data/precomputed_input_ids"
    os.makedirs(PRECOMPUTE_DIR, exist_ok=True)

    executor = ThreadPoolExecutor(max_workers=48)
    # 初始化 Tokenizer 和模型
    tokenizer = Tokenizer("../weights")
    device = "cuda"
    tokenizer.surface_fsq.to(device)
    tokenizer.edge_fsq.to(device)
    tokenizer.surface_fsq.eval()
    tokenizer.edge_fsq.eval()

    dataset = load_dataset_fast(
        json_path="dataset_split.json",
        root_dir="/cache/yanko/dataset/abc-reorder-p32/organized/",
    )
    inputs = dataset["train"]  # 你的文件路径列表
    inputs = inputs  # 替换成你加载的真实列表

    dataset = BatchedCADDataset(inputs)
    # batch_size=16 意味着一次 DataLoader 吐出 16 个文件 * 24 个旋转 = 384 个模型！
    dataloader = DataLoader(
        dataset, batch_size=8, num_workers=64, collate_fn=collate_fn
    )

    print("Starting ultra-fast batched precomputation on GPU...")
    for batch_data in tqdm(dataloader):
        if not batch_data:
            continue

        all_variants = []
        file_paths = []
        for file_path, variants in batch_data:
            file_paths.append(file_path)
            all_variants.extend(variants)  # 打平成巨大的列表

        # 把这 384 个模型的所有 face 和 edge 拼成一个巨大的 Tensor
        flat_face_norms = [
            v["face_norms"] for v in all_variants if len(v["face_norms"]) > 0
        ]
        flat_edge_norms = [
            v["edge_norms"] for v in all_variants if len(v["edge_norms"]) > 0
        ]

        with torch.no_grad():
            if flat_face_norms:
                giant_face_tensor = torch.tensor(
                    np.concatenate(flat_face_norms), dtype=torch.float32, device=device
                )
                giant_face_vq = np.concatenate(
                    [
                        tokenizer.encode_surface(chunk).cpu().numpy()
                        for chunk in torch.split(giant_face_tensor, VAE_MAX_BATCH)
                    ],
                    axis=0,
                )
            else:
                giant_face_vq = np.empty((0, 4))

            if flat_edge_norms:
                giant_edge_tensor = torch.tensor(
                    np.concatenate(flat_edge_norms), dtype=torch.float32, device=device
                )
                # 核心改动：同理处理 Edge
                giant_edge_vq = np.concatenate(
                    [
                        tokenizer.encode_edge(chunk).cpu().numpy()
                        for chunk in torch.split(giant_edge_tensor, VAE_MAX_BATCH)
                    ],
                    axis=0,
                )
            else:
                giant_edge_vq = np.empty((0, 2))
            # ---------------------------------------------------

        face_idx = 0
        edge_idx = 0
        variant_idx = 0

        for file_path in file_paths:
            file_stem = Path(file_path).stem
            save_path = os.path.join(PRECOMPUTE_DIR, f"{file_stem}.pt")

            # 如果存在则跳过 (强烈建议把这步提前到 Dataset 初始化前做，这里仅保留安全检查)
            if os.path.exists(save_path):
                for _ in range(24):
                    face_idx += len(all_variants[variant_idx]["face_norms"])
                    edge_idx += len(all_variants[variant_idx]["edge_norms"])
                    variant_idx += 1
                continue

            # 提前为这个文件准备好它专属的 24 份数据切片
            variants_subset = []
            face_vq_subset = []
            edge_vq_subset = []

            for _ in range(24):
                variant = all_variants[variant_idx]
                num_faces = len(variant["face_norms"])
                num_edges = len(variant["edge_norms"])

                variants_subset.append(variant)
                # Numpy 的切片是视图 (View)，瞬间完成，不占额外内存
                face_vq_subset.append(giant_face_vq[face_idx : face_idx + num_faces])
                edge_vq_subset.append(giant_edge_vq[edge_idx : edge_idx + num_edges])

                face_idx += num_faces
                edge_idx += num_edges
                variant_idx += 1

            # 主线程工作结束，把切好的物料全部扔给后台线程！
            executor.submit(
                process_and_save_file,
                save_path,
                variants_subset,
                face_vq_subset,
                edge_vq_subset,
            )
    print("Waiting for all background writing tasks to finish...")
    executor.shutdown(wait=True)
