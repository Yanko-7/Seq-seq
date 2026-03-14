"""
BRep Sequence Format Specification
==================================

This module implements the serialization of Boundary Representation (B-Rep) data
into a flat sequence of discrete tokens suitable for autoregressive transformers.

Sequence Structure
------------------
The sequence consists entirely of discrete tokens. Continuous geometric features
(surfaces and curves) are represented by two parts:
  1. Bounding Box (BBox): 6 discrete tokens representing spatial bounds.
  2. Geometry VQ: Quantized discrete indices from a pre-trained VQ-VAE.
This allows the entire B-Rep topology, spatial location, and local geometry to be
processed uniformly as a standard text-like sequence.

Structure Hierarchy:
-------------------------------------------------------------------------------
[BOS]
  |
  +--- [FACE_START]
  |      |
  |      +--- [BBox Token 1..6] (Face Bounding Box: min/max X, Y, Z)
  |      |
  |      +--- [Surface VQ Token 1..4]
  |      |
  |      +--- [LOOP_START] (Outer Loop)
  |      |      |
  |      |      +--- [EDGE_NEW] [BBox Token 1..6] [Curve VQ Token 1..2]
  |      |      |
  |      |      |
  |      |      +--- [EDGE_REF] [Relative Ref idx] (refers to a previously defined edge)
  |      |      |
  |      |      ...
  |      |
  |      +--- [LOOP_END]
  |      |
  |      +--- [LOOP_START] (Inner Loop 1) ... [LOOP_END]
  |      +--- [LOOP_START] (Inner Loop 2) ... [LOOP_END]
  |
  +--- [FACE_END]
  |
  ... (Next Face) ...
  |
[EOS]
-------------------------------------------------------------------------------

Position ID Logic:
------------------
Because the sequence is fully discrete, standard autoregressive position tracking
applies. Every token in the sequence (including topological markers, BBox coordinates,
and VQ-VAE geometry indices) increments the Position ID by exactly 1.

Edge ID Management:
-------------------
- Edge IDs are recycled. Once an edge is referenced (EDGE_REF), its ID token
  is freed and can be reused by a subsequent EDGE_NEW.
- This keeps the relative referencing vocabulary size small, even for complex
  models with high edge counts.

Geometry Representation:
------------------------
- Spatial BBox: 6 sequential tokens representing (X_min, Y_min, Z_min, X_max, Y_max, Z_max).
  These act as a global spatial prompt for the subsequent local geometry.
- Surface Geometry: Encoded as a fixed-length sequence of discrete tokens mapping
  to the Surface VQ-VAE codebook.
- Curve Geometry: Encoded as a fixed-length sequence of discrete tokens mapping
  to the Curve VQ-VAE codebook.
"""


# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass
from enum import IntEnum

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from collections import Counter, deque
from einops import rearrange
import torch
import numpy as np
from src.vq_vae import EdgeFSQVAE, SurfaceFSQVAE
# ============================================================================
# Spatial resolution configuration
# ============================================================================

SPATIAL_RESOLUTION = 1024
EDGE_ID_COUNT = 984

FACE_VQ_TOKEN_NUM = 4
EDGE_VQ_TOKEN_NUM = 2
BBOX_TOKEN_COUNT = 6

# ============================================================================
# Token vocabulary definition
# ============================================================================


@dataclass
class BRepData:
    face_points: np.ndarray
    edge_points: np.ndarray
    outer_edge_indices: np.ndarray
    face_outer_offsets: np.ndarray
    inner_edge_indices: np.ndarray
    inner_loop_offsets: np.ndarray
    face_inner_offsets: np.ndarray


class BRepTokenType(IntEnum):
    """BRep Token classification

    Token layout:
    - 0-31: special control tokens
    - 32-1055: coordinate tokens (1024)
    - 544-1023: Edge ID tokens (480, recycled)
    """

    # === special control tokens (0-31) ===
    PAD = 0
    BOS = 1
    EOS = 2

    # Face structure
    FACE_START = 3
    FACE_END = 4

    # BBox structure
    BBOX_START = 5
    BBOX_END = 6

    # Geometry placeholders
    SURFACE_GEOM = 7
    CURVE_GEOM = 8

    # Loop structure
    LOOP_START = 9
    LOOP_END = 10

    # Edge types
    EDGE_NEW = 11
    EDGE_REF = 12

    # Reserved (13-31)
    EDGE_REF_LAST = 13

    GEOM_START = 14
    GEOM_END = 15


# === Token range definitions ===
COORD_TOKEN_OFFSET = 32
COORD_TOKEN_MIN = COORD_TOKEN_OFFSET
COORD_TOKEN_MAX = COORD_TOKEN_OFFSET + SPATIAL_RESOLUTION - 1  # 1055

EDGE_ID_OFFSET = COORD_TOKEN_MAX + 1
EDGE_ID_MIN = EDGE_ID_OFFSET
EDGE_ID_MAX = EDGE_ID_OFFSET + EDGE_ID_COUNT - 1

FACE_VQ_TOKEN_OFFSET = EDGE_ID_MAX + 1
FACE_VQ_TOKEN_MIN = FACE_VQ_TOKEN_OFFSET
FACE_VQ_TOKEN_MAX = FACE_VQ_TOKEN_OFFSET + 1024 - 1  # Assuming 1024 face VQ tokens
EDGE_VQ_TOKEN_OFFSET = FACE_VQ_TOKEN_MAX + 1
EDGE_VQ_TOKEN_MIN = EDGE_VQ_TOKEN_OFFSET
EDGE_VQ_TOKEN_MAX = EDGE_VQ_TOKEN_OFFSET + 1024 - 1  #


BREP_VOCAB_SIZE = 4096  # 32 control + 1024 coord + 1024 face VQ + 1024 edge VQ


def coord_to_token(value: float, resolution: int = SPATIAL_RESOLUTION) -> int:
    value = np.clip(value, -1.0, 1.0)
    value_normalized = (value + 1.0) / 2.0
    quantized = int(value_normalized * (resolution - 1) + 0.5)
    quantized = min(max(quantized, 0), resolution - 1)
    return quantized + COORD_TOKEN_OFFSET


def token_to_coord(token_id: int, resolution: int = SPATIAL_RESOLUTION) -> float:
    quantized = token_id - COORD_TOKEN_OFFSET
    value_normalized = quantized / (resolution - 1)
    return value_normalized * 2.0 - 1.0


def is_coord_token(token_id: int) -> bool:
    return COORD_TOKEN_MIN <= token_id <= COORD_TOKEN_MAX


def is_face_vq_token(token_id: int) -> bool:
    return FACE_VQ_TOKEN_MIN <= token_id <= FACE_VQ_TOKEN_MAX


def is_edge_vq_token(token_id: int) -> bool:
    return EDGE_VQ_TOKEN_MIN <= token_id <= EDGE_VQ_TOKEN_MAX


def edge_id_to_token(edge_id: int) -> int:
    if edge_id < 0 or edge_id >= EDGE_ID_COUNT:
        raise ValueError(f"Edge ID {edge_id} out of range [0, {EDGE_ID_COUNT})")
    return edge_id + EDGE_ID_OFFSET


def token_to_edge_id(token_id: int) -> int:
    return token_id - EDGE_ID_OFFSET


def is_edge_id_token(token_id: int) -> bool:
    return EDGE_ID_MIN <= token_id <= EDGE_ID_MAX


def bbox_to_tokens(bbox: np.ndarray, resolution: int = SPATIAL_RESOLUTION) -> List[int]:
    # Reorder [x1, y1, z1, x2, y2, z2] -> [z1, z2, y1, y2, x1, x2]
    v = np.asarray(bbox).ravel()[[2, 5, 1, 4, 0, 3]]

    # Vectorized implementation of coord_to_token logic
    # 1. Clip -> Normalize -> Scale -> Round (+0.5 then cast)
    v = (np.clip(v, -1.0, 1.0) + 1.0) * 0.5 * (resolution - 1) + 0.5

    # 2. Clip result -> Offset
    return (np.clip(v, 0, resolution - 1).astype(int) + COORD_TOKEN_OFFSET).tolist()


def tokens_to_bbox(
    tokens: List[int], resolution: int = SPATIAL_RESOLUTION
) -> np.ndarray:
    if len(tokens) != 6:
        raise ValueError(f"Expected 6 tokens, got {len(tokens)}")

    # Decode tokens to values
    # Order is Z -> Y -> X: [z_min, z_max, y_min, y_max, x_min, x_max]
    vals = [token_to_coord(t, resolution) for t in tokens]
    z_min, z_max, y_min, y_max, x_min, x_max = vals

    # Return standard [x_min, y_min, z_min, x_max, y_max, z_max]
    return np.array(
        [x_min, y_min, z_min, x_max, y_max, z_max],
        dtype=np.float32,
    )


def sort_points_by_corners(points: np.ndarray) -> np.ndarray:
    if points.ndim == 2:
        if tuple(points[0]) > tuple(points[-1]):
            points = points[::-1]

    elif points.ndim == 3:
        ops = [
            (points[0, 0], points),
            (points[0, -1], points[:, ::-1]),
            (points[-1, 0], points[::-1, :]),
            (points[-1, -1], points[::-1, ::-1]),
        ]
        points = min(ops, key=lambda x: tuple(x[0]))[1]

        if points.shape[0] == points.shape[1] and tuple(points[0, 1]) > tuple(
            points[1, 0]
        ):
            points = points.transpose(1, 0, 2)

    return np.ascontiguousarray(points)


def normalize_points_with_bbox(points: np.ndarray):
    points = sort_points_by_corners(points)
    pts_flat = points.reshape(-1, 3)
    vmin, vmax = pts_flat.min(0), pts_flat.max(0)
    center = (vmin + vmax) / 2
    span = (vmax - vmin).max()
    scale = 2.0 / span if span > 1e-6 else 1.0
    return (points - center) * scale, np.concatenate([vmin, vmax])


def denormalize_points_with_bbox(norm_points: np.ndarray, bbox: np.ndarray):
    vmin, vmax = bbox[:3], bbox[3:]
    center = (vmin + vmax) / 2
    span = (vmax - vmin).max()
    scale = 2.0 / span if span > 1e-6 else 1.0
    return norm_points / scale + center


class EdgeRegistry:
    def __init__(self, edge_counts: Dict[int, int]):
        self._edge_counts = edge_counts
        self._seen_counts = {e: 0 for e in edge_counts}
        self._edge_to_token: Dict[int, int] = {}
        self._available: deque[int] = deque()
        self._next_id: int = 0

    def register(self, edge_id: int) -> Tuple[BRepTokenType, int]:
        self._seen_counts[edge_id] += 1
        seen = self._seen_counts[edge_id]
        total = self._edge_counts[edge_id]

        if seen == 1:
            token_id = self._available.popleft() if self._available else self._next_id
            if token_id == self._next_id:
                self._next_id += 1

            if token_id >= EDGE_ID_COUNT:
                raise ValueError(f"Concurrent live edges exceed {EDGE_ID_COUNT}")

            self._edge_to_token[edge_id] = token_id
            return BRepTokenType.EDGE_NEW, token_id

        token_id = self._edge_to_token[edge_id]
        if seen == total:
            self._available.append(token_id)
            del self._edge_to_token[edge_id]
            return BRepTokenType.EDGE_REF_LAST, token_id

        return BRepTokenType.EDGE_REF, token_id


ORTHOGONAL_ROTATIONS = np.array(
    [
        # 面向 +X (4种)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        # 面向 +Y (4种)
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
        # 面向 -X (4种)
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        # 面向 -Y (4种)
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
        # 面向 +Z (4种)
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
        # 面向 -Z (4种)
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    ],
    dtype=np.float32,
)


def build_face_adjacency(
    outer_edges, face_outer_offsets, inner_edges, inner_loop_offsets, face_inner_offsets
):
    num_faces = len(face_outer_offsets) - 1
    if num_faces <= 0:
        return []

    num_edges = (
        int(max(np.max(outer_edges, initial=-1), np.max(inner_edges, initial=-1))) + 1
    )

    edge_to_faces = [[] for _ in range(num_edges)]

    for f_id in range(num_faces):
        e_ids = list(
            outer_edges[face_outer_offsets[f_id] : face_outer_offsets[f_id + 1]]
        )

        for l_idx in range(face_inner_offsets[f_id], face_inner_offsets[f_id + 1]):
            e_ids.extend(
                inner_edges[inner_loop_offsets[l_idx] : inner_loop_offsets[l_idx + 1]]
            )

        for e_id in set(e_ids):
            edge_to_faces[e_id].append(f_id)

    adj = [set() for _ in range(num_faces)]
    for faces in edge_to_faces:
        for u, v in combinations(faces, 2):
            adj[u].add(v)
            adj[v].add(u)

    return adj


def _bfs_reorder_faces(
    face_points,
    outer_edge_indices,
    face_outer_offsets,
    inner_edge_indices,
    inner_loop_offsets,
    face_inner_offsets,
):
    num_faces = len(face_points)
    if num_faces <= 1:
        return (
            face_points,
            outer_edge_indices,
            face_outer_offsets,
            inner_edge_indices,
            inner_loop_offsets,
            face_inner_offsets,
        )

    centroids = face_points.mean(axis=(1, 2))

    adj = build_face_adjacency(
        outer_edge_indices,
        face_outer_offsets,
        inner_edge_indices,
        inner_loop_offsets,
        face_inner_offsets,
    )

    sorted_indices = np.lexsort((centroids[:, 0], centroids[:, 1], centroids[:, 2]))
    start = int(sorted_indices[0])

    visited = np.zeros(num_faces, dtype=bool)
    order = []
    queue = deque([start])
    visited[start] = True

    while queue:
        cur = queue.popleft()
        order.append(cur)

        nbs = [n for n in adj[cur] if not visited[n]]
        if nbs:
            nbs.sort(key=lambda n: tuple(centroids[n][::-1]))
            for nb in nbs:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    if len(order) < num_faces:
        remaining = [int(i) for i in sorted_indices if not visited[i]]
        order.extend(remaining)

    order = np.array(order, dtype=np.int64)
    new_face_points = face_points[order]

    new_outer_edges = []
    new_face_outer_offsets = [0]
    for i in order:
        s, e = face_outer_offsets[i], face_outer_offsets[i + 1]
        new_outer_edges.extend(outer_edge_indices[s:e])
        new_face_outer_offsets.append(len(new_outer_edges))

    new_inner_edges = []
    new_inner_loop_offsets = [0]
    new_face_inner_offsets = [0]
    for i in order:
        for j in range(face_inner_offsets[i], face_inner_offsets[i + 1]):
            s, e = inner_loop_offsets[j], inner_loop_offsets[j + 1]
            new_inner_edges.extend(inner_edge_indices[s:e])
            new_inner_loop_offsets.append(len(new_inner_edges))
        new_face_inner_offsets.append(len(new_inner_loop_offsets) - 1)

    return (
        new_face_points,
        np.array(new_outer_edges, dtype=np.int32),
        np.array(new_face_outer_offsets, dtype=np.int32),
        np.array(new_inner_edges, dtype=np.int32),
        np.array(new_inner_loop_offsets, dtype=np.int32),
        np.array(new_face_inner_offsets, dtype=np.int32),
    )


class TokenValidationStatus(IntEnum):
    SUCCESS = 0
    UNEXPECTED_TOKEN = 1
    TRUNCATED_STREAM = 2
    MISSING_EOS = 3
    NESTED_FACE = 4
    UNCLOSED_FACE = 5
    FACE_END_WITHOUT_START = 6
    LOOP_OUTSIDE_FACE = 7
    UNCLOSED_LOOP = 8
    LOOP_END_WITHOUT_START = 9
    NESTED_LOOP = 10
    EDGE_OUTSIDE_LOOP = 11
    INVALID_BBOX = 12
    INVALID_FACE_VQ = 13
    INVALID_EDGE_VQ = 14
    INVALID_EDGE_ID = 15
    DUPLICATE_EDGE_ID = 16
    UNDEFINED_EDGE_REF = 17


class Tokenizer(torch.nn.Module):
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(
                f"The checkpoint directory does not exist: {str(checkpoint_dir)}"
            )

        self.bos_id = int(BRepTokenType.BOS)
        self.eos_id = int(BRepTokenType.EOS)
        self.pad_id = int(BRepTokenType.PAD)

        # Load VQ-VAEs for discrete geometry tokens
        self.surface_fsq = SurfaceFSQVAE.load_from_checkpoint(
            checkpoint_dir / "surf-fsq.ckpt"
        )
        self.edge_fsq = EdgeFSQVAE.load_from_checkpoint(
            checkpoint_dir / "edge-fsq.ckpt"
        )

        # Ensure models are in eval mode
        self.surface_fsq.eval()
        self.edge_fsq.eval()

    @property
    def vocab_size(self) -> int:
        return BREP_VOCAB_SIZE

    def get_eos_token_id(self) -> int:
        return self.eos_id

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def get_pad_token_id(self) -> int:
        return self.pad_id

    def encode_surface(self, face_ncs) -> torch.tensor:
        _, surf_id = self.surface_fsq.encode(face_ncs.permute(0, 3, 1, 2))
        surf_id = surf_id.flatten(-2, -1)
        return surf_id

    def encode_edge(self, edge_ncs) -> torch.tensor:
        _, edge_id = self.edge_fsq.encode(edge_ncs.permute(0, 2, 1))
        return edge_id

    def encode(
        self,
        face_points: np.ndarray,
        edge_points: np.ndarray,
        outer_edge_indices: np.ndarray,
        face_outer_offsets: np.ndarray,
        inner_edge_indices: np.ndarray,
        inner_loop_offsets: np.ndarray,
        face_inner_offsets: np.ndarray,
        device: None,
        max_length: int = -1,
    ) -> torch.Tensor:
        """Serializes B-Rep topology and geometry into a 1D tensor of discrete tokens."""
        if device is None:
            device = next(self.surface_fsq.parameters()).device

        if self.surface_fsq.device != device or self.edge_fsq.device != device:
            self.surface_fsq.to(device)
            self.edge_fsq.to(device)

        num_faces = len(face_points)
        input_ids: List[int] = []

        edge_counts = Counter(outer_edge_indices.tolist() + inner_edge_indices.tolist())
        registry = EdgeRegistry(edge_counts)

        face_bboxes, face_norms = [], []
        for pts in face_points:
            norm_pts, bbox = normalize_points_with_bbox(pts)
            face_norms.append(norm_pts)
            face_bboxes.append(bbox)

        edge_bboxes, edge_norms = [], []
        for pts in edge_points:
            norm_pts, bbox = normalize_points_with_bbox(pts)
            edge_norms.append(norm_pts)
            edge_bboxes.append(bbox)

        if not face_norms or not edge_norms:
            raise ValueError("Empty face or edge points.")

        with torch.no_grad():
            face_tensors = torch.tensor(
                np.stack(face_norms), dtype=torch.float32, device=device
            )
            face_vq_indices = self.encode_surface(face_tensors).cpu().numpy()
            edge_tensors = torch.tensor(
                np.stack(edge_norms), dtype=torch.float32, device=device
            )
            edge_vq_indices = self.encode_edge(edge_tensors).cpu().numpy()

        def emit(token_id: Union[int, BRepTokenType]):
            input_ids.append(int(token_id))

        def emit_bbox(bbox: np.ndarray):
            for tok in bbox_to_tokens(bbox, SPATIAL_RESOLUTION):
                emit(tok)

        def emit_loop(edge_ids: Iterable[int]):
            if not len(edge_ids):
                return
            emit(BRepTokenType.LOOP_START)

            for eid in edge_ids:
                tok_type, tid = registry.register(int(eid))
                emit(tok_type)
                emit(edge_id_to_token(tid))

                if tok_type == BRepTokenType.EDGE_NEW:
                    emit_bbox(edge_bboxes[eid])
                    for vq_idx in edge_vq_indices[eid]:
                        emit(vq_idx + EDGE_VQ_TOKEN_OFFSET)

            emit(BRepTokenType.LOOP_END)

        emit(BRepTokenType.BOS)
        for fi in range(num_faces):
            emit(BRepTokenType.FACE_START)
            emit_bbox(face_bboxes[fi])
            for vq_idx in face_vq_indices[fi]:
                emit(vq_idx + FACE_VQ_TOKEN_OFFSET)

            emit_loop(
                outer_edge_indices[face_outer_offsets[fi] : face_outer_offsets[fi + 1]]
            )
            for j in range(face_inner_offsets[fi], face_inner_offsets[fi + 1]):
                emit_loop(
                    inner_edge_indices[
                        inner_loop_offsets[j] : inner_loop_offsets[j + 1]
                    ]
                )

            emit(BRepTokenType.FACE_END)
        emit(BRepTokenType.EOS)

        if max_length > 0:
            input_ids = input_ids[:max_length]
        return torch.tensor(input_ids, dtype=torch.long)

    def id_to_string(self, token_id: int) -> str:
        if token_id in BRepTokenType._value2member_map_:
            return BRepTokenType(token_id).name
        elif is_coord_token(token_id):
            coord = token_to_coord(token_id)
            return f"COORD({coord:.3f})"
        elif is_edge_id_token(token_id):
            edge_id = token_to_edge_id(token_id)
            return f"EDGE_ID({edge_id})"
        else:
            return f"UNK({token_id})"

    def ids_to_string(self, token_ids: torch.Tensor) -> str:
        id_list = token_ids.cpu().tolist()
        tokens = []
        for tid in id_list:
            if tid in BRepTokenType._value2member_map_:
                tokens.append(BRepTokenType(tid).name)
            elif is_coord_token(tid):
                coord = token_to_coord(tid)
                tokens.append(f"COORD({coord:.3f})")
            elif is_edge_id_token(tid):
                edge_id = token_to_edge_id(tid)
                tokens.append(f"EDGE_ID({edge_id})")
            else:
                tokens.append(f"UNK({tid})")
        return " ".join(tokens)

    def _parse_face_tokens(self, tokens: List[int], idx: int):
        if idx + 1 + BBOX_TOKEN_COUNT + FACE_VQ_TOKEN_NUM > len(tokens):
            raise ValueError("Truncated token stream after FACE_START.")

        bbox_tokens = tokens[idx + 1 : idx + 1 + BBOX_TOKEN_COUNT]
        face_vqs_tokens = tokens[
            idx + 1 + BBOX_TOKEN_COUNT : idx + 1 + BBOX_TOKEN_COUNT + FACE_VQ_TOKEN_NUM
        ]

        if not all(is_coord_token(t) for t in bbox_tokens):
            raise ValueError(
                f"Expected {BBOX_TOKEN_COUNT} coordinate tokens for face bbox at index {idx}."
            )
        if not all(is_face_vq_token(t) for t in face_vqs_tokens):
            raise ValueError(
                f"Expected {FACE_VQ_TOKEN_NUM} face VQ tokens at index {idx + 1 + BBOX_TOKEN_COUNT}."
            )

        face_vqs_tokens = [t - FACE_VQ_TOKEN_OFFSET for t in face_vqs_tokens]
        return (
            tokens_to_bbox(bbox_tokens, SPATIAL_RESOLUTION),
            face_vqs_tokens,
            idx + 1 + BBOX_TOKEN_COUNT + FACE_VQ_TOKEN_NUM,
        )

    def _parse_edge_tokens(self, tokens: List[int], idx: int):
        if idx + 1 + BBOX_TOKEN_COUNT + EDGE_VQ_TOKEN_NUM > len(tokens):
            raise ValueError("Truncated token stream after EDGE_NEW.")

        edge_id = tokens[idx + 1]
        bbox_tokens = tokens[idx + 2 : idx + 2 + BBOX_TOKEN_COUNT]
        edge_vq_tokens = tokens[
            idx + 2 + BBOX_TOKEN_COUNT : idx + 2 + BBOX_TOKEN_COUNT + EDGE_VQ_TOKEN_NUM
        ]

        if not is_edge_id_token(edge_id):
            raise ValueError(
                f"Expected EDGE_ID token at index {idx + 1}, got {edge_id}."
            )
        if not all(is_coord_token(t) for t in bbox_tokens):
            raise ValueError(
                f"Expected {BBOX_TOKEN_COUNT} coordinate tokens for edge bbox at index {idx}."
            )
        if not all(is_edge_vq_token(t) for t in edge_vq_tokens):
            raise ValueError(
                f"Expected {EDGE_VQ_TOKEN_NUM} edge VQ tokens at index {idx + 2 + BBOX_TOKEN_COUNT}."
            )
        edge_vq_tokens = [t - EDGE_VQ_TOKEN_OFFSET for t in edge_vq_tokens]
        return (
            edge_id,
            tokens_to_bbox(bbox_tokens, SPATIAL_RESOLUTION),
            edge_vq_tokens,
            idx + 2 + BBOX_TOKEN_COUNT + EDGE_VQ_TOKEN_NUM,
        )

    def _decode_face_vq_codes(self, face_vqs: torch.Tensor):
        """
        Args:
            vq_codes: (N, 4) 形状的张量，包含面的 VQ Token。
        Returns:
            face_points: (N, H, W, 3) 形状的面网格点云数据。
        """
        z = self.surface_fsq.quantizer.indices_to_codes(face_vqs).permute(0, 2, 1)
        uv_ncs_faces = self.surface_fsq.decode(z.unflatten(-1, (2, 2))).sample
        uv_ncs_faces = (
            rearrange(uv_ncs_faces, "b d ... -> b ... d").float().cpu().numpy()
        )
        return uv_ncs_faces

    def _decode_edge_vq_codes(self, edge_vqs: torch.Tensor):
        """
        Args:
            vq_codes: (M, 2) 形状的张量，包含边的 VQ Token。
        Returns:
            edge_points: (M, L, 3) 形状的边曲线点云数据。
        """
        geomZ_edges = self.edge_fsq.quantizer.indices_to_codes(edge_vqs).permute(
            0, 2, 1
        )
        uv_ncs_edges = self.edge_fsq.decode(geomZ_edges).sample
        uv_ncs_edges = (
            rearrange(uv_ncs_edges, "b d ... -> b ... d").float().detach().cpu().numpy()
        )
        return uv_ncs_edges

    def decode(self, tensor: torch.Tensor) -> BRepData:
        tokens = tensor.tolist()
        device = tensor.device

        face_bboxes, face_vqs = [], []
        edge_bboxes, edge_vqs = [], []
        active_edges = {}

        outer_edge_indices, face_outer_offsets = [], [0]
        inner_edge_indices, inner_loop_offsets, face_inner_offsets = [], [0], [0]

        idx, in_face, loop_idx_in_face = 0, False, 0

        while idx < len(tokens):
            tok = tokens[idx]

            if tok in (self.bos_id, self.pad_id):
                idx += 1
                continue
            elif tok == self.eos_id:
                break

            if tok == BRepTokenType.FACE_START:
                if in_face:
                    raise ValueError(f"Nested FACE_START at index {idx}.")

                bbox, face_vqs_tokens, idx = self._parse_face_tokens(tokens, idx)
                face_bboxes.append(bbox)
                face_vqs.append(face_vqs_tokens)
                in_face, loop_idx_in_face = True, 0

            elif tok == BRepTokenType.FACE_END:
                if not in_face:
                    raise ValueError(f"FACE_END without FACE_START at {idx}.")

                face_outer_offsets.append(len(outer_edge_indices))
                face_inner_offsets.append(len(inner_loop_offsets) - 1)
                in_face, idx = False, idx + 1

            elif tok == BRepTokenType.LOOP_START:
                if not in_face:
                    raise ValueError(f"LOOP_START outside of face at {idx}.")

                idx += 1
                current_loop = []

                while idx < len(tokens) and tokens[idx] != BRepTokenType.LOOP_END:
                    ltok = tokens[idx]

                    if ltok == BRepTokenType.EDGE_NEW:
                        edge_id, edge_bbox, edge_vqs_tokens, idx = (
                            self._parse_edge_tokens(tokens, idx)
                        )

                        canonical_idx = len(edge_bboxes)
                        edge_bboxes.append(edge_bbox)
                        edge_vqs.append(edge_vqs_tokens)

                        active_edges[edge_id] = canonical_idx
                        current_loop.append(canonical_idx)

                    elif ltok in (BRepTokenType.EDGE_REF, BRepTokenType.EDGE_REF_LAST):
                        if idx + 2 > len(tokens):
                            raise ValueError(
                                f"Truncated token stream after {BRepTokenType(ltok).name}."
                            )

                        edge_id = tokens[idx + 1]
                        if edge_id not in active_edges:
                            raise ValueError(
                                f"Invalid {BRepTokenType(ltok).name} {edge_id} at {idx}."
                            )

                        if ltok == BRepTokenType.EDGE_REF_LAST:
                            canonical_idx = active_edges.pop(edge_id)
                        else:
                            canonical_idx = active_edges[edge_id]

                        current_loop.append(canonical_idx)
                        idx += 2
                    else:
                        raise ValueError(f"Invalid token {ltok} inside loop at {idx}.")

                if idx >= len(tokens) or tokens[idx] != BRepTokenType.LOOP_END:
                    raise ValueError("Unterminated LOOP_START.")
                idx += 1

                if loop_idx_in_face == 0:
                    outer_edge_indices.extend(current_loop)
                else:
                    inner_edge_indices.extend(current_loop)
                    inner_loop_offsets.append(len(inner_edge_indices))
                loop_idx_in_face += 1

            else:
                raise ValueError(f"Unexpected token {tok} at index {idx}.")

        if in_face:
            raise ValueError("Sequence ended with an unclosed FACE_START.")

        face_points, edge_points = [], []

        if face_vqs:
            fvq_tensor = torch.tensor(face_vqs, dtype=torch.long, device=device)
            with torch.no_grad():
                face_pts_norm = self._decode_face_vq_codes(fvq_tensor)
            for pts, bbox in zip(face_pts_norm, face_bboxes):
                face_points.append(denormalize_points_with_bbox(pts, bbox))

        if edge_vqs:
            evq_tensor = torch.tensor(edge_vqs, dtype=torch.long, device=device)
            with torch.no_grad():
                edge_pts_norm = self._decode_edge_vq_codes(evq_tensor)
            for pts, bbox in zip(edge_pts_norm, edge_bboxes):
                edge_points.append(denormalize_points_with_bbox(pts, bbox))

        if face_points is None or edge_points is None:
            raise ValueError("face_points or edge_points is None after decoding.")

        return BRepData(
            face_points=np.array(face_points, dtype=np.float32),
            edge_points=np.array(edge_points, dtype=np.float32),
            outer_edge_indices=np.array(outer_edge_indices, dtype=np.int32),
            face_outer_offsets=np.array(face_outer_offsets, dtype=np.int32),
            inner_edge_indices=np.array(inner_edge_indices, dtype=np.int32),
            inner_loop_offsets=np.array(inner_loop_offsets, dtype=np.int32),
            face_inner_offsets=np.array(face_inner_offsets, dtype=np.int32),
        )

    def decode_stream(
        self,
        token_stream: Iterable[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Iterator[Dict[str, Any]]:
        if device is None:
            device = next(self.surface_fsq.parameters()).device

        def _token_generator():
            for tensor in token_stream:
                for tok in tensor.view(-1).tolist():
                    if tok not in (BRepTokenType.BOS, BRepTokenType.PAD):
                        yield tok

        tokens = _token_generator()
        active_edges = {}
        edge_count = 0

        try:
            while True:
                tok = next(tokens)
                if tok == BRepTokenType.EOS:
                    break

                if tok != BRepTokenType.FACE_START:
                    raise ValueError(f"Expected FACE_START or EOS, got {tok}.")

                face_bbox = tokens_to_bbox(
                    [next(tokens) for _ in range(6)], SPATIAL_RESOLUTION
                )
                face_vq_toks = [next(tokens) - FACE_VQ_TOKEN_OFFSET for _ in range(4)]

                outer_loop = []
                inner_loops = []

                while True:
                    ltok = next(tokens)
                    if ltok == BRepTokenType.FACE_END:
                        break
                    if ltok != BRepTokenType.LOOP_START:
                        raise ValueError(
                            f"Expected LOOP_START or FACE_END, got {ltok}."
                        )

                    current_loop = []
                    while True:
                        etok = next(tokens)
                        if etok == BRepTokenType.LOOP_END:
                            break

                        if etok == BRepTokenType.EDGE_NEW:
                            edge_id_tok = next(tokens)
                            edge_bbox = tokens_to_bbox(
                                [next(tokens) for _ in range(6)], SPATIAL_RESOLUTION
                            )
                            edge_vq_toks = [
                                next(tokens) - EDGE_VQ_TOKEN_OFFSET for _ in range(2)
                            ]

                            evq_tensor = torch.tensor(
                                [edge_vq_toks], dtype=torch.long, device=device
                            )
                            with torch.no_grad():
                                edge_pts_norm = (
                                    self.edge_fsq.decode(evq_tensor).cpu().numpy()[0]
                                )

                            edge_pts = denormalize_points_with_bbox(
                                edge_pts_norm, edge_bbox
                            )

                            active_edges[edge_id_tok] = edge_count
                            current_loop.append(edge_count)

                            yield {
                                "event": "edge_new",
                                "id": edge_count,
                                "points": np.array(edge_pts, dtype=np.float32),
                                "bbox": edge_bbox,
                            }
                            edge_count += 1

                        elif etok == BRepTokenType.EDGE_REF:
                            edge_id_tok = next(tokens)
                            if edge_id_tok not in active_edges:
                                raise ValueError(f"Invalid EDGE_REF {edge_id_tok}.")

                            canonical_idx = active_edges.pop(edge_id_tok)
                            current_loop.append(canonical_idx)

                            yield {
                                "event": "edge_ref",
                                "id": canonical_idx,
                            }
                        else:
                            raise ValueError(
                                f"Expected EDGE_NEW, EDGE_REF, or LOOP_END, got {etok}."
                            )

                    if not outer_loop:
                        outer_loop = current_loop
                    else:
                        inner_loops.append(current_loop)

                    yield {
                        "event": "loop_completed",
                        "is_outer": not bool(inner_loops),
                        "edge_indices": current_loop,
                    }

                fvq_tensor = torch.tensor(
                    [face_vq_toks], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    face_pts_norm = self.surface_fsq.decode(fvq_tensor).cpu().numpy()[0]

                face_pts = denormalize_points_with_bbox(face_pts_norm, face_bbox)

                yield {
                    "event": "face_completed",
                    "points": np.array(face_pts, dtype=np.float32),
                    "bbox": face_bbox,
                    "outer_loop": outer_loop,
                    "inner_loops": inner_loops,
                }

        except StopIteration:
            raise ValueError(
                "Truncated token stream: reached end of iterator prematurely."
            )

    def validate_tokens(
        self, tensor: Union[torch.Tensor, List[int]]
    ) -> Tuple[TokenValidationStatus, str, int]:
        tokens = tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor
        idx, in_face, in_loop, active_edges = 0, False, False, set()

        def has_length(required_len: int) -> bool:
            return idx + required_len <= len(tokens)

        def is_valid_sequence(offset: int, length: int, validator_func) -> bool:
            return all(
                validator_func(t) for t in tokens[idx + offset : idx + offset + length]
            )

        while idx < len(tokens):
            tok = tokens[idx]

            if tok in (self.bos_id, self.pad_id):
                idx += 1
                continue

            if tok == self.eos_id:
                if in_loop:
                    return (
                        TokenValidationStatus.UNCLOSED_LOOP,
                        "EOS with open LOOP",
                        idx,
                    )
                if in_face:
                    return (
                        TokenValidationStatus.UNCLOSED_FACE,
                        "EOS with open FACE",
                        idx,
                    )
                return TokenValidationStatus.SUCCESS, "Success", idx

            if tok == BRepTokenType.FACE_START:
                if in_face:
                    return TokenValidationStatus.NESTED_FACE, "Nested FACE_START", idx
                if in_loop:
                    return (
                        TokenValidationStatus.UNCLOSED_LOOP,
                        "FACE_START inside LOOP",
                        idx,
                    )
                if not has_length(1 + BBOX_TOKEN_COUNT + FACE_VQ_TOKEN_NUM):
                    return (
                        TokenValidationStatus.TRUNCATED_STREAM,
                        "Truncated after FACE_START",
                        idx,
                    )
                if not is_valid_sequence(1, BBOX_TOKEN_COUNT, is_coord_token):
                    return (
                        TokenValidationStatus.INVALID_BBOX,
                        "Invalid FACE bbox",
                        idx + 1,
                    )
                if not is_valid_sequence(
                    1 + BBOX_TOKEN_COUNT, FACE_VQ_TOKEN_NUM, is_face_vq_token
                ):
                    return (
                        TokenValidationStatus.INVALID_FACE_VQ,
                        "Invalid FACE VQ",
                        idx + 1 + BBOX_TOKEN_COUNT,
                    )

                in_face = True
                idx += 1 + BBOX_TOKEN_COUNT + FACE_VQ_TOKEN_NUM

            elif tok == BRepTokenType.FACE_END:
                if not in_face:
                    return (
                        TokenValidationStatus.FACE_END_WITHOUT_START,
                        "FACE_END without START",
                        idx,
                    )
                if in_loop:
                    return (
                        TokenValidationStatus.UNCLOSED_LOOP,
                        "FACE_END with open LOOP",
                        idx,
                    )
                in_face, idx = False, idx + 1

            elif tok == BRepTokenType.LOOP_START:
                if not in_face:
                    return (
                        TokenValidationStatus.LOOP_OUTSIDE_FACE,
                        "LOOP_START outside FACE",
                        idx,
                    )
                if in_loop:
                    return TokenValidationStatus.NESTED_LOOP, "Nested LOOP_START", idx
                in_loop, idx = True, idx + 1

            elif tok == BRepTokenType.LOOP_END:
                if not in_loop:
                    return (
                        TokenValidationStatus.LOOP_END_WITHOUT_START,
                        "LOOP_END without START",
                        idx,
                    )
                in_loop, idx = False, idx + 1

            elif tok == BRepTokenType.EDGE_NEW:
                if not in_loop:
                    return (
                        TokenValidationStatus.EDGE_OUTSIDE_LOOP,
                        "EDGE_NEW outside LOOP",
                        idx,
                    )
                if not has_length(2 + BBOX_TOKEN_COUNT + EDGE_VQ_TOKEN_NUM):
                    return (
                        TokenValidationStatus.TRUNCATED_STREAM,
                        "Truncated after EDGE_NEW",
                        idx,
                    )

                edge_id = tokens[idx + 1]
                if not is_edge_id_token(edge_id):
                    return (
                        TokenValidationStatus.INVALID_EDGE_ID,
                        f"Invalid EDGE_ID: {edge_id}",
                        idx + 1,
                    )
                if not is_valid_sequence(2, BBOX_TOKEN_COUNT, is_coord_token):
                    return (
                        TokenValidationStatus.INVALID_BBOX,
                        "Invalid EDGE bbox",
                        idx + 2,
                    )
                if not is_valid_sequence(
                    2 + BBOX_TOKEN_COUNT, EDGE_VQ_TOKEN_NUM, is_edge_vq_token
                ):
                    return (
                        TokenValidationStatus.INVALID_EDGE_VQ,
                        "Invalid EDGE VQ",
                        idx + 2 + BBOX_TOKEN_COUNT,
                    )
                if edge_id in active_edges:
                    return (
                        TokenValidationStatus.DUPLICATE_EDGE_ID,
                        f"Duplicate active EDGE_ID: {edge_id}",
                        idx + 1,
                    )

                active_edges.add(edge_id)
                idx += 2 + BBOX_TOKEN_COUNT + EDGE_VQ_TOKEN_NUM

            elif tok in (BRepTokenType.EDGE_REF, BRepTokenType.EDGE_REF_LAST):
                if not in_loop:
                    return (
                        TokenValidationStatus.EDGE_OUTSIDE_LOOP,
                        "EDGE_REF outside LOOP",
                        idx,
                    )
                if not has_length(2):
                    return (
                        TokenValidationStatus.TRUNCATED_STREAM,
                        "Truncated after EDGE_REF",
                        idx,
                    )

                edge_id = tokens[idx + 1]
                if not is_edge_id_token(edge_id):
                    return (
                        TokenValidationStatus.INVALID_EDGE_ID,
                        f"Invalid EDGE_ID: {edge_id}",
                        idx + 1,
                    )
                if edge_id not in active_edges:
                    return (
                        TokenValidationStatus.UNDEFINED_EDGE_REF,
                        f"Undefined EDGE_REF: {edge_id}",
                        idx + 1,
                    )

                if tok == BRepTokenType.EDGE_REF_LAST:
                    active_edges.remove(edge_id)
                idx += 2

            else:
                return (
                    TokenValidationStatus.UNEXPECTED_TOKEN,
                    f"Unexpected token: {tok}",
                    idx,
                )

        if in_loop:
            return TokenValidationStatus.UNCLOSED_LOOP, "Missing LOOP_END", idx
        if in_face:
            return TokenValidationStatus.UNCLOSED_FACE, "Missing FACE_END", idx
        return TokenValidationStatus.MISSING_EOS, "Missing EOS", idx


if __name__ == "__main__":
    # local in ../
    # export PYTHONPATH=./:$PYTHONPATH
    # python src/tokenizer
    tokenizer = Tokenizer(checkpoint_dir="weights")
    tokenizer.to("cuda")
    tokenizer.eval()
    test_data_path = "test/00000144_2fc54fcd110d4f49969163c4_step_007_8_18.npz"
    from scripts.convert2bin import tokenize_fn

    res = list(tokenize_fn(test_data_path, tokenizer, True))
    idx = 0
    for tokens in res:
        brep = tokenizer.decode(tokens.to("cuda"))
        from scripts.visual import PointGridVisualizer

        vis = PointGridVisualizer()
        vis.visualize(brep, save_path=f"output_face_{idx}.png", title=f"Face {idx}")
        idx += 1
