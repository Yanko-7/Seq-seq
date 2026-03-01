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

from enum import IntEnum

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from collections import deque
import torch
import numpy as np
from src.vq_vae import EdgeFSQVAE, SurfaceFSQVAE

# ============================================================================
# Spatial resolution configuration
# ============================================================================

SPATIAL_RESOLUTION = 1024
EDGE_ID_COUNT = 984

# ============================================================================
# Token vocabulary definition
# ============================================================================


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
    SEP = 13
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


def normalize_points_with_bbox(points: np.ndarray):
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


def _bfs_reorder_faces(
    face_points,
    edge_adjacency,
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

    adj = [set() for _ in range(num_faces)]
    for u, v in edge_adjacency:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

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


class EdgeRegistry:
    """Track serialized edges with ID recycling.

    Each edge is New'd once and Ref'd at most once. After being Ref'd,
    the token ID is freed and can be reused by the next New edge.
    """

    def __init__(self):
        self._edge_to_token: Dict[int, int] = {}  # real edge_id -> token_edge_id
        self._available: deque[int] = deque()  # FIFO queue of freed IDs
        self._next_id: int = 0  # next fresh ID

    def register(self, edge_id: int) -> Tuple[bool, int]:
        """Register an edge, return (is_new, token_edge_id)"""
        if edge_id in self._edge_to_token:
            # EDGE_REF: return the token ID, then free it for reuse (FIFO)
            token_id = self._edge_to_token.pop(edge_id)
            self._available.append(token_id)  # 入队：将释放的 ID 放入队尾
            return False, token_id

        # EDGE_NEW: allocate oldest freed ID, or a fresh one
        if self._available:
            token_id = self._available.popleft()  # 出队：从队首获取最早释放的 ID
        else:
            token_id = self._next_id
            self._next_id += 1

        if token_id >= EDGE_ID_COUNT:
            raise ValueError(f"Concurrent live edges exceed {EDGE_ID_COUNT}")

        self._edge_to_token[edge_id] = token_id
        return True, token_id

    def reset(self):
        self._edge_to_token.clear()
        self._available.clear()
        self._next_id = 0


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


def _bfs_reorder_faces(
    face_points,
    edge_adjacency,
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

    adj = [set() for _ in range(num_faces)]
    for u, v in edge_adjacency:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

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


class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
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
        registry = EdgeRegistry()

        # 1. Pre-compute Normalization, BBoxes, and VQ-VAE discrete indices
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
        if len(face_norms) == 0 or len(edge_norms) == 0:
            raise ValueError("Face points or edge points are empty, cannot encode.")

        with torch.no_grad():
            # Note: Adjust .encode() depending on your exact LightningModule signature
            # (e.g., if it returns a tuple, extract the indices tensor).
            face_tensors = torch.tensor(
                np.stack(face_norms), dtype=torch.float32, device=device
            )
            face_vq_indices = self.encode_surface(face_tensors).cpu().numpy()
            # Expected shape: (num_faces, 4)

            edge_tensors = torch.tensor(
                np.stack(edge_norms), dtype=torch.float32, device=device
            )
            edge_vq_indices = self.encode_edge(edge_tensors).cpu().numpy()
            # Expected shape: (num_edges, 2)

        # 2. Sequence Emission Helpers
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

        # 3. Build the discrete sequence
        emit(BRepTokenType.BOS)

        for fi in range(num_faces):
            emit(BRepTokenType.FACE_START)
            emit_bbox(face_bboxes[fi])

            for vq_idx in face_vq_indices[fi]:
                emit(vq_idx + FACE_VQ_TOKEN_OFFSET)

            # Outer loops
            outer = outer_edge_indices[
                face_outer_offsets[fi] : face_outer_offsets[fi + 1]
            ]
            emit_loop(outer)

            # Inner loops
            for j in range(face_inner_offsets[fi], face_inner_offsets[fi + 1]):
                inner = inner_edge_indices[
                    inner_loop_offsets[j] : inner_loop_offsets[j + 1]
                ]
                emit_loop(inner)

            emit(BRepTokenType.FACE_END)

        emit(BRepTokenType.EOS)

        if max_length > 0:
            input_ids = input_ids[:max_length]

        return torch.tensor(input_ids, dtype=torch.long)

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

    def decode(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Decodes a 1D token tensor back into reconstructed B-Rep topology and geometric points."""
        tokens = tensor.tolist()
        device = tensor.device

        faces = []
        all_edges = []
        active_edge_tokens = {}  # Tracks recycled tokens -> canonical edge index

        current_face = None
        current_loop = None
        idx = 0

        # 1. Parse Topology and Token Data
        while idx < len(tokens):
            tok = tokens[idx]

            if tok in (BRepTokenType.BOS, BRepTokenType.PAD):
                idx += 1
                continue
            elif tok == BRepTokenType.EOS:
                break

            elif tok == BRepTokenType.FACE_START:
                bbox_toks = tokens[idx + 1 : idx + 7]
                bbox = tokens_to_bbox(bbox_toks, SPATIAL_RESOLUTION)
                idx += 7

                surf_vq_toks = [t - FACE_VQ_TOKEN_OFFSET for t in tokens[idx : idx + 4]]
                idx += 4

                current_face = {
                    "bbox": bbox,
                    "surface_vq": surf_vq_toks,
                    "outer_loop": [],
                    "inner_loops": [],
                }
                faces.append(current_face)

            elif tok == BRepTokenType.FACE_END:
                idx += 1

            elif tok == BRepTokenType.LOOP_START:
                current_loop = []
                idx += 1

            elif tok == BRepTokenType.LOOP_END:
                if current_face is not None and current_loop is not None:
                    if not current_face["outer_loop"]:
                        current_face["outer_loop"] = current_loop
                    else:
                        current_face["inner_loops"].append(current_loop)
                current_loop = None
                idx += 1

            elif tok == BRepTokenType.EDGE_NEW:
                edge_id_tok = tokens[idx + 1]
                idx += 2

                bbox_toks = tokens[idx : idx + 6]
                bbox = tokens_to_bbox(bbox_toks, SPATIAL_RESOLUTION)
                idx += 6

                curve_vq_toks = [
                    t - EDGE_VQ_TOKEN_OFFSET for t in tokens[idx : idx + 2]
                ]
                idx += 2

                canonical_edge_id = len(all_edges)
                all_edges.append({"bbox": bbox, "curve_vq": curve_vq_toks})
                active_edge_tokens[edge_id_tok] = canonical_edge_id

                if current_loop is not None:
                    current_loop.append(canonical_edge_id)

            elif tok == BRepTokenType.EDGE_REF:
                edge_id_tok = tokens[idx + 1]
                idx += 2

                # Recover edge and free the token
                if edge_id_tok in active_edge_tokens:
                    canonical_edge_id = active_edge_tokens.pop(edge_id_tok)
                    if current_loop is not None:
                        current_loop.append(canonical_edge_id)
            else:
                idx += 1

        # 2. Reconstruct Point Clouds via VQ-VAE decoders
        if faces:
            face_vqs = torch.tensor(
                [f["surface_vq"] for f in faces], dtype=torch.long, device=device
            )
            with torch.no_grad():
                # Note: Adjust .decode() depending on your exact LightningModule signature
                face_points_norm = self.surface_fsq.decode(face_vqs).cpu().numpy()
            for i, f in enumerate(faces):
                f["points"] = denormalize_points_with_bbox(
                    face_points_norm[i], f["bbox"]
                )

        if all_edges:
            edge_vqs = torch.tensor(
                [e["curve_vq"] for e in all_edges], dtype=torch.long, device=device
            )
            with torch.no_grad():
                edge_points_norm = self.edge_fsq.decode(edge_vqs).cpu().numpy()
            for i, e in enumerate(all_edges):
                e["points"] = denormalize_points_with_bbox(
                    edge_points_norm[i], e["bbox"]
                )

        return {"faces": faces, "edges": all_edges}

    def decode_stream(
        self,
        token_stream: Iterable[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Iterator[Any]:
        """A generator that yields structures as they are constructed autoregressively (Optional to implement)."""
        raise NotImplementedError(
            "Streaming decode for discrete tokens not yet implemented."
        )
