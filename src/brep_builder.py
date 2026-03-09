import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple, List

from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.GeomAPI import (
    GeomAPI_PointsToBSpline,
    GeomAPI_PointsToBSplineSurface,
)
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.TopoDS import TopoDS_Shape

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import write_step_file
from OCC.Display.OCCViewer import OffscreenRenderer


class bcolors:
    OKGREEN = "\033[92m"
    ENDC = "\033[0m"


@dataclass
class BRepData:
    face_points: np.ndarray
    edge_points: np.ndarray
    outer_edge_indices: np.ndarray
    face_outer_offsets: np.ndarray
    inner_edge_indices: np.ndarray
    inner_loop_offsets: np.ndarray
    face_inner_offsets: np.ndarray


class BRepBuilder:
    def __init__(
        self,
        data: BRepData,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dist_threshold: float = 0.05,
        sewing_tolerance: float = 1e-3,
        eval_mode: bool = False,
    ):
        self.data = data
        self.device = device
        self.dist_threshold = dist_threshold
        self.sewing_tolerance = sewing_tolerance
        self.eval_mode = eval_mode

    @staticmethod
    def _get_slice(offsets: np.ndarray, idx: int, max_len: int) -> Tuple[int, int]:
        return offsets[idx], offsets[idx + 1] if idx + 1 < len(offsets) else max_len

    def _get_face_edge_adj(self) -> List[List[int]]:
        adj, d = [], self.data
        for i in range(len(d.face_points)):
            edges = []
            os, oe = self._get_slice(d.face_outer_offsets, i, len(d.outer_edge_indices))
            edges.extend(d.outer_edge_indices[os:oe])
            if d.face_inner_offsets is not None and len(d.face_inner_offsets) > 0:
                is_, ie_ = self._get_slice(
                    d.face_inner_offsets, i, len(d.inner_loop_offsets)
                )
                for j in range(is_, ie_):
                    ls, le = self._get_slice(
                        d.inner_loop_offsets, j, len(d.inner_edge_indices)
                    )
                    edges.extend(d.inner_edge_indices[ls:le])
            adj.append(edges)
        return adj

    def _optimize_connections(self) -> Tuple[np.ndarray, np.ndarray]:
        f_pts, e_pts = self.data.face_points.copy(), self.data.edge_points.copy()
        if self.eval_mode or len(e_pts) == 0:
            print(f"{bcolors.OKGREEN}[Eval Mode No Optimization]{bcolors.ENDC}")
            return f_pts, e_pts

        print(f"{bcolors.OKGREEN}[Joint Optimization]{bcolors.ENDC}")
        num_e = len(e_pts)

        # 1. 拓扑点对齐 (Edge Snapping)
        endpoints = np.concatenate([e_pts[:, 0, :], e_pts[:, -1, :]], axis=0)
        pairs = cKDTree(endpoints).query_pairs(self.dist_threshold)

        G = nx.Graph()
        G.add_nodes_from(range(2 * num_e))
        G.add_edges_from(pairs)

        new_endpoints = np.zeros_like(endpoints)
        for comp in nx.connected_components(G):
            comp_idx = list(comp)
            new_endpoints[comp_idx] = endpoints[comp_idx].mean(axis=0)

        weights = np.linspace(0, 1, e_pts.shape[1])[:, None]
        for i in range(num_e):
            s_shift = new_endpoints[i] - e_pts[i, 0]
            e_shift = new_endpoints[i + num_e] - e_pts[i, -1]
            e_pts[i] += s_shift * (1 - weights) + e_shift * weights

        # 2. 面-边拟合优化 (Surface PyTorch Joint Optimization)
        face_edge_adj = self._get_face_edge_adj()
        surf = torch.tensor(f_pts, dtype=torch.float32, device=self.device)
        face_edges = [
            torch.tensor(e_pts[adj], dtype=torch.float32, device=self.device)
            for adj in face_edge_adj
        ]

        surf_offset = nn.Parameter(
            torch.zeros((len(f_pts), 1, 1, 3), device=self.device)
        )
        optimizer = torch.optim.AdamW(
            [surf_offset], lr=1e-3, betas=(0.95, 0.999), weight_decay=1e-6
        )

        for _ in range(200):
            surf_updated = surf + surf_offset
            loss = 0
            for s_p, e_p in zip(surf_updated, face_edges):
                if len(e_p) == 0:
                    continue
                s_p, e_p = s_p.reshape(-1, 3), e_p.reshape(-1, 3).detach()
                dist = torch.cdist(s_p, e_p, p=2)
                loss += dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()

            optimizer.zero_grad()
            (loss / len(f_pts)).backward()
            optimizer.step()

        return (surf + surf_offset).detach().cpu().numpy(), e_pts

    def _pts2curve(self, points: np.ndarray):
        arr = TColgp_Array1OfPnt(1, points.shape[0])
        for i, p in enumerate(points):
            arr.SetValue(i + 1, gp_Pnt(*map(float, p)))
        return GeomAPI_PointsToBSpline(arr).Curve()

    def _pts2surface(self, points: np.ndarray):
        u_len, v_len = points.shape[:2]
        arr = TColgp_Array2OfPnt(1, u_len, 1, v_len)
        for i in range(u_len):
            for j in range(v_len):
                arr.SetValue(i + 1, j + 1, gp_Pnt(*map(float, points[i, j])))
        return GeomAPI_PointsToBSplineSurface(arr).Surface()

    def _build_wire(self, indices: np.ndarray, edges: List):
        wire_maker = BRepBuilderAPI_MakeWire()
        for idx in indices:
            wire_maker.Add(edges[idx])
        return wire_maker.Wire()

    def build(self) -> TopoDS_Shape:
        f_pts, e_pts = self._optimize_connections()
        d = self.data

        edges = [BRepBuilderAPI_MakeEdge(self._pts2curve(pts)).Edge() for pts in e_pts]
        sewing = BRepBuilderAPI_Sewing(self.sewing_tolerance)

        for i, f_p in enumerate(f_pts):
            surf = self._pts2surface(f_p)

            os, oe = self._get_slice(d.face_outer_offsets, i, len(d.outer_edge_indices))
            face_maker = BRepBuilderAPI_MakeFace(
                surf, self._build_wire(d.outer_edge_indices[os:oe], edges), True
            )

            if d.face_inner_offsets is not None and len(d.face_inner_offsets) > 0:
                is_, ie_ = self._get_slice(
                    d.face_inner_offsets, i, len(d.inner_loop_offsets)
                )
                for j in range(is_, ie_):
                    ls, le = self._get_slice(
                        d.inner_loop_offsets, j, len(d.inner_edge_indices)
                    )
                    face_maker.Add(self._build_wire(d.inner_edge_indices[ls:le], edges))

            if face_maker.IsDone():
                sewing.Add(face_maker.Face())

        sewing.Perform()
        return sewing.SewedShape()


def save_model_and_image(
    shape: TopoDS_Shape,
    step_path: str = "output.step",
    img_path: str = "output.png",
    resolution: tuple = (1920, 1080),
):
    # 1. 导出实体模型 (STEP格式)
    write_step_file(shape, step_path)

    # 2. 离屏渲染并保存图片 (支持PNG/JPEG)
    renderer = OffscreenRenderer(screen_size=resolution)
    renderer.DisplayShape(shape, update=True)
