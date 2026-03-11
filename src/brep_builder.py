import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, List

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.GeomAPI import (
    GeomAPI_PointsToBSpline,
    GeomAPI_PointsToBSplineSurface,
)
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPSOLID
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Shell
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume
from OCC.Core.TopTools import TopTools_ListOfShape
from typing import Optional
from OCC.Core.TopoDS import TopoDS_Wire
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Shape, ShapeFix_Wire
from OCC.Extend.DataExchange import write_step_file

# from OCC.Display.OCCViewer import OffscreenRenderer
from OCC.Display.SimpleGui import init_display

from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.ShapeFix import ShapeFix_Edge
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GeomLProp import GeomLProp_SLProps


def is_bspline_cylinder(face, tol=1e-3, steps=10):
    surf = BRep_Tool.Surface(face)
    surf_type = GeomAdaptor_Surface(surf).GetType()

    if surf_type == GeomAbs_Cylinder:
        return True
    if surf_type in (GeomAbs_Plane, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus):
        return False

    umin, umax, vmin, vmax = surf.Bounds()
    k1, k2 = [], []

    for i in range(steps + 1):
        u = umin + (umax - umin) * i / steps
        for j in range(steps + 1):
            v = vmin + (vmax - vmin) * j / steps
            props = GeomLProp_SLProps(surf, u, v, 2, 1e-6)
            if props.IsCurvatureDefined():
                k1.append(props.MaxCurvature())
                k2.append(props.MinCurvature())

    if not k1:
        return False

    num_zero, num_inf = 1e-6, 1e5

    def get_radius(k):
        return num_inf if abs(k) < num_zero else abs(1.0 / k)

    r1_ref, r2_ref = get_radius(k1[0]), get_radius(k2[0])

    for cur_k1, cur_k2 in zip(k1[1:], k2[1:]):
        if abs(get_radius(cur_k1) - r1_ref) > tol:
            return False
        if abs(get_radius(cur_k2) - r2_ref) > tol:
            return False

    is_flat_k1 = abs(k1[0]) < num_zero
    is_flat_k2 = abs(k2[0]) < num_zero

    return (is_flat_k1 and not is_flat_k2) or (not is_flat_k1 and is_flat_k2)


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

    @classmethod
    def load_npz(cls, path):
        with np.load(path) as data:
            return cls(**{k: data[k] for k in data.files})


@staticmethod
def get_slice(offsets: np.ndarray, idx: int, max_len: int = -1) -> Tuple[int, int]:
    return offsets[idx], offsets[idx + 1] if idx + 1 < len(offsets) else max_len


def pts2curve(cls, points: np.ndarray):
    arr = TColgp_Array1OfPnt(1, points.shape[0])
    for i, p in enumerate(points):
        arr.SetValue(i + 1, gp_Pnt(*map(float, p)))
    return GeomAPI_PointsToBSpline(arr, 3, 7, GeomAbs_C2, 0.005).Curve()


def save_model_and_image(
    shape: TopoDS_Shape,
    step_path: str = "output.step",
    img_path: str = "output.png",
    resolution: tuple = (1920, 1080),
):
    # 1. 导出实体模型 (STEP格式)
    write_step_file(shape, step_path)

    display, start_display, add_menu, add_function_to_menu = init_display()

    display.DisplayShape(shape, update=True)
    # # 2. 离屏渲染并保存图片 (支持PNG/JPEG)
    # renderer = OffscreenRenderer(screen_size=resolution)
    # renderer.DisplayShape(shape, update=True)


def build_healed_wire(
    edge_indices: List[int], edges: List, tolerance: float = 1e-3
) -> Optional[TopoDS_Wire]:
    """
    输入边的索引列表，输出一个经过几何修复的健康 Wire。

    :param edge_indices: 构成环的边索引列表，例如 [0, 5, 2]
    :param edges: 全局 TopoDS_Edge 列表
    :param tolerance: 缝合容差 (毫米)
    :return: 修复后的 TopoDS_Wire，如果输入为空则返回 None
    """
    if not len(edge_indices):
        return None

    wire_data = ShapeExtend_WireData()
    for idx in edge_indices:
        wire_data.Add(edges[idx])
    fixer = ShapeFix_Wire()
    fixer.Load(wire_data)
    fixer.SetPrecision(tolerance)
    fixer.SetMaxTolerance(tolerance * 10)
    fixer.SetMinTolerance(1e-5)
    fixer.SetFixReorderMode(1)  # 自动理顺边的连接顺序 (无视图论方向乱序)
    fixer.SetFixConnectedMode(1)  # 强行捏合微小的端点缝隙
    fixer.SetClosedWireMode(True)  # 确保首尾绝对闭合成环
    fixer.SetFixSelfIntersectionMode(1)  # 解开微小的自交
    fixer.Perform()
    return fixer.Wire()


def extract_topology_pure_graph(data):
    num_edges = len(data.edge_points)
    rows, cols = [], []

    def process_loop(edges):
        N = len(edges)
        if N == 0:
            return
        if N == 1:
            e = edges[0]
            rows.extend([2 * e, 2 * e + 1])
            cols.extend([2 * e + 1, 2 * e])
            return

        for i in range(N):
            e1, e2 = edges[i], edges[(i + 1) % N]
            p1, p2 = data.edge_points[e1][[0, -1]], data.edge_points[e2][[0, -1]]

            dist = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
            idx1, idx2 = np.unravel_index(dist.argmin(), dist.shape)

            n1, n2 = 2 * e1 + idx1, 2 * e2 + idx2
            rows.extend([n1, n2])
            cols.extend([n2, n1])

    for f in range(len(data.face_outer_offsets) - 1):
        process_loop(
            data.outer_edge_indices[
                data.face_outer_offsets[f] : data.face_outer_offsets[f + 1]
            ]
        )
        for li in range(data.face_inner_offsets[f], data.face_inner_offsets[f + 1]):
            process_loop(
                data.inner_edge_indices[
                    data.inner_loop_offsets[li] : data.inner_loop_offsets[li + 1]
                ]
            )

    adj_matrix = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(2 * num_edges, 2 * num_edges),
    )
    n_components, labels = connected_components(adj_matrix, directed=False)

    # 3. (Zero For-Loops)
    # 因为 labels 的一维索引严格对应 2*e + is_end，reshape(-1, 2) 会直接生成 [num_edges, 2] 的映射矩阵
    edge_vertex_adj = labels.reshape(-1, 2)

    # 获取所有端点坐标展平为 [2*num_edges, 3]
    all_pts = data.edge_points[:, [0, -1], :].reshape(-1, 3)

    # 向量化按聚类簇求均值
    sums = np.zeros((n_components, 3))
    np.add.at(sums, labels, all_pts)
    counts = np.bincount(labels, minlength=n_components)[:, None]

    unique_vertices = sums / counts

    return unique_vertices, edge_vertex_adj


def snap_edges_to_vertices(edge_points, unique_vertices, edge_vertex_adj):
    """
    将 32 个点的离散边，平滑地形变、吸附到全新的全局顶点上。

    :param edge_points: 原始边点云，形状 (N_edges, 32, 3)
    :param unique_vertices: 刚才求出的全局唯一顶点，形状 (N_vertices, 3)
    :param edge_vertex_adj: 边到顶点的映射矩阵，形状 (N_edges, 2)
    :return: 严丝合缝的全新边点云，形状 (N_edges, 32, 3)
    """
    N_edges, N_pts, _ = edge_points.shape

    # 1. 提取每条边【应该去】的目标起点和终点，形状 (N_edges, 3)
    target_starts = unique_vertices[edge_vertex_adj[:, 0]]
    target_ends = unique_vertices[edge_vertex_adj[:, 1]]

    # 2. 计算首尾的物理偏差向量 (Delta)
    delta_starts = target_starts - edge_points[:, 0, :]
    delta_ends = target_ends - edge_points[:, -1, :]

    # 3. 生成 0 到 1 的线性权重分布，形状 (32,)
    # 例如：[0., 0.032, 0.064, ..., 0.967, 1.]
    weights = np.linspace(0, 1, N_pts)

    # 4. 利用 NumPy 广播机制，生成整条曲线的形变补偿矩阵，形状 (N_edges, 32, 3)
    # 公式：Delta_i = (1 - w_i) * Delta_start + w_i * Delta_end
    curve_deformation = (
        delta_starts[:, np.newaxis, :] * (1 - weights)[np.newaxis, :, np.newaxis]
        + delta_ends[:, np.newaxis, :] * weights[np.newaxis, :, np.newaxis]
    )

    # 5. 将补偿矩阵加回原曲线
    snapped_edges = edge_points + curve_deformation

    return snapped_edges


def collect_face_outerwire_edges(face_idx, data) -> np.ndarray:
    os, oe = get_slice(data.face_outer_offsets, face_idx, len(data.outer_edge_indices))
    if os == oe:
        return np.array([], dtype=int)
    edge_idxes = data.outer_edge_indices[os:oe]
    return edge_idxes


def collect_face_innerwire_edges(face_idx, data) -> np.ndarray:
    is_, ie_ = get_slice(
        data.face_inner_offsets, face_idx, len(data.inner_loop_offsets)
    )
    if is_ == ie_:
        return np.array([], dtype=int)
    edge_idxes = []
    for j in range(is_, ie_):
        ls, le = get_slice(data.inner_loop_offsets, j, len(data.inner_edge_indices))
        edge_idxes.append(data.inner_edge_indices[ls:le])
    edge_idxes = np.concat(edge_idxes, dtype=int)
    return edge_idxes


def collect_face_edges_points(face_idx, edge_points, data) -> np.ndarray:
    edge_idxes = np.concat(
        [
            collect_face_outerwire_edges(face_idx, data),
            collect_face_innerwire_edges(face_idx, data),
        ],
        axis=0,
    )
    return edge_points[edge_idxes]


def pts2surface(points: np.ndarray) -> Geom_BSplineSurface:
    u_len, v_len = points.shape[:2]
    arr = TColgp_Array2OfPnt(1, u_len, 1, v_len)
    for i in range(u_len):
        for j in range(v_len):
            arr.SetValue(i + 1, j + 1, gp_Pnt(*map(float, points[i, j])))
    return GeomAPI_PointsToBSplineSurface(arr, 3, 8, GeomAbs_C2, 0.01).Surface()


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()  # noqa
        # assert ok


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()  # noqa
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face


def build_compsolid_from_face_soup(faces_list):
    """
    接收一堆可能包含内部隔断的面片，自动计算并生成 CompSolid 或 Solid。
    """
    volume_maker = BOPAlgo_MakerVolume()

    # 1. 把你所有的 TopoDS_Face 装进 OCCT 的列表里
    shape_list = TopTools_ListOfShape()
    for face in faces_list:
        shape_list.Append(face)

    # 2. 将这堆面设置为求解器的参数
    volume_maker.SetArguments(shape_list)

    # 3. 极其关键的开关：是否处理面的相交？
    # 如果你的面在边界处已经有了完美的 Edge 共享（咱们前面算出来的），设为 False 会极快。
    # 如果你怀疑面和面之间有穿模，设为 True，它会自动帮你切开并求交线。
    volume_maker.SetIntersect(False)

    # 4. 执行多面体空间分割算法 (The Magic Happens Here)
    volume_maker.Perform()

    # 5. 检查是否报错
    if volume_maker.HasErrors():
        print("致命错误：无法从这些面中寻找到任何封闭的体积！")
        # 在实际工程中，这里可以打印 volume_maker.GetReport() 来查错
        return None

    # 6. 获取最终的 3D 拓扑结构
    result_shape = volume_maker.Shape()
    global_fixer = ShapeFix_Shape(result_shape)
    global_fixer.SetPrecision(0.01)
    global_fixer.SetMaxTolerance(0.01)
    global_fixer.Perform()
    result_shape = global_fixer.Shape()
    # 诊断输出：看看它到底是个啥
    if result_shape.ShapeType() == TopAbs_COMPSOLID:
        print("🎉 成功：生成了一个复合实体 (CompSolid)！")
    elif result_shape.ShapeType() == TopAbs_SOLID:
        print("✅ 成功：生成了一个普通实体 (Solid)！")
    else:
        print(
            f"⚠️ 警告：生成的形状类型是 {result_shape.ShapeType()}，可能面没有完全闭合。"
        )

    return result_shape


if __name__ == "__main__":
    data = BRepData.load_npz("sample3.npz")

    # edge start
    unique_vertices, edge_vertex_adj = extract_topology_pure_graph(data)

    edge_points = snap_edges_to_vertices(
        data.edge_points, unique_vertices, edge_vertex_adj
    )

    surface_points = torch.from_numpy(data.face_points.copy()).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # [N_faces, 32, 32, 3]

    face_2_edge_points = [
        torch.from_numpy(
            collect_face_edges_points(f_idx, edge_points, data).reshape(-1, 3)
        ).to(device=surface_points.device, dtype=surface_points.dtype)
        for f_idx in range(len(data.face_points))
    ]  # [N_faces, np.array(N_edges_in_face * 32, 3)]

    with torch.no_grad():
        for i in range(len(surface_points)):
            e_p = face_2_edge_points[i]
            if len(e_p) == 0:
                continue
            s_p_flat = surface_points[i].reshape(-1, 3)
            # 1. 计算边框 (Cookie Cutter) 的物理包围盒与对角线尺度
            e_min, _ = torch.min(e_p, dim=0)
            e_max, _ = torch.max(e_p, dim=0)
            e_scale = torch.norm(e_max - e_min)

            # 2. 计算面团 (Cookie Dough) 的物理包围盒与对角线尺度
            s_min, _ = torch.min(s_p_flat, dim=0)
            s_max, _ = torch.max(s_p_flat, dim=0)
            s_center = (s_max + s_min) / 2.0
            s_scale = torch.norm(s_max - s_min)

            # 如果面不够大，强行以中心为原点放大，确保比边界大 5%
            if s_scale < 1.05 * e_scale:
                scale_factor = (1.05 * e_scale) / (s_scale + 1e-8)
                surface_points[i] = (
                    surface_points[i] - s_center
                ) * scale_factor + s_center

    surf_offset = nn.Parameter(
        torch.zeros(
            (len(surface_points), 1, 1, 3),
            device=surface_points.device,
            dtype=surface_points.dtype,
        )
    )
    optimizer = torch.optim.AdamW(
        [surf_offset], lr=1e-3, betas=(0.95, 0.999), weight_decay=1e-6
    )

    for _ in range(200):
        surf_updated = surface_points + surf_offset
        loss = 0
        for s_p, e_p in zip(surf_updated, face_2_edge_points):
            if len(e_p) == 0:
                continue
            s_p, e_p = s_p.reshape(-1, 3), e_p.reshape(-1, 3).detach()
            dist = torch.cdist(s_p, e_p, p=2)
            loss += dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()

        optimizer.zero_grad()
        (loss / len(surf_updated)).backward()
        optimizer.step()

    surface_points = (surface_points + surf_offset).detach().cpu().numpy()

    edges = [BRepBuilderAPI_MakeEdge(pts2curve(pts)).Edge() for pts in edge_points]

    ################# VISUAL #####################

    display, start_display, add_menu, add_function_to_menu = init_display()
    for v in unique_vertices:
        ver = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, v)))
        display.DisplayShape(ver.Vertex(), update=True)

    # edge finish

    # wire

    faces = []
    for f_idx in range(len(data.face_points)):
        edge_idxes_in_loop = collect_face_outerwire_edges(f_idx, data)
        out_wire = build_healed_wire(edge_idxes_in_loop, edges)
        in_w_b, in_w_e = (
            data.face_inner_offsets[f_idx],
            data.face_inner_offsets[f_idx + 1],
        )
        in_wire = []
        for li in range(in_w_b, in_w_e):
            edge_idxes_in_loop = data.inner_edge_indices[
                data.inner_loop_offsets[li] : data.inner_loop_offsets[li + 1]
            ]
            inner_wire = build_healed_wire(edge_idxes_in_loop, edges)
            if inner_wire is not None:
                in_wire.append(inner_wire)

        surf = pts2surface(surface_points[f_idx])
        face_maker = BRepBuilderAPI_MakeFace(surf, out_wire, True)
        for w in in_wire:
            w.Reverse()
            face_maker.Add(w)
        raw_face = face_maker.Face()
        fix_wires(raw_face)
        add_pcurves_to_edges(raw_face)
        fix_wires(raw_face)
        face = fix_face(raw_face)
        if is_bspline_cylinder(face, tol=0.01):
            print(
                f"Face {f_idx} is a cylinder, skipping sewing to preserve parametric form."
            )
        faces.append(face)

    sewing = BRepBuilderAPI_Sewing(0.01)
    for face in faces:
        sewing.Add(face)
    sewing.Perform()
    sewn_shape = sewing.SewedShape()
    if isinstance(sewn_shape, TopoDS_Shell):
        maker = BRepBuilderAPI_MakeSolid()
        maker.Add(sewn_shape)
        maker.Build()
        compound = maker.Solid()
    else:
        # Typically if we don't sew to make a shell
        # then we will create a compound
        compound = sewn_shape
    global_fixer = ShapeFix_Shape(compound)
    global_fixer.SetPrecision(0.01)
    global_fixer.SetMaxTolerance(0.01)
    global_fixer.Perform()
    final_perfect_solid = global_fixer.Shape()
    brc = BRepCheck_Analyzer(final_perfect_solid)
    is_valid = brc.IsValid()

    if not is_valid:
        print("模型无效！可能存在自交、未闭合或几何错误。")
        # 进一步获取具体的错误信息通常需要遍历模型的子元素
        # brc.State(shape) 可以返回具体的 BRepCheck_Status 枚举值
    else:
        print("模型有效。")
    display.DisplayShape(final_perfect_solid, update=True)
    display.FitAll()
    start_display()
