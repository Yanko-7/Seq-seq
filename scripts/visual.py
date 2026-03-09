"""
A simple class to visualize point grids using matplotlib.

We could try to avoid explicit dependency on
pytorch and Open Cascade in here
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from src.tokenizer import BRepData


class PointGridVisualizer:
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize

    def visualize(
        self,
        brep_data: BRepData,
        save_path: str = "brep_visualization.png",
        title="B-Rep Point Grid",
    ):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        all_points = []

        # 1. 渲染面 (Faces)
        num_faces = len(brep_data.face_points)
        colors = cm.tab20(np.linspace(0, 1, max(num_faces, 1)))

        for i, face_grid in enumerate(brep_data.face_points):
            pts = face_grid.reshape(-1, 3)
            if len(pts) == 0:
                continue

            all_points.append(pts)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[colors[i % 20]],
                s=5,
                alpha=0.6,
                edgecolors="none",
            )

        # 2. 渲染边 (Edges)
        for edge_curve in brep_data.edge_points:
            pts = edge_curve.reshape(-1, 3)
            if len(pts) == 0:
                continue

            all_points.append(pts)
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2], color="black", linewidth=2.0, alpha=0.9
            )

        # 3. 统一坐标轴比例 (等比例显示)
        if all_points:
            concat_pts = np.vstack(all_points)
            min_pt, max_pt = concat_pts.min(axis=0), concat_pts.max(axis=0)
            center = (max_pt + min_pt) / 2
            max_range = (max_pt - min_pt).max() / 2.0

            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            ax.set_box_aspect([1, 1, 1])

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()

        # 核心修改：保存高分辨率图片并关闭画布
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
