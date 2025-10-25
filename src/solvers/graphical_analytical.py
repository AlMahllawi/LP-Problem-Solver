from itertools import combinations
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# pylint: disable=no-name-in-module, ungrouped-imports
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from ..utils import to_subscript


class GraphicalAnalyticalSolver:
    """
    Manages the state and logic for the Graphical solving method.
    """

    def __init__(self, c, a_ub, b_ub, a_eq, b_eq, goal):
        self.c = c
        self.a_ub = a_ub
        self.b_ub = b_ub
        self.a_eq = a_eq
        self.b_eq = b_eq
        self.goal = goal
        self.c_original = self.c.copy()

    def solve(self):
        """Solves the LP problem using scipy.linprog."""
        c_to_solve = -self.c_original if self.goal == "max" else self.c_original
        bounds = [(0, None)] * len(self.c)

        a_ub_combined = self.a_ub.copy()
        b_ub_combined = self.b_ub.copy()

        if self.a_eq is not None and self.a_eq.size > 0:
            a_ub_combined = np.vstack([self.a_ub, self.a_eq, -self.a_eq])
            b_ub_combined = np.concatenate([self.b_ub, self.b_eq, -self.b_eq])

        result = linprog(
            c_to_solve,
            A_ub=self.a_ub if self.a_ub.size else None,
            b_ub=self.b_ub if self.b_ub.size else None,
            A_eq=self.a_eq if self.a_eq.size else None,
            b_eq=self.b_eq if self.b_eq.size else None,
            bounds=bounds,
            method="highs",
        )

        feasible_points = None
        if len(self.c) == 2:
            intersections = self.find_intersections(a_ub_combined, b_ub_combined)
            feasible_points = self.find_feasible_points(
                intersections, a_ub_combined, b_ub_combined
            )
        elif len(self.c) == 3:
            intersections = self.find_intersections_3d(a_ub_combined, b_ub_combined)
            feasible_points = self.find_feasible_points_3d(
                intersections, a_ub_combined, b_ub_combined
            )

        return result, feasible_points, a_ub_combined, b_ub_combined

    def find_intersections(self, a_ub, b_ub):
        """Finds intersection points of constraint lines for a 2-variable problem."""
        if a_ub.size == 0:
            return []
        axis_lines = [(np.array([1, 0]), 0), (np.array([0, 1]), 0)]
        lines = list(zip(a_ub, b_ub)) + axis_lines
        points = set()
        for (a1, b1), (a2, b2) in combinations(lines, 2):
            try:
                p = np.linalg.solve([a1, a2], [b1, b2])
                if np.all(p >= -1e-9):
                    points.add(tuple(np.round(p, 6).astype(float)))
            except np.linalg.LinAlgError:
                continue
        return sorted(list(points))

    def find_feasible_points(self, intersections, a_ub, b_ub):
        """Checks which intersection points are feasible."""
        feasible_points = []
        if not intersections and a_ub.size == 0:
            pass

        if intersections:
            for px, py in intersections:
                if px >= -1e-6 and py >= -1e-6:
                    valid = True
                    if a_ub.size > 0:
                        for i, a in enumerate(a_ub):
                            if np.dot(a, [px, py]) - b_ub[i] > 1e-6:
                                valid = False
                                break
                    if valid:
                        feasible_points.append([px, py])

        origin_valid = True
        if a_ub.size > 0:
            if np.any(b_ub < -1e-6):
                origin_valid = False

        if origin_valid:
            if not any(np.allclose([0, 0], p) for p in feasible_points):
                feasible_points.append([0, 0])

        if feasible_points:
            feasible_points_np = np.array(feasible_points).round(decimals=6)
            feasible_points = np.unique(feasible_points_np, axis=0)

        return np.array(feasible_points)

    def find_intersections_3d(self, a_ub, b_ub):
        """Finds intersection points of constraint planes for a 3-variable problem."""
        if a_ub.size == 0:
            return []
        axis_planes = [
            (np.array([1, 0, 0]), 0),
            (np.array([0, 1, 0]), 0),
            (np.array([0, 0, 1]), 0),
        ]
        planes = list(zip(a_ub, b_ub)) + axis_planes
        points = set()
        for (a1, b1), (a2, b2), (a3, b3) in combinations(planes, 3):
            try:
                a_matrix = np.array([a1, a2, a3])
                b_vector = np.array([b1, b2, b3])
                p = np.linalg.solve(a_matrix, b_vector)
                if np.all(p >= -1e-9):
                    points.add(tuple(np.round(p, 6).astype(float)))
            except np.linalg.LinAlgError:
                continue
        return sorted(list(points))

    def find_feasible_points_3d(self, intersections, a_ub, b_ub):
        """Checks which intersection points are feasible for a 3-variable problem."""
        feasible_points = []
        if not intersections and a_ub.size == 0:
            pass

        if intersections:
            for p in intersections:
                px, py, pz = p[0], p[1], p[2]
                if px >= -1e-6 and py >= -1e-6 and pz >= -1e-6:
                    valid = True
                    if a_ub.size > 0:
                        for i, a in enumerate(a_ub):
                            if np.dot(a, [px, py, pz]) - b_ub[i] > 1e-6:
                                valid = False
                                break
                    if valid:
                        feasible_points.append([px, py, pz])

        origin_valid = True
        if a_ub.size > 0:
            if np.any(b_ub < -1e-6):
                origin_valid = False

        if origin_valid:
            if not any(np.allclose([0, 0, 0], p) for p in feasible_points):
                feasible_points.append([0, 0, 0])

        if feasible_points:
            feasible_points_np = np.array(feasible_points).round(decimals=6)
            feasible_points = np.unique(feasible_points_np, axis=0)

        return np.array(feasible_points)

    def plot_problem(self, a_ub, b_ub, result, feasible_points):
        """Plots the feasible region and solution for a 2-variable LP problem."""
        if (
            a_ub.size == 0
            and (feasible_points is None or feasible_points.size == 0)
            and not result.success
        ):
            return False

        _, ax = plt.subplots(figsize=(8, 8))

        ax.set_title("Feasible Region & Optimal Point", fontsize=18)
        ax.set_xlabel("x₁", fontsize=16)
        ax.set_ylabel("x₂", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        if feasible_points is not None and feasible_points.size > 0:
            max_val = np.max(feasible_points) * 1.5
            plot_limit = max(max_val, 10)
        else:
            plot_limit = max(b_ub.max() if len(b_ub) > 0 else 10, 10) * 1.2

        ax.set_xlim(-plot_limit * 0.05, plot_limit)
        ax.set_ylim(-plot_limit * 0.05, plot_limit)

        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(a_ub)))
        d = np.linspace(-plot_limit, plot_limit, 400)

        for i, (a1, a2) in enumerate(a_ub):
            label = f"C{to_subscript(i+1)}"
            if np.isclose(a2, 0):
                if not np.isclose(a1, 0):
                    ax.axvline(
                        b_ub[i] / a1, color=colors[i], linewidth=1.8, label=label
                    )
            else:
                y = (b_ub[i] - a1 * d) / a2
                ax.plot(d, y, color=colors[i], linewidth=1.8, label=label)

        if feasible_points is not None:
            points_for_hull = list(feasible_points)
        else:
            points_for_hull = []

        boundary_points_to_check = []

        plot_corners = [[plot_limit, 0], [0, plot_limit], [plot_limit, plot_limit]]
        boundary_points_to_check.extend(plot_corners)

        for i, (a1, a2) in enumerate(a_ub):
            b_i = b_ub[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                p_y_axis = [0, b_i / a2]
                if 0 <= p_y_axis[1] <= plot_limit:
                    boundary_points_to_check.append(p_y_axis)

                p_x_axis = [b_i / a1, 0]
                if 0 <= p_x_axis[0] <= plot_limit:
                    boundary_points_to_check.append(p_x_axis)

                p_right_edge = [plot_limit, (b_i - a1 * plot_limit) / a2]
                if 0 <= p_right_edge[1] <= plot_limit:
                    boundary_points_to_check.append(p_right_edge)

                p_top_edge = [(b_i - a2 * plot_limit) / a1, plot_limit]
                if 0 <= p_top_edge[0] <= plot_limit:
                    boundary_points_to_check.append(p_top_edge)

        cleaned_boundary_points = []
        for p in boundary_points_to_check:
            if np.all(np.isfinite(p)):
                cleaned_boundary_points.append(p)

        for p in cleaned_boundary_points:
            point = np.array(p)
            is_feasible = True

            if point[0] < -1e-6 or point[1] < -1e-6:
                is_feasible = False
                continue

            if a_ub.size > 0:
                for i, a in enumerate(a_ub):
                    if np.dot(a, point) - b_ub[i] > 1e-6:
                        is_feasible = False
                        break

            if is_feasible:
                points_for_hull.append(list(point))

        if not points_for_hull:
            all_hull_points = np.array([])
        else:
            all_hull_points = np.array(points_for_hull).round(decimals=6)
            all_hull_points = np.unique(all_hull_points, axis=0)

        if feasible_points is not None and feasible_points.size > 0:
            x_offset = plot_limit * 0.015
            y_offset = plot_limit * 0.015

            for i, (px, py) in enumerate(feasible_points):
                ax.plot(px, py, "bo")
                label = f"P{to_subscript(i+1)}"
                ax.text(
                    px + x_offset, py + y_offset, label, fontsize=12, color="darkblue"
                )

        if all_hull_points.shape[0] >= 3:
            try:
                hull = ConvexHull(all_hull_points)
                polygon = all_hull_points[hull.vertices]
                patch = patches.Polygon(
                    polygon, facecolor="green", alpha=0.25, label="Feasible Region"
                )
                ax.add_patch(patch)
            except QhullError as e:
                print(f"Could not compute Convex Hull: {e}")
        elif all_hull_points.shape[0] == 2:
            ax.plot(
                all_hull_points[:, 0],
                all_hull_points[:, 1],
                "g-",
                linewidth=5,
                alpha=0.3,
                label="Feasible Region",
            )
        elif all_hull_points.shape[0] == 1:
            ax.plot(
                all_hull_points[0, 0],
                all_hull_points[0, 1],
                "go",
                markersize=15,
                alpha=0.3,
                label="Feasible Region",
            )

        if result.success and len(result.x) >= 2:
            x_sol, y_sol = result.x[0], result.x[1]
            ax.plot(x_sol, y_sol, "r*", markersize=14, label="Optimal Solution")

        ax.legend(loc="upper right", fontsize=16)
        plt.show()
        return True

    def plot_3d_problem(self, a_ub, b_ub, result, feasible_points):
        """Plots the feasible region and solution for a 3-variable LP problem."""
        if (
            a_ub.size == 0
            and (feasible_points is None or feasible_points.size == 0)
            and not result.success
        ):
            return False

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_title("Feasible Region & Optimal Point", fontsize=18)
        ax.set_xlabel("x₁", fontsize=16)
        ax.set_ylabel("x₂", fontsize=16)
        ax.set_zlabel("x₃", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)

        if feasible_points is not None and feasible_points.size > 0:
            max_val = np.max(feasible_points) * 1.5
            plot_limit = max(max_val, 10)
        else:
            plot_limit = max(b_ub.max() if b_ub.size > 0 else 10, 10) * 1.2

        ax.set_xlim(0, plot_limit)
        ax.set_ylim(0, plot_limit)
        ax.set_zlim(0, plot_limit)

        d = np.linspace(0, plot_limit, 10)
        xx, yy = np.meshgrid(d, d)

        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(a_ub)))

        for i, (a1, a2, a3) in enumerate(a_ub):
            with np.errstate(divide="ignore", invalid="ignore"):
                color = colors[i % len(colors)]
                if not np.isclose(a3, 0):
                    zz = (b_ub[i] - a1 * xx - a2 * yy) / a3
                    ax.plot_surface(xx, yy, zz, alpha=0.2, color=color)
                elif not np.isclose(a2, 0):
                    xx_p, zz_p = np.meshgrid(d, d)
                    yy_p = (b_ub[i] - a1 * xx_p - a3 * zz_p) / a2
                    ax.plot_surface(xx_p, yy_p, zz_p, alpha=0.2, color=color)
                elif not np.isclose(a1, 0):
                    yy_p, zz_p = np.meshgrid(d, d)
                    xx_p = (b_ub[i] - a2 * yy_p - a3 * zz_p) / a1
                    ax.plot_surface(xx_p, yy_p, zz_p, alpha=0.2, color=color)

        if feasible_points is not None and feasible_points.shape[0] >= 4:
            try:
                hull = ConvexHull(feasible_points)
                for simplex in hull.simplices:
                    triangle = feasible_points[simplex]
                    poly = Poly3DCollection(
                        [triangle], alpha=0.25, facecolor="g", edgecolor="k"
                    )
                    ax.add_collection3d(poly)
            except QhullError as e:
                print(f"Could not compute Convex Hull for 3D plot: {e}")

        if feasible_points is not None and feasible_points.size > 0:
            ax.scatter(
                feasible_points[:, 0],
                feasible_points[:, 1],
                feasible_points[:, 2],
                c="b",
                marker="o",
                s=30,
                label="Feasible Vertices",
            )
            for i, p in enumerate(feasible_points):
                ax.text(
                    p[0],
                    p[1],
                    p[2],
                    f" P{to_subscript(i+1)}",
                    size=10,
                    zorder=1,
                    color="darkblue",
                )

        if result.success and len(result.x) >= 3:
            x_sol, y_sol, z_sol = result.x[0], result.x[1], result.x[2]
            ax.scatter(
                x_sol,
                y_sol,
                z_sol,
                c="r",
                marker="*",
                s=200,
                label="Optimal Solution",
                zorder=10,
            )

        ax.legend(loc="upper right", fontsize=12)
        plt.show()
        return True
