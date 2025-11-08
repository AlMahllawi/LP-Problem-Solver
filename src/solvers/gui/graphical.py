"""
UI components for the Graphical and Analytical method.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ..graphical import GraphicalAnalyticalSolver
from ...utils import to_subscript, format_number


class GraphicalSolverUI:
    """
    Manages the UI for the graphical and analytical solving method.
    """

    def __init__(self, parent_frame, app):
        self.parent_frame = parent_frame
        self.app = app
        self.c_original = None

        self.results_frame = ttk.Frame(self.parent_frame)
        self.results_frame.grid(row=2, column=0, sticky="nsew")
        self.results_frame.rowconfigure(0, weight=1)
        self.results_frame.columnconfigure(0, weight=1)

        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(
            self.results_frame,
            orient="vertical",
            command=self.results_canvas.yview,
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self.results_canvas.configure(yscrollcommand=scroll.set)

        self.scrollable_frame = ttk.Frame(self.results_canvas)
        self.results_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            ),
        )

    def show(self):
        """Shows the graphical solver UI."""
        self.results_frame.grid()

    def hide(self):
        """Hides the graphical solver UI."""
        self.results_frame.grid_remove()

    def get_data(self):
        """Retrieves and validates data, formatting it for scipy.linprog."""
        try:
            if "c" not in self.app.entry_widgets:
                messagebox.showerror("Error", "Please click 'Setup Problem' first.")
                return (None,) * 6

            c = [float(e.get()) for e in self.app.entry_widgets["c"]]
            self.c_original = np.array(c)
            goal = self.app.goal.get()
            num_constraints = self.app.num_constraints.get()
            a_ub, b_ub, a_eq, b_eq = [], [], [], []

            for i in range(num_constraints):
                row = [float(e.get()) for e in self.app.entry_widgets["a_ub"][i]]
                b_val = float(self.app.entry_widgets["b_ub"][i].get())
                sign = self.app.entry_widgets["signs"][i].get()

                if sign == "≤":
                    a_ub.append(row)
                    b_ub.append(b_val)
                elif sign == "≥":
                    a_ub.append([-x for x in row])
                    b_ub.append(-b_val)
                elif sign == "=":
                    a_eq.append(row)
                    b_eq.append(b_val)

            return (
                np.array(c),
                np.array(a_ub),
                np.array(b_ub),
                np.array(a_eq),
                np.array(b_eq),
                goal,
            )
        except (ValueError, tk.TclError):
            messagebox.showerror(
                "Input Error", "Please fill all numeric fields correctly."
            )
            return (None,) * 6
        except (AttributeError, KeyError):
            messagebox.showerror(
                "Error", "Please click 'Setup Problem' to generate fields first."
            )
            return (None,) * 6

    def solve(self):
        """Solves the LP problem using scipy.linprog and plots."""
        c, a_ub, b_ub, a_eq, b_eq, goal = self.get_data()
        if c is None:
            return

        solver = GraphicalAnalyticalSolver(c, a_ub, b_ub, a_eq, b_eq, goal)
        result, feasible_points, a_ub_combined, b_ub_combined = solver.solve()

        self.display_results(result, goal, feasible_points)

        if len(c) in [2, 3]:
            if not solver.plot_problem(
                a_ub_combined, b_ub_combined, result, feasible_points
            ):
                messagebox.showinfo(
                    "Plot Info", "Cannot plot an unbounded problem with no constraints."
                )
        elif len(c) > 3:
            messagebox.showinfo(
                "Plot Info", "Plotting is only supported for 2 or 3 variables."
            )

    def display_results(self, result, goal, feasible_points):
        """Formats and displays the solution using labels."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        container = self.scrollable_frame
        if result.success:
            val = -result.fun if goal == "max" else result.fun
            ttk.Label(container, text="Solution Found!", style="Header.TLabel").pack(
                anchor="w", pady=(0, 10)
            )
            ttk.Label(container, text=f"Optimal Z: {format_number(val)}").pack(
                anchor="w"
            )
            ttk.Label(container, text="Optimal Point:").pack(anchor="w", pady=(5, 0))
            for i, v in enumerate(result.x):
                ttk.Label(
                    container, text=f"  x{to_subscript(i + 1)} = {format_number(v)}"
                ).pack(anchor="w")
            optimal_x = np.round(result.x, 6)
        else:
            ttk.Label(container, text=result.message, style="Header.TLabel").pack(
                anchor="w"
            )
            optimal_x = None

        if feasible_points is not None and feasible_points.size > 0:
            ttk.Separator(container, orient="horizontal").pack(
                fill="x", pady=20, padx=20
            )
            ttk.Label(
                container, text="Feasible Points & Z-Values:", style="Header.TLabel"
            ).pack(anchor="w", pady=(0, 10))

            for i, p in enumerate(feasible_points):
                if p.ndim != 1:
                    continue

                point_vec = p
                z_val = np.dot(self.c_original, point_vec)
                label = f"P{to_subscript(i + 1)}"
                point_coords = ", ".join([format_number(coord) for coord in p])
                point_str = f"{label}: ({point_coords})"
                z_str = f"Z = {format_number(z_val)}"
                is_optimal = optimal_x is not None and np.allclose(point_vec, optimal_x)

                point_frame = ttk.Frame(container)
                point_frame.pack(fill="x", padx=5, pady=2)
                lbl_style = "Optimal.TLabel" if is_optimal else "Feasible.TLabel"
                main_lbl = ttk.Label(
                    point_frame, text=f"{point_str:<40} {z_str}", style=lbl_style
                )
                main_lbl.pack(side="left", fill="x", expand=True)

                if is_optimal:
                    ttk.Label(
                        point_frame,
                        text="  (Optimal)",
                        style=lbl_style,
                        font=("Courier", 16, "bold"),
                    ).pack(side="left")
