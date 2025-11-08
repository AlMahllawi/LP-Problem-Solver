"""
UI components for the Simplex method.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ..simplex import SimplexSolver
from ...utils import format_number


class SimplexSolverUI:
    """
    Manages the UI for the Simplex solving method.
    """

    def __init__(self, parent_frame, app):
        self.parent_frame = parent_frame
        self.app = app
        self.solver = None

        self.results_frame = ttk.Frame(self.parent_frame)
        self.results_frame.grid(row=2, column=0, sticky="nsew")
        self.results_frame.rowconfigure(1, weight=1)
        self.results_frame.columnconfigure(0, weight=1)

        self.tableau_canvas = tk.Canvas(self.results_frame)
        self.scrollable_frame = ttk.Frame(self.tableau_canvas)

        self.tableau_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )

        tableau_scroll_y = ttk.Scrollbar(
            self.results_frame,
            orient="vertical",
            command=self.tableau_canvas.yview,
        )
        tableau_scroll_x = ttk.Scrollbar(
            self.results_frame,
            orient="horizontal",
            command=self.tableau_canvas.xview,
        )
        self.tableau_canvas.configure(
            yscrollcommand=tableau_scroll_y.set, xscrollcommand=tableau_scroll_x.set
        )

        self.tableau_canvas.grid(row=1, column=0, sticky="nsew")
        tableau_scroll_y.grid(row=1, column=1, sticky="ns")
        tableau_scroll_x.grid(row=2, column=0, sticky="ew")

        self.final_results_frame = ttk.Frame(self.results_frame)
        self.final_results_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.tableau_canvas.configure(
                scrollregion=self.tableau_canvas.bbox("all")
            ),
        )

    def show(self):
        """Shows the simplex solver UI."""
        self.results_frame.grid()

    def hide(self):
        """Hides the simplex solver UI."""
        self.results_frame.grid_remove()

    def get_data(self):
        """Retrieves and validates data, returning it raw for Simplex solver."""
        try:
            if "c" not in self.app.entry_widgets:
                messagebox.showerror("Error", "Please click 'Setup Problem' first.")
                return (None,) * 6

            c = [float(e.get()) for e in self.app.entry_widgets["c"]]
            goal = self.app.goal.get()
            num_vars = self.app.num_vars.get()
            num_constraints = self.app.num_constraints.get()
            a, b, signs = [], [], []

            for i in range(num_constraints):
                row = [float(e.get()) for e in self.app.entry_widgets["a_ub"][i]]
                b_val = float(self.app.entry_widgets["b_ub"][i].get())
                sign = self.app.entry_widgets["signs"][i].get()
                a.append(row)
                b.append(b_val)
                signs.append(sign)

            return (np.array(c), np.array(a), np.array(b), signs, goal, num_vars)
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
        """Initializes the SimplexSolver, solves the problem, and displays the steps."""
        c, a, b, signs, goal, num_vars = self.get_data()
        if c is None:
            return

        self.solver = SimplexSolver(c, a, b, signs, goal, num_vars)
        try:
            self.solver.initialize_tableau()
        except ValueError as e:
            messagebox.showerror("Simplex Error", f"Failed to initialize tableau: {e}")
            return

        history = []
        # Initial state
        history.append(
            {
                "tableau": self.solver.tableau.copy(),
                "pivot": None,
                "basis_vars": list(self.solver.basis_vars),
                "var_names": list(self.solver.var_names),
                "phase": self.solver.phase,
            }
        )

        while not (
            self.solver.is_optimal
            or self.solver.is_unbounded
            or self.solver.is_infeasible
        ):
            pivot_info = self.solver.step()
            if history:
                history[-1]["pivot"] = pivot_info

            history.append(
                {
                    "tableau": self.solver.tableau.copy(),
                    "pivot": None,
                    "basis_vars": list(self.solver.basis_vars),
                    "var_names": list(self.solver.var_names),
                    "phase": self.solver.phase,
                }
            )

        self.display_steps(history)

    def display_steps(self, history):
        """Renders the sequence of Simplex tableaux with pivot highlighting."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        style = ttk.Style()
        style.configure("PivotRow.TFrame", background="lightyellow")
        style.configure("PivotCol.TFrame", background="lightblue")
        style.configure("PivotElement.TFrame", background="lightgreen")
        style.configure("PivotRow.TLabel", background="lightyellow")
        style.configure("PivotCol.TLabel", background="lightblue")
        style.configure("PivotElement.TLabel", background="lightgreen")

        previous_tableau = None
        for i, step_data in enumerate(history):
            tableau = step_data["tableau"].copy()
            if previous_tableau is not None and np.array_equal(
                tableau, previous_tableau
            ):
                continue
            previous_tableau = tableau.copy()
            pivot = step_data["pivot"]
            var_names = step_data["var_names"]
            basis_vars = step_data["basis_vars"]
            basis_var_names = [var_names[j] for j in basis_vars]

            step_frame = ttk.Frame(self.scrollable_frame, padding="10")
            step_frame.pack(fill="x", expand=True, pady=10, padx=5)

            num_cols = len(var_names) + 2
            for j in range(num_cols):
                step_frame.columnconfigure(j, weight=1)

            ttk.Label(step_frame, text=f"Tableau {i + 1}", style="Header.TLabel").grid(
                row=0, column=0, columnspan=num_cols, sticky="w"
            )

            header = ("Basis",) + tuple(var_names) + ("RHS",)
            for j, text in enumerate(header):
                cell_frame = ttk.Frame(step_frame, borderwidth=1, relief="solid")
                cell_frame.grid(row=1, column=j, sticky="nsew")
                lbl = ttk.Label(
                    cell_frame,
                    text=text,
                    font=("Helvetica", 16, "bold"),
                    anchor="center",
                )
                lbl.pack(expand=True, fill="both", padx=5, pady=2)

            for r in range(tableau.shape[0]):
                basis_text = ""
                is_obj_row = False
                if (step_data["phase"] == 1 and r == len(basis_var_names)) or (
                    step_data["phase"] == 2 and r == len(basis_var_names)
                ):
                    basis_text = f"{'-' if self.app.goal.get() == 'min' else ''}Z"
                    is_obj_row = True
                if step_data["phase"] == 1 and r == len(basis_var_names) + 1:
                    basis_text = "-W"
                    is_obj_row = True

                if not is_obj_row:
                    basis_text = basis_var_names[r]

                cell_frame = ttk.Frame(step_frame, borderwidth=1, relief="solid")
                cell_frame.grid(row=r + 2, column=0, sticky="nsew")
                lbl = ttk.Label(cell_frame, text=basis_text, anchor="center")
                lbl.pack(expand=True, fill="both", padx=5, pady=2)

                for c in range(tableau.shape[1]):
                    is_pivot_row = pivot and r == pivot["row"]
                    is_pivot_col = pivot and c == pivot["col"]
                    frame_style, label_style = "TFrame", "TLabel"

                    if is_pivot_row and is_pivot_col:
                        frame_style, label_style = (
                            "PivotElement.TFrame",
                            "PivotElement.TLabel",
                        )
                    elif is_pivot_row:
                        frame_style, label_style = "PivotRow.TFrame", "PivotRow.TLabel"
                    elif is_pivot_col:
                        frame_style, label_style = "PivotCol.TFrame", "PivotCol.TLabel"

                    cell_frame = ttk.Frame(
                        step_frame, style=frame_style, borderwidth=1, relief="solid"
                    )
                    cell_frame.grid(row=r + 2, column=c + 1, sticky="nsew")
                    val = tableau[r, c]
                    val_text = format_number(val)
                    lbl = ttk.Label(
                        cell_frame, text=val_text, style=label_style, anchor="e"
                    )
                    lbl.pack(expand=True, fill="both", padx=5, pady=2)

        self.display_final_solution()

    def display_final_solution(self):
        """Displays the final solution of the Simplex method."""
        for widget in self.final_results_frame.winfo_children():
            widget.destroy()

        if self.solver.is_optimal:
            final_z = self.solver.tableau[self.solver.z_row_index, -1]
            if self.app.goal.get() == "min":
                final_z = -final_z
            ttk.Label(
                self.final_results_frame,
                text="Optimal solution found!",
                style="Header.TLabel",
            ).pack(anchor="w")
            ttk.Label(
                self.final_results_frame, text=f"Optimal Z: {format_number(final_z)}"
            ).pack(anchor="w")
            ttk.Label(self.final_results_frame, text="Optimal Point:").pack(anchor="w")

            solution_vars = {}
            for i in range(self.solver.num_constraints):
                basis_var_index = self.solver.basis_vars[i]
                if basis_var_index < self.solver.num_vars:
                    solution_vars[basis_var_index] = self.solver.tableau[i, -1]

            for i in range(self.solver.num_vars):
                var_name = self.solver.var_names[i]
                val = solution_vars.get(i, 0.0)
                ttk.Label(
                    self.final_results_frame,
                    text=f"  {var_name} = {format_number(val)}",
                ).pack(anchor="w")

        elif self.solver.is_unbounded:
            ttk.Label(
                self.final_results_frame,
                text="Problem is UNBOUNDED.",
                style="Header.TLabel",
            ).pack(anchor="w")
        elif self.solver.is_infeasible:
            ttk.Label(
                self.final_results_frame,
                text="Problem is INFEASIBLE.",
                style="Header.TLabel",
            ).pack(anchor="w")
