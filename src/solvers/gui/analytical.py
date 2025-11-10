"""
UI components for the Analytical method.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ..analytical import AnalyticalSolver
from ...utils import format_number


class AnalyticalSolverUI:
    """
    Manages the UI for the Analytical solving method.
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

        table_scroll_y = ttk.Scrollbar(
            self.results_frame,
            orient="vertical",
            command=self.tableau_canvas.yview,
        )
        table_scroll_x = ttk.Scrollbar(
            self.results_frame,
            orient="horizontal",
            command=self.tableau_canvas.xview,
        )
        self.tableau_canvas.configure(
            yscrollcommand=table_scroll_y.set, xscrollcommand=table_scroll_x.set
        )

        self.tableau_canvas.grid(row=1, column=0, sticky="nsew")
        table_scroll_y.grid(row=1, column=1, sticky="ns")
        table_scroll_x.grid(row=2, column=0, sticky="ew")

        self.final_results_frame = ttk.Frame(self.results_frame)
        self.final_results_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.tableau_canvas.configure(
                scrollregion=self.tableau_canvas.bbox("all")
            ),
        )

    def show(self):
        """Shows the analytical solver UI."""
        self.results_frame.grid()

    def hide(self):
        """Hides the analytical solver UI."""
        self.results_frame.grid_remove()

    def get_data(self):
        """Retrieves and validates data, returning it raw for Analytical solver."""
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

            return (np.array(c), np.array(a), np.array(b), np.array(signs), goal, num_vars)
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
        """Initializes the AnalyticalSolver, solves the problem, and displays the results."""
        c, a, b, signs, goal, num_vars = self.get_data()
        if c is None:
            return

        self.solver = AnalyticalSolver(c, a, b, signs, goal, num_vars)
        try:
            self.solver.solve()
        except (ValueError, np.linalg.LinAlgError) as e:
            messagebox.showerror("Solver Error", f"An error occurred during solving: {e}")
            return

        self.display_results_table()
        self.display_final_solution()

    def display_results_table(self):
        """Renders the table of basic feasible solutions."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not self.solver.solutions:
            ttk.Label(self.scrollable_frame, text="No feasible solutions found.").pack()
            return

        num_cols = len(self.solver.var_names) + 1
        for j in range(num_cols):
            self.scrollable_frame.columnconfigure(j, weight=1)

        # --- Table Header ---
        header_labels = self.solver.var_names + ["Z"]
        for i, text in enumerate(header_labels):
            cell_frame = ttk.Frame(self.scrollable_frame, borderwidth=1, relief="solid")
            cell_frame.grid(row=0, column=i, sticky="nsew")
            lbl = ttk.Label(
                cell_frame,
                text=text,
                font=("Helvetica", 14, "bold"),
                anchor="center",
            )
            lbl.pack(expand=True, fill="both", padx=5, pady=2)

        # --- Table Rows ---
        for i, sol_data in enumerate(self.solver.solutions):
            row_num = i + 1
            # Variables
            for j, var_val in enumerate(sol_data["vars"]):
                cell_frame = ttk.Frame(self.scrollable_frame, borderwidth=1, relief="solid")
                cell_frame.grid(row=row_num, column=j, sticky="nsew")
                lbl = ttk.Label(
                    cell_frame, text=format_number(var_val), anchor="e"
                )
                lbl.pack(expand=True, fill="both", padx=5, pady=2)
            
            # Z value
            cell_frame = ttk.Frame(self.scrollable_frame, borderwidth=1, relief="solid")
            cell_frame.grid(row=row_num, column=num_cols - 1, sticky="nsew")
            lbl = ttk.Label(
                cell_frame, text=format_number(sol_data["z"]), anchor="e"
            )
            lbl.pack(expand=True, fill="both", padx=5, pady=2)


    def display_final_solution(self):
        """Displays the optimal solution found."""
        for widget in self.final_results_frame.winfo_children():
            widget.destroy()

        if self.solver.is_optimal and self.solver.optimal_solution:
            opt_sol = self.solver.optimal_solution
            
            ttk.Label(
                self.final_results_frame,
                text="Optimal Solution:",
                style="Header.TLabel",
            ).pack(anchor="w")
            
            ttk.Label(
                self.final_results_frame, text=f"Optimal Z: {format_number(opt_sol['z'])}"
            ).pack(anchor="w")

            ttk.Label(self.final_results_frame, text="Optimal Point:").pack(anchor="w")
            
            for i in range(self.solver.num_vars):
                var_name = self.solver.var_names[i]
                val = opt_sol["vars"][i]
                ttk.Label(
                    self.final_results_frame,
                    text=f"  {var_name} = {format_number(val)}",
                ).pack(anchor="w")
        else:
            ttk.Label(
                self.final_results_frame,
                text="No optimal solution found.",
                style="Header.TLabel",
            ).pack(anchor="w")