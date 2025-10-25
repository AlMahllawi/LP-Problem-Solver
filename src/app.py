"""
A GUI application for solving Linear Programming (LP) problems
using Tkinter and SciPy's linprog solver (for graphical method)
or a custom Simplex algorithm implementation.

It supports 2-variable problems (graphically) and n-variable problems
(analytically/simplex), with non-negativity constraints.
It provides a graphical visualization for 2/3-variable problems
and an iterable Simplex tableau for the Simplex method.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from .solvers.graphical_analytical import GraphicalAnalyticalSolver
from .solvers.simplex import SimplexSolver
from .utils import to_subscript, format_number


class App:
    """
    Main class for the Linear Programming Solver GUI application.
    Manages layout, user input, problem solving, and result visualization.
    """

    def __init__(self, master):
        """Initializes the GUI application."""
        self.master = master
        self.master.title("Linear Programming Problem Solver")
        self.master.geometry("1280x720")

        # --- Style Configuration ---
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 13))
        style.configure("TRadiobutton", font=("Helvetica", 13))
        style.configure("TButton", font=("Helvetica", 13, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 17, "bold"))
        style.configure("Optimal.TLabel", background="lightgreen")
        style.configure("Feasible.TLabel", background="lightyellow")
        style.configure("PivotRow.TLabel", background="lightyellow")
        style.configure("PivotCol.TLabel", background="lightblue")
        style.configure("PivotElement.TLabel", background="lightgreen")
        style.configure(
            "Normal.TLabel", background=style.lookup("TFrame", "background")
        )

        # --- Initialize Attributes ---
        self.paned_window = None
        self.left_pane = None
        self.main_canvas = None
        self.scrollable_frame = None
        self.results_frame = None
        self.setup_frame = None
        self.matrix_frame = None
        self.graphical_results_frame = None
        self.simplex_results_frame = None
        self.graphical_results_canvas = None
        self.graphical_results_scrollable_frame = None
        self.tableau_canvas = None
        self.tableau_scrollable_frame = None
        self.simplex_final_results_frame = None
        self.simplex_solver = None
        self.entry_widgets = {}

        # --- State Variables ---
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)
        self.goal = tk.StringVar(value="max")
        self.solve_method = tk.StringVar(value="graphical")

        # --- Main Layout ---
        self.paned_window = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True)

        self.left_pane = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(self.left_pane, weight=1)

        # --- Scrollable Frame Setup ---
        self.main_canvas = tk.Canvas(self.left_pane)
        v_scrollbar = ttk.Scrollbar(
            self.left_pane, orient="vertical", command=self.main_canvas.yview
        )
        h_scrollbar = ttk.Scrollbar(
            self.left_pane, orient="horizontal", command=self.main_canvas.xview
        )
        self.main_canvas.configure(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            ),
        )
        self.main_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        self.master.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- Results ---
        self.results_frame = ttk.Frame(self.paned_window, padding="15")
        self.paned_window.add(self.results_frame, weight=1)

        # --- Input Frames ---
        self.setup_frame = ttk.Frame(self.scrollable_frame, padding="15")
        self.setup_frame.pack(fill="x", padx=10, pady=5)

        self.matrix_frame = ttk.Frame(self.scrollable_frame, padding="15")
        self.matrix_frame.pack(fill="x", padx=10, pady=5)

        # --- Create Widgets ---
        self.create_setup_widgets()
        self.create_results_widgets()
        self.toggle_results_view()

    def _on_mousewheel(self, event):
        """
        Handles mouse wheel or trackpad scrolling events to move the main canvas.
        This function provides cross-platform support for scrolling.
        """
        if event.delta:
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def create_setup_widgets(self):
        """Creates the initial widgets for problem setup."""

        # --- Solving Method ---
        ttk.Label(self.setup_frame, text="Solving Method:", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Radiobutton(
            self.setup_frame,
            text="Graphically & Analytically",
            variable=self.solve_method,
            value="graphical",
            command=self.toggle_results_view,
        ).grid(row=1, column=0, sticky="w")

        ttk.Radiobutton(
            self.setup_frame,
            text="Simplex",
            variable=self.solve_method,
            value="simplex",
            command=self.toggle_results_view,
        ).grid(row=1, column=1, sticky="w")

        # --- Goal ---
        ttk.Label(
            self.setup_frame, text="Optimization Goal:", style="Header.TLabel"
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(15, 10))

        ttk.Radiobutton(
            self.setup_frame, text="Maximize", variable=self.goal, value="max"
        ).grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(
            self.setup_frame, text="Minimize", variable=self.goal, value="min"
        ).grid(row=3, column=1, sticky="w")

        # --- Variables and Constraints ---
        ttk.Label(self.setup_frame, text="Number of Variables:").grid(
            row=4, column=0, sticky="w", pady=(15, 5)
        )
        ttk.Entry(self.setup_frame, textvariable=self.num_vars, width=8).grid(
            row=4, column=1, sticky="w"
        )

        ttk.Label(self.setup_frame, text="Number of Constraints:").grid(
            row=5, column=0, sticky="w", pady=5
        )
        ttk.Entry(self.setup_frame, textvariable=self.num_constraints, width=8).grid(
            row=5, column=1, sticky="w"
        )

        # --- Setup Button ---
        ttk.Button(
            self.setup_frame, text="Setup Problem", command=self.create_matrix_entries
        ).grid(row=6, column=0, columnspan=2, pady=20)

    def create_matrix_entries(self):
        """Dynamically creates entry fields for objective function and constraints."""

        if hasattr(self, "graphical_results_scrollable_frame"):
            for widget in self.graphical_results_scrollable_frame.winfo_children():
                widget.destroy()
        if hasattr(self, "simplex_final_results_frame"):
            for widget in self.simplex_final_results_frame.winfo_children():
                widget.destroy()
        if hasattr(self, "tableau_scrollable_frame"):
            for widget in self.tableau_scrollable_frame.winfo_children():
                widget.destroy()

        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        try:
            num_vars = self.num_vars.get()
            num_constraints = self.num_constraints.get()
            if num_vars <= 0 or num_constraints <= 0:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showerror(
                "Input Error",
                "Please enter positive integers for variables and constraints.",
            )
            return

        ttk.Label(
            self.matrix_frame, text="Objective Function", style="Header.TLabel"
        ).grid(row=0, column=0, columnspan=num_vars + 1, sticky="w", pady=5)
        self.entry_widgets["c"] = []
        for j in range(num_vars):
            label = f"x{to_subscript(j+1)}"
            ttk.Label(self.matrix_frame, text=label).grid(row=1, column=j + 1)
            entry = ttk.Entry(self.matrix_frame, width=8)
            entry.grid(row=2, column=j + 1)
            self.entry_widgets["c"].append(entry)

        ttk.Label(self.matrix_frame, text="Constraints", style="Header.TLabel").grid(
            row=3, column=0, columnspan=num_vars + 3, sticky="w", pady=(20, 5)
        )
        (
            self.entry_widgets["a_ub"],
            self.entry_widgets["b_ub"],
            self.entry_widgets["signs"],
        ) = ([], [], [])

        for j in range(num_vars):
            label = f"x{to_subscript(j+1)}"
            ttk.Label(self.matrix_frame, text=label).grid(row=4, column=j + 1)
        ttk.Label(self.matrix_frame, text="RHS").grid(row=4, column=num_vars + 2)

        for i in range(num_constraints):
            row_entries = []
            ttk.Label(self.matrix_frame, text=f"C{to_subscript(i+1)}:").grid(
                row=i + 5, column=0, sticky="e", padx=(0, 5)
            )
            for j in range(num_vars):
                entry = ttk.Entry(self.matrix_frame, width=8)
                entry.grid(row=i + 5, column=j + 1)
                row_entries.append(entry)

            sign = ttk.Combobox(
                self.matrix_frame,
                values=["≤", "≥", "="],
                width=5,
                state="readonly",
            )
            sign.set("≤")
            sign.grid(row=i + 5, column=num_vars + 1)
            self.entry_widgets["signs"].append(sign)
            b_entry = ttk.Entry(self.matrix_frame, width=8)
            b_entry.grid(row=i + 5, column=num_vars + 2)
            self.entry_widgets["a_ub"].append(row_entries)
            self.entry_widgets["b_ub"].append(b_entry)

        ttk.Button(
            self.matrix_frame, text="Solve", command=self.solve_linear_program
        ).grid(
            row=num_constraints + 5,
            column=0,
            columnspan=num_vars + 3,
            pady=20,
        )

    def create_results_widgets(self):
        """Creates the text area to display results in the right pane."""

        self.results_frame.rowconfigure(2, weight=1)
        self.results_frame.columnconfigure(0, weight=1)

        developed_by_text = (
            "Developed By: Mohammad AlMahllawi, Yusuf Nasr and Yusuf Mostafa"
        )
        ttk.Label(self.results_frame, text=developed_by_text).grid(
            row=0, column=0, sticky="w", padx=5, pady=(0, 5)
        )

        ttk.Label(self.results_frame, text="Results", style="Header.TLabel").grid(
            row=1, column=0, sticky="w", padx=5
        )

        # --- Graphical Results ---
        self.graphical_results_frame = ttk.Frame(self.results_frame)
        self.graphical_results_frame.grid(row=2, column=0, sticky="nsew")
        self.graphical_results_frame.rowconfigure(0, weight=1)
        self.graphical_results_frame.columnconfigure(0, weight=1)

        self.graphical_results_canvas = tk.Canvas(self.graphical_results_frame)
        self.graphical_results_canvas.grid(row=0, column=0, sticky="nsew")
        graphical_scroll = ttk.Scrollbar(
            self.graphical_results_frame,
            orient="vertical",
            command=self.graphical_results_canvas.yview,
        )
        graphical_scroll.grid(row=0, column=1, sticky="ns")
        self.graphical_results_canvas.configure(yscrollcommand=graphical_scroll.set)

        self.graphical_results_scrollable_frame = ttk.Frame(
            self.graphical_results_canvas
        )
        self.graphical_results_canvas.create_window(
            (0, 0), window=self.graphical_results_scrollable_frame, anchor="nw"
        )

        self.graphical_results_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.graphical_results_canvas.configure(
                scrollregion=self.graphical_results_canvas.bbox("all")
            ),
        )

        # --- Simplex Results ---
        self.simplex_results_frame = ttk.Frame(self.results_frame)
        self.simplex_results_frame.grid(row=2, column=0, sticky="nsew")
        self.simplex_results_frame.rowconfigure(1, weight=1)
        self.simplex_results_frame.columnconfigure(0, weight=1)

        self.tableau_canvas = tk.Canvas(self.simplex_results_frame)
        tableau_scroll_y = ttk.Scrollbar(
            self.simplex_results_frame,
            orient="vertical",
            command=self.tableau_canvas.yview,
        )
        self.tableau_scrollable_frame = ttk.Frame(self.tableau_canvas)

        self.tableau_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.tableau_canvas.configure(
                scrollregion=self.tableau_canvas.bbox("all")
            ),
        )

        self.tableau_canvas.create_window(
            (0, 0), window=self.tableau_scrollable_frame, anchor="nw"
        )
        self.tableau_canvas.configure(yscrollcommand=tableau_scroll_y.set)

        tableau_scroll_x = ttk.Scrollbar(
            self.simplex_results_frame,
            orient="horizontal",
            command=self.tableau_canvas.xview,
        )
        self.tableau_canvas.configure(xscrollcommand=tableau_scroll_x.set)

        self.tableau_canvas.grid(row=1, column=0, sticky="nsew")
        tableau_scroll_y.grid(row=1, column=1, sticky="ns")
        tableau_scroll_x.grid(row=2, column=0, sticky="ew")

        self.simplex_final_results_frame = ttk.Frame(self.simplex_results_frame)
        self.simplex_final_results_frame.grid(
            row=3, column=0, sticky="ew", pady=(10, 0)
        )

    def toggle_results_view(self):
        """Shows or hides the results frames based on method selection."""
        method = self.solve_method.get()
        if method == "simplex":
            self.graphical_results_frame.grid_remove()
            self.simplex_results_frame.grid()

            if self.num_vars.get() in [2, 3]:
                messagebox.showinfo(
                    "Note",
                    "Graphical plotting is disabled for Simplex method."
                    + "Results will be shown in the tableau.",
                )

        else:
            self.simplex_results_frame.grid_remove()
            self.graphical_results_frame.grid()

    def get_data_for_linprog(self):
        """Retrieves and validates data, formatting it for scipy.linprog."""
        try:
            if "c" not in self.entry_widgets:
                messagebox.showerror("Error", "Please click 'Setup Problem' first.")
                return (None,) * 6

            c = [float(e.get()) for e in self.entry_widgets["c"]]
            goal = self.goal.get()
            num_constraints = self.num_constraints.get()
            a_ub, b_ub, a_eq, b_eq = [], [], [], []

            for i in range(num_constraints):
                row = [float(e.get()) for e in self.entry_widgets["a_ub"][i]]
                b_val = float(self.entry_widgets["b_ub"][i].get())
                sign = self.entry_widgets["signs"][i].get()

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

    def get_data_for_simplex(self):
        """Retrieves and validates data, returning it raw for Simplex solver."""
        try:
            if "c" not in self.entry_widgets:
                messagebox.showerror("Error", "Please click 'Setup Problem' first.")
                return (None,) * 6

            c = [float(e.get()) for e in self.entry_widgets["c"]]
            goal = self.goal.get()
            num_vars = self.num_vars.get()
            num_constraints = self.num_constraints.get()
            a, b, signs = [], [], []

            for i in range(num_constraints):
                row = [float(e.get()) for e in self.entry_widgets["a_ub"][i]]
                b_val = float(self.entry_widgets["b_ub"][i].get())
                sign = self.entry_widgets["signs"][i].get()

                if b_val < 0:
                    messagebox.showerror(
                        "Input Error",
                        f"Constraint {i+1} has a negative RHS ({b_val}). "
                        "Please multiply the constraint by -1 and flip the sign.",
                    )
                    return (None,) * 6

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

    def solve_linear_program(self):
        """Solves the LP problem using the selected method."""
        method = self.solve_method.get()

        if method == "simplex":
            self.setup_simplex()
        else:
            self.solve_graphically()

    def setup_simplex(self):
        """Initializes the SimplexSolver, solves the problem, and displays the steps."""
        c, a, b, signs, goal, num_vars = self.get_data_for_simplex()
        if c is None:
            return

        self.simplex_solver = SimplexSolver(c, a, b, signs, goal, num_vars)
        try:
            self.simplex_solver.initialize_tableau()
        except ValueError as e:
            messagebox.showerror("Simplex Error", f"Failed to initialize tableau: {e}")
            return

        # --- History generation ---
        history = []
        solver = self.simplex_solver

        # Initial state
        history.append(
            {
                "tableau": solver.tableau.copy(),
                "pivot": None,
                "basis_vars": list(solver.basis_vars),
                "var_names": list(solver.var_names),
                "phase": solver.phase,
            }
        )

        # Loop
        while not (solver.is_optimal or solver.is_unbounded or solver.is_infeasible):
            pivot_info = solver.step()  # This also updates the tableau

            # Add pivot to the previous state
            if history:
                history[-1]["pivot"] = pivot_info

            history.append(
                {
                    "tableau": solver.tableau.copy(),
                    "pivot": None,  # pivot for this new tableau is unknown yet
                    "basis_vars": list(solver.basis_vars),
                    "var_names": list(solver.var_names),
                    "phase": solver.phase,
                }
            )

        # Now display history
        self.display_simplex_steps(history)

    def display_simplex_steps(self, history):
        """Renders the sequence of Simplex tableaux with pivot highlighting."""

        for widget in self.tableau_scrollable_frame.winfo_children():
            widget.destroy()

        style = ttk.Style()
        style.configure("PivotRow.TFrame", background="lightyellow")
        style.configure("PivotCol.TFrame", background="lightblue")
        style.configure("PivotElement.TFrame", background="lightgreen")
        style.configure("PivotRow.TLabel", background="lightyellow")
        style.configure("PivotCol.TLabel", background="lightblue")
        style.configure("PivotElement.TLabel", background="lightgreen")

        for i, step_data in enumerate(history):
            tableau = step_data["tableau"].copy()
            tableau[-1, :] *= -1.0
            pivot = step_data["pivot"]
            var_names = step_data["var_names"]
            basis_vars = step_data["basis_vars"]
            basis_var_names = [var_names[j] for j in basis_vars]

            step_frame = ttk.Frame(self.tableau_scrollable_frame, padding="10")
            step_frame.pack(fill="x", expand=True, pady=10, padx=5)

            num_cols = len(var_names) + 2  # For Basis, vars, and RHS
            for j in range(num_cols):
                step_frame.columnconfigure(j, weight=1)

            ttk.Label(step_frame, text=f"Tableau {i + 1}", style="Header.TLabel").grid(
                row=0, column=0, columnspan=num_cols, sticky="w"
            )

            # --- Create Table using Labels in Frames ---
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
                # Basis variable name
                basis_text = (
                    basis_var_names[r] if r < len(basis_var_names) else "Z (Obj)"
                )
                cell_frame = ttk.Frame(step_frame, borderwidth=1, relief="solid")
                cell_frame.grid(row=r + 2, column=0, sticky="nsew")
                lbl = ttk.Label(cell_frame, text=basis_text, anchor="center")
                lbl.pack(expand=True, fill="both", padx=5, pady=2)

                for c in range(tableau.shape[1]):
                    is_pivot_row = pivot and r == pivot["row"]
                    is_pivot_col = pivot and c == pivot["col"]

                    frame_style = "TFrame"
                    label_style = "TLabel"

                    if is_pivot_row and is_pivot_col:
                        frame_style = "PivotElement.TFrame"
                        label_style = "PivotElement.TLabel"
                    elif is_pivot_row:
                        frame_style = "PivotRow.TFrame"
                        label_style = "PivotRow.TLabel"
                    elif is_pivot_col:
                        frame_style = "PivotCol.TFrame"
                        label_style = "PivotCol.TLabel"

                    cell_frame = ttk.Frame(
                        step_frame, style=frame_style, borderwidth=1, relief="solid"
                    )
                    cell_frame.grid(row=r + 2, column=c + 1, sticky="nsew")

                    val_text = format_number(tableau[r, c])
                    lbl = ttk.Label(
                        cell_frame, text=val_text, style=label_style, anchor="e"
                    )
                    lbl.pack(expand=True, fill="both", padx=5, pady=2)

        # --- Display Final Solution ---
        for widget in self.simplex_final_results_frame.winfo_children():
            widget.destroy()

        solver = self.simplex_solver
        if solver.is_optimal:
            final_z = -solver.tableau[-1, -1]

            ttk.Label(
                self.simplex_final_results_frame,
                text="Optimal solution found!",
                style="Header.TLabel",
            ).pack(anchor="w")
            ttk.Label(
                self.simplex_final_results_frame,
                text=f"Optimal Z: {format_number(final_z)}",
            ).pack(anchor="w")
            ttk.Label(self.simplex_final_results_frame, text="Optimal Point:").pack(
                anchor="w"
            )

            for i in range(solver.num_vars):
                var_name = solver.var_names[i]
                val = 0.0
                if i in solver.basis_vars:
                    row_index = solver.basis_vars.index(i)
                    val = solver.tableau[row_index, -1]
                ttk.Label(
                    self.simplex_final_results_frame,
                    text=f"  {var_name} = {format_number(val)}",
                ).pack(anchor="w")

        elif solver.is_unbounded:
            ttk.Label(
                self.simplex_final_results_frame,
                text="Problem is UNBOUNDED.",
                style="Header.TLabel",
            ).pack(anchor="w")
        elif solver.is_infeasible:
            ttk.Label(
                self.simplex_final_results_frame,
                text="Problem is INFEASIBLE.",
                style="Header.TLabel",
            ).pack(anchor="w")

    def solve_graphically(self):
        """Solves the LP problem using scipy.linprog and plots."""
        c, a_ub, b_ub, a_eq, b_eq, goal = self.get_data_for_linprog()
        if c is None:
            return

        solver = GraphicalAnalyticalSolver(c, a_ub, b_ub, a_eq, b_eq, goal)
        result, feasible_points, a_ub_combined, b_ub_combined = solver.solve()

        self.display_results(result, goal, solver.c_original, feasible_points)

        if len(c) == 2:
            if not solver.plot_problem(
                a_ub_combined, b_ub_combined, result, feasible_points
            ):
                messagebox.showinfo(
                    "Plot Info", "Cannot plot an unbounded problem with no constraints."
                )
        elif len(c) == 3:
            if not solver.plot_3d_problem(
                a_ub_combined, b_ub_combined, result, feasible_points
            ):
                messagebox.showinfo(
                    "Plot Info", "Cannot plot an unbounded problem with no constraints."
                )
        elif len(c) > 3:
            messagebox.showinfo(
                "Plot Info", "Plotting is only supported for 2 or 3 variables."
            )

    def display_results(self, result, goal, c_original, feasible_points):
        """Formats and displays the solution using labels."""

        for widget in self.graphical_results_scrollable_frame.winfo_children():
            widget.destroy()

        container = self.graphical_results_scrollable_frame

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
                z_val = np.dot(c_original, point_vec)

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
