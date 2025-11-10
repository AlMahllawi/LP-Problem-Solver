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
from .solvers.gui.graphical import GraphicalSolverUI
from .solvers.gui.simplex import SimplexSolverUI
from .solvers.gui.analytical import AnalyticalSolverUI
from .utils import to_subscript


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
        self._configure_styles()

        # --- Initialize Attributes ---
        self.paned_window = None
        self.left_pane = None
        self.main_canvas = None
        self.scrollable_frame = None
        self.results_frame = None
        self.setup_frame = None
        self.matrix_frame = None
        self.dev_label = None
        self.entry_widgets = {}
        self.graphical_solver_ui = None
        self.simplex_solver_ui = None
        self.analytical_solver_ui = None

        # --- State Variables ---
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=2)
        self.goal = tk.StringVar(value="max")
        self.solve_method = tk.StringVar(value="graphical")

        # --- Main Layout ---
        self._setup_layout()

        # --- Create Widgets ---
        self.create_setup_widgets()
        self.create_results_widgets()
        self.toggle_results_view()

    def _configure_styles(self):
        """Configures the ttk styles for the application."""
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

    def _setup_layout(self):
        """Sets up the main layout of the application."""
        self.paned_window = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True)

        self.left_pane = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(self.left_pane, weight=1)

        self._setup_scrollable_left_pane()

        self.results_frame = ttk.Frame(self.paned_window, padding="15")
        self.paned_window.add(self.results_frame, weight=1)

        self.setup_frame = ttk.Frame(self.scrollable_frame, padding="15")
        self.setup_frame.pack(fill="x", padx=10, pady=5)

        self.matrix_frame = ttk.Frame(self.scrollable_frame, padding="15")
        self.matrix_frame.pack(fill="x", padx=10, pady=5)

    def _setup_scrollable_left_pane(self):
        """Sets up the scrollable frame in the left pane."""
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

    def _on_mousewheel(self, event):
        """
        Handles mouse wheel scrolling, directing it to the correct canvas.
        """
        canvas_map = {
            self.left_pane: self.main_canvas,
            self.graphical_solver_ui.results_frame: self.graphical_solver_ui.results_canvas,
            self.simplex_solver_ui.results_frame: self.simplex_solver_ui.tableau_canvas,
            self.analytical_solver_ui.results_frame: self.analytical_solver_ui.table_canvas,
        }

        for pane, canvas in canvas_map.items():
            try:
                if pane.winfo_exists() and pane.winfo_containing(
                    event.x_root, event.y_root
                ):
                    if event.num == 4:
                        canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        canvas.yview_scroll(1, "units")
                    elif event.delta:
                        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    return
            except (KeyError, tk.TclError):
                continue

    def create_setup_widgets(self):
        """Creates the initial widgets for problem setup."""
        ttk.Label(self.setup_frame, text="Solving Method:", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        ttk.Radiobutton(
            self.setup_frame,
            text="Graphically",
            variable=self.solve_method,
            value="graphical",
            command=self.toggle_results_view,
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            self.setup_frame,
            text="Analytical",
            variable=self.solve_method,
            value="analytical",
            command=self.toggle_results_view,
        ).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(
            self.setup_frame,
            text="Simplex",
            variable=self.solve_method,
            value="simplex",
            command=self.toggle_results_view,
        ).grid(row=1, column=2, sticky="w")

        ttk.Label(
            self.setup_frame, text="Optimization Goal:", style="Header.TLabel"
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(15, 10))
        ttk.Radiobutton(
            self.setup_frame, text="Maximize", variable=self.goal, value="max"
        ).grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(
            self.setup_frame, text="Minimize", variable=self.goal, value="min"
        ).grid(row=3, column=1, sticky="w")

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

        ttk.Button(
            self.setup_frame, text="Setup Problem", command=self.create_matrix_entries
        ).grid(row=6, column=0, columnspan=2, pady=20)

    def create_matrix_entries(self):
        """Dynamically creates entry fields for objective function and constraints."""
        for ui in [
            self.graphical_solver_ui,
            self.simplex_solver_ui,
            self.analytical_solver_ui,
        ]:
            if ui:
                for widget in ui.scrollable_frame.winfo_children():
                    widget.destroy()
                if hasattr(ui, "final_results_frame"):
                    for widget in ui.final_results_frame.winfo_children():
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

        self._create_objective_function_entries(num_vars)
        self._create_constraints_entries(num_vars, num_constraints)

        ttk.Button(
            self.matrix_frame, text="Solve", command=self.solve_linear_program
        ).grid(
            row=num_constraints + 5,
            column=0,
            columnspan=num_vars + 3,
            pady=20,
        )

    def _create_objective_function_entries(self, num_vars):
        """Creates entries for the objective function."""
        ttk.Label(
            self.matrix_frame, text="Objective Function", style="Header.TLabel"
        ).grid(row=0, column=0, columnspan=num_vars + 1, sticky="w", pady=5)
        self.entry_widgets["c"] = []
        for j in range(num_vars):
            label = f"x{to_subscript(j + 1)}"
            ttk.Label(self.matrix_frame, text=label).grid(row=1, column=j + 1)
            entry = ttk.Entry(self.matrix_frame, width=8)
            entry.grid(row=2, column=j + 1)
            self.entry_widgets["c"].append(entry)

    def _create_constraints_entries(self, num_vars, num_constraints):
        """Creates entries for the constraints."""
        ttk.Label(self.matrix_frame, text="Constraints", style="Header.TLabel").grid(
            row=3, column=0, columnspan=num_vars + 3, sticky="w", pady=(20, 5)
        )
        self.entry_widgets.update({"a_ub": [], "b_ub": [], "signs": []})

        for j in range(num_vars):
            label = f"x{to_subscript(j + 1)}"
            ttk.Label(self.matrix_frame, text=label).grid(row=4, column=j + 1)
        ttk.Label(self.matrix_frame, text="RHS").grid(row=4, column=num_vars + 2)

        for i in range(num_constraints):
            row_entries = []
            ttk.Label(self.matrix_frame, text=f"C{to_subscript(i + 1)}:").grid(
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

    def create_results_widgets(self):
        """Creates the widgets for the results pane."""
        self.results_frame.rowconfigure(2, weight=1)
        self.results_frame.columnconfigure(0, weight=1)

        developed_by_text = (
            "Developed By: Mohammad AlMahllawi, Yusuf Nasr and Yusuf Mostafa"
        )
        self.dev_label = ttk.Label(
            self.results_frame, text=developed_by_text, wraplength=1
        )
        self.dev_label.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.dev_label.bind(
            "<Configure>",
            lambda e: self.dev_label.config(wraplength=self.dev_label.winfo_width()),
        )

        ttk.Label(self.results_frame, text="Results", style="Header.TLabel").grid(
            row=1, column=0, sticky="w", padx=5
        )

        self.graphical_solver_ui = GraphicalSolverUI(self.results_frame, self)
        self.simplex_solver_ui = SimplexSolverUI(self.results_frame, self)
        self.analytical_solver_ui = AnalyticalSolverUI(self.results_frame, self)

    def toggle_results_view(self):
        """Shows or hides the results frames based on method selection."""
        method = self.solve_method.get()
        if method == "simplex":
            self.graphical_solver_ui.hide()
            self.analytical_solver_ui.hide()
            self.simplex_solver_ui.show()
        elif method == "analytical":
            self.graphical_solver_ui.hide()
            self.simplex_solver_ui.hide()
            self.analytical_solver_ui.show()
        else:
            self.simplex_solver_ui.hide()
            self.analytical_solver_ui.hide()
            self.graphical_solver_ui.show()

    def solve_linear_program(self):
        """Solves the LP problem using the selected method."""
        method = self.solve_method.get()
        if method == "simplex":
            self.simplex_solver_ui.solve()
        elif method == "analytical":
            self.analytical_solver_ui.solve()
        else:
            self.graphical_solver_ui.solve()
