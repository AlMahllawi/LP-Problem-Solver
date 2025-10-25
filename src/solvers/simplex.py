import numpy as np
from ..utils import to_subscript, format_number


class SimplexSolver:
    """
    Manages the state and logic for the Simplex algorithm.
    Handles tableau creation, pivoting, and state detection (optimal, etc.).
    """

    def __init__(self, c, a, b, signs, goal, num_vars):
        self.c_original = np.array(c)
        self.a_original = np.array(a)
        self.b_original = np.array(b)
        self.signs = signs
        self.goal = goal
        self.num_vars = num_vars
        self.num_constraints = len(b)

        self.tableau = None
        self.basis_vars = []  # Stores indices of variables in the basis
        self.var_names = []  # Stores string names (e.g., 'x1', 's1', 'a1')
        self.phase = 2  # Start in Phase 2 unless artificials are needed
        self.phase_2_obj_row = None  # To store original Cj for Phase 2

        self.is_optimal = False
        self.is_unbounded = False
        self.is_infeasible = False

        # A log to report steps to the GUI
        self.status_log = []

    def initialize_tableau(self):
        """Builds the initial Simplex tableau from the problem data."""
        self.status_log.clear()

        # --- 1. Initialize variable names and basis ---
        self.var_names = [f"x{to_subscript(i+1)}" for i in range(self.num_vars)]
        self.basis_vars = [0] * self.num_constraints  # Placeholder

        num_slacks = 0
        num_surplus = 0
        num_artificials = 0
        needs_phase_1 = False

        # --- 2. Build A matrix with slack/surplus/artificial cols ---
        a_matrix = self.a_original.copy()

        for i in range(self.num_constraints):
            sign = self.signs[i]

            # Add slack/surplus/artificial columns to A
            slack_col = np.zeros((self.num_constraints, 1))
            surplus_col = np.zeros((self.num_constraints, 1))
            art_col = np.zeros((self.num_constraints, 1))

            if sign == "≤":  # <=
                num_slacks += 1
                slack_col[i, 0] = 1
                self.var_names.append(f"s{to_subscript(num_slacks)}")
                a_matrix = np.hstack([a_matrix, slack_col])
                self.basis_vars[i] = len(self.var_names) - 1  # s_i starts in basis

            elif sign == "≥":  # >=
                num_surplus += 1
                surplus_col[i, 0] = -1
                self.var_names.append(f"e{to_subscript(num_surplus)}")  # e for 'excess'
                a_matrix = np.hstack([a_matrix, surplus_col])

                num_artificials += 1
                art_col[i, 0] = 1
                self.var_names.append(f"a{to_subscript(num_artificials)}")
                a_matrix = np.hstack([a_matrix, art_col])
                self.basis_vars[i] = len(self.var_names) - 1  # a_i starts in basis
                needs_phase_1 = True

            elif sign == "=":  # =
                num_artificials += 1
                art_col[i, 0] = 1
                self.var_names.append(f"a{to_subscript(num_artificials)}")
                a_matrix = np.hstack([a_matrix, art_col])
                self.basis_vars[i] = len(self.var_names) - 1  # a_i starts in basis
                needs_phase_1 = True

        # --- 3. Build Objective Row (Cj row) ---
        total_vars = len(self.var_names)

        # Original problem objective row (Phase 2)
        obj_row_p2 = np.zeros(total_vars)
        obj_row_p2[: self.num_vars] = self.c_original

        # Store for later
        self.phase_2_obj_row = obj_row_p2.copy()

        if needs_phase_1:
            self.phase = 1
            self.status_log.append("Starting Phase 1: Minimize artificial variables.")
            # Phase 1 Obj: Minimize sum of artificials (using Maximization of negative sum)
            obj_row_p1 = np.zeros(total_vars)
            art_indices = [
                i for i, name in enumerate(self.var_names) if name.startswith("a")
            ]
            obj_row_p1[art_indices] = (
                -1.0
            )  # Max -sum(a_i) -> obj row is [0... -1, -1...]

            # Combine A and Cj
            self.tableau = np.hstack([a_matrix, self.b_original.reshape(-1, 1)])
            obj_row_with_z = np.append(obj_row_p1, 0.0)  # 0.0 for RHS
            self.tableau = np.vstack([self.tableau, obj_row_with_z])

            # Make obj row in terms of non-basics
            # For each a_i in basis, ADD its row to obj row (because we're maximizing negative sum)
            for i in range(self.num_constraints):
                if self.var_names[self.basis_vars[i]].startswith("a"):
                    self.tableau[-1, :] += self.tableau[i, :]

        else:  # No Phase 1 needed
            self.phase = 2
            self.status_log.append("Starting Phase 2: Solving problem.")
            self.tableau = np.hstack([a_matrix, self.b_original.reshape(-1, 1)])
            obj_row_with_z = np.append(obj_row_p2, 0.0)  # 0.0 for RHS
            self.tableau = np.vstack([self.tableau, obj_row_with_z])

            # Make obj row in terms of non-basics
            for i in range(self.num_constraints):
                basic_var_index = self.basis_vars[i]
                multiplier = self.tableau[-1, basic_var_index]
                if abs(multiplier) > 1e-9:
                    self.tableau[-1, :] -= multiplier * self.tableau[i, :]

        self.status_log.append("Initial tableau created.")

    def step(self):
        """Performs a single pivot operation and returns pivot info."""

        if self.is_optimal or self.is_unbounded or self.is_infeasible:
            return None

        self.status_log.clear()

        # --- 1. Check for Optimality ---
        obj_row = self.tableau[-1, :-1]

        is_optimal = False
        if self.phase == 1:
            # Phase 1 is Maximization of negative sum: optimal if obj_row <= 0
            is_optimal = np.all(obj_row <= 1e-9)
        else:  # Phase 2
            if self.goal == "max":
                is_optimal = np.all(obj_row <= 1e-9)
            else:
                is_optimal = np.all(obj_row >= -1e-9)

        if is_optimal:
            if self.phase == 1:
                obj_val = self.tableau[-1, -1]
                if abs(obj_val) > 1e-6:
                    # Artificials are non-zero. Infeasible.
                    self.is_infeasible = True
                    self.status_log.append(
                        f"Phase 1 complete. Z = {format_number(obj_val)} != 0."
                    )
                    self.status_log.append("Problem is INFEASIBLE.")
                    return None
                else:
                    # Phase 1 successful. Move to Phase 2.
                    self.status_log.append("Phase 1 complete. Switching to Phase 2.")
                    self.phase = 2

                    # Remove artificial columns from tableau
                    art_indices = [
                        i
                        for i, name in enumerate(self.var_names)
                        if name.startswith("a")
                    ]

                    # Keep only non-artificial columns
                    keep_indices = [
                        i for i in range(len(self.var_names)) if i not in art_indices
                    ]
                    keep_indices.append(-1)  # Keep RHS column

                    # Update tableau and variable names
                    self.tableau = self.tableau[:, keep_indices]
                    self.var_names = [
                        self.var_names[i]
                        for i in range(len(self.var_names))
                        if i not in art_indices
                    ]

                    # Update basis variables to remove artificials
                    new_basis_vars = []
                    for basis_idx in self.basis_vars:
                        if basis_idx not in art_indices:
                            # Adjust index after removing artificials
                            new_idx = basis_idx - len(
                                [i for i in art_indices if i < basis_idx]
                            )
                            new_basis_vars.append(new_idx)
                        else:
                            # Find a variable to replace the artificial in basis
                            # For now, we'll use the first available variable
                            new_basis_vars.append(0)
                    self.basis_vars = new_basis_vars

                    # Restore Phase 2 objective row
                    obj_row_p2 = self.phase_2_obj_row.copy()
                    # Remove artificial columns from objective row
                    obj_row_p2 = np.array(
                        [
                            obj_row_p2[i]
                            for i in range(len(obj_row_p2))
                            if i not in art_indices
                        ]
                    )

                    # Add Z (RHS) value
                    obj_row_with_z = np.append(obj_row_p2, 0.0)

                    # Add the new objective row to tableau
                    self.tableau = np.vstack([self.tableau[:-1, :], obj_row_with_z])

                    # Must make obj row in terms of non-basics
                    for i in range(self.num_constraints):
                        basic_var_index = self.basis_vars[i]
                        multiplier = self.tableau[-1, basic_var_index]
                        if abs(multiplier) > 1e-9:
                            self.tableau[-1, :] -= multiplier * self.tableau[i, :]

                    # Re-run step to check P2 optimality
                    return self.step()

            else:  # Phase 2 is optimal
                self.is_optimal = True
                self.status_log.append("Optimal solution found.")
                return None

        # --- 2. Find Pivot Column (Entering Variable) ---
        if self.phase == 1:  # Phase 1 is Maximization
            valid_indices = np.where(obj_row > 1e-9)[0]
            if len(valid_indices) > 0:
                pivot_col = valid_indices[np.argmax(obj_row[valid_indices])]
            else:
                self.is_optimal = True
                return None
        else:  # Phase 2
            if self.goal == "max":
                valid_indices = np.where(obj_row > 1e-9)[0]
                if len(valid_indices) > 0:
                    pivot_col = valid_indices[np.argmax(obj_row[valid_indices])]
                else:
                    self.is_optimal = True
                    return None
            else:
                valid_indices = np.where(obj_row < -1e-9)[0]
                if len(valid_indices) > 0:
                    pivot_col = valid_indices[np.argmin(obj_row[valid_indices])]
                else:
                    self.is_optimal = True
                    return None

        self.status_log.append(f"Entering variable: {self.var_names[pivot_col]}")

        # --- 3. Find Pivot Row (Leaving Variable) ---
        pivot_col_data = self.tableau[:-1, pivot_col]
        rhs_data = self.tableau[:-1, -1]

        if np.all(pivot_col_data <= 1e-9):
            self.is_unbounded = True
            self.status_log.append("Problem is UNBOUNDED.")
            return None

        # Min Ratio Test
        ratios = []
        for i in range(self.num_constraints):
            if pivot_col_data[i] > 1e-9:
                ratio = rhs_data[i] / pivot_col_data[i]
                if ratio >= -1e-9:  # Only consider non-negative ratios
                    ratios.append((ratio, i))
            else:
                ratios.append((np.inf, i))

        if all(ratio[0] == np.inf for ratio in ratios):
            self.is_unbounded = True
            self.status_log.append("Problem is UNBOUNDED.")
            return None

        pivot_row = min(ratios, key=lambda x: x[0])[1]

        self.status_log.append(
            f"Leaving variable: {self.var_names[self.basis_vars[pivot_row]]}"
        )

        # --- 4. Perform Pivot ---
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.status_log.append(
            f"Pivoting on element at (R{pivot_row+1}, C={self.var_names[pivot_col]})"
            f" = {format_number(pivot_element)}"
        )

        # Normalize the pivot row
        self.tableau[pivot_row, :] /= pivot_element

        # Eliminate the pivot column from other rows
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row, :]

        self.basis_vars[pivot_row] = pivot_col
        self.status_log.append("Pivot complete. Tableau updated.")

        return {"row": pivot_row, "col": pivot_col}
