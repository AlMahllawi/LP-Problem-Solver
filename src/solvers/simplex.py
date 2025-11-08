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

        if goal == "max":
            self.c_original = -self.c_original

        self.tableau = None
        self.basis_vars = []  # Stores indices of variables in the basis
        self.var_names = []  # Stores string names (e.g., 'x1', 's1', 'a1')
        self.phase = 2  # Start in Phase 2 unless artificials are needed
        self.z_row_index = -1  # Index of the Z-row (original objective)
        self.w_row_index = -1  # Index of the W-row (artificial objective)

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
        self.basis_vars = [0] * self.num_constraints

        num_slacks = 0
        num_surplus = 0
        num_artificials = 0
        needs_phase_1 = False

        # --- 2. Identify variable types and build A matrix ---
        a_matrix = self.a_original.copy()
        art_var_indices = []

        # First, add slack/surplus variables
        for i in range(self.num_constraints):
            sign = self.signs[i]
            if sign == "≤":
                num_slacks += 1
                slack_col = np.zeros((self.num_constraints, 1))
                slack_col[i, 0] = 1
                self.var_names.append(f"s{to_subscript(num_slacks)}")
                a_matrix = np.hstack([a_matrix, slack_col])
                self.basis_vars[i] = len(self.var_names) - 1
            elif sign == "≥":
                num_surplus += 1
                surplus_col = np.zeros((self.num_constraints, 1))
                surplus_col[i, 0] = -1
                self.var_names.append(f"e{to_subscript(num_surplus)}")
                a_matrix = np.hstack([a_matrix, surplus_col])
                needs_phase_1 = True
            elif sign == "=":
                needs_phase_1 = True

        # Then, add artificial variables together
        if needs_phase_1:
            for i in range(self.num_constraints):
                sign = self.signs[i]
                if sign == "≥" or sign == "=":
                    num_artificials += 1
                    art_col = np.zeros((self.num_constraints, 1))
                    art_col[i, 0] = 1
                    self.var_names.append(f"a{to_subscript(num_artificials)}")
                    a_matrix = np.hstack([a_matrix, art_col])
                    art_var_indices.append(len(self.var_names) - 1)
                    self.basis_vars[i] = len(self.var_names) - 1

        # --- 3. Build Objective Rows ---
        total_vars = len(self.var_names)
        self.tableau = np.hstack([a_matrix, self.b_original.reshape(-1, 1)])

        # Z-row (original objective)
        z_row = np.zeros(total_vars + 1)
        z_row[: self.num_vars] = self.c_original

        if needs_phase_1:
            self.phase = 1
            self.status_log.append("Starting Phase 1: Minimize artificial variables.")

            # W-row (artificial objective)
            w_row = np.zeros(total_vars + 1)
            w_row[art_var_indices] = 1.0  # Minimize sum of a_i

            # Add rows to tableau
            self.tableau = np.vstack([self.tableau, z_row, w_row])
            self.z_row_index = self.num_constraints
            self.w_row_index = self.num_constraints + 1

            # Make objective rows in terms of non-basics
            for i in range(self.num_constraints):
                if self.var_names[self.basis_vars[i]].startswith("a"):
                    self.tableau[self.z_row_index, :] -= (
                        self.tableau[self.z_row_index, self.basis_vars[i]]
                        * self.tableau[i, :]
                    )
                    self.tableau[self.w_row_index, :] -= self.tableau[i, :]

        else:  # No Phase 1 needed
            self.phase = 2
            self.status_log.append("Starting Phase 2: Solving problem.")
            self.tableau = np.vstack([self.tableau, z_row])
            self.z_row_index = self.num_constraints

            # Make obj row in terms of non-basics
            for i in range(self.num_constraints):
                basic_var_index = self.basis_vars[i]
                multiplier = self.tableau[self.z_row_index, basic_var_index]
                if abs(multiplier) > 1e-9:
                    self.tableau[self.z_row_index, :] -= multiplier * self.tableau[i, :]

        self.status_log.append("Initial tableau created.")

    def step(self):
        """Performs a single pivot operation and returns pivot info."""
        if self.is_optimal or self.is_unbounded or self.is_infeasible:
            return None

        self.status_log.clear()

        # --- 1. Determine active objective row ---
        if self.phase == 1:
            obj_row_index = self.w_row_index
        else:
            obj_row_index = self.z_row_index

        obj_row = self.tableau[obj_row_index, :-1]

        # --- 2. Check for Optimality ---
        if np.all(obj_row >= -1e-9):
            if self.phase == 1:
                w_val = -self.tableau[
                    self.w_row_index, -1
                ]  # W is min, so tableau has -W
                if abs(w_val) > 1e-6:
                    self.is_infeasible = True
                    self.status_log.append(
                        f"Phase 1 complete. W = {format_number(w_val)} != 0."
                    )
                    self.status_log.append("Problem is INFEASIBLE.")
                    return None
                else:
                    self.status_log.append("Phase 1 complete. Switching to Phase 2.")
                    self.phase = 2

                    # Remove W-row and artificial variable columns
                    art_indices = [
                        i
                        for i, name in enumerate(self.var_names)
                        if name.startswith("a")
                    ]
                    self.tableau = np.delete(self.tableau, self.w_row_index, axis=0)
                    self.tableau = np.delete(self.tableau, art_indices, axis=1)

                    # Update var_names and z_row_index
                    self.var_names = [
                        name
                        for i, name in enumerate(self.var_names)
                        if i not in art_indices
                    ]
                    self.z_row_index = self.num_constraints

                    # Re-run step to check P2 optimality
                    return self.step()
            else:
                self.is_optimal = True
                self.status_log.append("Optimal solution found.")
                return None

        # --- 3. Find Pivot Column (Entering Variable) ---
        pivot_col = np.argmin(obj_row)
        self.status_log.append(f"Entering variable: {self.var_names[pivot_col]}")

        # --- 4. Find Pivot Row (Leaving Variable) ---
        pivot_col_data = self.tableau[: self.num_constraints, pivot_col]
        rhs_data = self.tableau[: self.num_constraints, -1]

        if np.all(pivot_col_data <= 1e-9):
            self.is_unbounded = True
            print(pivot_col_data)
            self.status_log.append("Problem is UNBOUNDED.")
            return None

        ratios = []
        for i in range(self.num_constraints):
            if pivot_col_data[i] > 1e-9:
                ratio = rhs_data[i] / pivot_col_data[i]
                if ratio >= -1e-9:
                    ratios.append((ratio, i))
            else:
                ratios.append((np.inf, i))

        if not ratios or all(r[0] == np.inf for r in ratios):
            self.is_unbounded = True
            self.status_log.append("Problem is UNBOUNDED.")
            return None

        pivot_row = min(ratios, key=lambda x: x[0])[1]
        self.status_log.append(
            f"Leaving variable: {self.var_names[self.basis_vars[pivot_row]]}"
        )

        # --- 5. Perform Pivot ---
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.status_log.append(
            f"Pivoting on element at (R{pivot_row + 1}, C={self.var_names[pivot_col]})"
            f" = {format_number(pivot_element)}"
        )

        # Normalize pivot row
        self.tableau[pivot_row, :] /= pivot_element

        # Eliminate other rows
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row, :]

        self.basis_vars[pivot_row] = pivot_col
        self.status_log.append("Pivot complete. Tableau updated.")

        return {"row": pivot_row, "col": pivot_col}
