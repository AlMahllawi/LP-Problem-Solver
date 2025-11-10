import numpy as np
from ..utils import to_subscript, format_number
from itertools import combinations


class AnalyticalSolver:
    """
    Solves the LP problem by finding all basic feasible solutions.
    """

    def __init__(self, c, a, b, signs, goal, num_vars):
        self.c_original = np.array(c)
        self.a_original = np.array(a)
        self.b_original = np.array(b)
        self.signs = signs
        self.goal = goal
        self.num_vars = num_vars
        self.num_constraints = len(b)

        self.var_names = []
        self.solutions = []
        self.is_optimal = False
        self.optimal_solution = None

    def solve(self):
        """
        Enumerates all basic solutions, checks for feasibility,
        and calculates the objective function for feasible ones.
        """
        # --- 0. Ensure b >= 0 (pre-processing) ---
        a_proc = self.a_original.copy()
        b_proc = self.b_original.copy()
        signs_proc = self.signs.copy()

        for i in range(len(b_proc)):
            if b_proc[i] < 0:
                a_proc[i, :] *= -1
                b_proc[i] *= -1
                if signs_proc[i] == "≤":
                    signs_proc[i] = "≥"
                elif signs_proc[i] == "≥":
                    signs_proc[i] = "≤"

        # --- 1. Standardize the problem ---
        num_slacks = (signs_proc == "≤").sum()
        num_surplus = (signs_proc == "≥").sum()

        self.var_names = [f"x{to_subscript(i+1)}" for i in range(self.num_vars)]
        slack_surplus_names = []

        c_std = np.concatenate(
            [self.c_original, np.zeros(num_slacks + num_surplus)]
        )

        extra_cols = []
        s_idx = 1
        for i in range(self.num_constraints):
            if signs_proc[i] == "≤":
                slack_col = np.zeros((self.num_constraints, 1))
                slack_col[i, 0] = 1
                extra_cols.append(slack_col)
                slack_surplus_names.append(f"s{to_subscript(s_idx)}")
                s_idx += 1
            elif signs_proc[i] == "≥":
                surplus_col = np.zeros((self.num_constraints, 1))
                surplus_col[i, 0] = -1
                extra_cols.append(surplus_col)
                slack_surplus_names.append(f"s{to_subscript(s_idx)}")
                s_idx += 1
        
        if extra_cols:
            a_std = np.hstack([a_proc, *extra_cols])
        else:
            a_std = a_proc.copy()

        self.var_names.extend(slack_surplus_names)

        n = a_std.shape[1]  # Total variables
        m = self.num_constraints  # Number of constraints

        if n < m:
            raise ValueError("Number of variables cannot be less than constraints.")

        # --- 2. Iterate through all basic solutions ---
        var_indices = range(n)
        
        for basic_indices in combinations(var_indices, m):
            non_basic_indices = list(set(var_indices) - set(basic_indices))

            B = a_std[:, list(basic_indices)]

            try:
                x_b = np.linalg.solve(B, b_proc)
            except np.linalg.LinAlgError:
                continue

            # Check for feasibility
            if np.all(x_b >= -1e-9):
                solution = np.zeros(n)
                solution[list(basic_indices)] = x_b
                
                z = np.dot(c_std, solution)
                
                point_data = {
                    "vars": solution,
                    "z": z,
                    "basic_indices": list(basic_indices)
                }
                self.solutions.append(point_data)

        # --- 3. Find the optimal solution ---
        if not self.solutions:
            return

        if self.goal == "max":
            self.optimal_solution = max(self.solutions, key=lambda s: s["z"])
        else: # min
            self.optimal_solution = min(self.solutions, key=lambda s: s["z"])
        
        self.is_optimal = True