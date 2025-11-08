# Linear Programming (LP) Solver GUI

This is a graphical user interface (GUI) application built with Python and Tkinter, designed to solve linear programming problems using SciPy's optimization library (linprog).  
- Generates a plot of the feasible region and the optimal solution for 2 and 3 variables problems when solving graphically.
- Creates the Tableaus to show the step-by-step solution when solving with simplex.

## Setup and Installation

To run this application, you need to set up a Python virtual environment and install the required dependencies.

### Create a Virtual Environment

A virtual environment helps isolate the project's dependencies from your system's global Python packages.

#### Create the Environment
```sh
python -m venv .venv
```

#### Activate the Environment

| Operating System         | Command                       |
|--------------------------|-------------------------------|
| Windows (Command Prompt) | `venv\Scripts\activate`       |
| Windows (PowerShell)     | `.\venv\Scripts\Activate.ps1` |
| POSIX (macOS/Linux)      | `source venv/bin/activate`    |

## Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application
```sh
python run.py
```

## Create an Executable
```sh
pyinstaller LPPS.spec
```
