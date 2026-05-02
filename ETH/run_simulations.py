# run_simulations.py
import matplotlib.pyplot as plt
# Imports the plotting library for creating graphs.
import numpy as np
# Imports NumPy for math.
from physics_core import build_hamiltonian, diagonalize
# Imports the Hamiltonian builder and solver from our custom core file.
from diagnostics import calc_r_parameter, calc_expectation_values
# Imports the analysis tools from our custom diagnostics file.

def simulate_and_plot(L, J, Delta, W, title):
# Defines a master function to run a full simulation from start to finish.
    print(f"\n--- Starting Simulation for {title} ---")
    # Prints a header to the console.
    H = build_hamiltonian(L, J, Delta, W)
    # Constructs the Hamiltonian for the specified parameters and disorder strength W.
    E, V = diagonalize(H, L)
    # Solves for the energies and states in the Sz=0 sector.
    
    # --- 1. NEW: EIGENVALUE REPORT ---
    print("\n--- EIGENVALUE ANALYSIS ---")
    # Prints a header for the energy statistics.
    print(f"Ground State Energy (E_0): {E[0]:.4f}")
    # Displays the lowest energy level found.
    print(f"First Excited State (E_1): {E[1]:.4f}")
    # Displays the second lowest energy level.
    print(f"Energy Gap (Delta E):      {E[1] - E[0]:.4f}")
    # Displays the difference between the first two levels (important for phase transitions).
    print(f"Highest Energy State:      {E[-1]:.4f}")
    # Displays the maximum energy level.
    print("---------------------------\n")
    # Prints a closing line for the report.
    
    r_vals = calc_r_parameter(E)
    # Calculates the level spacing ratios for the spectrum.
    sz_vals, entropies = calc_expectation_values(E, V, L)
    # Calculates the physical spin values and entanglement for all states.
    
    # --- 2. PLOTTING (Now a 4-Panel Masterpiece) ---
    plt.figure(figsize=(20, 5)) # Made it wider to fit 4 graphs
    # Creates a figure window that is 20 units wide and 5 units tall.
    
    # Panel 1: Density of States (DOS)
    plt.subplot(1, 4, 1)
    # Selects the first slot in a 1x4 grid of plots.
    plt.hist(E, bins=30, color='orange', alpha=0.75, edgecolor='black')
    # Plots a histogram showing how the energy levels are distributed.
    plt.title("Density of States (DOS)")
    # Sets the title for the first plot.
    plt.xlabel("Energy (E)")
    # Labels the X-axis as Energy.
    plt.ylabel("Number of States")
    # Labels the Y-axis as the count of states.
    
    # Panel 2: Level Statistics
    plt.subplot(1, 4, 2)
    # Selects the second slot in the plot grid.
    plt.hist(r_vals, bins=30, density=True, alpha=0.7, color='blue')
    # Plots the histogram of the r-parameters to see the statistical distribution.
    plt.axvline(x=np.mean(r_vals), color='r', linestyle='dashed', linewidth=2, label=f'Mean r = {np.mean(r_vals):.3f}')
    # Draws a vertical line at the average r-value to distinguish between ETH and MBL.
    plt.title("Level Spacing Ratio P(r)")
    # Sets the title for the second plot.
    plt.xlabel("r")
    # Labels the X-axis.
    plt.ylabel("Probability Density")
    # Labels the Y-axis.
    plt.legend()
    # Adds a legend to the plot.
    
    # Panel 3: Entanglement Entropy
    plt.subplot(1, 4, 3)
    # Selects the third slot in the plot grid.
    plt.scatter(E, entropies, s=2, color='purple', alpha=0.5)
    # Creates a scatter plot of Entropy vs. Energy (thermal states form a "volume law" curve).
    plt.title("Von Neumann Entropy")
    # Sets the title for the third plot.
    plt.xlabel("Energy (E)")
    # Labels the X-axis.
    plt.ylabel("Entropy (S)")
    # Labels the Y-axis.
    
    # Panel 4: Local Observable
    plt.subplot(1, 4, 4)
    # Selects the fourth slot in the plot grid.
    plt.scatter(E, sz_vals, s=2, color='green', alpha=0.5)
    # Creates a scatter plot of the middle spin value vs. Energy.
    plt.title("<Sz> (Middle Spin) vs Energy")
    # Sets the title for the final plot.
    plt.xlabel("Energy (E)")
    # Labels the X-axis.
    plt.ylabel("Expectation Value")
    # Labels the Y-axis.
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    # Adds a large main title at the top of the entire figure.
    plt.tight_layout()
    # Adjusts spacing so the subplots don't overlap.
    plt.show()
    # Displays the final window with all four graphs.

# --- SYSTEM PARAMETERS ---
L = 10
# Sets the chain length to 10 spins.
J = 1.0
# Sets the exchange interaction strength to 1.0.
Delta = 1.0
# Sets the anisotropy (Z-direction interaction) to 1.0 (Heisenberg limit).

# --- RUN ETH PHASE ---
simulate_and_plot(L, J, Delta, W=0.5, title="Regime 1: ETH (W=0.5)")
# Runs simulation with low disorder (W=0.5), which should show thermalization behavior.

# --- RUN MBL PHASE ---
simulate_and_plot(L, J, Delta, W=5.0, title="Regime 2: MBL (W=5.0)")
# Runs simulation with high disorder (W=5.0), which should show localization behavior.
