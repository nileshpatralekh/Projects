# diagnostics.py
import numpy as np
# Imports NumPy for numerical calculations.
import scipy.sparse as sp
# Imports SciPy sparse module for operator handling.

def calc_r_parameter(E):
# Defines a function to calculate the ratio of consecutive energy level spacings.
    """Calculates the Level Spacing Ratio (r)."""
    # Docstring; r=0.53 signifies chaos (GOE), r=0.38 signifies localization (Poisson).
    gaps = np.diff(E)
    # Calculates the difference between adjacent energy levels.
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    # Computes the ratio of the smaller gap to the larger gap for each adjacent pair.
    return r
    # Returns an array of r-values for statistical analysis.

def calc_entanglement_entropy(state_vector, L):
# Defines a function to measure how "entangled" the left half of the chain is with the right half.
    """
    Calculates Von Neumann Entropy for a half-chain subsystem using SVD.
    """
    # Docstring; high entropy means thermal/chaotic, low entropy means localized.
    dim_A = 2**(L // 2)
    # Calculates the dimension of subsystem A (the first half of the chain).
    dim_B = 2**(L - L // 2)
    # Calculates the dimension of subsystem B (the second half of the chain).
    
    # Reshape the 1D state vector into a 2D matrix (Subsystem A x Subsystem B)
    psi_matrix = state_vector.reshape((dim_A, dim_B))
    # Interprets the state vector as a matrix representing the coupling between A and B.
    
    # Perform Singular Value Decomposition
    _, S, _ = np.linalg.svd(psi_matrix, full_matrices=False)
    # Decomposes the matrix to find singular values (S), which relate to entanglement.
    
    # The singular values squared are the eigenvalues of the reduced density matrix
    eigenvalues_rho = S**2
    # Squaring the singular values gives the probability weights of the reduced state.
    
    # Filter out zeros to avoid log(0) errors
    eigenvalues_rho = eigenvalues_rho[eigenvalues_rho > 1e-12]
    # Removes values near zero to prevent numerical errors during logarithm calculation.
    
    # S = - sum( lambda * ln(lambda) )
    entropy = -np.sum(eigenvalues_rho * np.log(eigenvalues_rho))
    # Computes the Von Neumann entropy formula.
    return entropy
    # Returns the entanglement entropy value for the given state.

def calc_expectation_values(E, V, L):
# Defines a function to calculate physical properties for every single energy state.
    """Calculates <Sz> for the middle spin across all states."""
    # Docstring; this checks if a specific spin "remembers" its initial state.
    sigma_z = sp.csr_matrix([[1, 0], [0, -1]])
    # Defines the Z-direction Pauli matrix.
    from physics_core import get_operator # Import helper to build the observable
    # Locally imports the operator builder from the core physics file.
    
    center_site = L // 2
    # Identifies the index of the spin at the center of the chain.
    Sz_center = get_operator(sigma_z, center_site, L).toarray()
    # Builds the full matrix for the Z-spin at the center site.
    
    exp_vals = []
    # Initializes a list to store the expectation values for each state.
    entropies = []
    # Initializes a list to store the entanglement entropy for each state.
    
    print("Calculating Observables and Entropy for all states...")
    # Progress update printed to the console.
    for n in range(len(E)):
    # Loops through every eigenstate found during diagonalization.
        vec = V[:, n]
        # Extracts the n-th eigenvector (eigenstate).
        # Expectation value <psi| Sz |psi>
        val = np.vdot(vec, np.dot(Sz_center, vec)).real
        # Computes the inner product (bra-ket) to find the average value of Sz in that state.
        exp_vals.append(val)
        # Adds the calculated expectation value to the list.
        
        # Calculate Entropy
        ent = calc_entanglement_entropy(vec, L)
        # Calculates the entanglement entropy for this specific eigenstate.
        entropies.append(ent)
        # Adds the entropy value to the list.
        
    return exp_vals, entropies
    # Returns the list of local spin values and entropies for all eigenstates.
