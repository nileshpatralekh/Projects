# physics_core.py
import numpy as np
# Imports the NumPy library for high-performance numerical and array operations.
import scipy.sparse as sp
# Imports SciPy's sparse module to handle large matrices efficiently without wasting memory on zeros.

def get_operator(op, site, L):
# Defines a function to "embed" a local 2x2 operator into the full L-site Hilbert space.
    I = sp.csr_matrix(np.eye(2))
    # Creates a 2x2 identity matrix in Compressed Sparse Row format.
    op_list = [I] * L
    # Initializes a list of identity matrices, one for each site in the spin chain.
    op_list[site] = op
    # Replaces the identity at the specific 'site' with the desired local operator (e.g., Sigma-X).
    full_op = op_list[0]
    # Starts the "Kronecker product" chain with the operator at the first site.
    for i in range(1, L):
    # Loops through the remaining sites to build the full multi-site operator.
        full_op = sp.kron(full_op, op_list[i], format='csr')
        # Computes the tensor product of the current operator and the next site, expanding the matrix size.
    return full_op
    # Returns the final 2^L x 2^L sparse matrix representing the operator on the whole chain.

def build_hamiltonian(L, J, Delta, W):
# Defines the function to construct the XXZ Hamiltonian with random field disorder.
    I = sp.csr_matrix(np.eye(2))
    # Creates a 2x2 sparse identity matrix.
    sigma_x = sp.csr_matrix([[0, 1], [1, 0]])
    # Defines the Pauli-X matrix as a sparse matrix.
    sigma_y = sp.csr_matrix([[0, -1j], [1j, 0]])
    # Defines the Pauli-Y matrix (using complex 'j') as a sparse matrix.
    sigma_z = sp.csr_matrix([[1, 0], [0, -1]])
    # Defines the Pauli-Z matrix as a sparse matrix.
    
    dim = 2**L
    # Calculates the total dimension of the Hilbert space (2 to the power of the number of sites).
    H = sp.csr_matrix((dim, dim))
    # Initializes an empty (all-zero) sparse matrix for the Hamiltonian.
    
    for i in range(L - 1):
    # Iterates through adjacent pairs of spins (nearest neighbors).
        Sx_i = get_operator(sigma_x, i, L)
        # Gets the Sigma-X operator acting on site i.
        Sx_j = get_operator(sigma_x, i+1, L)
        # Gets the Sigma-X operator acting on the neighbor site i+1.
        Sy_i = get_operator(sigma_y, i, L)
        # Gets the Sigma-Y operator acting on site i.
        Sy_j = get_operator(sigma_y, i+1, L)
        # Gets the Sigma-Y operator acting on site i+1.
        Sz_i = get_operator(sigma_z, i, L)
        # Gets the Sigma-Z operator acting on site i.
        Sz_j = get_operator(sigma_z, i+1, L)
        # Gets the Sigma-Z operator acting on site i+1.
        H += J * (Sx_i.dot(Sx_j) + Sy_i.dot(Sy_j) + Delta * Sz_i.dot(Sz_j))
        # Adds the interaction terms to the Hamiltonian (XX + YY + Delta*ZZ).
        
    np.random.seed(42) 
    # Sets a fixed seed for the random number generator to ensure simulation results are reproducible.
    h_fields = np.random.uniform(-W, W, L)
    # Generates L random magnetic field values chosen from a uniform distribution between -W and W.
    for i in range(L):
    # Loops through each site to apply the random field.
        Sz_i = get_operator(sigma_z, i, L)
        # Gets the Sigma-Z operator for the current site.
        H += h_fields[i] * Sz_i
        # Adds the random longitudinal field term (h * Sz) to the Hamiltonian.
        
    return H
    # Returns the completed Hamiltonian matrix.

def generate_sz0_basis(L):
# Defines a function to find states where the total spin in the Z direction is zero.
    """Finds all binary numbers from 0 to 2^L that have exactly L//2 ones."""
    # Docstring explaining that this identifies the "half-filling" sector of the Hilbert space.
    basis = []
    # Initializes an empty list to store the indices of the states in this sector.
    for i in range(2**L):
    # Iterates through every possible state index in the full Hilbert space.
        if bin(i).count('1') == L // 2:
        # Converts index to binary and checks if exactly half the bits are '1' (spins up).
            basis.append(i)
            # Adds the index to the basis list if the condition is met.
    return basis
    # Returns the list of indices that form the Sz=0 subspace.

def diagonalize(H_full, L):
# Defines a function to solve the Schrodinger equation for the Hamiltonian.
    """Slices the Hamiltonian to the Sz=0 sector, diagonalizes, and maps back."""
    # Docstring explaining that we only care about the specific spin sector for efficiency.
    basis = generate_sz0_basis(L)
    # Generates the indices for the Sz=0 sector.
    
    # 1. Convert to dense array to easily slice it
    H_dense = H_full.toarray()
    # Converts the sparse Hamiltonian to a standard dense NumPy array for indexing.
    
    # 2. Extract the 252 x 252 submatrix!
    H_sub = H_dense[np.ix_(basis, basis)]
    # Uses advanced indexing to extract only the rows and columns corresponding to the Sz=0 sector.
    
    print(f"Diagonalizing reduced Hamiltonian of size {len(basis)}x{len(basis)}...")
    # Prints the size of the matrix being solved (e.g., 252 for L=10).
    E, V_sub = np.linalg.eigh(H_sub)
    # Finds the eigenvalues (E) and eigenvectors (V_sub) of the Hermitian submatrix.
    
    # 3. V_sub is 252x252. We map these back to 1024-length vectors.
    V_full = np.zeros((2**L, len(E)))
    # Creates a large zero-filled matrix to hold the full-sized eigenvectors.
    for n in range(len(E)):
    # Loops through each calculated eigenvector.
        V_full[basis, n] = V_sub[:, n]
        # Places the sub-sector eigenvector values back into their correct original indices.
        
    return E, V_full
    # Returns the energy eigenvalues and the full-space eigenvectors.
