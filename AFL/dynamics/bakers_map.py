"""
Quantum Baker's map functions.
"""

import numpy as np
from scipy.linalg import block_diag

def dft(N, q_phase, p_phase):
    """
    Returns the NxN generalized DFT matrix with elements
    F[k,n] = 1/sqrt(N) * exp{(-i2π/N)(k + q_phase)(n + p_phase)}
    where q_phase and p_phase are between 0 and 1.

    This derives from a quantum system with quasiperiodic boundary conditions:
        ψ(q+1) = exp(i2π*q_phase)ψ(q)
        ϕ(p+1) = exp(-i2π*p_phase)ϕ(p)
    for which we need discretized q and p:
        q[j] = (j + p_phase) / N, j=0,...,N-1
        p[j] = (j + q_phase) / N, j=0,...,N-1
    The Fourier relation between these coordinates is
        ϕ(p[k]) = F[k,n]ψ(q[n]) --> |q[j]> = F|p[j]>
        F[k,n] = <q[k]|F|q[n]> = <p[k]|q[n]>
    """
    F = np.empty((N,N), dtype=np.complex128)
    for k in np.arange(N):
        for n in np.arange(N):
            F[k,n] = np.exp(-2j*np.pi * (k + q_phase) * (n + p_phase) / N) / np.sqrt(N)
    return F

def baker_unitary(N, N_split, q_phase, p_phase):
    """
    Returns the NxN unitary for the quantum Baker's map with quasiperiodic boundary conditions
        ψ(q+1) = exp(i2π*q_phase)ψ(q), ϕ(p+1) = exp(-i2π*p_phase)ϕ(p)
    N_split - the location of the Baker's map discontinuity, i.e. r = N_split / N.
    """
    F_N = dft(N, q_phase, p_phase)
    F_split1 = dft(N_split, q_phase, p_phase)
    F_split2 = dft(N - N_split, q_phase, p_phase)
    F_split = block_diag(F_split1, F_split2)
    U = np.conj(F_N) @ F_split
    assert np.allclose(U @ U.conj().T, np.eye(N)), 'Map is not unitary'
    return U

def perm_baker_unitary(block_sizes, permutation, q_phase, p_phase):
    """
    Returns the unitary for the quantum Baker's map with quasiperiodic boundary conditions
        ψ(q+1) = exp(i2π*q_phase)ψ(q), ϕ(p+1) = exp(-i2π*p_phase)ϕ(p)
    This function can handle an arbitrary number of Baker's blocks and permutations.
    block_sizes - list of the sizes of each Baker's block.
    permutation - list of where the Baker's blocks (1,...,n) respectively map to.
        Ex: block_sizes = (2,4,3), permutation = (3,1,2)
            q[0:2] maps to p[7:9]
            q[2:6] maps to p[0:4]
            q[6:9] maps to p[4:7]
    """
    num = block_sizes.size
    if permutation.size != num:
        raise IndexError("Number of blocks does not match permuatation.")
    
    N = np.sum(block_sizes)
    F_N = dft(N, q_phase, p_phase)

    permutation -= 1
    inv_perm = permutation[permutation]

    col = 0
    F_blocks = np.zeros((N,N), dtype=np.complex128)
    for i in np.arange(num):
        N_block = block_sizes[i]

        row = 0
        for j in np.arange(permutation[i]):
            row += block_sizes[inv_perm[j]]

        F_block = dft(N_block, q_phase, p_phase)
        F_blocks[row:row+N_block, col:col+N_block] = F_block

        col += N_block

    U = np.conj(F_N) @ F_blocks
    assert np.allclose(U @ U.conj().T, np.eye(N)), 'Map is not unitary'
    return U

def baker_R(N, PBC=False):
    """
    Constructs the baker's map R-symmetry operator.
    Classically, R(q,p) = (1 - q, 1 - p).
    R can be quantized for periodic or anti-periodic boundary conditions (PBC or APBC).
    R is a symmetry of the quantum bakers map only for APBC with r=1/2 (requires even N).
    """
    if PBC:
        return block_diag(1, baker_R(N - 1, PBC=False))

    R = np.zeros((N,N))
    for j in np.arange(N):
        R[j, N - 1 - j] = 1
    return R
