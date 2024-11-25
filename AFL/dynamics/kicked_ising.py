"""
Functions for the longitudinal-field Ising model with a transverse kick.
B. Bertini, P. Kos, and T. Prosen, Exact Spectral Form Factor in a Minimal Model of Many-Body Quantum Chaos, Phys. Rev. Lett. 121, 264101 (2018).
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import expm

def spin_matrices(spin, return_sparse=True, fmt='dia'):
    """
    Constructs the spin matrices, i.e. a representation of su(2) with dimension 2*spin+1.
    Returns a scipy sparse arrays in format 'fmt' (default DIAgonal).
    Set return_sparse=False to return standard numpy arrays.
    """
    assert (2.0*spin).is_integer() and spin >= 0, 'Invalid value of spin.'
    dim = int(2*spin + 1)
    m = spin - np.arange(dim)
    off_diag = np.sqrt(spin*(spin+1) - m[:-1]*m[1:]) / 2

    Sz = sparse.diags_array(m, format=fmt)
    Sx = sparse.diags_array([off_diag, off_diag], offsets=[-1,1], format=fmt)
    Sy = sparse.diags_array([1j*off_diag, -1j*off_diag], offsets=[-1,1], format=fmt)

    if not return_sparse:
        return Sx.toarray(), Sy.toarray(), Sz.toarray()
    return Sx, Sy, Sz

def wignerD(spin, vec):
    """
    Constructs the Wigner D-matrix (in the Sz basis) that rotates from z to the vector 'vec'.
    The columns of the D-matrix are also the spin eigenvectors in the 'vec' direction.
    
    There are significantly more efficient methods (e.g. recursion) to construct D-matrices,
    but this construction is not the limiting factor in the runtime, so this will do.
    """
    Sx, Sy, Sz = spin_matrices(spin, return_sparse=False)
    Svec = np.einsum('i,ijk', vec, np.array([Sx, Sy, Sz]), optimize='greedy')
    _, eigvecs = np.linalg.eigh(Svec) # Eigvals are returned in asceding order
    return np.flip(eigvecs, axis=1)

# Issue: wignerD returns in the total Sz basis, which is not the local Sz/tensored basis.
# Best thing to do is probably start with |all up> and do total lowing to get the others.
def spin_proj(spin, L, vec, sites, loc=None):
    assert (2.0*spin).is_integer() and spin >= 0, 'Invalid value of spin.'
    spin_dim = int(2*spin + 1)
    if loc is None:
        loc = L//2 - sites//2
    assert loc + sites <= L, 'Desired spin sites are out of bounds'

    eigvecs = wignerD(sites * spin, vec)
    projs = np.einsum('ik,jk->ijk', eigvecs, eigvecs.conj(), optimize='greedy')
    return None

def ising_unitary(spin, L, J, h, b):
    """
    Computes the kicked Ising unitary on a spin chain of length L (open boundary conditions).
    J - nearest neighbor coupling
    h - longitudinal field strength
    b - kicked transverse field strength
    """
    assert (2.0*spin).is_integer() and spin >= 0, 'Invalid value of spin.'
    spin_dim = int(2*spin + 1)
    Sx, _, Sz = spin_matrices(spin)

    dimH = spin_dim**L
    H_long = sparse.dia_array((dimH, dimH))
    H_kick = sparse.dia_array((dimH, dimH))

    # Nearest neighbor coupling with OBC
    for i in np.arange(L-1):
        left = sparse.kron(sparse.eye_array(spin_dim**i), Sz, format='dia')
        right = sparse.kron(Sz, sparse.eye_array(spin_dim**(L-i-2)), format='dia')
        H_long += J * sparse.kron(left, right, format='dia')
    # Longitudinal field and transverse kick
    for i in np.arange(L):
        left = sparse.eye_array(spin_dim**i)
        right = sparse.eye_array(spin_dim**(L-i-1))
        H_long += h * sparse.kron(left, sparse.kron(Sz, right, format='dia'), format='dia')
        H_kick += b * sparse.kron(left, sparse.kron(Sx, right, format='dia'), format='dia')

    # scipy.sparse.linalg.eigsh is not suited for this as it uses Lanczos,
    # but maybe we can do the top and bottom half of the eigenvalues separately?
    # TODO: try converting Hamiltonian to dense then passing to eigh
    # The Floquet unitary is in general very dense unless b=0.
    # Convert to CSC before passing to expm
    U_long = expm(-1j * sparse.csc_array(H_long)).toarray()
    U_kick = expm(-1j * sparse.csc_array(H_kick)).toarray()
    U = U_kick @ U_long
    assert np.allclose(U @ U.conj().T, np.eye(spin_dim**L)), 'Map is not unitary'
    return U
