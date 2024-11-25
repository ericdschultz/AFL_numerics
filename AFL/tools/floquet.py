"""
Useful functions for computing AFL entropy on a Floquet system with evolution operator U.
"""
import numpy as np
import scipy.linalg as linalg

def eig_sort(U):
    """Sorts eigenstates of a unitary U by quasienergy."""
    if np.allclose(U.conj().T @ U, np.eye(U.shape[0])):
        # schur over eig since eigvecs is guaranteed to be unitary
        # eigvecs[i,j] is the i-th component of the j-th eigenvector
        eigval_mat, eigvecs = linalg.schur(U)
        eigvals = np.diag(eigval_mat)
        # Sort states by quasienergy
        # Branch is (-π, π]
        energy = np.angle(eigvals)
        indices = np.argsort(energy)
        sorted_energy = energy[indices]
        sorted_eigvecs = eigvecs[:,indices].astype(np.complex128)
        return sorted_energy, sorted_eigvecs
    raise ValueError("Argument to eig_sort must be unitary.")

'''
The following functions deal with (purely numerical) branch cut issues of Floquet perturbations.
Quasienergies that move across the branch cut as the perturbation strength is tuned will shift the order of the basis (sorted by quasienergy).
We find this shift and appropriately reassign the value of the energies.
Assumes perturbations are adiabatic and not too large (fewer than half the quasienergies should cross the branch cut).

After finding this shift, we can construct the good basis as in first-order degenerate perturbation theory.
The good basis is adiabatically connected to the perturbed basis as we vary the perturbation strength.

Note about past attemps: I also tried a non-optimization approach that matched the shifted energy difference to the 'true' energy difference.
However, I realized I could not find the 'true' energy difference since matrices like U*.U_pert do NOT have 
the true difference as eigenvalues due to noncommutivity.
'''

def shift_basis_en(energy, en_pert):
    """
    Returns the shift of the index of the eigenbasis of U to match that of the perturbed vectors.
    This function works by minimizing the difference between the quasi-energies.
    
    WARNING: shift_basis_en may fail when energy spacing is on the order of the energy perturbation.
    This may occur when N is large, as the spectrum is bounded to the unit circle.
    This also occurs for large degeneracy, like for identity dynamics.
    """
    N = energy.size
    phases = np.exp(1j * energy)
    phases_pert = np.exp(1j * en_pert)
    
    # Minimize sum of squares
    shift = -1
    min_diff = N * np.pi
    for i in np.arange(N):
        # Essential that there is no branch cut near z=1 (phase = 0)
        diffs = np.imag(np.log(phases_pert / np.roll(phases, i)))
        tot_diff = np.sum(np.abs(diffs) ** 2)
        if tot_diff < min_diff:
            min_diff = tot_diff
            shift = i
    if shift < 0:
        raise RuntimeError("Failed to find unperturbed basis ordering that matched the perturbed basis.")
    return shift    

def shift_basis_vec(eigvecs, vec_pert):
    """
    Returns the shift of the index of the eigenbasis of U to match that of the perturbed vectors.
    This function works by minimizing the distance between the eigenbases w.r.t. the shift.
    
    WARNING: shift_basis_vec may fail in the degenerate case, so use shift_basis_proj instead.
    """
    N = eigvecs.shape[0]
    shift = -1
    min_dist = N
    for i in np.arange(N):
        '''
        Distance heuristic: N - Σ|<u,u'>|^2 where primed is perturbed.
        We dont use ||u'-u|| due to not caring for an overall phase on the eigenvectors.
        The heuristic is within [0,N] where U is NxN.
        Maximal when u and u' are orthogonal, minimal when u and u' are complex-collinear (differ by a phase).
        '''
        shifted_vecs = np.roll(eigvecs, i, axis=1)
        dots = np.einsum('ij,ij->j', shifted_vecs.conj(), vec_pert, optimize='greedy')
        dist = N - np.sum(np.abs(dots)**2)
        if dist < min_dist:
            min_dist = dist
            shift = i
    if shift < 0:
        raise RuntimeError("Failed to find unperturbed basis ordering that matched the perturbed basis.")
    return shift

def constr_proj(eigproj, indices, counts):
    """
    Constructs the projectors onto the subspaces given by indices and counts.
    
    eigproj: 1D eigenspace projectors
    indices: where the subspace starts in eigproj
    counts: dimension of the subspace (how many consecutive vectors)
    """
    N = eigproj.shape[0]
    num_spaces = indices.size
    proj = np.zeros((num_spaces, N, N), dtype=np.complex128)
    for i in np.arange(indices.size):
        for j in np.arange(counts[i]):
            proj[i,:,:] += eigproj[(indices[i] + j) % N,:,:]
    return proj

def shift_basis_proj(eigvecs, vec_pert, indices, counts):
    """
    Returns the shift of the index of the eigenbasis of U to match that of the perturbed vectors.
    This function works by minimizing the distance between the eigenbasis w.r.t. the shift.
    Accounts for degeneracy of U by using eigenspace projectors.
    """
    N = eigvecs.shape[0]
    # No degeneracy
    if indices.size == N:
        return shift_basis_vec(eigvecs, vec_pert)
    
    # First index labels the eigenvector, the second two are the projector
    eigproj = np.einsum('ij,kj->jik', eigvecs, eigvecs.conj(), optimize='greedy')
    proj_pert = np.einsum('ij,kj->jik', vec_pert, vec_pert.conj(), optimize='greedy')
    degen_proj = constr_proj(eigproj, indices, counts)
    
    shift = -1
    min_dist = N
    for i in np.arange(N):
        shifted_proj = constr_proj(np.roll(proj_pert, -i, axis=0), indices, counts)
        '''
        Distance heuristic: N - Σtr(P.P') where P' projects onto the perturbed space.
        This is equivalent to ||P-P'||^2 where ||A||^2 = tr(A*.A), and reduces to shift_basis_vec in the nondegenerate case.
        Maximal when P and P' are orthogonal, minimal when they coincide.
        '''
        dist = N - np.einsum('ijk,ikj', degen_proj, shifted_proj, optimize='greedy')
        if dist < min_dist:
            min_dist = dist
            shift = i
    if shift < 0:
        raise RuntimeError("Failed to find unperturbed basis ordering that matched the perturbed basis.")
    return shift
    
def proj_diag(start, stop, eigvecs, H_pert):
    """Diagonalize perturbed or perturbing Hamiltonian projected to the subspace indexed by start and stop."""
    degen_vecs = eigvecs[:,start:stop]
    # Perturbed quasi-Hamiltonian projected onto the degenerate subspace, in the unperturbed eigenbasis
    H_proj = degen_vecs.conj().T @ H_pert @ degen_vecs
    assert np.allclose(H_proj.conj().T, H_proj), 'Projected quasi-Hamiltonian is not Hermitian'
    _, good_basis = linalg.eigh(H_proj)
    new_vecs = degen_vecs @ good_basis
    eigvecs[:,start:stop] = new_vecs
    return eigvecs

def good_basis(U, U_pert):
    """
    Basis in degerate subspaces is arbitrary, so find the good basis with respect to the perturbed evolution U_pert,
    as in first-order degenerate perturbation theory.
    The good basis is adiabatically connected to the perturbed basis as we vary the perturbation strength.

    Returns the energies and (good) eigenvectors in the order that matches eig_sort(U_pert).
    Note this means the energy values and ordering may be shifted from eig_sort(U)
    if the perturbation causes energies to cross the branch cut.
    """
    en_pert, vec_pert = eig_sort(U_pert)
    return good_basis_shifted(U, en_pert, vec_pert)

def good_basis_set(unitaries):
    """
    Computes the good bases for a list of unitaries.
    Assumes they are sorted by ascending perturbation strength.
    Returns energies[i,j] and bases[i,j,k]
        i indexes the unitary
        j indexes the eigenvector
        k indexes the vector components
    """
    num = unitaries.shape[0]
    dim = unitaries.shape[1]
    energies = np.empty((num, dim))
    bases = np.empty((num, dim, dim), dtype=np.complex128)

    # Start with the largest perturbation and work downwards
    energies[-1,:], bases[-1,:] = eig_sort(unitaries[-1,:,:])
    for i in np.arange(num-2, -1, -1):
        energies[i,:], bases[i,:] = good_basis_shifted(unitaries[i,:,:], energies[i+1,:], bases[i+1,:])
    
    return energies, bases

def good_basis_shifted(U, en_pert, vec_pert):
    """
    Finds the good basis as in good_basis(), but explicitly uses the ordering of en_pert and vec_pert.
    These may differ from eig_sort(U_pert) due to previous branch cut crossings.
    The function assumes en_pert and vec_pert are sorted.
    Returns the energies and (good) eigenvectors in the order that matches vec_pert.

    WARNING: Has a fixed tolerance of 10^-10 when finding pseudo-degenerate subspaces.
             If using higher precision, this function should be edited.
    """
    # Energies are in (-π,π] and have about ~15 sig figs of precision.
    # This precision is likely lower due to numerical errors throughout the computations.
    # The function np.unique uses '==', so we need to manually implement the tolerance.
    energy, eigvecs = eig_sort(U)
    _, indices, counts = np.unique(np.around(energy, decimals=10), return_index=True, return_counts=True)
    # Account for pseudodegeneracy across the branch cut
    if np.isclose(energy[0] + 2*np.pi, energy[-1]):
        indices = np.delete(indices, 0)
        counts[-1] += counts[0]
        counts = np.delete(counts, 0)
    
    shift = shift_basis_proj(eigvecs, vec_pert, indices, counts)
    # Can compare to shift_bases_en for bug-checking. Note shift_basis_en fails for U=1.
    # shift_en = shift_basis_en(energy, en_pert)
    # assert shift == shift_en, 'Energy shift: {0}, vector shift: {1}\nEnergies: {2}'.format(shift_en, shift, energy/np.pi)
    
    '''
    When projected onto a degenerate subspace of U, U_pert may not be unitary
    due to the connection of eigenvectors outside the degenerate subspace.
    However, the quasi-Hamiltonian H_pert will still be Hermitian when projected.
    Symbolically, for some projector P, unitary U, and Hermitian H:
      (PUP)*(PUP)=PU*PUP=1 is not guaranteed, but (PHP)*=PHP is.
    We first keep U sorted and shift en_pert, vec_pert to match U for computational simplicity.
    After the computation, we will shift the indices of U to match the original vec_pert.
    '''
    vec_pert = np.roll(vec_pert, -shift, axis=1)
    en_pert = np.roll(en_pert, -shift)
    H_pert = vec_pert @ np.diag(en_pert) @ vec_pert.conj().T
    
    # Projected diagonalization
    N = energy.size
    for i in np.arange(indices.size):
        if counts[i] > 1:
            start = indices[i]
            stop = start + counts[i]
            # Pseudodegeneracy across the branch cut
            if stop > N:
                degen_shift = stop % N
                vecs_shift = np.roll(eigvecs, degen_shift, axis=1)
                H_shift = np.roll(H_pert, degen_shift, axis=1)
                vecs_shift = proj_diag(0, stop + degen_shift, vecs_shift, H_shift)
                eigvecs = np.roll(vecs_shift, -degen_shift, axis=1)
            else:
                eigvecs = proj_diag(start, stop, eigvecs, H_pert)
            
    assert np.allclose(eigvecs.conj().T @ U @ eigvecs,
                       np.diag(np.exp(1j*energy))), 'New basis is not an eigenbasis'
    
    energy = np.roll(energy, shift)
    '''
    Perturbation should not shift the majority of the spectrum over the branch cut,
    so we can adjust the branch of the energies that crossed the cut to keep them adiabatic/sorted.
    For example, an energy that goes π --> π+ϵ gets assigned to -π+ϵ, so we adjust it back.
    '''
    if shift < N//2:
        energy[:shift] -= 2*np.pi
    else:
        energy[shift-1:] += 2*np.pi
    eigvecs = np.roll(eigvecs, shift, axis=1)
    return energy, eigvecs
