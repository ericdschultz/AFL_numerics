"""
Functions for computing AFL entropy.

Possible improvements:
  - Only compute positive eigenvalues (power iteration, Lanczos, LOBPCG)
  - Use better einsum libraries like opt-einsum
  - Reshape returns a view (not copy) when possible, but the stride could be causing cache misses
"""
import numpy as np
import scipy.linalg as linalg
from scipy.special import xlogy
import AFL.tools.floquet as floquet

def time_estimate(dim, dtype=np.float64):
    """Slight overestimation for the time it takes to reach machine precision for chaotic AFL."""
    # Takes O(log(dim)) steps to get close to the bound (for chaotic systems)
    # After that, we lose O(1) bit of precision per time step
    return 5 * int(np.log(dim)) - (np.finfo(dtype).machep) // 2

def AFL_entropy(weights, X, U, max_time=None):
    """
    Computes the AFL entropy.
    
    weights: parameterize the mixed state ρ = Σ weights[i]|i><i|
      |i> are stationary states so ρ is time-invariant
      Eigenstates are sorted by quasienergy on (-π,π]
    X: initial partition of unity (list of matrices)
    U: unitary time evolution matrix
    max_time: time steps to compute
    
    Returns list of the AFL entropies at each time step.
    """
    energy, eigvecs = floquet.eig_sort(U)
    return AFL_entropy_helper(weights, X, U, eigvecs, max_time=max_time)

def degen_AFL_entropy(weights, X, U, U_pert, max_time=None):
    """Compute the AFL entropy in the degenerate case using the good basis."""
    energy, eigvecs = floquet.good_basis(U, U_pert)
    return AFL_entropy_helper(weights, X, U, eigvecs, max_time=max_time)

def AFL_entropy_set(weights, partitions, unitaries, max_time=None):
    """
    Compute the AFL entropy for a set of unitaries.
    unitaries: iterable of different sized unitaries.
    partitions: iterable of respective partitions
    weights: iterable of weight lists
    If there are fewer partitions or weights than unitaries, the function wraps back around.
    To compute on a single partition X, for example, use [X] as the argument.
    """
    num = len(unitaries)
    if max_time is None:
        max_dim = max([U.shape[0] for U in unitaries])
        max_time = time_estimate(max_dim)
    entropies = np.empty((num, max_time))
    for i in np.arange(num):
        entropy = AFL_entropy(weights[i % len(weights)], partitions[i % len(partitions)], unitaries[i], max_time=max_time)
        max_time = min(max_time, entropy.size)
        entropies[i,:max_time] = entropy[:max_time]
    return entropies[:,:max_time]

def AFL_entropy_perts(weights, X, unitaries, max_time=None):
    """
    Compute the AFL entropy for a set of unitaries adiabatically connected by a perturbation.
    Assumes unitaries are sorted by ascending perturbation strength.
    Returns entropies[i,j]
        i indexes the unitary
        j indexes the timestep
    """
    energies, bases = floquet.good_basis_set(unitaries)
    num = unitaries.shape[0]
    if max_time is None:
        max_time = time_estimate(unitaries.shape[1])
    entropies = np.empty((num, max_time))
    for i in np.arange(num):
        entropy = AFL_entropy_helper(weights, X, unitaries[i,:,:], bases[i,:,:], max_time=max_time)
        max_time = min(max_time, entropy.size)
        entropies[i,:max_time] = entropy[:max_time]
    return entropies[:,:max_time]

def thermal_AFL_entropy(betas, X, U, max_time=None):
    """
    Computes AFL entropy on a set of thermal states with inverse temperatures 'betas'.
    Returns entropies[i,j] and bounds[i] (AFL entropy bound)
        i indexes the inverse temperature
        j indexes the time step
    """
    num = betas.size
    dim = U.shape[1]
    if max_time is None:
        max_time = time_estimate(dim)
    entropies = np.empty((num, max_time))
    bounds = np.empty(num)

    energy, eigvecs = floquet.eig_sort(U)
    for i in np.arange(num):
        boltzmann = np.exp(-betas[i] * energy)
        weights = boltzmann / np.sum(boltzmann)
        bounds[i] = np.log(np.log(dim)) - np.sum(xlogy(weights, weights))
        entropy = AFL_entropy_helper(weights, X, U, eigvecs, max_time=max_time)
        max_time = min(max_time, entropy.size)
        entropies[i,:max_time] = entropy[:max_time]

    return entropies[:,:max_time], bounds

def thermal_AFL_set(betas, X, unitaries, max_time=None):
    """
    Computes AFL entropy for a set of unitaries on a set of thermal states with inverse temperatures 'betas'.
    Returns entropies[i,j,k] and bounds[i,j] (AFL entropy bound)
        i indexes the unitary
        j indexes the inverse temperature
        k indexes the time step
    """
    num_b = betas.size
    num_U = unitaries.shape[0]
    dim = unitaries.shape[1]
    if max_time is None:
        max_time = time_estimate(dim)
    entropies = np.empty((num_U, num_b, max_time))
    bounds = np.empty((num_U, num_b))

    energies, bases = floquet.good_basis_set(unitaries)
    for i in np.arange(num_U):
        for j in np.arange(num_b):
            boltzmann = np.exp(-betas[j] * energies[i,:])
            weights = boltzmann / np.sum(boltzmann)
            bounds[i,j] = np.log(dim) - np.sum(xlogy(weights, weights))
            entropy = AFL_entropy_helper(weights, X, unitaries[i,:,:], bases[i,:,:], max_time=max_time)
            max_time = min(max_time, entropy.size)
            entropies[i,j,:max_time] = entropy[:max_time]

    return entropies[:,:,:max_time], bounds

def AFL_entropy_helper(weights, X, U, eigvecs, max_time=None, tol=1e-13):
    """
    Does the heavy lifting for AFL_entropy.
    Runs until numerical precision or max_time is reached.
    """
    N = U.shape[0]
    if max_time is None:
        max_time = time_estimate(N)
    
    # Construct ρ^(1/2)
    sqrt_state = np.einsum('a,ia,ja->ij', np.sqrt(weights), eigvecs, eigvecs.conj(), optimize='greedy')
    assert np.allclose(sqrt_state @ U, U @ sqrt_state), 'State is not stationary'
    full_state = np.einsum('ij,kl->ijkl', sqrt_state, sqrt_state.conj(), optimize='greedy')
    assert np.isclose(np.einsum('ijij', full_state), 1.0), 'Initial state is not normalized'
    
    # Optimal contraction
    UX = np.einsum('ij,ajk->aik', U, X, optimize='greedy')
    # UXc = UX.conj()
    # path = np.einsum_path('nij,nab,jkbc->ikac', UX, UXc, full_state, optimize='optimal')[0]

    # Using transfer takes more memory but has fewer contractions per time step
    transfer = np.einsum('aij,akl->ijkl', UX, UX.conj(), optimize='greedy')
    path = np.einsum_path('ijab,jkbc->ikac', transfer, full_state, optimize='optimal')[0]
    
    # Compute entropy
    entropies = np.empty(max_time)
    entropies[-1] = 0
    for i in np.arange(max_time):
        #full_state[:] = np.einsum('nij,nab,jkbc->ikac', UX, UXc, full_state, optimize=path)
        full_state[:] = np.einsum('ijab,jkbc->ikac', transfer, full_state, optimize=path)
        flat_state = full_state.reshape((N*N, N*N))
        assert np.isclose(np.trace(flat_state), 1.0), 'State is not normalized'
        
        eigs = linalg.eigvalsh(flat_state)
        assert eigs[0] > -1e-12, 'State is not positive'
        pos_eigs = eigs[np.nonzero(eigs > 0)]
        entropies[i] = -np.sum(pos_eigs * np.log(pos_eigs))
        
        # Check whether we have reached machine precision
        rel_diff = np.abs((entropies[i] - entropies[i-1]) / entropies[i])
        if rel_diff < tol:
            entropies = entropies[:i+1]
            break
        
    return entropies

    
def AFL_entropy_helper_alt(weights, X, U, max_time, eigvecs):
    """
    Computes AFL entropy by directly purifying to N^2 x N^2 matrices.
    This is significantly slower than the other method (perhaps due to memory accesses?).
    This function goes unused, but remains here to document our progress.
    """
    N = U.shape[0]
    
    # Construct purification
    GNS_vec = np.zeros(N*N, dtype=np.complex128)
    for i in np.arange(N):
        eigvec = eigvecs[:,i]
        GNS_vec += np.sqrt(weights[i]) * np.kron(eigvec, eigvec)
    GNS_state = np.einsum('i,j->ij', GNS_vec, GNS_vec.conj(), optimize='greedy')
    assert np.isclose(np.trace(GNS_state), 1.0), 'Initial state is not normalized'
    
    # Optimal contraction
    I = np.eye(N)
    UX = np.einsum('ij,ajk->aik', np.kron(U, I), np.kron(X, I), optimize='greedy')
    UXc = UX.conj()
    path = np.einsum_path('nij,jk,nlk', UX, GNS_state, UXc, optimize='optimal')[0]
    
    # Compute entropy
    entropies = np.empty(max_time)
    for i in np.arange(max_time):
        GNS_state[:] = np.einsum('nij,jk,nlk', UX, GNS_state, UXc, optimize=path)
        assert np.isclose(np.trace(GNS_state), 1.0), 'State is not normalized'
        
        eigs = linalg.eigvalsh(GNS_state)
        assert eigs[0] > -1e-12, 'State is not positive'
        pos_eigs = eigs[np.nonzero(eigs > 0)]
        entropies[i] = -np.sum(pos_eigs * np.log(pos_eigs))
        
    return entropies

'''
Functions using Numba's @jit.
Unfortunately, both functions seem to be slower than bare Numpy (optimal einsum is very good).
The alternative (N^2,N^2) method does better here than the (N,N,N,N) tensor.
'''

'''
from numba import jit, prange

def AFL_entropy_jit(weights, X, U):
    """ Compute the AFL entropy using Numba's @jit """
    energy, eigvecs = floquet.eig_sort(U)
    return AFL_entropy_helper_jit(weights, X, U, eigvecs)

def AFL_entropy_alt_jit(weights, X, U):
    """ Compute the AFL entropy using Numba's @jit (alternative method) """
    energy, eigvecs = floquet.eig_sort(U)
    return AFL_entropy_helper_alt_jit(weights, X, U, eigvecs)

@jit(parallel=True)
def AFL_entropy_helper_jit(weights, X, U, eigvecs, tol=1e-13):
    """
    Version of AFL_entropy_helper compatible with Numba @jit (no einsum)
    """
    N = U.shape[0]
    max_time = 10 * int(np.log(N)) + 50

    # Purification
    sqrt_state = np.empty((N,N), dtype=np.complex128)
    for i in np.arange(N):
        for j in np.arange(N):
            sqrt_state[i,j] = np.sum(np.sqrt(weights) * eigvecs[i,:] * np.conj(eigvecs[j,:]))

    state = np.empty((N,N,N,N), dtype=np.complex128)
    transfer = np.empty_like(state)
    
    UX = np.empty_like(X, dtype=np.complex128)
    for i in np.arange(X.shape[0]):
        UX[i,:,:] = U @ X[i,:,:].astype(np.complex128)

    for i in np.arange(N):
        for j in np.arange(N):
            for a in np.arange(N):
                for b in np.arange(N):
                    state[i,j,a,b] = sqrt_state[i,j] * np.conj(sqrt_state[a,b])
                    transfer[i,j,a,b] = np.sum(UX[:,i,j] * np.conj(UX[:,a,b]))

    new_state = np.empty_like(state)
    entropies = np.empty(max_time)
    entropies[-1] = 0
    for t in np.arange(max_time):
        for i in prange(N):
            for k in prange(N):
                for a in prange(N):
                    for c in prange(N):
                        new_state[i,k,a,c] = np.sum(transfer[i,:,a,:] * state[:,k,:,c])
        state[:] = new_state[:]
        mat_state = state.reshape((N*N, N*N))
        eigs = np.linalg.eigvalsh(mat_state)
        pos_eigs = eigs[np.nonzero(eigs > 0)]
        entropies[t] = -np.sum(pos_eigs * np.log(pos_eigs))

        # Check whether we have reached machine precision
        rel_diff = np.abs((entropies[t] - entropies[t-1]) / entropies[t])
        if rel_diff < tol:
            entropies = entropies[:t+1]
            break

    return entropies

@jit
def AFL_entropy_helper_alt_jit(weights, X, U, eigvecs, tol=1e-13):
    """ Version of AFL_entropy_helper_alt compatible with Numba @jit """
    N = U.shape[0]
    max_time = 10 * int(np.log(N)) + 50

    GNS_vec = np.zeros(N*N, dtype=np.complex128)
    for i in np.arange(N):
        eigvec = eigvecs[:,i]
        GNS_vec += np.sqrt(weights[i]) * np.kron(eigvec, eigvec)
    GNS_mat = np.outer(GNS_vec, GNS_vec.conj())
    
    I = np.eye(N, dtype=np.complex128)
    UX = np.empty((X.shape[0], N*N, N*N), dtype=np.complex128)
    for i in np.arange(X.shape[0]):
        UX_bare = U @ X[i,:,:].astype(np.complex128)
        UX[i,:,:] = np.kron(UX_bare, I)
    
    entropies = np.empty(max_time)
    entropies[-1] = 0
    for t in np.arange(max_time):
        new_state = np.zeros_like(GNS_mat)
        for i in np.arange(X.shape[0]):
            new_state[:] += UX[i,:,:] @ GNS_mat @ UX[i,:,:].conj().T
        GNS_mat = new_state

        eigs = np.linalg.eigvalsh(GNS_mat)
        pos_eigs = eigs[np.nonzero(eigs > 0)]
        entropies[t] = -np.sum(pos_eigs * np.log(pos_eigs))

        # Check whether we have reached machine precision
        rel_diff = np.abs((entropies[t] - entropies[t-1]) / entropies[t])
        if rel_diff < tol:
            entropies = entropies[:t+1]
            break

    return entropies
'''


''' Pure state AFL functions. Have not been updated to not need max_time.'''


def pure_AFL_entropy(X, U, max_time):
    """
    Computes AFL entropy of the pure stationary states.
    Returns entropies[i,j] and energy[i]
        i indexes the eigenstate sorted by quasienergy
        j indexes the time step
    """
    energy, eigvecs = floquet.eig_sort(U)
    entropies = pure_AFL_helper(eigvecs, X, U, max_time)
    return entropies, energy

def degen_pure_AFL(X, U, U_pert, max_time):
    """Compute AFL entropy of the pure states in the degenerate case using the good basis."""
    energy, eigvecs = floquet.good_basis(U, U_pert)
    entropies = pure_AFL_helper(eigvecs, X, U, max_time)
    return entropies, energy

def pure_AFL_set(X, unitaries, max_time):
    """
    Compute the AFL entropy of the pure states for a set of unitaries.
    Assumes unitaries are sorted by ascending perturbation strength.
    Returns entropies[i,j,k] and energies[i,j]
        i indexes the unitary
        j indexes the eigenstate sorted by quasienergy
        k indexes the time step
    """
    energies, bases = floquet.good_basis_set(unitaries)
    num = unitaries.shape[0]
    N = unitaries.shape[1]
    entropies = np.empty((num, N, max_time))
    for i in np.arange(num):
        entropies[i,:,:] = pure_AFL_helper(bases[i,:,:], X, unitaries[i,:,:], max_time)
    return entropies, energies

def pure_AFL_helper(eigvecs, X, U, max_time):
    """Does the heavy lifting for computing the pure state AFL entropy."""
    N = U.shape[0]
    
    # Eigenstate density matrices
    states = np.einsum('ij,kj->jik', eigvecs, eigvecs.conj(), optimize='greedy')
    assert np.allclose(states @ U, U @ states), 'State is not stationary'
    
    # Optimal contraction
    UX = np.einsum('ij,ajk->aik', U, X, optimize='greedy')
    UXc = UX.conj()
    path = np.einsum_path('nij,nab,mjb->mia', UX, UXc, states, optimize='optimal')[0]
    
    # Compute entropy
    entropies = np.empty((N, max_time))
    eigs = np.empty((N, N))
    for i in np.arange(max_time):
        states[:] = np.einsum('nij,nab,mjb->mia', UX, UXc, states, optimize=path)
        assert np.allclose(np.einsum('ijj', states, optimize='greedy'),
                           np.ones(N)), 'State is not normalized'
        
        eigs[:] = np.linalg.eigvalsh(states)
        assert np.all(eigs[:,0] > -1e-12), 'State is not positive'
        eigs[eigs < 0] = 0 # Removes small negatives arising from numerical error
        entropies[:,i] = -np.sum(xlogy(eigs, eigs), axis=1)

    return entropies
    

''' Other helpful functions. '''

def sym_dims(V):
    """
    Given a Hermitian symmetry operator V, returns the (pseudo-)degeneracies of the eigenvalues.
    This gives the dimensions of the Krylov subspaces of the dynamics with distinct charges.
    """
    if np.allclose(V, V.conj().T):
        eigvals = np.linalg.eigvalsh(V)
        _, __, counts = np.unique(np.around(eigvals, decimals=10), return_index=True, return_counts=True)
        return counts
    raise ValueError('Argument to sym_dims must be Hermitian.')

def sym_bound(V, print_dims=False):
    """
    HAS NOT BEEN UPDATED WITH THE MORE GENERAL BOUND
    Given a Hermitian symmetry operator V, returns the associated dimensional bound of AFL entropy
    for a full-rank density matrix (faithful state).
    Each D-dimensional Krylov space of the N-dimensional Hilbert space has AFL bound 2*log(D).
    The full bound is the sum of each of these bounds weighted by their proportion of the Hilbert space D/N,
    with the additional Shannon entropy of picking the Krylov subspace.
    """
    dims = sym_dims(V)
    if print_dims:
        print(dims)
    p = dims / np.sum(dims)
    shannon = -np.sum(p * np.log(p))
    bound = 2 * np.sum(p * np.log(dims)) + shannon
    return bound

'''
The AFL quantum channel has Kraus operators (U X_i) applied repeatedly.
The Louiville/Hilbert-Schmidt representation is then a transfer matrix:
  T = Σ_i (U X_i) tensor (U X_i)*
where '*' means complex conjugation in given basis.
T may not be diagonalizable, but will have generalized eigenvalues |λ| <= 1.
Unity is always an eigenvalue with left-eigenvector corresponding to Id (trace dual of channel is unital).

AFL entropy asymptotically approaches its dimensional bound as
  S(t) = S_max - const*exp(-μt)
for constant exp(-μ) < 1, the modulus of the subleading eigenvalue of T.
'''

def transfer_eigvals(U, X, modulus=False):
    """
    Returns the eigenvalues of the transfer matrix corresponding to the AFL quantum channel.
    If modulus=True, instead return the modulus of the eigvals sorted in descending order.
    """
    k = X.shape[0]
    N = U.shape[0]
    T = np.zeros((N*N, N*N), dtype=np.complex128)
    UX = np.einsum('ab,ibc->iac', U, X, optimize='greedy')
    for i in np.arange(X.shape[0]):
        UXi = UX[i,:,:]
        T += np.kron(UXi, UXi.conj())
    eigvals = linalg.eigvals(T)

    if modulus:
        return np.flip(np.sort(np.abs(eigvals)))
    return eigvals

def transfer_eigvals_set(unitaries, X, modulus=False):
    """
    Returns the eigenvalues of the transfer matrices corresponding to the AFL quantum channels.
    If modulus=True, instead return the modulus of the eigvals sorted in descending order for each unitary.
    """
    num = unitaries.shape[0]
    dim = unitaries.shape[1]
    dtype = np.float64 if modulus else np.complex128
    eigvals = np.empty((num, dim*dim), dtype=dtype)
    for i in np.arange(num):
        eigvals[i,:] = transfer_eigvals(unitaries[i], X, modulus=modulus)
    return eigvals
