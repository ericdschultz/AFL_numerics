"""
Constructs various partitions/POVMs for use in AFL entropy.
"""
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import schur
from AFL.dynamics.bakers_map import dft

def projector_partition(dim, num):
    """
    Partition is a set of num projectors on a dim-dimensional space.
    Each projector has rank dim/num and is diagonal.
    If num does not divide dim, the last projector has smaller rank.
    This is measuring in the q-basis of the cat map.
    """
    X = np.zeros((num, dim, dim))
    rank = int(np.ceil(dim / num))
    for i in np.arange(num):
        start = i * rank
        end = min(start + rank, dim)
        for j in np.arange(start, end):
            X[i,j,j] = 1
    return X
    
def conjugate_partition(X, phase=0):
    """
    Returns the Fourier conjugate to the partition X.
    'phase' is an optional parameter for the genearlized/shifted DFT.
    See AFL.dynamics.bakers_map for details.
    """
    F = dft(X.shape[1], q_phase=phase, p_phase=phase)
    P = np.einsum('ji,ajk,kl->ail', F.conj(), X, F, optimize='greedy')
    return P
    
def prob_measurement(K, p):
    """
    Returns Kraus operators corresponding to performing, with probability p,
    the measurement associated with the input Kraus operators K.
    """
    dim = K.shape[1]
    X1 = np.sqrt(p) * K
    X2 = np.array([np.sqrt(1-p) * np.eye(dim)])
    X = np.append(X1, X2, axis=0)
    return X

def anticomm_partition(U, num, mode=3):
    """
    Returns a "local" partition in the presence of an anticommuting unitary.

    If a unitary anticommutes with U, it induces a tensor product (C^2 ⊗ H) where
    C^2 is a pseudospin. In some basis, the dynamics looks like U = Sz ⊗ Uz.
    From this, we construct a partition (Xs ⊗ X) where X is a projector partition and Xs
    is a projective measurement of pseudospin or is identity.

    U - unitary dynamics
    num - number of projectors in each pseudospin subspace (size of X)
    mode - 1,2,3 corresponding to measuring pseudospin x,y,z OR 0 for identity (Xs partition)
    """
    if mode not in range(4):
        raise ValueError("mode must be in [0,1,2,3]")

    N = U.shape[0]
    M = N // 2
    if N % 2 != 0:
        raise ValueError("Unitary must be even-dimensional.")
    if not np.allclose(U.conj().T @ U, np.eye(U.shape[0])):
        raise ValueError("Argument is not unitary.")

    # schur over eig since eigvecs is guaranteed to be unitary
    # eigvecs[i,j] is the i-th component of the j-th eigenvector
    eigval_mat, eigvecs = schur(U)
    eigvals = np.diag(eigval_mat)
    indices = np.argsort(np.angle(eigvals)) # Branch is (-π, π]
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:,indices]

    if not np.allclose(eigvals[:M], -eigvals[M:]):
        raise RuntimeError("No pseudospin basis found.")
    
    # Must perform a change of basis of the form W = (R ⊗ V).T
    # for pseudospin rotation R and some (random) unitary V.
    # Otherwise, partition will commute with the dynamics.
    # Note the transpose since we are acting direclty on the basis instead of on components.
    rots = [np.eye(2), np.array([[1,1],[1,-1]]) / np.sqrt(2), np.array([[1,1],[1j,-1j]]) / np.sqrt(2)]
    R = rots[mode % 3]
    V = unitary_group.rvs(M)
    vecs = eigvecs @ np.kron(R,V)

    part = sym_partition(vecs, np.array([0,M]), num)
    if mode > 0:
        return part
    # K==num unless num is larger than a pseudospin sector
    K = part.shape[0] // 2
    return part[:K,:,:] + part[K:,:,:]
    

def sym_op_partition(V, num, mixing=False, randomize=False):
    """
    Given a Hermitian symmetry operator V, returns a partition of symmetry eigenstate projectors.
    """
    # eigvecs[i,j] is the i-th component of the j-th eigenvector
    eigvals, eigvecs = np.linalg.eigh(V)
    _, indices = np.unique(np.around(eigvals, decimals=10), return_index=True)
    return sym_partition(eigvecs, indices, num, mixing=mixing, randomize=randomize)

def sym_partition(eigvecs, indices, num, mixing=False, randomize=False):
    """
    Returns a symmetric partition.
    
    eigens - Charged eigvecstates sorted by charge
    indices - Indices where each distinct charge begins
    mixing is False
        The partition does not mix eigenspaces of distinct eigenvalues.
        num - how many projectors per distinct eigenspace
    mixing is True
        num - how many projectors cover the whole space.
        The projectors may mix eigenspaces of different eigenvalues.
    randomize
        False - The eigenbasis given is used as is.
        True - The basis within each eigenspace is randomly rotated.
               Note the eigenspace projectors are basis-independent but the partition is not.
    """
    N = eigvecs.shape[0]
    indices = np.append(indices, N)
    counts = np.diff(indices)
    if randomize:
        eigvecs = eigvecs.astype(np.complex128)
        for i in np.arange(len(counts)):
            dim = counts[i]
            start = indices[i]
            stop = start + dim
            eigspace = eigvecs[:,start:stop]
            for _ in range(4):
                V = unitary_group.rvs(dim)
                eigspace[:,:] = eigspace @ V
            eigvecs[:,start:stop] = eigspace
    eigproj = np.einsum('ij,kj->jik', eigvecs, eigvecs.conj(), optimize='greedy')

    if mixing:
        X = np.zeros((num, N, N), dtype=np.complex128)
        rank = int(np.ceil(N / num))
        for i in np.arange(num):
            start = i * rank
            stop = min(start + rank, N)
            for j in np.arange(start, stop):
                X[i,:,:] += eigproj[j,:,:]
        return X
    
    X = np.zeros((num * indices.size, N, N), dtype=np.complex128)
    ii = 0
    for i in np.arange(indices.size - 1):
        rank = int(np.ceil(counts[i] / num))
        for j in np.arange(num):
            start = j*rank + indices[i]
            stop = min(start + rank, indices[i+1])
            # The break and 'ii' (instead of i*rank+j) are used here
            # in case some eigspace has dimension counts[i] < num.
            if stop <= start:
                break
            for k in np.arange(start, stop):
                X[ii,:,:] += eigproj[k,:,:]
            ii += 1
    if ii < X.shape[0]:
        X = X[:ii,:,:]
    return X


''' Partitions specifically for quantum maps on a 2-torus phase space. '''

def clock_partition(N, num, phase=0):
    """This is the exponentiated projector partition (includes phases)."""
    C = np.zeros((num, N, N), dtype=np.complex128)
    rank = int(np.ceil(N / num))
    for i in np.arange(N):
        start = i * rank
        end = min(start + rank, N)
        for j in np.arange(start, end):
            C[i,j,j] = np.exp(2j*np.pi/N * (j + phase))
    return C

def weyl_partition(N, phase=0):
    """Returns a partition consisting of the clock and shift maps. Note this is the forward shift."""
    clock = np.diag(np.exp(2j*np.pi/N * (np.arange(N) + phase)))
    shift = np.roll(np.eye(N), 1, axis=0)
    return np.array([clock, shift]) / np.sqrt(2)

def get_qmap_partition(name, N, num, phase=0):
    """
    Convenience function for partitions identified by name.
    Partitions here are specifically for quantum maps on a 2-torus phase space.

    WARNING: For now, all partitions using momentum are for periodic boundary conditions.
    """
    if name == 'qproj':
        return projector_partition(N, num)
    if name == 'pproj':
        q_proj = projector_partition(N, num)
        return conjugate_partition(q_proj, phase=phase)
    if name == 'clock':
        return clock_partition(N, num)
    if name == 'qpproj':
        q_proj = projector_partition(N, num//2)
        p_proj = conjugate_partition(q_proj, phase=phase)
        return np.append(q_proj, p_proj, axis=0) / np.sqrt(2)
    if name == 'clockshift':
        clock = clock_partition(N, num//2)
        shift = conjugate_partition(clock.conj(), phase=phase)
        return np.append(clock, shift, axis=0) / np.sqrt(2)
    if name == 'weyl':
        return weyl_partition(N)
    if name == 'randproj':
        projs = projector_partition(N, num).astype(np.complex128)
        # Rotate basis randomly a few times
        # Rotating only once doesn't seem sufficient
        for _ in range(4):
            V = unitary_group.rvs(N)
            projs[:,:,:] = np.einsum('ij,ajk,lk->ail', V, projs, V.conj(), optimize='greedy')
        return projs
    raise Exception('Invalid partition name.')
