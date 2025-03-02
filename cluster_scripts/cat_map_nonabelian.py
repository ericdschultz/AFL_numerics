"""
Computes AFL entropy over time on the perturbed cat map.
Fixed perturbation and dimension, varying partition type.
Saves the entropies in a .npy file.
The first element of each array is the size of the partition in each charge sector.
The partition types (in the order of the saved .npy) are
    randproj, W-symmetric, R-symmetric, nonabelian (commutant)

Command:
python -O cat_map_nonabelian.py a b c d pert N psize

a b c d - cat map matrix elements
pert - perturbation strength
N - Hilbert space dimension
psize - total size of the partition
        number of projectors in each charge sector:
        R - psize / s
        W - psize / 2
        nonabelian - 2*psize/(s+3) <-- center has dimension (s+3)/2
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
import warnings

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions
from AFL.tools.floquet import eig_sort

def main(matrix, pert, N, psize):
    A = np.array(matrix).reshape((2,2))

    kmax = cat_map.max_pert(A, 1, 0)
    if pert > kmax:
        raise ValueError('Perturbation strength exceeds Anosov bound of {0}'.format(kmax))

    shear = cat_map.typ_pshear(N, pert)
    U = cat_map.cat_unitary_gauss(N, A) @ shear
    _, eigs = eig_sort(U)
    weights = np.ones(N) / N
    max_time = entropy.time_estimate(N)

    file = 'cat_nonabel_{}_k{}_N{}_proj{}'.format('-'.join(map(str, matrix)),
                                               str(pert).replace('.','p'),
                                               N, psize)
    
    s = np.gcd(A[0,1], A[1,1]-1)
    if s % 2 == 0:
        s //= 2
    
    centerdim = (s + 3) // 2
    psizes = np.array([psize, psize//s, psize//2, psize//centerdim])
    if not (psize % 2 == 0 and psize % s == 0 and psize % centerdim == 0):
        warnings.warn("Desired total partition size not divisible by abelian dimensions. Partitions will have different sizes.")
    
    Xs = []
    Xs.append(partitions.get_qmap_partition('randproj', N, psizes[0]))

    vecs, inds = cat_map.cat_W_vecs(N)
    Xs.append(partitions.sym_partition(vecs, inds, psizes[2], randomize=True))

    vecs, inds = cat_map.cat_R_vecs(N, s)
    Xs.append(partitions.sym_partition(vecs, inds, psizes[1], randomize=True))

    rep_dims = cat_map.nonabelian_dims(N, s)
    rep_basis = cat_map.rep_to_qbasis(N, s)
    Xs.append(partitions.nonabelian_partition(rep_dims, rep_basis, psizes[3]))


    # Multiprocessing
    pool = Pool(4)
    AFL_func = partial(entropy.AFL_entropy_helper, weights, U=U, eigvecs=eigs, max_time=max_time)
    entropies = pool.map(AFL_func, Xs, 1)

    pool.close()

    # Make all entropy lists of the same length
    min_time = min(map(len, entropies))
    ent_arr = np.array([ent[:min_time] for ent in entropies])

    data = np.insert(ent_arr, 0, psizes, axis=1)
    np.save('data/' + file, data)
    # # The entropies reach max precision at different times, so save them separately
    # labels = ['rand', 'R', 'W', 'comm']
    # for i in range(4):
    #     data = np.insert(entropies[i], 0, psizes[i])
    #     file = 'cat_nonabel_{}_{}'.format(labels[i], file_end)
    #     np.save('data/' + file, data)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=int, nargs=4,
                        help="matrix elements a b c d")
    parser.add_argument("pert", type=float,
                        help="perturbation strength")
    parser.add_argument("N", type=int,
                        help="Hilbert space dimension")
    parser.add_argument("psize", type=int,
                        help="size of the partitions")
    args = parser.parse_args()
    main(args.matrix, args.pert, args.N, args.psize)
