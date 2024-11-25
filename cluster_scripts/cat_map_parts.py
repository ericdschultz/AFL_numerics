"""
Computes AFL entropy over time on the perturbed cat map.
Fixed perturbation and dimension; varying partition size.
Saves the entropies in an .npy file.
The first element of each array is partition size.

This file runs each dimension in parallel using multiprocessing.

Command:
python -O cat_map_parts.py a b c d pert N partition psizes

a b c d - cat map matrix elements
pert - perturbation strength
N - Hilbert space dimension
partition - qmap partition name (see partitions.py)
psizes - size of the partitions
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions
from AFL.tools.floquet import eig_sort

def main(matrix, pert, N, partition, psizes, test=False):
    A = np.array(matrix).reshape((2,2))
    U = cat_map.cat_unitary_gauss(N, A, pert)
    _, eigs = eig_sort(U)
    weights = np.ones(N) / N
    max_time = int(3 * np.log(N) / np.log(min(psizes)))

    file = 'cat_parts_{}_k{}_N{}_{}'.format('-'.join(map(str, matrix)),
                                         str(pert).replace('.','p'),
                                         N, partition)
    
    # Multiprocessing
    pool = Pool(len(psizes))

    qmap_func = partial(partitions.get_qmap_partition, partition, N)
    Xs = pool.map(qmap_func, psizes)

    AFL_func = partial(entropy.AFL_entropy_helper, weights, U=U, eigvecs=eigs, max_time=max_time)
    entropies = pool.map(AFL_func, Xs, 1)

    pool.close()

    # Make all entropy lists of the same length
    min_time = min(map(len, entropies))
    ent_arr = np.array([ent[:min_time] for ent in entropies])

    data = np.insert(ent_arr, 0, psizes, axis=1)
    if not test:
        np.save('data/' + file, data)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=int, nargs=4,
                        help="matrix elements a b c d")
    parser.add_argument("pert", type=float,
                        help="perturbation strength")
    parser.add_argument("N", type=int,
                        help="Hilbert space dimension")
    parser.add_argument("partition",
                        help="qmap partition name")
    parser.add_argument("psizes", type=int, nargs='+',
                        help="sizes of the partitions")
    parser.add_argument("-test", action="store_true",
                        help="Will not save result. For testing purposes.")
    args = parser.parse_args()
    main(args.matrix, args.pert, args.N, args.partition, args.psizes, test=args.test)
