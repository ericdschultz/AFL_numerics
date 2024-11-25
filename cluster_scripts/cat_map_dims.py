"""
Computes AFL entropy over time on the perturbed cat map.
Fixed perturbation and partition; varying dimension.
Saves the entropies in an .npy file.
The first element of each array is the Hilbert space dimension

This file runs each dimension in parallel using multiprocessing.

Command:
python -O cat_map_dims.py a b c d pert partition psize dims

a b c d - cat map matrix elements
pert - perturbation strength
partition - qmap partition name (see partitions.py)
psize - size of the partition
dims - the Hilbert space dimensions to compute
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions
from AFL.tools.floquet import eig_sort

def AFL_args(A, pert, partition, psize, N):
    U = cat_map.cat_unitary_gauss(N, A, pert)
    _, eigs = eig_sort(U)
    X = partitions.get_qmap_partition(partition, N, psize)
    weights = np.ones(N) / N
    return (weights, X, U, eigs)

def main(matrix, pert, partition, psize, dims, test=False):
    A = np.array(matrix).reshape((2,2))

    file = 'cat_dims_{}_k{}_{}{}_N{}-{}'.format('-'.join(map(str, matrix)),
                                         str(pert).replace('.','p'),
                                         partition, psize, dims[0], dims[-1])
    
    # Multiprocessing
    pool = Pool(len(dims))

    arg_func = partial(AFL_args, A, pert, partition, psize)
    args = pool.map(arg_func, dims, 1)
    entropies = pool.starmap(entropy.AFL_entropy_helper, args, 1)

    pool.close()

    # Make all entropy lists of the same length
    min_time = min(map(len, entropies))
    ent_arr = np.array([ent[:min_time] for ent in entropies])

    data = np.insert(ent_arr, 0, dims, axis=1)
    if not test:
        np.save('data/' + file, data)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=int, nargs=4,
                        help="matrix elements a b c d")
    parser.add_argument("pert", type=float,
                        help="perturbation strength")
    parser.add_argument("partition",
                        help="qmap partition name")
    parser.add_argument("psize", type=int,
                        help="size of the partition")
    parser.add_argument("dims", type=int, nargs='+',
                        help="Hilbert space dimensions")
    parser.add_argument("-test", action="store_true",
                        help="Will not save result. For testing purposes.")
    args = parser.parse_args()
    main(args.matrix, args.pert, args.partition, args.psize, args.dims, test=args.test)
