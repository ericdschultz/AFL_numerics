"""
Computes AFL entropy over time on the perturbed cat map.
Fixed perturbation and partition; varying dimension.
This is specifically for the cat W and R symmetries, so the partition is symmetrized
Saves the entropies in an .npy file.
The first element of each array is the Hilbert space dimension

This file runs each dimension in parallel using multiprocessing.

Command:
python -O cat_map_sym.py sym a b c d pert psize dims -mix -rand

sym - 'W' or 'R' for the respective cat map symmetry
a b c d - cat map matrix elements
pert - perturbation strength
psize - size of the partition in each charge sector
dims - the Hilbert space dimensions to compute
mix - If included, Kraus operators can mix different charge sectors
      and psize refers to the total number of partitions
rand - If included, randomize the basis in each charge sector
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions
from AFL.tools.floquet import eig_sort

def AFL_args(sym, A, pert, psize, mix, rand, N):
    U = cat_map.cat_unitary_gauss(N, A, pert)
    _, eigs = eig_sort(U)
    weights = np.ones(N) / N
    if sym == 'W':
        vecs, inds = cat_map.cat_W_vecs(N)
    else:
        vecs, inds = cat_map.cat_R_vecs(N, A[0,1])
    X = partitions.sym_partition(vecs, inds, psize, mixing=mix, randomize=rand)
    return (weights, X, U, eigs)

def main(sym, matrix, pert, psize, dims, mix=False, rand=True):
    A = np.array(matrix).reshape((2,2))

    file = 'cat_sym{}_{}_k{}_m{}r{}{}_N{}-{}'\
            .format(sym, '-'.join(map(str, matrix)), str(pert).replace('.','p'),
                    str(mix)[0], str(rand)[0], psize, dims[0], dims[-1])

    # Multiprocessing
    pool = Pool(len(dims))

    arg_func = partial(AFL_args, sym, A, pert, psize, mix, rand)
    args = pool.map(arg_func, dims, 1)
    entropies = pool.starmap(entropy.AFL_entropy_helper, args, 1)

    pool.close()

    # Make all entropy lists of the same length
    min_time = min(map(len, entropies))
    ent_arr = np.array([ent[:min_time] for ent in entropies])

    data = np.insert(ent_arr, 0, dims, axis=1)
    np.save('data/' + file, data)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("sym",
                        help="Cat map symmetry: 'W' or 'R'")
    parser.add_argument("matrix", type=int, nargs=4,
                        help="matrix elements a b c d")
    parser.add_argument("pert", type=float,
                        help="perturbation strength")
    parser.add_argument("psize", type=int,
                        help="size of the partition in each sector")
    parser.add_argument("dims", type=int, nargs='+',
                        help="Hilbert space dimensions")
    parser.add_argument("-mix", action="store_true",
                        help="Kraus operators can mix charge sectors")
    parser.add_argument("-rand", action="store_true",
                        help="Randomize basis in each charge sector")
    args = parser.parse_args()
    if not (args.sym == 'W' or args.sym == 'R'):
        raise ValueError("Invalid cat map symmetry.")
    main(args.sym, args.matrix, args.pert, args.psize, args.dims, mix=args.mix, rand=args.rand)
