"""
Computes AFL entropy over time on the perturbed cat map.
Fixed partition and dimension; varying perturbation.
Saves the entropies in an .npy file.
The first element of each array is the perturbation strength.

This file runs each perturbation in parallel using multiprocessing.

Command:
python -O cat_map_perts.py [-perts PERTS] a b c d partition psize N

perts (optional) - list of perturbation strengths; default [0,0.05,0.1,0.15,0.2,0.25]
a b c d - cat map matrix
partition - qmap partition name (see partitions.py)
psize - size of the partition
N - Hilbert space dimension

To time this module:
python -O -m timeit -s "import cat_map_perts" -n [number of loops] -r [number of repeats]
    "cat_map_perts.main([a,b,c,d], partition, psize, N)
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions
import AFL.tools.floquet as floquet

def main(matrix, partition, psize, N, perts=np.arange(0, 0.3, 0.05), test=False):
    A = np.array(matrix).reshape((2,2))

    kmax = cat_map.max_pert(A, 1, 0)
    if any(pert > kmax for pert in perts):
        raise ValueError('Perturbation strength exceeds Anosov bound of {0}'.format(kmax))

    U = cat_map.cat_unitary_gauss(N, A)
    unitaries = np.array([U @ cat_map.typ_pshear(N, pert) for pert in perts])
    _, bases = floquet.good_basis_set(unitaries)
    
    X = partitions.get_qmap_partition(partition, N, psize)
    weights = np.ones(N) / N
    max_time = entropy.time_estimate(N)

    file = 'cat_perts_{}_N{}_{}{}'.format('-'.join(map(str,matrix)), N, partition, psize)
    
    # Multiprocessing
    pool = Pool(len(perts))

    AFL_func = partial(entropy.AFL_entropy_helper, weights, X, max_time=max_time)
    entropies = pool.starmap(AFL_func, zip(unitaries, bases), 1)

    pool.close()

    # Make all entropy lists of the same length
    min_time = min(map(len, entropies))
    ent_arr = np.array([ent[:min_time] for ent in entropies])

    data = np.insert(ent_arr, 0, perts, axis=1)
    if not test:
        np.save('data/' + file, data)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=int, nargs=4,
                        help="matrix elements a b c d")
    parser.add_argument("partition",
                        help="qmap partition name")
    parser.add_argument("psize", type=int,
                        help="size of the partition")
    parser.add_argument("N", type=int,
                        help="Hilbert space dimension")
    parser.add_argument("-perts", type=float, nargs='+',
                        default=np.arange(0, 0.3, 0.05),
                        help="perturbation strengths")
    parser.add_argument("-test", action="store_true",
                        help="Will not save result. For testing purposes.")
    args = parser.parse_args()
    main(args.matrix, args.partition, args.psize, args.N, perts=args.perts, test=args.test)
