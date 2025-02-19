"""
Computes AFL entropy over time on the perturbed cat map.
Saves the entropies in an .npy file.
The first element of each array is the perturbation strength.

This file computes sequentially and is for testing the performance of the parallel version.

Command:
python -O cat_map_perts_seq.py [-perts PERTS] a b c d partition psize N

N - Hilbert space dimension
partition - qmap partition name (see partitions.py)
psize - size of the partition
perts (optional) - list of perturbation strengths; default [0,0.05,0.1,0.15,0.2,0.25]
"""
import argparse
import numpy as np

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions

def main(matrix, partition, psize, N, perts=np.arange(0, 0.3, 0.05)):
    A = np.array(matrix).reshape((2,2))

    kmax = cat_map.max_pert(A, 1, 0)
    if any(pert > kmax for pert in perts):
        raise ValueError('Perturbation strength exceeds Anosov bound of {0}'.format(kmax))

    X = partitions.get_qmap_partition(partition, N, psize)
    weights = np.ones(N) / N
    max_time = entropy.time_estimate(N)

    file = 'cat_perts_seq_{}_N{}_{}{}'.format('-'.join(map(str,matrix)), N, partition, psize)
    
    unitaries = np.empty((len(perts), N, N), dtype=np.complex128)
    U = cat_map.cat_unitary_gauss(N, A)
    for i in range(len(perts)):
        unitaries[i,:,:] = U @ cat_map.typ_pshear(N, perts[i])
    entropies = entropy.AFL_entropy_perts(weights, X, unitaries, max_time=max_time)

    data = np.insert(entropies, 0, perts, axis=1)
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
    args = parser.parse_args()
    main(args.matrix, args.partition, args.psize, args.N, perts=args.perts)
