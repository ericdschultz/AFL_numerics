"""
Computes AFL entropy over time on the perturbed cat map.
This is specifically for the cat W and R symmetries, so the partition is symmetrized.
Fixed partition and dimension; varying perturbation strength for the symmetry breaking shear.
Saves the entropies in an .npy file.
The first element of each array is the perturbation strength of the symmetry breaking shear.

This file runs each perturbation in parallel using multiprocessing.

Command:
python -O cat_map_aprx_sym.py sym a b c d psize -mix -rand N sympert perts

sym - 'W' or 'R' for the respective cat map symmetry,
      or 'none' for no symmetry (uses random partition)
a b c d - cat map matrix elements
psize - size of the partition in each charge sector
mix - If included, Kraus operators can mix different charge sectors
      and psize refers to the total number of partitions
rand - If included, randomize the basis in each charge sector
N - Hilbert space dimension
sympert - perturbation strength of symmetric shear
perts - list of perturbation strengths for symmetry breaking shear
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

import AFL.dynamics.cat_map as cat_map
import AFL.tools.entropy as entropy
import AFL.tools.partitions as partitions
import AFL.tools.floquet as floquet

def main(sym, matrix, psize, N, sympert, perts, mix=False, rand=False):
    A = np.array(matrix).reshape((2,2))

    kmax = cat_map.max_pert(A, sympert, max(perts))
    if kmax < 1.0:
        raise ValueError('Perturbation exceeds Anosov bound; scale by < {0}'.format(kmax))

    if sym == 'W':
        vecs, inds = cat_map.cat_W_vecs(N)
        X = partitions.sym_partition(vecs, inds, psize, mixing=mix, randomize=rand)
    elif sym == 'R':
        s = np.gcd(A[0,1], A[1,1]-1)
        if s % 2 == 0:
            s //= 2
        if N % s != 0:
            raise ValueError("The chosen cat map and dimension is not R-symmetric.")
        vecs, inds = cat_map.cat_R_vecs(N, s)
        X = partitions.sym_partition(vecs, inds, psize, mixing=mix, randomize=rand)
    elif sym == 'none':
        X = partitions.get_qmap_partition('randproj', N, psize)
    else:
        raise ValueError("Invalid cat map symmetry.")
    
    weights = np.ones(N) / N
    max_time = entropy.time_estimate(N)

    U = cat_map.cat_unitary_gauss(N, A) @ cat_map.typ_pshear(N, sympert)
    unitaries = np.array([U @ cat_map.symbreak_qshear(N, pert) for pert in perts])
    _, bases = floquet.good_basis_set(unitaries)
    

    file = 'cat_aprxsym{}_{}_k{}_m{}r{}{}_N{}_kq{}-{}'\
            .format(sym, '-'.join(map(str, matrix)), str(sympert).replace('.','p'),
                    str(mix)[0], str(rand)[0], psize, N,
                    str(perts[0]).replace('.','p'), str(perts[-1]).replace('.','p'))
    
    # Multiprocessing
    pool = Pool(len(perts))

    AFL_func = partial(entropy.AFL_entropy_helper, weights, X, max_time=max_time)
    entropies = pool.starmap(AFL_func, zip(unitaries, bases), 1)

    pool.close()

    # Make all entropy lists of the same length
    min_time = min(map(len, entropies))
    ent_arr = np.array([ent[:min_time] for ent in entropies])

    data = np.insert(ent_arr, 0, perts, axis=1)
    np.save('data/' + file, data)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("sym",
                        help="cat map symmetry 'W' or 'R', or 'none' for no symmetry")
    parser.add_argument("matrix", type=int, nargs=4,
                        help="matrix elements a b c d")
    parser.add_argument("psize", type=int,
                        help="size of the partition")
    parser.add_argument("-mix", action="store_true",
                        help="Kraus operators can mix charge sectors")
    parser.add_argument("-rand", action="store_true",
                        help="Randomize basis in each charge sector")
    parser.add_argument("N", type=int,
                        help="Hilbert space dimension")
    parser.add_argument("sympert", type=float,
                        help="perturbation strength of symmetric shear")
    parser.add_argument("perts", type=float, nargs='+',
                        help="perturbation strengths of symmetry breaking shear")
    args = parser.parse_args()
    main(args.sym, args.matrix, args.psize, args.N, args.sympert, perts=args.perts, mix=args.mix, rand=args.rand)
