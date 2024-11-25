"""
Quantum cat map functions.
"""

import numpy as np
import warnings

def cat_unitary(N, A, kappa):
    """
    Converts the classical cat map matrix A and perturbation kappa into a quantum unitary time evolution.
    Assumes there are no hidden symmetries that require use of the 'Gauss averages' function.
    """
    A11, A12, A21, A22 = A[0,0], A[0,1], A[1,0], A[1,1]
    if not (A11*A22 - A12*A21 == 1 and A11 + A22 > 2):
        raise ValueError("Invalid cat map matrix.")

    if np.gcd(A12, A22 - 1) > 1:
        warnings.warn('\n  This cat map requires computing the Gauss sum. Use cat_unitary_gauss() instead.')

    kmax = max_pert(A)
    if kappa > kmax:
        warnings.warn('\n  Perturbation strength {0:.2f} exceeds the Anosov bound of {1:.2f}'.format(kappa, kmax))

    j,k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    term1 = np.pi * (A22*j*j - 2*j*k + A11*k*k) / (N*A12)
    term2 = kappa * N * np.sin(2*np.pi*k / N) / (2*np.pi)
    
    U = np.sqrt(A12/N) * np.exp(1j * (term1 + term2))
    assert np.allclose(U @ U.conj().T, np.eye(N)), 'Map is not unitary'
    return U

def cat_unitary_set(N, A, kappas):
    """Computes the set of cat map (without Gauss sum) unitaries with perturbations 'kappas'."""
    unitaries = np.empty((kappas.size, N, N), dtype=np.complex128)
    for i in np.arange(kappas.size):
        unitaries[i,:,:] = cat_unitary(N, A, kappas[i])
    return unitaries
    
def cat_periodicity(N, A, max_time=None):
    """
    Computes the periodicity of the (unperturbed) quantum cat map.
    Returns None upon failure.
    """
    if max_time is None:
        max_time = 5 * N
    U = cat_unitary_gauss(N, A, 0)
    U_next = np.eye(N)
    for t in np.arange(max_time):
        U_next = U @ U_next
        diags = np.diag(U_next)
        # Period reached when U^t = exp(iϕ)I
        if np.allclose(U_next, np.diag(diags)) and np.allclose(diags, diags[0]):
            return t+1
    return None

def max_pert(A, max_gen=1.0):
    """
    Computes the maximum perturbation strength below which the cat map remains Anosov.
    The bound appears in [Bäcker 2003] and includes a maximum over the derivative of the generating Hamiltonian
    of the perturbation. Perturbations are often normalized so this is unity, so we have 'max_gen=1' by default.
    """
    tr = A[0,0] + A[1,1]
    num = np.sqrt(tr**2 - 4) - tr + 2
    den = 2 * max_gen * np.sqrt(1 + A[1,1]**2)
    return num / den

def cat_R_vecs(N, s):
    """
    Sorts the eigenvectors of the cat map R operator by eigenvalue.
    Specifically, the R operator where β = 1/s.
    """
    # eigvecs has s charge blocks, each of size M (symmetric case)
    M = N // s
    inds = np.arange(M*s)
    # R in the q-basis is <j|R|k> = δ(jk)exp(i2πj/s)
    indmap = s*(inds % M) + inds // M
    charge_inds = np.arange(0, N, M)

    # Tail in the nonsymmetric case (s does not divide N)
    # The tail is size r = N % s, so the first r charge blocks have one extra vector
    r = N % s
    if r != 0:
        R = np.arange(r)
        indmap = np.insert(indmap, M*(R + 1), M*s + R)
        charge_inds = np.append(np.arange(0, (M+1)*r, M+1), np.arange((M+1)*r, N, M))

    eye = np.eye(N)
    # eigvecs[i,j] is the i-th component of the j-th eigenvector
    eigvecs = eye[:,indmap]
    return eigvecs, charge_inds

def cat_W(N):
    """
    Constructs the 4|N cat map symmetry operator W.
    Classically, W(q,p) = (1/2 - q, 1/2 - p).
    Quantization is possible for even N, where W|j> = (-1)^j |N/2 - j>.
    W is a symmetry when 4|N and W commutes with the applied perturbations.
    """
    if N % 2 != 0:
        raise ValueError("Cat map W is not defined for odd N.")
    W = np.zeros((N,N))
    for j in np.arange(N):
        W[N//2 - j, j] = (-1) ** j
    if N % 4 != 0:
        W = 1j*W # Ensures W is always Hermitian
    return W

def cat_W_vecs(N):
    """
    If 4|N, diagonalizes the cat map symmetry operator W (sorted by eigenvalue).
    We also return the indices where each charge sector begins.
    """
    if N % 4 != 0:
        raise ValueError("Cat map W is not defined for odd N.")
    
    '''
    W|j> = (-1)^j|N/2-j>
    
        0(8)              4
      7     1          5     3
    6         2  --> 6         2  (with phase factors)
      5     3          7     1
         4                0

        0(10)              5
      9     1           6     4
    8         2  -->  7         3  (with phase factors)
    7         3       8         2
      6     4           9     1
         5                0(10)
    '''
    # eigvecs[i,j] is the i-th component of the j-th eigenvector
    eigvecs = np.zeros((N,N))
    # +1 eigenvalues
    # We must alternate between symmetric and antisymmetric
    for i in range(1, N//2):
        eigvecs[i - N//4, i] = 1 / np.sqrt(2)
        eigvecs[N//2 + N//4 - i, i] = (1 - 2*(i % 2)) / np.sqrt(2)
    # This includes the two fixed points at N/4 and 3N/4
    # which are normalized differently
    eigvecs[-N//4, 0] = 1
    eigvecs[N//4, N//2] = 1

    # -1 eigenvalues
    # The opposite parity combinations
    for i in range(N//2 + 1, N):
        eigvecs[i - N//4, i] = 1 / np.sqrt(2)
        eigvecs[N//2 + N//4 - i, i] = (2*(i % 2) - 1) / np.sqrt(2)

    indices = np.array([0, N//2 + 1])
    return eigvecs, indices


''' Cat map (and helper functions) including the full Gauss sum below. '''

def cat_unitary_gauss(N, A, kappa):
    """
    Converts the classical cat map matrix A and perturbation kappa into a quantum unitary time evolution.
    Can handle hidden symmetries by computing the Gauss sum explicity.
    """
    A11, A12, A21, A22 = A[0,0], A[0,1], A[1,0], A[1,1]
    if not (A11*A22 - A12*A21 == 1 and A11 + A22 > 2):
        raise ValueError("Invalid cat map matrix.")
             
    kmax = max_pert(A)
    if kappa > kmax:
        warnings.warn('\n  Perturbation strength {0:.2f} exceeds the Anosov bound of {1:.2f}'.format(kappa, kmax))

    # Index-independent Gauss sum factor
    gcd = np.gcd(N, A12)
    a = N * A11 // gcd
    b = A12 // gcd
    assert np.gcd(a, b) == 1, 'a and b must be coprime'
    P = 1 / np.sqrt(b)
    if b % 2 == 0:
        jac = jacobi(b, a)
        a_reduced = a % 8
        P *= jac * np.exp(1j * np.pi * a_reduced / 4)
    else:
        jac = jacobi(a, b)
        P *= jac * np.exp(-1j * np.pi * (b - 1) / 4)

    # Index-dependent things
    U = np.empty((N,N), dtype=np.complex128)
    for j in np.arange(N):
        for k in np.arange(N):
            unpert = np.pi * (A22*j*j - 2*j*k + A11*k*k) / (N*A12)
            shear = kappa * N * np.sin(2*np.pi*k / N) / (2*np.pi)
            # Gauss sum things
            T = 0
            c_num = 2 * (A11*k - j)
            c = c_num // gcd
            if c_num % gcd == 0 and (a*b + c) % 2 == 0:
                if a*b % 2 == 0:
                    a_inv = pow(a.item(), -1, mod=b.item())
                    # modulo avoids issues with large N
                    a_reduced = a % (8*b)
                    c_reduced = c % (8*b)
                    num = (a_reduced * (a_inv * c_reduced)**2) % (8*b)
                    T = np.exp(-2j*np.pi * num / (8*b))
                else:
                    a4_inv = pow(4*a.item(), -1, mod=b.item())
                     # modulo avoids issues with large N
                    a_reduced = a % b
                    c_reduced = c % b
                    num = (2 * a_reduced * (a4_inv * c_reduced)**2) % b
                    T = np.exp(-2j*np.pi * num / b)
            # The unitary
            U[j,k] = np.sqrt(A12/N) * np.exp(1j * (unpert + shear)) * P * T
    
    assert np.allclose(U @ U.conj().T, np.eye(N)), 'Map is not unitary'
    return U
    
    
def jacobi(a, b):
    """
    Computes the Jacobi symbol (a/b). This is 1, 0, or -1.
    a is an integer and b is a positive, odd integer.
    See https://en.wikipedia.org/wiki/Jacobi_symbol#Calculating_the_Jacobi_symbol for the algorithm.
    WARNING: Not sure if this will work for negative 'a' as Python and C modulo work differently.
    """
    if not (b > 0 and b % 2 == 1):
        raise ValueError("b must be positive and odd")
    out = 1
    a = a % b # Reduce
    while a != 0:
        # Factors of 2
        while a % 2 == 0:
            a = a // 2
            r = b % 8
            if r == 3 or r == 5:
                out = -out
        # Quadratic reciprocity allows swapping a,b
        # with possibly a change of sign
        if a % 4 == 3 and b % 4 == 3:
            out = -out
        a,b = b,a
        # Reduce again
        a = a % b
    if  b == 1:
        return out
    return 0
