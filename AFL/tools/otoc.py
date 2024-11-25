"""
Functions for computing out-of-time-ordered correlators.
"""
import numpy as np

def pure_otoc(states, W, V, U, max_time, sub_init=False):
    """
    Computes the OTOC <[W(t),V][W(t),V]*> on a set of pure states.
    If sub_init, then compute <ZZ*> for Z=[W(t),V]-[W,V] instead.
    """
    otocs = np.empty((states.shape[0], max_time), dtype=np.complex128)
    init_comm = W @ V - V @ W
    
    W_new = W
    for i in np.arange(max_time):
        W_new = U.conj().T @ W_new @ U
        commutator = W_new @ V - V @ W_new
        if sub_init:
            commutator -= init_comm
        otocs[:,i] = np.einsum('ji,jk,lk,li->i', states.conj(), commutator, commutator.conj(), states, optimize='greedy')
        
    Re_otoc = np.real(otocs)
    Im_otoc = np.imag(otocs)
    assert np.allclose(Im_otoc, np.zeros_like(otocs)), 'Computed OTOC was not real'
    return Re_otoc

def otoc(state, W, V, U, max_time, sub_init=False):
    """
    Computes the OTOC <[W(t),V][W(t),V]*> on a density matrix.
    If sub_init, then compute <ZZ*> for Z=[W(t),V]-[W,V] instead.
    """
    otocs = np.empty(max_time, dtype=np.complex128)
    init_comm = W @ V - V @ W
    
    W_new = W
    for i in np.arange(max_time):
        W_new = U.conj().T @ W_new @ U
        commutator = W_new @ V - V @ W_new
        if sub_init:
            commutator -= init_comm
        otocs[i] = np.einsum('ij,jk,ik', state, commutator, commutator.conj())
        
    Re_otoc = np.real(otocs)
    Im_otoc = np.imag(otocs)
    assert np.allclose(Im_otoc, np.zeros_like(otocs)), 'Computed OTOC was not real'
    return Re_otoc
