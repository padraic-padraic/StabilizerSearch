from .gram_schmidt import ortho_projector

import numpy as np

def calculate_overlap(states, target):    
    tot = 0
    for state in states:
        olp = np.sum(state.H*target)
        tot += np.abs(olp.conj()*olp)
    return tot