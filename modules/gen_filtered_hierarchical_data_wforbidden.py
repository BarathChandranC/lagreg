"""
gen_filtered_hierarchical_data.py

This script contains functions for generating data following the procedure described in our paper "How transformers learn structured data: insights from hierarchical filtering."
It leverages numpy for numerical operations and multiprocessing for parallel generation of a large number of samples (N_trials).
The data is saved in a numpy file with the following format:
[q,l,sigma,x0s,xis,M,restrictions] where:
- q: Size of the vocabulary
- q_eff: number of non-zero transitions for each parent
- l: Depth of the tree
- sigma: Standard deviation of the noise generating the grammar
- x0s: Array of root nodes of dim (N_trials)
- xis: Array of leaf nodes of dim (2**l,N_trials)
- M: Transition rates matrix of dim (q,q,q)
- restrictions: forbidden child-symbol pairs

Functions:
- get_P_xlevel_root(M, level): Computes the probability matrices for a given level (i.e. Eq. (3) of our paper).
- get_leaves(x0, M, l, k, probs): Generates leaves from a root x0.
- get_M(q, sigma): Generates the transition rates matrix M.
- gen_tree(sample_index, M, l, q, k, probs, seed): Generates one root/leaves pair from an index-derived RNG stream.

Dependencies:
- numpy
- scipy.special.softmax
- multiprocessing.Pool
- multiprocessing.cpu_count
"""

import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.special import softmax


def get_P_xlevel_root(M,level):
    M_L = np.sum(M,axis=2)
    M_R = np.sum(M,axis=1)
    q = M_L.shape[0]
    leaves_indices = np.arange(2**(level+1))[2**level:]
    leaves_indices_binary = [bin(leaves_indices[i])[2:][1:] for i in range(len(leaves_indices))]
    probs = np.empty((q,q,2**level))
    probs_prev = np.empty((q,q,2**level))
    for i in range(2**level):
        probs[:,:,i] = np.eye(q)
        probs_prev[:,:,i] = np.eye(q)
        for j in range(level):
            if leaves_indices_binary[i][j] == '0':
                probs[:,:,i] = probs_prev[:,:,i]@M_L
            else:
                probs[:,:,i] = probs_prev[:,:,i]@M_R
            probs_prev[:,:,:] = probs[:,:,:]
    return probs 


def get_leaves(x0, M, l, level, probs, rng=None):
    rng = np.random if rng is None else rng

    def get_branches(S,M):
        # Get the two branches from a given state S using the standard rule
        p_flat = M[S,:,:].ravel()
        ind = np.arange(len(p_flat))
        return np.unravel_index(rng.choice(ind, p=p_flat),M[S,:,:].shape)
    
    def get_nodes(x0,probs,level):
        nodes = np.empty(2**level,dtype=np.int8)
        q = probs.shape[0]
        for i in range(2**level):
            nodes[i] = rng.choice(np.arange(q),p=probs[x0,:,i])
        return nodes
    
    x = get_nodes(x0,probs,level).astype(np.int8)

    for i in range(level+1,l+1):
        x_new = np.empty(2**i,dtype=np.int8)
        for j in range(len(x)):
            x_new[2*j],x_new[2*j+1] = get_branches(x[j],M)
        x = x_new
    return x


def get_M(q,sigma):
    h = np.log(np.zeros((q,q,q))) # Actually parametrize with logits
    M = np.empty((q,q,q))
    tuples = np.array([(i,j) for i in range(q) for j in range(q)]) # Generate the non-overlaping partitions
    np.random.shuffle(tuples)

    for i in range(q):
        for j in range(i*q,(i+1)*q):
            k,l = tuples[j]
            h[i,k,l] = sigma*np.random.randn()
        M[i,:,:] = softmax(h[i,:,:])
    return M


""" def get_M_wforbidden(q,sigma,n_forbidden):
    h = np.log(np.zeros((q,q,q))) # Actually parametrize with logits
    M = np.empty((q,q,q))
    tuples = np.array([(i,j) for i in range(q) for j in range(q)]) # Generate the non-overlaping partitions that do NOT include all transitions
    np.random.shuffle(tuples)
    # randomly generate n_forbidden tuples for which the transition is forbidden
    forbidden_tuples_idx = np.random.choice(len(tuples),size=n_forbidden,replace=False)
    for i in range(q):
        for j in range(i*q,(i+1)*q):
            k,l = tuples[j]
            if j not in forbidden_tuples_idx:
                h[i,k,l] = sigma*np.random.randn()
            else:
                print('forbidding',(k,l))
        M[i,:,:] = softmax(h[i,:,:])
    return M,[tuples[i] for i in forbidden_tuples_idx] """


def get_M_wforbidden(q, sigma, q_eff, rng=None):
    rng = np.random if rng is None else rng

    h = np.full((q,q,q), -np.inf) # Actually parametrize with logits
    M = np.empty((q,q,q))
    tuples = np.array([(i,j) for i in range(q) for j in range(q)]) # Generate the non-overlaping partitions that do NOT include all transitions
    rng.shuffle(tuples)
    forbidden_tuples = np.empty((q,q-q_eff,2))

    for i in range(q):
        for j in range(i*q,i*q + q_eff):
            k,l = tuples[j]
            h[i,k,l] = sigma*rng.randn()
        for j in range(i*q + q_eff,i*q + q):
            forbidden_tuples[i,(j-q_eff)%q,:] = tuples[j][:]
        M[i,:,:] = softmax(h[i,:,:])

    forbidden_tuples = forbidden_tuples.reshape(-1,2).astype(int)
    forbidden_tuples = [tuple(row) for row in forbidden_tuples]
    return M,forbidden_tuples


def gen_tree(sample_index, M, l, q, k, probs, seed):
    """Generate one sample from a seed derived only from its dataset index."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index]))
    x0 = rng.integers(q)
    leaves = get_leaves(x0, M, l, k, probs, rng=rng)
    return x0,leaves


def generate_dataset(q, l, sigma, q_eff, seed, N_trials=int(1e6), k=0, workers=None):
    if N_trials <= 0:
        raise ValueError("N_trials must be positive")
    workers = cpu_count() if workers is None else int(workers)
    if workers <= 0:
        raise ValueError("workers must be positive")

    # Preserve the historical seed-0 grammar while giving every sample an
    # index-derived RNG stream that is independent of worker scheduling.
    grammar_rng = np.random.RandomState(seed)
    M,restrictions = get_M_wforbidden(q, sigma, q_eff, rng=grammar_rng)
    probs = get_P_xlevel_root(M,k)

    x0s = np.empty(N_trials)
    xis = np.empty((int(2**l),N_trials),dtype=np.int8) # Save as integers to save memory
    worker = partial(gen_tree, M=M, l=l, q=q, k=k, probs=probs, seed=seed)
    if workers == 1:
        runs = map(worker, range(N_trials))
        for h, (root, leaves) in enumerate(runs):
            x0s[h], xis[:, h] = root, leaves
    else:
        chunksize = max(1, N_trials // (workers * 16))
        with Pool(workers) as pool:
            for h, (root, leaves) in enumerate(pool.imap(worker, range(N_trials), chunksize=chunksize)):
                x0s[h], xis[:, h] = root, leaves
    
    return np.array([q,l,sigma,x0s,xis,M,restrictions],dtype=object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the controlled hierarchical dataset deterministically.")
    parser.add_argument("--n-trials", type=int, default=int(1e6))
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factorized-layers", type=int, default=0)
    parser.add_argument("--q-eff", type=int, default=4)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parents[1] / "data" / (
            f"labeled_data_restrictedfixed_{args.q}_{args.depth}_{args.sigma}_{args.q_eff}_{args.seed}.npy"
        )
    output = output.expanduser().resolve()
    if output.suffix != ".npy":
        output = output.with_suffix(".npy")

    data = generate_dataset(
        args.q,
        args.depth,
        args.sigma,
        args.q_eff,
        args.seed,
        N_trials=args.n_trials,
        k=args.factorized_layers,
        workers=args.workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, data)
    print(f"Saved {args.n_trials} samples to {output}")
