"""Training utilities script of parametric pacmap/paramrepulsor.
"""

def convert_pairs(pair_neighbors, pair_FP, pair_MN, N):
    pair_neighbors = pair_neighbors[:, 1].reshape((N, -1))
    pair_FP = pair_FP[:, 1].reshape((N, -1))
    pair_MN = pair_MN[:, 1].reshape((N, -1))
    return pair_neighbors, pair_FP, pair_MN
