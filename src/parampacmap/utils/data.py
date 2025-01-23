"""Data related utility functions."""
import numba
import numpy as np
from annoy import AnnoyIndex


def generate_pair(
    X, n_neighbors, n_MN, n_FP, distance="euclidean", verbose=True, random_state=None
):
    """Generate pairs for the dataset.
    
    Args:
        X: Input data array
        n_neighbors: Number of neighbors to find
        n_MN: Number of mid-near pairs
        n_FP: Number of far pairs
        distance: Distance metric to use
        verbose: Whether to print progress
        random_state: Random seed for reproducibility

    Returns:
        pair_neighbors: Nearest neighbor pairs
        pair_MN: Mid-near pairs
        pair_FP: Far pairs
        tree: Annoy index
    """
    n, dim = X.shape
    # sample more neighbors than needed
    n_neighbors_extra = min(n_neighbors + 50, n - 1)
    tree = AnnoyIndex(dim, metric=distance)
    if random_state is not None:
        tree.set_seed(random_state)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    option = distance_to_option(distance=distance)

    nbrs = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_neighbors_extra), dtype=np.float32)

    for i in range(n):
        nbrs_ = tree.get_nns_by_item(i, n_neighbors_extra + 1)
        nbrs[i, :] = nbrs_[1:]
        # This line is subject to change
        for j in range(n_neighbors_extra):
            knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
    print_verbose("Found nearest neighbor", verbose)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    print_verbose("Calculated sigma", verbose)
    scaled_dist = scale_dist(knn_distances, sig, nbrs)
    print_verbose("Found scaled dist", verbose)
    X = X.astype(np.float32)
    pair_neighbors = sample_neighbors_pair(X, scaled_dist, nbrs, np.int32(n_neighbors))
    if random_state is None:
        pair_MN = sample_MN_pair(X, np.int32(n_MN), np.int32(option))
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_MN = sample_MN_pair_deterministic(X, np.int32(n_MN), random_state, option)
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, np.int32(n_FP), random_state
        )
    return pair_neighbors, pair_MN, pair_FP, tree


def distance_to_option(distance="euclidean"):
    """A helper function that translates distance metric to int options.
    Such a translation is useful for numba acceleration.
    """
    if distance == "euclidean":
        option = 0
    elif distance == "manhattan":
        option = 1
    elif distance == "angular":
        option = 2
    elif distance == "hamming":
        option = 3
    else:
        raise NotImplementedError(
            "Distance other than euclidean, manhattan,"
            + "angular or hamming is not supported"
        )
    return option


def print_verbose(msg, verbose, **kwargs):
    if verbose:
        print(msg, **kwargs)


@numba.njit("f4(f4[:])")
def l2_norm(x):
    """
    L2 norm of a vector.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])")
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])")
def manhattan_dist(x1, x2):
    """
    Manhattan distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i] - x2[i])
    return result


@numba.njit("f4(f4[:],f4[:])")
def angular_dist(x1, x2):
    """
    Angular (i.e. cosine) distance between two vectors.
    """
    x1_norm = np.maximum(l2_norm(x1), 1e-20)
    x2_norm = np.maximum(l2_norm(x2), 1e-20)
    result = 0.0
    for i in range(x1.shape[0]):
        result += x1[i] * x2[i]
    return np.sqrt(2.0 - 2.0 * result / x1_norm / x2_norm)


@numba.njit("f4(f4[:],f4[:])")
def hamming_dist(x1, x2):
    """
    Hamming distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@numba.njit()
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:  # euclidean
        return euclid_dist(x1, x2)
    elif distance_index == 1:  # manhattan
        return manhattan_dist(x1, x2)
    elif distance_index == 2:  # angular
        return angular_dist(x1, x2)
    elif distance_index == 3:  # hamming
        return hamming_dist(x1, x2)


@numba.njit("i4[:](i4,i4,i4[:])", nogil=True)
def sample_FP(n_samples, maximum, reject_ind):
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(maximum)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(reject_ind.shape[0]):
                if j == reject_ind[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit("i4[:,:](f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True)
def sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors):
    n = X.shape[0]
    pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i * n_neighbors + j][0] = i
            pair_neighbors[i * n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](i4,f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True)
def sample_neighbors_pair_basis(n_basis, X, scaled_dist, nbrs, n_neighbors):
    """Sample Nearest Neighbor pairs for additional data."""
    n = X.shape[0]
    pair_neighbors = np.empty((n * n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i * n_neighbors + j][0] = n_basis + i
            pair_neighbors[i * n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](f4[:,:],i4,i4)", nogil=True)
def sample_MN_pair(X, n_MN, option=0):
    """Sample Mid Near pairs."""
    n = X.shape[0]
    pair_MN = np.empty((n * n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_index=option
                )
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i * n_MN + jj][0] = i
            pair_MN[i * n_MN + jj][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4,i4,i4)", nogil=True)
def sample_MN_pair_deterministic(X, n_MN, random_state, option=0):
    """Sample Mid Near pairs using the given random state."""
    n = X.shape[0]
    pair_MN = np.empty((n * n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            # Shifting the seed to prevent sampling the same pairs
            np.random.seed(random_state + i * n_MN + jj)
            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_index=option
                )
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i * n_MN + jj][0] = i
            pair_MN[i * n_MN + jj][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP):
    """Sample Further pairs."""
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            FP_index = sample_FP(
                n_FP,
                n,
                pair_neighbors[i * n_neighbors : i * n_neighbors + n_neighbors, 1],
            )
            pair_FP[i * n_FP + k][0] = i
            pair_FP[i * n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4,i4)", parallel=True, nogil=True)
def sample_FP_pair_deterministic(X, pair_neighbors, n_neighbors, n_FP, random_state):
    """Sample Further pairs using the given random state."""
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            np.random.seed(random_state + i * n_FP + k)
            FP_index = sample_FP(
                n_FP,
                n,
                pair_neighbors[i * n_neighbors : i * n_neighbors + n_neighbors, 1],
            )
            pair_FP[i * n_FP + k][0] = i
            pair_FP[i * n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True)
def scale_dist(knn_distance, sig, nbrs):
    """Scale the distance"""
    n, num_neighbors = knn_distance.shape
    scaled_dist = np.zeros((n, num_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(num_neighbors):
            scaled_dist[i, j] = knn_distance[i, j] ** 2 / sig[i] / sig[nbrs[i, j]]
    return scaled_dist

