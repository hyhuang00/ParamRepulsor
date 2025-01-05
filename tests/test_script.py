import numpy as np

from parampacmap import ParamPaCMAP


def test_3_to_1():
    A = np.random.randn(1_000, 3)
    R = ParamPaCMAP(n_components=1).fit_transform(A)
    assert R.shape[0] == A.shape[0]
    assert R.shape[1] == 1


def main():
    A = np.random.randn(1_000, 20)
    R = ParamPaCMAP(num_workers=0).fit_transform(A)
    return R


def test_basic_usage():
    R = main()
    assert R.shape[0] == 1000
    assert R.shape[1] == 2


def test_pair_saving():
    A = np.random.randn(1_000, 3)
    reducer = ParamPaCMAP(n_components=2, save_pairs=True)
    R = reducer.fit_transform(A)
    assert reducer.pair_neighbors.shape[0] == 10000
    assert reducer.pair_neighbors.shape[1] == 2
    assert reducer.pair_FP.shape[0] == 20000
    assert reducer.pair_FP.shape[1] == 2
    assert reducer.pair_MN.shape[0] == 5000
    assert reducer.pair_MN.shape[1] == 2
    assert reducer._num_samples == 1000

if __name__ == "__main__":
    result = main()
    print(result)
