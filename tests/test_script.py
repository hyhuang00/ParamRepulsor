import numpy as np

from parampacmap import ParamPaCMAP


def test_3_to_1():
    A = np.random.randn(1_000, 3)
    R = ParamPaCMAP(n_components=1, num_epochs=1).fit_transform(A)
    assert R.shape[0] == A.shape[0]
    assert R.shape[1] == 1


def main():
    A = np.random.randn(1_000, 20)
    R = ParamPaCMAP(num_workers=0, num_epochs=1).fit_transform(A)
    return R


def test_basic_usage():
    R = main()
    assert R.shape[0] == 1000
    assert R.shape[1] == 2


def test_fit_transform_same_as_fit_then_transform():
    # Arrange
    np.random.seed(21)  # Set seed for input data generation
    A = np.random.randn(1_000, 20)
    P1 = ParamPaCMAP(num_workers=1, seed=42, num_epochs=1)
    # persistent_workers option needs num_workers > 0 (for .transform)
    P2 = ParamPaCMAP(num_workers=1, seed=42, num_epochs=1, save_pairs=True)

    # Act
    R1 = P1.fit_transform(A)
    P2.fit(A)
    R2 = P2.transform(A)

    # Assert
    R1 = np.round(R1, 3)
    R2 = np.round(R2, 3)
    assert np.allclose(R1, R2, rtol=5e-3)


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
