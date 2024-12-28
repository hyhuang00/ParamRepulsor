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
    A = np.random.randn(1_000, 20)
    P1 = ParamPaCMAP(num_workers=1, seed=42, num_epochs=1)
    # persistent_workers option needs num_workers > 0 (for .transform)
    P2 = ParamPaCMAP(num_workers=1, seed=42, num_epochs=1)

    # Act
    R1 = P1.fit_transform(A)
    P2.fit(A)
    R2 = P2.transform(A)

    # Assert
    R1 = np.round(R1, 3)
    R2 = np.round(R2, 3)
    assert np.allclose(R1, R2, rtol=5e-3)


if __name__ == "__main__":
    result = main()
    print(result)
