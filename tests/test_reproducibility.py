import numpy as np
import pytest

from parampacmap import ParamPaCMAP


@pytest.fixture
def array_fixture():
    np.random.seed(1992)
    return np.random.randn(1_000, 20)


@pytest.fixture
def fixed_reducer():
    return ParamPaCMAP(seed=21, num_epochs=1)


def test_seed_reproducibility(array_fixture, fixed_reducer):
    # Arrange
    A = array_fixture

    # Act
    R1 = fixed_reducer.fit_transform(A)
    R2 = ParamPaCMAP(seed=21, num_epochs=1).fit_transform(A)

    # Assert
    R1 = np.round(R1, 3)
    R2 = np.round(R2, 3)
    assert R1.shape[0] == A.shape[0]
    assert R1.shape[1] == 2
    assert np.allclose(R1, R2, atol=1e-2)


def test_instantiation_with_defaults(array_fixture):
    # Arrange
    A = array_fixture

    # Act
    R1 = ParamPaCMAP(num_epochs=1).fit_transform(A)
    R2 = ParamPaCMAP(num_epochs=1).fit_transform(A)

    # Assert
    assert R1.shape[0] == A.shape[0]
    assert R1.shape[1] == 2
    assert not np.allclose(R1, R2)


def test_seed_reproducibility_with_multiple_workers(array_fixture, fixed_reducer):
    # Arrange
    A = array_fixture

    # Act
    R1 = fixed_reducer.fit_transform(A)
    R2 = ParamPaCMAP(seed=21, num_workers=2, num_epochs=1).fit_transform(A)
    R3 = ParamPaCMAP(seed=21, num_workers=4, num_epochs=1).fit_transform(A)

    # Assert
    assert R1.shape[0] == A.shape[0]
    assert R1.shape[1] == 2
    assert np.allclose(R1, R2, rtol=1e-4)
    assert np.allclose(R1, R3, rtol=1e-4)
