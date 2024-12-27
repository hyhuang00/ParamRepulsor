import numpy as np
import pytest

from parampacmap import ParamPaCMAP


@pytest.fixture
def array_fixture():
    np.random.seed(1992)
    return np.random.randn(1_000, 20)


def test_seed_reproducibility(array_fixture):
    A = array_fixture
    seed = 21
    R1 = ParamPaCMAP(seed=seed).fit_transform(A)
    R2 = ParamPaCMAP(seed=seed).fit_transform(A)
    assert R1.shape[0] == A.shape[0]
    assert R1.shape[1] == 2
    assert np.allclose(R1, R2)


def test_instantiation_with_defaults(array_fixture):
    A = array_fixture
    R1 = ParamPaCMAP().fit_transform(A)
    R2 = ParamPaCMAP().fit_transform(A)
    assert R1.shape[0] == A.shape[0]
    assert R1.shape[1] == 2
    assert not np.allclose(R1, R2)
