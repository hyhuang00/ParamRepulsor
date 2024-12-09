import numpy as np

from parampacmap import ParamPaCMAP


def main():
    A = np.random.randn(100, 20)
    r = ParamPaCMAP(num_workers=0).fit_transform(A)
    assert r.shape[0] == A.shape[0]
    assert r.shape[1] == 2


def test_basic_usage():
    main()


if __name__ == "__main__":
    main()
