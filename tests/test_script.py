import numpy as np

from parampacmap import ParamPaCMAP


def main():
    A = np.random.randn(100, 20)
    R = ParamPaCMAP(num_workers=0).fit_transform(A)
    assert R.shape[0] == A.shape[0]
    assert R.shape[1] == 2
    return R


def test_basic_usage():
    r = main()


if __name__ == "__main__":
    result = main()
    print(result)
