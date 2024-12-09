from parampacmap import ParamPaCMAP

import numpy as np


def main():

    A = np.random.randn(1000, 20)

    r = ParamPaCMAP(num_workers=0).fit_transform(A)

    print(r)


def test_basic_usage():
    main()

if __name__ == "__main__":
    main()
