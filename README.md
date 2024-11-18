# ParamRepulsor

This is the code repository for the NeurIPS 2024 paper "Navigating the Effect of Parametrization for Dimensionality Reduction".

## How to install
This repository can be installed locally via pip by the following command:

```bash
git clone https://github.com/hyhuang00/ParamRepulsor.git
cd ParamRepulsor
pip install .
```

## How to use our algorithm
ParamPaCMAP/ParamRepulsor is fully scikit-learn compatible, meaning that it can be
used as any other scikit-learn based algorithm.
After the installation, you can use our algorithm by:

```python
import parampacmap

# Initialize the reducer. Notice that by default, the stronger paramrepulsor
# algorithm will be used.
reducer = parampacmap.ParamPaCMAP()
X_low = reducer.fit_transform(X)  # Substitute your data here.
```


## Citation
If you have referred to our research in your publication, or you used the ParamRepulsor/ParamPaCMAP algorithm in this repository, please cite our paper using the following bibtex:

```
@inproceedings{huang2024navigating,
  title={Navigating the Effect of Parametrization for Dimensionality Reduction},
  author={Huang, Haiyang and Wang, Yingfan and Rudin, Cynthia},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}```
