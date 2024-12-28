"""The PaCMAP dataset and dataloader.
"""

import os
from typing import Optional

import numpy as np
import torch

if os.environ.get("TORCH_DEVICE", "") == "cpu":
    TORCH_DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
else:
    TORCH_DEVICE = torch.device("cpu")

class PaCMAPDataset(torch.utils.data.Dataset):
    """The PaCMAP dataset for training.
    Input:
        data: A numpy.ndarray of the shape [N, D] that consists the dataset.
        nn_pairs: A numpy.ndarray of the shape [N, np_nn] that consists the
            nearest neighbor pairs.
        fp_pairs: A numpy.ndarray of the shape [N, np_fp] that consists the
            farther pairs.
        mn_pairs: A numpy.ndarray of the shape [N, np_mn] that consists the
            mid-near pairs.
    """

    def __init__(
        self, data, nn_pairs, fp_pairs, mn_pairs, reshape=None, dtype=torch.float32
    ):
        self.data = data
        self.nn_pairs = nn_pairs
        self.fp_pairs = fp_pairs
        self.mn_pairs = mn_pairs
        self.reshape = reshape
        self._dtype = dtype
        if self.reshape is not None:
            assert np.product(self.reshape) == self.data.shape[-1]
            new_shape = [
                self.data.shape[0],
            ] + self.reshape
            self.data = self.data.reshape(new_shape)

    def __getitem__(self, index):
        """Given a series of index, return all of the points that connect to it."""
        basis = torch.tensor(self.data[index], dtype=self._dtype)
        nn_pair = self.nn_pairs[index]
        nn_pairs = torch.tensor(self.data[nn_pair], dtype=self._dtype)
        fp_pair = self.fp_pairs[index]
        fp_pairs = torch.tensor(self.data[fp_pair], dtype=self._dtype)
        mn_pair = self.mn_pairs[index]
        mn_pairs = torch.tensor(self.data[mn_pair], dtype=self._dtype)
        return basis, nn_pairs, fp_pairs, mn_pairs

    def __len__(self):
        return len(self.data)


class FastDataloader:
    """A customized dataloader for PaCMAP.
    Input:
        data: A numpy.ndarray of the shape [N, D] that consists the dataset.
        nn_pairs: A numpy.ndarray of the shape [N, np_nn] that consists the
            nearest neighbor pairs.
        fp_pairs: A numpy.ndarray of the shape [N, np_fp] that consists the
            farther pairs.
        mn_pairs: A numpy.ndarray of the shape [N, np_mn] that consists the
            mid-near pairs.
    """

    def __init__(
        self,
        data: np.ndarray,
        nn_pairs: np.ndarray,
        fp_pairs: np.ndarray,
        mn_pairs: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size=1024,
        device=TORCH_DEVICE,
        shuffle: bool = False,
        reshape=None,
        dtype=torch.float32,
        seed: Optional[int] = None,
    ):
        self.data = torch.tensor(data, dtype=torch.float32).to(device).to(dtype)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels).to(device)
        self.n_items = data.shape[0]
        self.n_batches = (self.n_items + batch_size - 1) // batch_size
        self.nn_pairs = nn_pairs
        self.fp_pairs = fp_pairs
        self.mn_pairs = mn_pairs
        self.batch_size = batch_size
        self.reshape = reshape
        self._dtype = dtype
        self.device = device
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self._epoch = 0
        if self.reshape is not None:
            assert (
                np.product(self.reshape) == self.data.shape[-1]
            ), f"{np.product(self.reshape)}, {self.data.shape[-1]}"
            new_shape = [
                self.data.shape[0],
            ] + self.reshape
            self.data = self.data.reshape(new_shape)

    def __iter__(self):
        self.idx = 0
        
        if self.seed is not None:
            epoch_seed = self.seed + self._epoch
            torch.manual_seed(epoch_seed)
        
        self.nn_iter = torch.tensor(self.nn_pairs, device=self.device).int()
        self.fp_iter = torch.tensor(self.fp_pairs, device=self.device).int()
        self.mn_iter = torch.tensor(self.mn_pairs, device=self.device).int()
        
        if not self.shuffle:
            self.indices = None
        else:
            # Create index
            self.indices = torch.randperm(self.n_items, device=self.device)
            self.nn_iter = torch.index_select(self.nn_iter, dim=0, index=self.indices)
            self.fp_iter = torch.index_select(self.fp_iter, dim=0, index=self.indices)
            self.mn_iter = torch.index_select(self.mn_iter, dim=0, index=self.indices)
        
        self._epoch += 1
        return self

    def __next__(self):
        if self.idx >= self.n_batches:
            raise StopIteration
        begin = self.idx * self.batch_size
        end = (self.idx + 1) * self.batch_size
        basis_label = None
        if self.indices is None:
            basis = self.data[begin:end]
            if self.labels is not None:
                basis_label = self.labels[begin:end]
        else:
            basis = torch.index_select(self.data, dim=0, index=self.indices[begin:end])
            if self.labels is not None:
                basis_label = torch.index_select(
                    self.labels, dim=0, index=self.indices[begin:end]
                )

        nn_iter = self.nn_iter[begin:end].flatten()
        fp_iter = self.fp_iter[begin:end].flatten()
        mn_iter = self.mn_iter[begin:end].flatten()
        nn_pairs = torch.index_select(self.data, dim=0, index=nn_iter)
        fp_pairs = torch.index_select(self.data, dim=0, index=fp_iter)
        mn_pairs = torch.index_select(self.data, dim=0, index=mn_iter)
        batch = torch.concat((basis, nn_pairs, fp_pairs, mn_pairs), dim=0)
        self.idx += 1
        n_items = basis.shape[0]
        return n_items, batch, basis_label

    def __len__(self):
        return self.n_batches


class FastNSDataloader:
    """A customized, Negative Sampling Based dataloader for PaCMAP.
    Input:
        data: A numpy.ndarray of the shape [N, D] that consists the dataset.
        labels: A numpy.ndarray of the shape [N,] that consists the labels.
            By default, labels of -1 is ignored.
        nn_pairs: A numpy.ndarray of the shape [N, np_nn] that consists the
            nearest neighbor pairs.
        fp_pairs: A numpy.ndarray of the shape [N, np_fp] that consists the
            farther pairs.
        mn_pairs: A numpy.ndarray of the shape [N, np_mn] that consists the
            mid-near pairs.
    """

    def __init__(
        self,
        data: np.ndarray,
        nn_pairs: np.ndarray,
        fp_pairs: np.ndarray,
        mn_pairs: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size=1024,
        device=TORCH_DEVICE,
        shuffle: bool = False,
        reshape=None,
        dtype=torch.float32,
        seed=None,
    ):
        self.data = torch.tensor(data).to(device).to(dtype)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels).to(device)
        self.n_items = data.shape[0]
        self.n_batches = (self.n_items + batch_size - 1) // batch_size
        self.nn_pairs = nn_pairs
        self.fp_pairs = fp_pairs
        self.num_fp = fp_pairs.shape[1]
        self.mn_pairs = mn_pairs
        self.batch_size = batch_size
        self.reshape = reshape
        self._dtype = dtype
        self.device = device
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=seed)
        if self.reshape is not None:
            assert np.product(self.reshape) == self.data.shape[-1]
            new_shape = [
                self.data.shape[0],
            ] + self.reshape
            self.data = self.data.reshape(new_shape)

    def __iter__(self):
        self.idx = 0
        self.nn_iter = torch.tensor(self.nn_pairs, device=self.device).int()
        self.mn_iter = torch.tensor(self.mn_pairs, device=self.device).int()
        if not self.shuffle:
            self.indices = None
        else:
            # Create index
            self.indices = torch.randperm(self.n_items, device=self.device)
            self.nn_iter = torch.index_select(self.nn_iter, dim=0, index=self.indices)
            self.mn_iter = torch.index_select(self.mn_iter, dim=0, index=self.indices)
        return self

    def __next__(self):
        if self.idx >= self.n_batches:
            raise StopIteration
        begin = self.idx * self.batch_size
        end = (self.idx + 1) * self.batch_size
        basis_label = None
        if self.indices is None:
            basis = self.data[begin:end]
            if self.labels is not None:
                basis_label = self.labels[begin:end]
        else:
            basis = torch.index_select(self.data, dim=0, index=self.indices[begin:end])
            if self.labels is not None:
                basis_label = torch.index_select(
                    self.labels, dim=0, index=self.indices[begin:end]
                )

        n_items = basis.shape[0]
        nn_iter = self.nn_iter[begin:end].flatten()
        fp_iter = torch.randint(
            0,
            self.n_items,
            (self.num_fp * n_items,),
            dtype=torch.int,
            device=self.device,
        )
        mn_iter = self.mn_iter[begin:end].flatten()
        nn_pairs = torch.index_select(self.data, dim=0, index=nn_iter)
        fp_pairs = torch.index_select(self.data, dim=0, index=fp_iter)
        mn_pairs = torch.index_select(self.data, dim=0, index=mn_iter)
        batch = torch.concat((basis, nn_pairs, fp_pairs, mn_pairs), dim=0)
        self.idx += 1
        return n_items, batch, basis_label

    def __len__(self):
        return self.n_batches


class FastIBNSDataloader:
    """A customized, In-batch Negative Sampling Based dataloader for PaCMAP.

    Note that IBNS dataloader applies a permutation on NN to obtain negative samples. Only indices are repeated.
    Input:
        data: A numpy.ndarray of the shape [N, D] that consists the dataset.
        nn_pairs: A numpy.ndarray of the shape [N, np_nn] that consists the
            nearest neighbor pairs.
        fp_pairs: A numpy.ndarray of the shape [N, np_fp] that consists the
            farther pairs.
        mn_pairs: A numpy.ndarray of the shape [N, np_mn] that consists the
            mid-near pairs.
    """

    def __init__(
        self,
        data: np.ndarray,
        nn_pairs: np.ndarray,
        fp_pairs: np.ndarray,
        mn_pairs: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size=1024,
        device=TORCH_DEVICE,
        shuffle: bool = False,
        reshape=None,
        dtype=torch.float32,
        seed=None,
    ):
        self.data = torch.tensor(data).to(device).to(dtype)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels).to(device)
        self.n_items = data.shape[0]
        self.n_batches = (self.n_items + batch_size - 1) // batch_size
        self.nn_pairs = nn_pairs
        self.num_nn = nn_pairs.shape[1]
        self.fp_pairs = fp_pairs
        self.num_fp = fp_pairs.shape[1]
        self.mn_pairs = mn_pairs
        self.batch_size = batch_size
        self.reshape = reshape
        self._dtype = dtype
        self.device = device
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=seed)
        self.full_perms = self.num_fp // self.num_nn
        self.remainder_perms = self.num_fp % self.num_nn
        if self.reshape is not None:
            assert np.product(self.reshape) == self.data.shape[-1]
            new_shape = [
                self.data.shape[0],
            ] + self.reshape
            self.data = self.data.reshape(new_shape)

    def __iter__(self):
        self.idx = 0
        self.nn_iter = torch.tensor(self.nn_pairs, device=self.device).int()
        self.mn_iter = torch.tensor(self.mn_pairs, device=self.device).int()
        if not self.shuffle:
            self.indices = None
        else:
            # Create index
            self.indices = torch.randperm(self.n_items, device=self.device)
            self.nn_iter = torch.index_select(self.nn_iter, dim=0, index=self.indices)
            self.mn_iter = torch.index_select(self.mn_iter, dim=0, index=self.indices)
        return self

    def __next__(self):
        if self.idx >= self.n_batches:
            raise StopIteration
        begin = self.idx * self.batch_size
        end = (self.idx + 1) * self.batch_size
        basis_label = None
        if self.indices is None:
            basis = self.data[begin:end]
            if self.labels is not None:
                basis_label = self.labels[begin:end]
        else:
            basis = torch.index_select(self.data, dim=0, index=self.indices[begin:end])
            if self.labels is not None:
                basis_label = torch.index_select(
                    self.labels, dim=0, index=self.indices[begin:end]
                )

        n_items = basis.shape[0]
        nn_iter = self.nn_iter[begin:end].flatten()
        full_permutations = [
            torch.randperm(nn_iter.shape[0], device=self.device)
            for _ in range(self.full_perms)
        ]
        # Change fp_iter generation. Now it should generate from nn only.
        fp_indices = torch.cat(full_permutations)
        if self.remainder_perms > 0:
            fp_indices = torch.cat(
                [
                    fp_indices,
                    torch.randperm(nn_iter.shape[0], device=self.device)[
                        : self.remainder_perms * n_items
                    ],
                ]
            )
        mn_iter = self.mn_iter[begin:end].flatten()
        nn_pairs = torch.index_select(self.data, dim=0, index=nn_iter)
        mn_pairs = torch.index_select(self.data, dim=0, index=mn_iter)
        batch = torch.concat((basis, nn_pairs, mn_pairs), dim=0)
        self.idx += 1
        return n_items, batch, fp_indices, basis_label

    def __len__(self):
        return self.n_batches


class NegativeSamplingDataset(torch.utils.data.Dataset):
    """The Negative Sampling dataset for Dimensionality Reduction.
    Input:
        data: A numpy.ndarray of the shape [N, D] that consists the dataset.
        nn_pairs: A numpy.ndarray of the shape [N, np_nn] that consists the
            nearest neighbor pairs.
        fp_pairs: A numpy.ndarray of the shape [N, np_fp] that consists the
            farther pairs.
        mn_pairs: A numpy.ndarray of the shape [N, np_mn] that consists the
            mid-near pairs.
    """

    def __init__(
        self,
        data,
        nn_pairs,
        fp_pairs,
        mn_pairs,
        seed=0,
        reshape=None,
        dtype=torch.float32,
    ):
        self.data = data
        self.nn_pairs = nn_pairs
        self.num_fp = fp_pairs.shape[1]
        self.mn_pairs = mn_pairs
        self.reshape = reshape
        self.data_size = data.shape[0]
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self._dtype = dtype
        if self.reshape is not None:
            assert np.product(self.reshape) == self.data.shape[-1]
            new_shape = [
                self.data.shape[0],
            ] + self.reshape
            self.data = self.data.reshape(new_shape)

    def __getitem__(self, index):
        """Given a series of index, return all of the points that connect to it."""
        basis = torch.tensor(self.data[index], dtype=self._dtype)
        nn_pair = self.nn_pairs[index]
        nn_pairs = torch.tensor(self.data[nn_pair], dtype=self._dtype)
        mn_pair = self.mn_pairs[index]
        mn_pairs = torch.tensor(self.data[mn_pair], dtype=self._dtype)
        fp_pair = self.rng.choice(self.data_size, self.num_fp, replace=False)
        fp_pairs = torch.tensor(
            self.data[fp_pair], dtype=self._dtype
        )  # FP pairs are randomly sampled
        return basis, nn_pairs, fp_pairs, mn_pairs

    def __len__(self):
        return len(self.data)


class TensorDataset(torch.utils.data.Dataset):
    """The tensor dataset for inference.
    Input:
        data: A numpy.ndarray of the shape [N, D] that consists the dataset.
        shuffle_pairs: A boolean value.
    """

    def __init__(self, data, reshape=None, dtype=torch.float32):
        self.data = data
        self.reshape = reshape
        self._dtype = dtype
        if self.reshape is not None:
            assert np.product(self.reshape) == self.data.shape[-1]
            new_shape = [
                self.data.shape[0],
            ] + self.reshape
            self.data = self.data.reshape(new_shape)

    def __getitem__(self, index):
        basis = torch.tensor(self.data[index], dtype=self._dtype)
        return basis

    def __len__(self):
        return len(self.data)
