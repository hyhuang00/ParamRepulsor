"""Implementing the ParamRepulsor/ParamPaCMAP Algorithm as a sklearn estimator."""

import logging
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from sklearn import decomposition, preprocessing
from sklearn.base import BaseEstimator

from parampacmap import training
from parampacmap.models import dataset, module, TORCH_DEVICE
from parampacmap.utils import data, utils

logger = logging.getLogger(__name__)


def pacmap_weight_schedule(epoch: int):
    """Weight schedule for PaCMAP/ParamPaCMAP."""
    if epoch < 100:
        w_mn = 10 * (100 - epoch) + 0.03 * epoch
        w_nn = 2.0
        w_fp = 1.0
    elif epoch < 200:
        w_mn = 3.0
        w_nn = 3.0
        w_fp = 1.0
    else:
        w_mn = 0.0
        w_nn = 1.0
        w_fp = 1.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def paramrep_weight_schedule(epoch: int):
    """Weight schedule for ParamRepulsor."""
    if epoch < 200:
        w_nn = 4.0
        w_fp = 8.0
        w_mn = 0.0
    else:
        w_nn = 1.0
        w_fp = 8.0
        w_mn = -12.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def paramrep_const_schedule(epoch: int):
    """Const schedule for ParamRepulsor."""
    w_nn = 10.0
    w_fp = 1.0
    w_mn = 1.0
    const = np.array([w_nn, w_fp, w_mn])
    return const


class ParamPaCMAP(BaseEstimator):
    """Parametric PaCMAP implemented with Pytorch."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        n_FP: int = 20,
        n_MN: int = 5,
        distance: str = "euclidean",
        optim_type: str = "Adam",
        lr: float = 1e-3,
        lr_schedule: Optional[bool] = None,
        apply_pca: bool = True,
        apply_scale: Optional[str] = None,
        model_dict: Optional[dict] = utils.DEFAULT_MODEL_DICT,
        intermediate_snapshots: Optional[list] = [],
        loss_weight: Optional[list] = [1, 1, 1],
        batch_size: int = 1024,
        data_reshape: Optional[list] = None,
        num_epochs: int = 450,
        verbose: bool = False,
        weight_schedule: Callable = paramrep_weight_schedule,
        const_schedule: Optional[Callable] = paramrep_const_schedule,
        num_workers: int = 1,
        dtype: torch.dtype = torch.float32,
        embedding_init: str = "pca",
        seed: Optional[int] = None,
        save_pairs: bool = False,
    ):
        super().__init__()
        self.n_components = n_components  # output_dims
        self.n_neighbors = n_neighbors
        self.n_FP = n_FP
        self.n_MN = n_MN
        self.distance = distance
        self.optim_type = optim_type
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.apply_pca = apply_pca
        self.apply_scale = apply_scale
        # Placeholder for the model. The model is initialized during fit.
        self.model = None
        self.model_dict = model_dict
        self.intermediate_snapshots = intermediate_snapshots
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.data_reshape = data_reshape
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.weight_schedule = weight_schedule
        self.num_workers = num_workers
        self._dtype = dtype
        self._scaler = None
        self._projector = None
        self.time_profiles = None
        self.const_schedule = const_schedule

        # Pair-saving related variables
        self.save_pairs = save_pairs
        self._pairs_saved = False
        self._num_samples = None
        self.pair_neighbors = None
        self.pair_MN = None
        self.pair_FP = None
        self.device = TORCH_DEVICE
        self._pairs_saved = False
        if self._dtype == torch.float32:
            torch.set_float32_matmul_precision("medium")
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)  # Untested: mutli-GPU reproducibility
            # np.random.seed(seed)  # does not seem needed - even on CPU-only.
        if embedding_init not in ["pca", "random"]:
            raise ValueError(
                f"Embedding init mode '{embedding_init}' is not supported."
            )
        self.embedding_init = embedding_init

    def _scale_input(self, X: np.ndarray, input_dims: int) -> Tuple[np.ndarray, int]:
        # Data Preprocessing
        if input_dims > 100 and self.apply_pca:
            self._projector = decomposition.PCA(n_components=100)
            X = self._projector.fit_transform(X)
            input_dims = X.shape[1]
        if self.apply_scale == "standard":
            self._scaler = preprocessing.StandardScaler()
            X = self._scaler.fit_transform(X)
        elif self.apply_scale == "minmax":
            self._scaler = preprocessing.MinMaxScaler()
            X = self._scaler.fit_transform(X)
        return (X, input_dims)

    def fit(
        self,
        X: np.ndarray,
        profile_only: bool = False,
        per_layer: bool = False,
    ) -> None:
        # Reset random states at the start of fit
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        fit_begin = time.perf_counter()
        input_dims = X.shape[1]

        # Data Preprocessing
        X, input_dims = self._scale_input(X, input_dims)

        self.model = (
            module.ParamPaCMAP(
                input_dims=input_dims,
                output_dims=self.n_components,
                model_dict=self.model_dict,
                n_samples=X.shape[0],
            )
            .to(self.device)
            .to(self._dtype)
        )
        self.loss = module.PaCMAPLoss(
            weight=self.loss_weight,
        ).to(self._dtype)
        self.intermediate_outputs = []

        # Constructing dataloader
        if self.save_pairs and self._pairs_saved:
            if X.shape[0] != self._num_samples:
                logger.warning("Number of samples has changed. Are you sure you want"
                " to use the saved pairs?")
            pair_neighbors, pair_MN, pair_FP = (
                self.pair_neighbors,
                self.pair_MN,
                self.pair_FP,
            )
        else:
            pair_neighbors, pair_MN, pair_FP, _ = data.generate_pair(
                X,
                n_neighbors=self.n_neighbors,
                n_MN=self.n_MN,
                n_FP=self.n_FP,
                distance=self.distance,
                verbose=False,
                random_state=self.seed,  # critical line for reproducibility
            )
            if self.save_pairs:
                self.pair_neighbors = pair_neighbors
                self.pair_MN = pair_MN
                self.pair_FP = pair_FP
                self._pairs_saved = True
                self._num_samples = X.shape[0]

        nn_pairs, fp_pairs, mn_pairs = training.convert_pairs(
            pair_neighbors, pair_FP, pair_MN, X.shape[0]
        )

        train_loader_ctor = dataset.FastDataloader

        # For non-parametric version, we will use the indices as the input.
        if not self.model.is_parametric:
            self._init_embedding(X)
            X = np.arange(X.shape[0])
            data_dtype = torch.int32
        else:
            data_dtype = self._dtype

        train_loader = train_loader_ctor(
            data=X,
            nn_pairs=nn_pairs,
            fp_pairs=fp_pairs,
            mn_pairs=mn_pairs,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=True,
            reshape=self.data_reshape,
            dtype=data_dtype,
            # seed=self.seed,  # TODO: may be relevant if dataset.FastDataloader is not used?
        )
        test_set = dataset.TensorDataset(
            data=X, reshape=self.data_reshape, dtype=data_dtype
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=max(1, self.num_workers),
            persistent_workers=True,
        )

        parameter_set = [{"params": self.model.backbone.parameters()}]
        # Construct optimizer
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.optim_type == "Adam":
            optimizer = optim.Adam(parameter_set, lr=self.lr)
        elif self.optim_type == "SGD":
            optimizer = optim.SGD(parameter_set, lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optim_type}")

        if profile_only:
            epoch_begin = time.perf_counter()
            print(
                f"Time Profile: Before Epoch\n"
                f"Preparation:{(epoch_begin - fit_begin):03.3f}s\n"
            )
            self._tune_weight(epoch=0)
            self._profile_epoch(train_loader, optimizer)
            self._embedding = None
            return

        for epoch in range(self.num_epochs):
            if epoch in self.intermediate_snapshots:
                if per_layer:
                    result = self._inference_per_layer(test_loader)
                else:
                    result = self._inference(test_loader)
                self.intermediate_outputs.append(result)
            # Tune the weights
            self._tune_weight(epoch=epoch)

            # Perform training for one epoch
            self._train_epoch(train_loader, epoch, optimizer)

        if per_layer:
            self._embedding = self._inference_per_layer(test_loader)
        else:
            self._embedding = self._inference(test_loader)

    def _tune_weight(self, epoch: int):
        """Automatically tune the weight."""
        # Decide weight based on the functions
        weight = self.weight_schedule(epoch)
        if self.const_schedule is not None:
            const = self.const_schedule(epoch)
        else:
            const = None
        self.loss.update_weight(weight, const)

    def _train_epoch(
        self, train_loader, epoch, optimizer: optim.Optimizer, has_labels: bool = False
    ):
        """Perform a single epoch of training."""
        for batch in train_loader:
            optimizer.zero_grad()
            num_items, model_input, model_label = batch
            if has_labels:
                model_output, predicted_labels = self.model(model_input)
                predicted_labels = predicted_labels[:num_items]
            else:
                model_output = self.model(model_input)
                predicted_labels = None
            basis = model_output[:num_items]
            nn_pairs = model_output[num_items : num_items * (self.n_neighbors + 1)]
            fp_pairs = model_output[
                num_items
                * (self.n_neighbors + 1) : num_items
                * (self.n_neighbors + self.n_FP + 1)
            ]
            mn_pairs = model_output[num_items * (self.n_neighbors + self.n_FP + 1) :]
            basis = torch.unsqueeze(basis, 1)
            nn_pairs = nn_pairs.view(num_items, self.n_neighbors, nn_pairs.shape[1])
            fp_pairs = fp_pairs.view(num_items, self.n_FP, fp_pairs.shape[1])
            mn_pairs = mn_pairs.view(num_items, self.n_MN, mn_pairs.shape[1])
            loss = self.loss(
                basis,
                nn_pairs,
                fp_pairs,
                mn_pairs,
                predicted_labels=predicted_labels,
                labels=model_label,
            )
            loss.backward()
            optimizer.step()
        if ((epoch + 1) % 20 == 0 or epoch == 0) and self.verbose:
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f},",
                flush=True,
            )

    def _train_epoch_ib(
        self, train_loader, epoch, optimizer: optim.Optimizer, has_labels: bool = False
    ):
        for batch in train_loader:
            optimizer.zero_grad()
            num_items, model_input, model_label, fp_indices = batch
            if has_labels:
                model_output, predicted_labels = self.model(model_input)
                predicted_labels = predicted_labels[:num_items]
            else:
                model_output = self.model(model_input)
                predicted_labels = None
            basis = model_output[:num_items]
            nn_pairs = model_output[num_items : num_items * (self.n_neighbors + 1)]
            mn_pairs = model_output[num_items * (self.n_neighbors + 1) :]
            fp_pairs = torch.index_select(nn_pairs, dim=0, index=fp_indices)
            basis = torch.unsqueeze(basis, 1)
            nn_pairs = nn_pairs.view(num_items, self.n_neighbors, nn_pairs.shape[1])
            fp_pairs = fp_pairs.view(num_items, self.n_FP, fp_pairs.shape[1])
            mn_pairs = mn_pairs.view(num_items, self.n_MN, mn_pairs.shape[1])
            loss = self.loss(
                basis,
                nn_pairs,
                fp_pairs,
                mn_pairs,
                predicted_labels=predicted_labels,
                labels=model_label,
            )
            loss.backward()
            optimizer.step()
        if ((epoch + 1) % 20 == 0 or epoch == 0) and self.verbose:
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f},",
                flush=True,
            )

    def _profile_epoch(
        self, train_loader, optimizer: optim.Optimizer, has_labels: bool = False
    ):
        """Perform a single epoch of training with detailed profiling."""
        time_profiles = []
        batch_begin = time.perf_counter()
        for batch in train_loader:
            torch.cuda.synchronize()
            time_dataloader = time.perf_counter()
            optimizer.zero_grad()
            # The pairs are under the format (i, num_pairs, ...)
            num_items, model_input = batch
            torch.cuda.synchronize()
            time_reshape = time.perf_counter()
            # Use the model to perform forward
            if has_labels:
                model_output, predicted_labels = self.model(model_input)
                predicted_labels = predicted_labels[:num_items]
            else:
                model_output = self.model(model_input)
                predicted_labels = None
            basis = model_output[:num_items]
            nn_pairs = model_output[num_items : num_items * (self.n_neighbors + 1)]
            fp_pairs = model_output[
                num_items
                * (self.n_neighbors + 1) : num_items
                * (self.n_neighbors + self.n_FP + 1)
            ]
            mn_pairs = model_output[num_items * (self.n_neighbors + self.n_FP + 1) :]
            torch.cuda.synchronize()
            time_forward = time.perf_counter()
            basis = torch.unsqueeze(basis, 1)
            nn_pairs = nn_pairs.view(num_items, self.n_neighbors, nn_pairs.shape[1])
            fp_pairs = fp_pairs.view(num_items, self.n_FP, fp_pairs.shape[1])
            mn_pairs = mn_pairs.view(num_items, self.n_MN, mn_pairs.shape[1])
            loss = self.loss(basis, nn_pairs, fp_pairs, mn_pairs)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            time_backward = time.perf_counter()
            time_series = [
                time_dataloader - batch_begin,
                time_reshape - time_dataloader,
                time_forward - time_reshape,
                time_backward - time_forward,
            ]
            batch_begin = time_backward
            time_profiles.append(time_series)
        self.time_profiles = np.array(time_profiles)
        # Generate a profile report
        time_summary = np.sum(self.time_profiles, axis=0)
        summary_text = (
            f"Time Profile: Sum in Epoch\n"
            f"Dataloader: {time_summary[0]:03.3f}s\n"
            f"Reshape:    {time_summary[1]:03.3f}s\n"
            f"Forward:    {time_summary[2]:03.3f}s\n"
            f"Backward:   {time_summary[3]:03.3f}s\n"
        )
        print(summary_text)

    def _inference(self, test_loader):
        """Perform a pure inference for the model."""
        results = []
        with torch.inference_mode():
            for batch in test_loader:
                result = self.model(batch.to(self.device))
                if isinstance(result, tuple):
                    result = result[0]  # Remove predicted labels
                results.append(result.detach())
            results = torch.concatenate(results)
            results = results.float().cpu().numpy()
        return results

    def _inference_per_layer(self, test_loader):
        """Perform a pure inference for the model."""
        self.model.set_output_per_layer(True)
        results = []
        with torch.inference_mode():
            for batch in test_loader:
                result = self.model(
                    batch.to(self.device)
                )  # A list of multiple embeddings
                if isinstance(result, tuple):
                    result = result[0]  # Remove predicted labels
                results.append(result)
        all_same_size = all(len(result) == len(results[0]) for result in results)
        assert all_same_size
        num_layers = len(results[0])
        layer_results = []
        for i in range(num_layers):
            sub_result = [result[i] for result in results]
            layer_result = torch.concatenate(sub_result).float().cpu().numpy()
            layer_results.append(layer_result)
        self.model.set_output_per_layer(False)
        return layer_results

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, per_layer: bool = False
    ):
        self.fit(X, per_layer=per_layer)
        if len(self.intermediate_outputs) == 0:
            return self._embedding
        return self._embedding, self.intermediate_outputs

    def _prepare_test_loader(self, X: np.ndarray) -> torch.utils.data.DataLoader:
        if self.model.is_parametric:
            if self._projector is not None:
                X = self._projector.transform(X)
            if self._scaler is not None:
                X = self._scaler.transform(X)
        else:
            X = np.arange(X.shape[0])
        data_dtype = self._dtype if self.model.is_parametric else torch.int32
        test_set = dataset.TensorDataset(
            data=X, reshape=self.data_reshape, dtype=data_dtype
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return test_loader

    def transform(self, X: np.ndarray, per_layer: bool = False) -> np.ndarray:
        test_loader = self._prepare_test_loader(X)
        if per_layer:
            return self._inference_per_layer(test_loader)
        return self._inference(test_loader)

    def _init_embedding(self, X: np.ndarray) -> None:
        with torch.inference_mode():
            state_dict = self.model.backbone.state_dict()
            if self.embedding_init == "pca":
                state_dict["weight"] = (
                    torch.tensor(
                        X[:, : self.n_components],
                        dtype=self._dtype,
                        device=self.device,
                    )
                    * 0.01
                )
            elif self.embedding_init == "random":
                state_dict["weight"] = state_dict["weight"] * 0.0001
            self.model.backbone.load_state_dict(state_dict)
