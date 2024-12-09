import os

import torch
from torch import nn

if os.environ.get("TORCH_DEVICE", "") == "cpu":
    TORCH_DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
else:
    TORCH_DEVICE = torch.device("cpu")

class SinLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class ANNLayer(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        bias: bool = True,
        device: torch.device = TORCH_DEVICE,
        dtype: torch.dtype = torch.float32,
        activation: str = "relu",
        residual_connection: bool = False,
    ):
        super().__init__()
        self.bias = bias
        self.linear = nn.Linear(
            in_dims, out_dims, bias=bias, device=device, dtype=dtype
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "sin":
            self.activation = SinLayer()
        elif activation is not None:
            raise ValueError(f"Activation {activation} is not yet supported.")
        else:
            self.activation = None
        self.residual_connection = residual_connection and in_dims == out_dims

    def eye_init(self):
        nn.init.eye_(self.linear.weight)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        output = self.linear(x)
        if self.activation is not None:
            output = self.activation(output)
        if self.residual_connection:
            return output + x
        return output


class ANN(nn.Module):
    """A simple Feed-Forward Neural Network that serve as the backbone."""

    def __init__(
        self,
        layer_size,
        eye_init: bool = False,
        bias: bool = True,
        device: torch.device = TORCH_DEVICE,
        dtype: torch.dtype = torch.float32,
        activation: str = "relu",
        residual_connection: bool = False,
    ):
        super().__init__()
        layers = []
        num_layers = len(layer_size) - 1
        self.activation = activation
        for i in range(num_layers):
            layer = ANNLayer(
                in_dims=layer_size[i],
                out_dims=layer_size[i + 1],
                bias=bias,
                device=device,
                dtype=dtype,
                activation=None if i == num_layers - 1 else self.activation,
                residual_connection=residual_connection,
            )
            if eye_init:
                layer.eye_init()
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self._output_per_layer = False

    def set_output_per_layer(self, value):
        self._output_per_layer = value

    def forward(self, x: torch.Tensor):
        if self._output_per_layer:
            outputs = []
            for layer in self.layers:
                x = layer(x)
                outputs.append(x.detach())
            return outputs
        for layer in self.layers:
            x = layer(x)
        return x


class ParamPaCMAP(nn.Module):
    def __init__(
        self,
        input_dims: int = 100,
        output_dims: int = 2,
        model_dict: dict = {},
        n_samples: int = None,
        is_parametric: bool = True,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.model_dict = model_dict
        self._output_per_layer = False
        self.backbone = self.get_backbone(input_dims, output_dims, model_dict)
        self.is_parametric = is_parametric

    def set_output_per_layer(self, value: bool):
        self._output_per_layer = value
        if value and not (
            isinstance(self.backbone, ANN) or isinstance(self.backbone, nn.Embedding)
        ):
            raise ValueError("CNN does not support output per layer yet.")
        if isinstance(self.backbone, ANN):
            self.backbone.set_output_per_layer(value)


    def get_backbone(
        self, input_dims=100, output_dims=2, model_dict: dict = {}
    ) -> nn.Module:
        backbone = model_dict["backbone"]
        device = TORCH_DEVICE

        # Fully Connected Layers
        if backbone == "ANN":
            eye_init = model_dict.get("eye_init", False)
            dtype = model_dict.get("dtype", torch.float32)
            residual = model_dict.get("residual", False)
            bias = model_dict.get("bias", True)
            activation = model_dict.get("activation", "relu")
            layer_size = [input_dims] + model_dict["layer_size"] + [output_dims]
            model = ANN(
                layer_size=layer_size,
                eye_init=eye_init,
                bias=bias,
                device=device,
                dtype=dtype,
                activation=activation,
                residual_connection=residual,
            )
            return model
        elif backbone == "CNN":
            module_list = []
            for i in range(len(model_dict["conv_size"])):
                # InChannel, OutChannel, Size
                module_list.append(
                    nn.Conv2d(
                        model_dict["conv_size"][i][0],
                        model_dict["conv_size"][i][1],
                        model_dict["conv_size"][i][2],
                        padding="same",
                    )
                )
                module_list.append(nn.BatchNorm2d(model_dict["conv_size"][i][1]))
                module_list.append(nn.ReLU())
            module_list.append(nn.Flatten())  # Flatten the intermediate layer
            layer_size = model_dict["layer_size"] + [output_dims]
            for i in range(len(layer_size) - 1):
                module_list.append(nn.Linear(layer_size[i], layer_size[i + 1]))
                module_list.append(nn.ReLU())
            module_list = module_list[:-1]
            return nn.Sequential(*module_list)
        elif backbone == "embedding":
            assert self.n_samples is not None
            embedding = nn.Embedding(self.n_samples, output_dims)
            return embedding
        else:
            raise NotImplementedError("Unsupported model backbone style.")

    def forward(self, sample):
        embedding = self.backbone(sample)
        return embedding


class PaCMAPLoss(nn.Module):
    def __init__(
        self,
        weight,
        thresholds=[None, None, None],
        exponents=[2, 2, 2],
        consts=[10, 1, 10000],
        label_weight=1e-1,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.nnloss = NNLoss(
            weight[0],
            threshold=thresholds[0],
            exponent=exponents[0],
            const=consts[0],
            device=TORCH_DEVICE,
        )
        self.fploss = FPLoss(
            weight[1],
            threshold=thresholds[1],
            exponent=exponents[1],
            const=consts[1],
            device=TORCH_DEVICE,
        )
        self.mnloss = MNLoss(
            weight[2],
            threshold=thresholds[2],
            exponent=exponents[2],
            const=consts[2],
            device=TORCH_DEVICE,
        )
        self.label_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.label_weight = label_weight

    def forward(
        self, basis, nn_pairs, fp_pairs, mn_pairs, predicted_labels=None, labels=None
    ):
        # Based on the labels, generate the outputs
        nn_loss = self.nnloss(basis, nn_pairs)
        fp_loss = self.fploss(basis, fp_pairs)
        mn_loss = self.mnloss(basis, mn_pairs)
        loss = nn_loss + fp_loss + mn_loss
        if labels is not None:
            label_loss = self.label_weight * self.label_loss(predicted_labels, labels)
            loss += label_loss
        return loss

    def update_weight(self, weight, const=None) -> None:
        self.weight = weight
        self.nnloss.weight = weight[0]
        self.fploss.weight = weight[1]
        self.mnloss.weight = weight[2]
        if const is not None:
            self.nnloss.const.fill_(const[0])
            self.fploss.const.fill_(const[1])
            self.mnloss.const.fill_(const[2])


class NNLoss(nn.Module):
    """NN Loss of PaCMAP."""

    def __init__(
        self, weight, threshold=None, exponent=2, const=10, device=TORCH_DEVICE,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.exponent = exponent
        self.multiplier = 0
        self.const = torch.tensor(
            [
                const,
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, basis, pair_components):
        diff = pair_components - basis  # N, P, D
        norm = torch.linalg.norm(diff, dim=2)
        d2 = norm**self.exponent + 1  # dist squared
        loss = d2 / (self.const + d2)
        if self.threshold is not None:
            loss = torch.where(loss > self.threshold, loss, 1.0)
        loss = torch.sum(loss, dim=1)
        return self.weight * torch.mean(loss)


class FPLoss(nn.Module):
    """FP Loss of PaCMAP."""

    def __init__(
        self, weight, threshold=None, exponent=2, const=1, device=TORCH_DEVICE,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.exponent = exponent
        self.const = torch.tensor(
            [
                const,
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, basis, pair_components):
        diff = pair_components - basis  # N, P, D
        norm = torch.linalg.norm(diff, dim=2)
        d2 = norm**self.exponent + 1  # dist squared
        loss = self.const / (self.const + d2)
        if self.threshold is not None:
            loss = torch.where(loss < self.threshold, loss, 0.0)
        loss = torch.sum(loss, dim=1)
        return self.weight * torch.mean(loss)


class MNLoss(nn.Module):
    """MN Loss of PaCMAP."""

    def __init__(
        self,
        weight,
        threshold=None,
        exponent=2,
        const=10000,
        device=TORCH_DEVICE,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.exponent = exponent
        self.const = torch.tensor(
            [
                const,
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, basis, pair_components):
        diff = pair_components - basis  # N, P, D
        norm = torch.linalg.norm(diff, dim=2)
        d2 = norm**self.exponent + 1  # dist squared
        loss = d2 / (self.const + d2)
        if self.threshold is not None:
            loss = torch.where(loss > self.threshold, loss, 1.0)
        loss = torch.sum(loss, dim=1)
        return self.weight * torch.mean(loss)
