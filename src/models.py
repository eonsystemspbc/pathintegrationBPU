from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from torch import nn

from .config import INPUT_DIM, OUTPUT_DIM


@dataclass(frozen=True)
class ModelMasks:
    sensory_indices: list[int]
    output_indices: list[int]


class CXBPU(nn.Module):
    def __init__(
        self,
        recurrent: sparse.spmatrix | np.ndarray | torch.Tensor,
        sensory_indices: list[int],
        output_indices: list[int],
        K: int,
        reset_each_timestep: bool = False,
    ) -> None:
        super().__init__()
        if sparse.issparse(recurrent):
            rec_array = recurrent.toarray().astype(np.float32)
        elif isinstance(recurrent, torch.Tensor):
            rec_array = recurrent.detach().cpu().numpy().astype(np.float32)
        else:
            rec_array = np.asarray(recurrent, dtype=np.float32)
        if rec_array.ndim != 2 or rec_array.shape[0] != rec_array.shape[1]:
            raise ValueError("recurrent matrix must be square.")
        if not sensory_indices:
            raise ValueError("sensory_indices cannot be empty.")
        if not output_indices:
            raise ValueError("output_indices cannot be empty.")
        self.N = int(rec_array.shape[0])
        self.K = int(K)
        self.reset_each_timestep = bool(reset_each_timestep)
        self.register_buffer("W_rec", torch.as_tensor(rec_array, dtype=torch.float32))
        self.register_buffer(
            "sensory_indices", torch.as_tensor(sensory_indices, dtype=torch.long)
        )
        self.register_buffer("output_indices", torch.as_tensor(output_indices, dtype=torch.long))
        scale_in = 1.0 / math.sqrt(INPUT_DIM)
        scale_out = 1.0 / math.sqrt(max(len(output_indices), 1))
        self.W_in = nn.Parameter(torch.empty(len(sensory_indices), INPUT_DIM))
        self.b_in = nn.Parameter(torch.zeros(len(sensory_indices)))
        self.W_out = nn.Parameter(torch.empty(OUTPUT_DIM, len(output_indices)))
        self.b_out = nn.Parameter(torch.zeros(OUTPUT_DIM))
        nn.init.uniform_(self.W_in, -scale_in, scale_in)
        nn.init.uniform_(self.W_out, -scale_out, scale_out)

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != INPUT_DIM:
            raise ValueError(f"inputs must have shape [batch, T, {INPUT_DIM}]")
        batch, T, _ = inputs.shape
        if h0 is None:
            h = inputs.new_zeros((batch, self.N))
        else:
            h = h0
        outputs: list[torch.Tensor] = []
        rec_t = self.W_rec.t()
        for t in range(T):
            if self.reset_each_timestep:
                h = inputs.new_zeros((batch, self.N))
            injection = inputs[:, t, :] @ self.W_in.t() + self.b_in
            for microstep in range(self.K):
                next_h = h @ rec_t
                if microstep == 0:
                    next_h = next_h.index_add(
                        1,
                        self.sensory_indices,
                        injection,
                    )
                h = torch.relu(next_h)
            readout = h.index_select(1, self.output_indices)
            outputs.append(readout @ self.W_out.t() + self.b_out)
        return torch.stack(outputs, dim=1)


class GRUBaseline(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(INPUT_DIM, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, OUTPUT_DIM)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(inputs)
        return self.out(h)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def assert_bpu_trainable_surface(model: CXBPU) -> None:
    expected = ["W_in", "b_in", "W_out", "b_out"]
    observed = trainable_parameter_names(model)
    if observed != expected:
        raise AssertionError(f"CXBPU trainable surface mismatch: {observed} != {expected}")
    if model.W_rec.requires_grad:
        raise AssertionError("W_rec must be frozen.")

