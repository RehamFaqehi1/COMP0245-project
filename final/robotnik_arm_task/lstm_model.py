"""LSTM behaviour-cloning policy for the COMP0245 Target37 experiment."""

from __future__ import annotations

import torch
from torch import nn


class LSTMPolicy(nn.Module):
    def __init__(
        self,
        input_size=37,
        output_size=11,
        hidden_size=128,
        num_layers=2,
        dropout=0.15,
        sequence_length=25,
        dense_sizes=(256, 256, 128, 64),
    ):
        super().__init__()

        if sequence_length < 1 or num_layers < 1:
            raise ValueError(
                "sequence_length and num_layers must be positive"
            )

        if not dense_sizes or any(int(size) < 1 for size in dense_sizes):
            raise ValueError("All dense layer sizes must be positive")

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.sequence_length = int(sequence_length)
        self.dense_sizes = tuple(int(size) for size in dense_sizes)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        layers = []
        previous_size = self.hidden_size

        for size in self.dense_sizes:
            layers.extend([
                nn.Linear(previous_size, size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            previous_size = size

        layers.append(nn.Linear(previous_size, self.output_size))
        self.head = nn.Sequential(*layers)

        self._step_buffer = None

    def reset_state(self):
        self._step_buffer = None

    def forward_sequence(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("Expected a 3D sequence tensor")

        if sequence.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"received {sequence.shape[-1]}"
            )

        features, _ = self.lstm(sequence)
        return self.head(features[:, -1, :])

    def forward(self, x):
        # Training/inference with a complete sequence.
        if x.ndim == 3:
            return self.forward_sequence(x)

        # Stateful one-observation-at-a-time inference.
        if x.ndim != 2 or x.shape != (1, self.input_size):
            raise ValueError(
                f"Expected (1, {self.input_size}) for single-step input"
            )

        row = x.unsqueeze(1)

        if self._step_buffer is None:
            self._step_buffer = row.expand(
                1, self.sequence_length, self.input_size
            ).clone()
        else:
            self._step_buffer = torch.cat(
                (self._step_buffer[:, 1:, :], row),
                dim=1,
            )

        return self.forward_sequence(self._step_buffer)