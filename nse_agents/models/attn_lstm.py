"""Peephole LSTM with a temporal attention layer (PLSTM-TAL, Heliyon 2024).

PyTorch ships no peephole LSTM, so the cell is written out here. The peephole
connections let each gate see the cell state ``c_{t-1}`` directly rather than
only the hidden state, which is the paper's specific architectural claim; the
temporal attention layer then pools the hidden sequence by learned relevance
instead of taking the last step.

Both are implemented as switchable flags rather than hardcoded, so the
ablations that isolate their contribution ("does attention actually help, or
does a vanilla LSTM do as well?") run the identical code path with one
argument changed. An ablation that compares two separately-written models is
comparing two implementations, not two ideas.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PeepholeLSTMCell(nn.Module):
    """LSTM cell whose input, forget and output gates read the cell state."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        # Peephole weights are diagonal: each gate unit sees only its own cell
        # unit, which is what makes them cheap (3H parameters, not 3H^2).
        self.p_i = nn.Parameter(torch.zeros(hidden_size))
        self.p_f = nn.Parameter(torch.zeros(hidden_size))
        self.p_o = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.x2h.weight, self.h2h.weight):
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.x2h.bias)
        # Forget-gate bias at 1.0: the standard fix for an LSTM that forgets
        # everything early in training.
        with torch.no_grad():
            self.x2h.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.x2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i + self.p_i * c)
        f = torch.sigmoid(f + self.p_f * c)
        g = torch.tanh(g)
        c_new = f * c + i * g
        o = torch.sigmoid(o + self.p_o * c_new)  # output peephole sees the new state
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class VanillaLSTMWrapper(nn.Module):
    """cuDNN LSTM behind the same interface, for the peephole ablation."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = PeepholeLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        outputs = []
        for t in range(steps):
            h, c = self.cell(x[:, t], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over the time axis."""

    def __init__(self, hidden_size: int, attn_size: int = 32):
        super().__init__()
        self.project = nn.Linear(hidden_size, attn_size)
        self.score = nn.Linear(attn_size, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(torch.tanh(self.project(hidden))).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


class PLSTMTAL(nn.Module):
    """Directional classifier: sequence in, one logit out.

    ``attention_weights`` from the forward pass are what make this agent
    explainable at the technical level -- they say which of the last 30 days
    the call actually rested on, and are surfaced verbatim in the agent's
    rationale rather than being decorative.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        attn_size: int = 32,
        dropout: float = 0.2,
        peephole: bool = True,
        attention: bool = True,
    ):
        super().__init__()
        self.peephole = peephole
        self.attention = attention
        self.encoder = (
            PeepholeLSTM(input_size, hidden_size)
            if peephole
            else VanillaLSTMWrapper(input_size, hidden_size)
        )
        self.attn = TemporalAttention(hidden_size, attn_size) if attention else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        if self.attn is not None:
            context, weights = self.attn(hidden)
            self.last_attention = weights.detach()
        else:
            context = hidden[:, -1]
            self.last_attention = None
        return self.head(self.dropout(context)).squeeze(-1)

    def describe(self) -> str:
        parts = ["peephole-LSTM" if self.peephole else "vanilla-LSTM"]
        parts.append("temporal-attention" if self.attention else "last-step-pooling")
        return " + ".join(parts)
