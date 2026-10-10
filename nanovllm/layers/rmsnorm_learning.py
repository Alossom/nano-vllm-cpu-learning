import math
import torch
from torch import nn

class RMSNorm(nn.Module):
	def __init__(self, hidden_size: int, eps: float = 1e-6):
		super().__init__()

		if type(hidden_size) is not int or hidden_size <= 0:
			raise ValueError("hidden_size must be a positive integer")

		eps = float(eps)
		if not math.isfinite(eps) or eps <= 0:
			raise ValueError("eps must be finite and positive")

		self.hidden_size = hidden_size
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32, device="cpu"))

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		if hidden_states.ndim < 1:
			raise ValueError("Input must have at least one dimension")

		if hidden_states.shape[-1] != self.hidden_size:
			raise ValueError(f"Expected last dimension {self.hidden_size}, " f"got {hidden_states.shape[-1]}")

		if hidden_states.device.type != "cpu":
			raise ValueError("This reference implementation requires CPU input")

		if hidden_states.dtype != torch.float32:
			raise ValueError("This reference implementation requires float32 inpu")

		if(self.weight.device != hidden_states.device or self.weight.dtype != hidden_states.dtype):
			raise ValueError("Weight and input must have the same device and dtype")

		mean_square = hidden_states.square().mean(dim=-1, keepdim=True)

		inverse_rms = torch.rsqrt(mean_square + self.eps)

		normalized = hidden_states * inverse_rms

		return normalized * self.weight
