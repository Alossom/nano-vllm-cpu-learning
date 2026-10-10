import math

import torch

class ReferenceRoPE:
	def __init__(self, head_dim: int, theta: float = 10000.0):
		if type(head_dim) is not int or head_dim <= 0:
			raise ValueError("head_dim must be a positive integer")

		if head_dim % 2 != 0:
			raise ValueError("head_dim must be even")

		theta = float(theta)
		if not math.isfinite(theta) or theta <= 1.0:
			raise ValueError("theta must be finite and greater than 1")

		self.head_dim = head_dim
		self.theta = theta

		exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu") / head_dim

		self.inv_freq = 1.0 / (theta ** exponents)

	def _rotate(self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
		half = self.head_dim // 2

		first = tensor[..., :half]
		second = tensor[..., half:]

		rotated_first = first * cos - second * sin
		rotated_second = first * sin + second * cos

		return torch.cat((rotated_first, rotated_second), dim=-1)

	@torch.no_grad()
	def forward(self, queries: torch.Tensor, keys: torch.Tensor, positions: list[int] | tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
		positions = tuple(positions)

		if not positions:
			raise ValueError("positions must not be empty")

		if any(type(position) is not int or position < 0 for position in positions):
			raise ValueError("Positions must be nonnegative integers")

		for name, tensor in (("queries", queries), ("keys", keys)):
			if tensor.ndim != 3:
				raise ValueError(f"{name} must have shape [tokens, heads, head_dim]")

			if (tensor.shape[0] != len(positions) or tensor.shape[1] <= 0 or tensor.shape[2] != self.head_dim):
				raise ValueError(f"Invalid {name} shape")

			if tensor.device.type != "cpu":
				raise ValueError(f"{name} must be on CPU")

			if tensor.dtype != torch.float32:
				raise ValueError(f"{name} must use float32")

		position_tensor = torch.tensor(positions, dtype=torch.float32, device="cpu")

		angles = (position_tensor.unsqueeze(1) * self.inv_freq.unsqueeze(0))

		cos = angles.cos().unsqueeze(1)
		sin = angles.sin().unsqueeze(1)

		rotated_queries = self._rotate(queries, cos, sin)
		rotated_keys = self._rotate(keys, cos, sin)

		return rotated_queries, rotated_keys
