import torch
from torch import nn

class SwiGLUMLP(nn.Module):
	def __init__(self, hidden_size: int, intermediate_size: int):
		super().__init__()

		if type(hidden_size) is not int or hidden_size <= 0:
			raise ValueError("hidden_size must be a positive integer")

		if type(intermediate_size) is not int or intermediate_size <= 0:
			raise ValueError("intermediate_size must be a positive integer")

		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size

		self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device="cpu", dtype=torch.float32)
		self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device="cpu", dtype=torch.float32)
		self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device="cpu", dtype=torch.float32)

	@torch.no_grad()
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		if hidden_states.ndim != 2:
			raise ValueError("hidden_states must have shape [tokens, hidden_size]")

		if hidden_states.shape[-1] != self.hidden_size:
			raise ValueError(f"Expected last dimension {self.hidden_size}, " f"got {hidden_states.shape[-1]}")

		if hidden_states.shape[0] == 0:
			raise ValueError("hidden_states must contain at least one token")

		if(hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float32):
			raise ValueError("This reference implementation requires CPU float32 input")

		gate = self.gate_proj(hidden_states)
		up = self.up_proj(hidden_states)

		activated_gate = torch.nn.functional.silu(gate)
		intermediate = activated_gate * up

		return self.down_proj(intermediate)
