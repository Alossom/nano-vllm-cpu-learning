import torch
from torch import nn
from nanovllm.engine.input_builder_learning import BatchInputs
from nanovllm.engine.kv_cache_learning import KVCache
from nanovllm.layers.attention_learning import ReferenceAttention
from nanovllm.layers.rmsnorm_learning import RMSNorm
from nanovllm.layers.rotary_learning import ReferenceRoPE

class ModelAttention(nn.Module):
	def __init__(self, hidden_size: int, num_query_heads: int, num_kv_heads: int, head_dim: int, cache: KVCache, rms_norm_eps: float = 1e-6, rope_theta: float = 10000.0):
		super().__init__()

		dimensions = {"hidden_size": hidden_size, "num_query_heads": num_query_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}

		for name, value in dimensions.items():
			if type(value) is not int or value <= 0:
				raise ValueError(f"{name} must be a positive integer")

		if num_query_heads % num_kv_heads != 0:
			raise ValueError("Query heads must be divisible by KV heads")

		if head_dim % 2 != 0:
			raise ValueError("head_dim must be even")

		if (cache.num_kv_heads != num_kv_heads or cache.head_dim != head_dim):
			raise ValueError("KVCache dimensions do not match Attention")

		self.hidden_size = hidden_size
		self.num_query_heads = num_query_heads
		self.num_kv_heads = num_kv_heads
		self.head_dim = head_dim

		query_size = num_query_heads * head_dim
		kv_size = num_kv_heads * head_dim

		self.q_proj = nn.Linear(hidden_size, query_size, bias=False, device="cpu", dtype=torch.float32)
		self.k_proj = nn.Linear(hidden_size, kv_size, bias=False, device="cpu", dtype=torch.float32)
		self.v_proj = nn.Linear(hidden_size, kv_size, bias=False, device="cpu", dtype=torch.float32)
		self.o_proj = nn.Linear(query_size, hidden_size, bias=False, device="cpu", dtype=torch.float32)

		self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
		self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

		self.rope = ReferenceRoPE(head_dim, theta=rope_theta)
		self.attention = ReferenceAttention(cache, num_query_heads=num_query_heads)

	@torch.no_grad()
	def forward(self, batch: BatchInputs, hidden_states: torch.Tensor) -> torch.Tensor:
		if hidden_states.ndim != 2:
			raise ValueError("hidden_states must have shape [tokens, hidden_size]")

		if hidden_states.shape[1] != self.hidden_size:
			raise ValueError("Unexpected hidden size")

		num_tokens = hidden_states.shape[0]

		if num_tokens == 0 or num_tokens != len(batch.input_ids):
			raise ValueError("Hidden-state token count does not match batch")

		if(hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float32):
			raise ValueError("This reference implementation requires CPU float32 input")

		queries = self.q_proj(hidden_states).reshape(num_tokens, self.num_query_heads, self.head_dim)
		keys = self.k_proj(hidden_states).reshape(num_tokens, self.num_kv_heads, self.head_dim)
		values = self.v_proj(hidden_states).reshape(num_tokens, self.num_kv_heads, self.head_dim)

		queries = self.q_norm(queries)
		keys = self.k_norm(keys)

		queries, keys = self.rope.forward(queries, keys, batch.positions)

		attended = self.attention.forward(batch, queries, keys, values)

		merged = attended.reshape(num_tokens, self.num_query_heads * self.head_dim)

		return self.o_proj(merged)
