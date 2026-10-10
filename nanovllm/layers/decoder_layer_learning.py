import torch
from torch import nn
from nanovllm.engine.input_builder_learning import BatchInputs
from nanovllm.engine.kv_cache_learning import KVCache
from nanovllm.layers.model_attention_learning import ModelAttention
from nanovllm.layers.mlp_learning import SwiGLUMLP
from nanovllm.layers.rmsnorm_learning import RMSNorm

class DecoderLayer(nn.Module):
	def __init__(self, hidden_size: int, intermediate_size: int, num_query_heads: int, num_kv_heads: int, head_dim: int, cache: KVCache, rms_norm_eps: float = 1e-6, rope_theta: float = 10000.0):
		super().__init__()

		self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
		self.self_attn = ModelAttention(hidden_size=hidden_size, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, cache=cache, rms_norm_eps=rms_norm_eps, rope_theta=rope_theta)
		self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
		self.mlp = SwiGLUMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

	@torch.no_grad()
	def forward(self, batch: BatchInputs, hidden_states: torch.Tensor) -> torch.Tensor:
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		hidden_states = self.self_attn(batch, hidden_states)

		hidden_states = residual + hidden_states

		residual = hidden_states

		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)

		hidden_states = residual + hidden_states

		return hidden_states
