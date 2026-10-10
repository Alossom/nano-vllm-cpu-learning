import torch
from torch import nn
from nanovllm.engine.input_builder_learning import BatchInputs
from nanovllm.engine.kv_cache_learning import KVCache
from nanovllm.layers.decoder_layer_learning import DecoderLayer
from nanovllm.layers.rmsnorm_learning import RMSNorm

class TinyCausalLM(nn.Module):
	def __init__(self, vocab_size: int, hidden_size: int, intermediate_size: int, num_layers: int, num_query_heads: int, num_kv_heads: int, head_dim: int, num_blocks: int, block_size: int =256, rms_norm_eps: float = 1e-6, rope_theta: float = 10000.0, tie_word_embeddings: bool = False):
		super().__init__()

		dimensions = {"vocab_size": vocab_size, "hidden_size": hidden_size, "intermediate_size": intermediate_size, "num_layers": num_layers, "num_query_heads": num_query_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "num_blocks": num_blocks, "block_size": block_size}

		for name, value in dimensions.items():
			if type(value) is not int or value <= 0:
				raise ValueError(f"{name} must be a positive integer")

		if num_query_heads % num_kv_heads != 0:
			raise ValueError("Query heads must be divisible by KV heads")

		if head_dim % 2 != 0:
			raise ValueError("head_dim must be even")

		if type(tie_word_embeddings) is not bool:
			raise ValueError("tie_word_embeddings must be a boolean")

		self.tie_word_embeddings = tie_word_embeddings
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_blocks = num_blocks
		self.block_size = block_size

		self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device="cpu", dtype=torch.float32)

		self.caches = [KVCache(num_blocks=num_blocks, block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_dim) for _ in range(num_layers)]

		self.layers = nn.ModuleList([DecoderLayer(hidden_size=hidden_size, intermediate_size=intermediate_size, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, cache=cache, rms_norm_eps=rms_norm_eps, rope_theta=rope_theta) for cache in self.caches])

		self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

		self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device="cpu", dtype=torch.float32)

		if tie_word_embeddings:
			self.lm_head.weight = self.embed_tokens.weight

	@torch.no_grad()
	def forward_hidden(self, batch: BatchInputs) -> torch.Tensor:
		if len(batch.input_ids) == 0:
			raise ValueError("Batch must contain at least one token")

		for token_id in batch.input_ids:
			if type(token_id) is not int:
				raise ValueError("Token ids must be Python integers")

			if not 0 <= token_id < self.vocab_size:
				raise ValueError(f"Token id out of vocabulary range: {token_id}")

		input_ids = torch.tensor(batch.input_ids, dtype=torch.long, device="cpu")

		hidden_states = self.embed_tokens(input_ids)

		for layer in self.layers:
			hidden_states = layer(batch, hidden_states)

		return self.norm(hidden_states)

	@torch.no_grad()
	def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
		if(hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size):
			raise ValueError("hidden_states must have shape [tokens, hidden_size]")

		if(hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float32):
			raise ValueError("This reference implementation requires CPU float32 input")

		return self.lm_head(hidden_states)

	@torch.no_grad()
	def forward(self, batch: BatchInputs) -> torch.Tensor:
		hidden_states = self.forward_hidden(batch)
		return self.compute_logits(hidden_states)
