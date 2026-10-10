from dataclasses import dataclass

def require_positive_int(name: str, value: int) -> None:
	if type(value) is not int or value <= 0:
		raise ValueError(f"{name} must be a positive integer")

@dataclass(frozen=True)
class CpuConfig:
	max_num_seqs: int = 4
	max_num_batched_tokens: int = 128

	max_model_len: int = 1024

	eos: int = -1
	kvcache_block_size: int = 256
	num_kvcache_blocks: int = 8

	def __post_init__(self) -> None:
		for name in("max_num_seqs", "max_num_batched_tokens", "max_model_len", "kvcache_block_size", "num_kvcache_blocks"):
			require_positive_int(name, getattr(self, name))

		if type(self.eos) is not int or self.eos < -1:
			raise ValueError("eos must be -1 or a nonnegative integer")

		if self.kvcache_block_size != 256:
			raise ValueError("This learning implementation uses block_size=256")

		if self.max_num_seqs > self.max_num_batched_tokens:
			raise ValueError("max_num_seqs must not exceed " "max_num_batched_tokens")

		if self.max_model_len > self.kv_token_capacity:
			raise ValueError("max_model_len exceeds total KV token capacity")

	@property
	def kv_token_capacity(self) -> int:
		return self.num_kvcache_blocks * self. kvcache_block_size

	def validate_model_limits(self, *, max_position_embeddings: int) -> None:
		require_positive_int("max_position_embeddings", max_position_embeddings)

		if self.max_model_len > max_position_embeddings:
			raise ValueError("max_model_len exceeds model position limit")

	def validate_request(self, *, prompt_tokens: int, max_tokens: int) -> None:
		require_positive_int("prompt_tokens", prompt_tokens)
		require_positive_int("max_tokens", max_tokens)

		requested_total = prompt_tokens + max_tokens

		if requested_total > self.max_model_len:
			raise ValueError("Request exceeds max_model_len: " f"prompt={prompt_tokens}, " f"max_tokens={max_tokens}, " f"limit+{self.max_model_len}")

	def estimate_kv_cache_bytes(self, *, num_layers: int, num_kv_heads: int, head_dim: int, element_size: int = 4) -> int:
		for name, value in(("num_layers", num_layers), ("num_kv_heads", num_kv_heads), ("head_dim", head_dim), ("element_size", element_size)):
			require_positive_int(name, value)

		return(2 * num_layers * self.kv_token_capacity * num_kv_heads * head_dim * element_size)
