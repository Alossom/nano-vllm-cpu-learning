import json
import math
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Qwen3ModelConfig:
	vocab_size: int
	hidden_size: int
	intermediate_size: int
	num_layers: int
	num_query_heads: int
	num_kv_heads: int
	head_dim: int

	rms_norm_eps: float
	rope_theta: float
	max_position_embeddings: int

	tie_word_embeddings: bool
	eos_token_ids: tuple[int, ...]

	@classmethod
	def from_dict(cls, data: dict):
		if not isinstance(data, dict):
			raise ValueError("Model config must be a dictionary")

		if data.get("model_type") != "qwen3":
			raise ValueError("Only dense Qwen3 configurations are supported")

		if data.get("hidden_act") != "silu":
			raise ValueError("Only SiLU/SwiGLU is supported")

		attention_bias = data.get("attention_bias", False)
		if attention_bias is not False:
			raise ValueError("Attention bias is not supported")

		if data.get("quantization_config") is not None:
			raise ValueError("Quantized checkpointa are not supported")

		if data.get("rope_scaling") is not None:
			raise ValueError("RoPE scaling is not supported")

		if data.get("rope_parameters") is not None:
			raise ValueError("Nested rope_parameters are not supported yet")

		use_sliding_window = data.get("use_sliding_window", False)

		if type(use_sliding_window) is not bool:
			raise ValueError("use_sliding_window must be a boolean")

		if use_sliding_window is not False:
			raise ValueError("Sliding-window attention is not supported")

		layer_types = data.get("layer_types")
		if layer_types is not None:
			if(not isinstance(layer_types, list) or any(item != "full_attention" for item in layer_types)):
				raise ValueError("Only full_attention layers are supported")

		def positive_int(name):
			value = data.get(name)

			if type(value) is not int or value <= 0:
				raise ValueError(f"{name} must be a positive integer")

			return value

		def positive_float(name):
			value = data.get(name)

			if(type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
				raise ValueError(f"{name} must be finite and positive")

			return float(value)

		vocab_size = positive_int("vocab_size")
		hidden_size = positive_int("hidden_size")
		intermediate_size = positive_int("intermediate_size")
		num_layers = positive_int("num_hidden_layers")
		num_query_heads = positive_int("num_attention_heads")
		num_kv_heads = positive_int("num_key_value_heads")

		if num_query_heads % num_kv_heads != 0:
			raise ValueError("Query heads must be divisible by KV heads")

		if data.get("head_dim") is None:
			if hidden_size % num_query_heads != 0:
				raise ValueError("Cannot infer head_dim from hidden_size and heads")

			head_dim = hidden_size // num_query_heads
		else:
			head_dim = positive_int("head_dim")

		if head_dim % 2 != 0:
			raise ValueError("head_dim must be even")

		rms_norm_eps = positive_float("rms_norm_eps")
		rope_theta = positive_float("rope_theta")

		if rope_theta <= 1:
			raise ValueError("rope_theta must be greater than 1")

		max_position_embeddings = positive_int("max_position_embeddings")
		tie_word_embeddings = data.get("tie_word_embeddings", False)
		if type(tie_word_embeddings) is not bool:
			raise ValueError("tie_word_embeddings must be a boolean")

		raw_eos = data.get("eos_token_id")

		if raw_eos is None:
			eos_token_ids = ()
		elif type(raw_eos) is int:
			eos_token_ids = (raw_eos,)
		elif isinstance(raw_eos, list):
			eos_token_ids = tuple(raw_eos)
		else:
			raise ValueError("eos_token_id must be an integer, list, or null")

		if any(type(token_id) is not int or not 0 <= token_id < vocab_size for token_id in eos_token_ids):
			raise ValueError("Invalid EOS token id")

		eos_token_ids = tuple(dict.fromkeys(eos_token_ids))

		if layer_types is not None and len(layer_types) != num_layers:
			raise ValueError("layer_types length does not match num_hidden_layers")

		return cls(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_layers=num_layers, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, rms_norm_eps=rms_norm_eps, rope_theta=rope_theta, max_position_embeddings=max_position_embeddings, tie_word_embeddings=tie_word_embeddings, eos_token_ids=eos_token_ids)

	@classmethod
	def from_json(cls, path: str | Path):
		with Path(path).open("r", encoding="utf-8") as file:
			data = json.load(file)

		return cls.from_dict(data)

	def model_kwargs(self, num_blocks: int, block_size: int = 256) -> dict:
		for name, value in(("num_blocks", num_blocks), ("block_size", block_size)):
			if type(value) is not int or value <= 0:
				raise ValueError(f"{name} must be a positive integer")

		return {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size, "num_layers": self.num_layers, "num_query_heads": self.num_query_heads, "num_kv_heads": self.num_kv_heads, "head_dim": self.head_dim, "num_blocks": num_blocks, "block_size": block_size, "rms_norm_eps": self.rms_norm_eps, "rope_theta": self.rope_theta, "tie_word_embeddings": self.tie_word_embeddings}

	def kv_cache_bytes(self, num_blocks: int, block_size: int = 256, bytes_per_element: int = 4) -> int:
		for name, value in (("num_blocks", num_blocks), ("block_size", block_size), ("bytes_per_element", bytes_per_element)):
			if type(value) is not int or value <= 0:
				raise ValueError(f"{name} must be a positive integer")

		return(2 * self.num_layers * num_blocks * block_size * self.num_kv_heads * self.head_dim * bytes_per_element)
