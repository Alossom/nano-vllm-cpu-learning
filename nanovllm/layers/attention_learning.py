import math
import torch

from nanovllm.engine.input_builder_learning import BatchInputs
from nanovllm.engine.kv_cache_learning import KVCache

class ReferenceAttention:
	def __init__(self, cache: KVCache, num_query_heads: int | None = None):
		self.cache = cache
		self.num_kv_heads = cache.num_kv_heads

		if num_query_heads is None:
			num_query_heads = self.num_kv_heads

		if type(num_query_heads) is not int or num_query_heads <= 0:
			raise ValueError("num_query_heads must be a positive integer")

		if num_query_heads % self.num_kv_heads != 0:
			raise ValueError("Query heads must be divisible by KV heads")

		self.num_query_heads = num_query_heads
		self.group_size = num_query_heads // self.num_kv_heads
		self.scale = 1.0 / math.sqrt(cache.head_dim)

	@torch.no_grad()
	def forward(self, batch: BatchInputs, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
		token_count = len(batch.input_ids)
		head_dim = self.cache.head_dim

		q_shape = (token_count, self.num_query_heads, head_dim)
		kv_shape = (token_count, self.num_kv_heads, head_dim)

		for name, tensor, expected_shape in (("queries", queries, q_shape), ("keys", keys, kv_shape), ("values", values, kv_shape)):
			if tuple(tensor.shape) != expected_shape:
				raise ValueError(f"{name} must have shape {expected_shape}")

			if tensor.device.type != "cpu":
				raise ValueError(f"{name} must be on CPU")

			if tensor.dtype != torch.float32:
				raise ValueError(f"{name} must use float32")

		self.cache.write(batch.slot_mapping, keys, values)

		output = torch.empty_like(queries)

		for index in range(len(batch.seq_ids)):
			begin = batch.query_start_loc[index]
			end = batch.query_start_loc[index + 1]
			query_len = end - begin

			q = queries[begin:end]
			k, v = self.cache.read(batch.block_tables[index], batch.context_lens[index])

			if not(torch.isfinite(q).all().item() and torch.isfinite(k).all().item() and torch.isfinite(v).all().item()):
				raise RuntimeError("Non-finite query or cache K/V")

			q_groups = q.reshape(query_len, self.num_kv_heads, self.group_size, head_dim).permute(1, 2, 0, 3)
			k_groups = k.transpose(0,1).unsqueeze(1)
			v_groups = v.transpose(0,1).unsqueeze(1)

			scores = (q_groups @ k_groups.transpose(-2,-1)) * self.scale

			query_positions = torch.tensor(batch.positions[begin:end], dtype=torch.long, device="cpu")

			key_positions = torch.arange(batch.context_lens[index], dtype=torch.long, device="cpu")

			allowed = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))

			scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))

			weights = torch.softmax(scores, dim=-1)

			attended = weights @ v_groups

			output[begin:end] = attended.permute(2, 0, 1, 3).reshape(query_len, self.num_query_heads, head_dim)

		return output
