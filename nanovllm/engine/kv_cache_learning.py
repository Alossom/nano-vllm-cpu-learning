import torch

class KVCache:
	def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int):
		if min(num_blocks, block_size, num_kv_heads, head_dim) <= 0:
			raise ValueError("All cache dimensions must be positive")

		self.num_blocks = num_blocks
		self.block_size = block_size
		self.num_kv_heads = num_kv_heads
		self.head_dim = head_dim
		self.num_slots = num_blocks * block_size

		shape = (num_blocks, block_size, num_kv_heads, head_dim)

		self.k_cache = torch.full(shape, float("nan"), dtype=torch.float32, device="cpu")
		self.v_cache = torch.full(shape, float("nan"), dtype=torch.float32, device="cpu")

	def _flat(self, cache: torch.Tensor) -> torch.Tensor:
		return cache.view(self.num_slots, self.num_kv_heads, self.head_dim)

	@torch.no_grad()
	def write(self, slot_mapping: list[int] | tuple[int, ...], keys: torch.Tensor, values: torch.Tensor):
		slots = tuple(slot_mapping)

		if not slots:
			raise ValueError("Cannot write an empty token batch")

		if any(type(slot) is not int for slot in slots):
			raise TypeError("Slot ids must be Python integers")

		if any(slot < 0 or slot >= self.num_slots for slot in slots):
			raise ValueError("slot id is outside cache capacity")

		if len(set(slots)) != len(slots):
			raise ValueError("Duplicate write slots in one batch")

		expected_shape = (len(slots), self.num_kv_heads, self.head_dim)

		for name, tensor in (("keys", keys), ("values", values)):
			if tuple(tensor.shape) != expected_shape:
				raise ValueError(f"{name} must have shape {expected_shape}")

			if tensor.device.type != "cpu":
				raise ValueError(f"{name} must be on CPU")

			if tensor.dtype != torch.float32:
				raise ValueError(f"{name} must use float32")

		indices = torch.tensor(slots, dtype=torch.long, device="cpu")

		self._flat(self.k_cache).index_copy_(0, indices, keys)
		self._flat(self.v_cache).index_copy_(0, indices, values)

	def read(self, block_table: list[int] | tuple[int, ...], context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
		if type(context_len) is not int or context_len < 0:
			raise ValueError("context_len must be a positive integer")

		required_blocks = (context_len + self.block_size - 1) // self.block_size

		if len(block_table) < required_blocks:
			raise ValueError("Insufficient block mappings")

		table = tuple(block_table[:required_blocks])

		if any(type(block_id) is not int for block_id in table):
			raise TypeError("Block ids must be Python integers")

		if any(block_id < 0 or block_id >= self.num_blocks for block_id in table):
			raise ValueError("Physical block id is outside cache capacity")

		positions = torch.arange(context_len, dtype=torch.long, device="cpu")
		physical_blocks = torch.tensor(table, dtype=torch.long, device="cpu")

		logical_blocks = positions // self.block_size
		offsets = positions % self.block_size

		block_ids = physical_blocks[logical_blocks]

		slots = physical_blocks[logical_blocks] * self.block_size + offsets

		keys = self._flat(self.k_cache).index_select(0, slots)
		values = self._flat(self.v_cache).index_select(0, slots)

		return keys, values
