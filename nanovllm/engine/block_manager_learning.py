import hashlib
import struct
from nanovllm.engine.sequence_learning import Sequence
from nanovllm.engine.block_pool_learning import BlockPool

class BlockManager:
	def __init__(self, num_blocks: int, block_size: int):
		if block_size <= 0:
			raise ValueError("block_size must be positive")

		if block_size != Sequence.block_size:
			raise ValueError("BlockManager and Sequence must use the same block size")

		self.block_size = block_size
		self.pool = BlockPool(num_blocks)
		self.hash_to_block_id: dict[bytes, int] = {}

		self._block_keys: dict[int, bytes] = {}
		self._block_tokens: dict[int, tuple[int, ...]] = {}

	@property
	def blocks(self):
		return self.pool.blocks

	@property
	def free_block_ids(self):
		return tuple(block.block_id for block in self.pool.blocks if block.ref_count == 0)

	@property
	def used_block_ids(self):
		return{block.block_id for block in self.pool.blocks if block.ref_count > 0}

	def _missing_blocks(self, seq: Sequence) -> int:
		required = seq.num_blocks
		allocated = len(seq.block_table)

		if allocated > required:
			raise RuntimeError("Sequence has more allocated blocks than expected")

		return required - allocated

	def can_allocate(self, seq: Sequence) -> int:
		if seq.block_table:
			raise ValueError("Sequence already has block mappings")

		matched = self._find_cached_blocks(seq)

		used_hits = sum(self.pool.blocks[block_id].ref_count > 0 for block_id in matched)

		required_free = seq.num_blocks - used_hits

		if self.pool.num_free_blocks < required_free:
			return -1

		return len(matched)

	def allocate(self, seq: Sequence, num_cached_blocks: int):
		if seq.block_table:
			raise ValueError("Sequence already has block mappings")

		matched = self._find_cached_blocks(seq)

		if not 0 <= num_cached_blocks <= len(matched):
			raise RuntimeError("Requested cached prefix is unavailable")

		reused = matched[:num_cached_blocks]

		used_hits = sum(self.pool.blocks[block_id].ref_count > 0 for block_id in reused)

		required_free = seq.num_blocks - used_hits

		if self.pool.num_free_blocks < required_free:
			raise RuntimeError("Not enough free blocks for this sequence")

		for block_id in reused:
			block = self.pool.blocks[block_id]

			if block.ref_count > 0:
				self.pool.retain(block_id)
			else:
				self.pool.reactivate(block_id)

			seq.block_table.append(block_id)

		new_blocks = seq.num_blocks - num_cached_blocks

		for _ in range(new_blocks):
			seq.block_table.append(self._acquire_block())

		seq.num_cached_tokens = num_cached_blocks * self.block_size

	def can_append(self, seq: Sequence) -> bool:
		missing = self._missing_blocks(seq)
		return self.pool.num_free_blocks >= missing

	def may_append(self, seq: Sequence):
		missing = self._missing_blocks(seq)

		if self.pool.num_free_blocks < missing:
			raise RuntimeError("Not enough free blocks to extend sequence")

		for _ in range(missing):
			seq.block_table.append(self._acquire_block())

	def deallocate(self, seq: Sequence):
		for block_id in reversed(seq.block_table):
			self.pool.release(block_id)

		seq.block_table.clear()
		seq.num_cached_tokens = 0

	def hash_blocks(self, seq: Sequence):
		completed = (seq.num_cached_tokens + seq.num_scheduled_tokens)

		if (seq.num_cached_tokens < 0 or seq.num_scheduled_tokens < 0 or completed > seq.num_tokens):
			raise RuntimeError("Invalid cache progress")

		start = seq.num_cached_tokens // self.block_size
		end = completed // self.block_size

		if start == end:
			return

		if end > len(seq.block_table):
			raise RuntimeError("Completed tokens exceed allocated capacity")

		prefix = None

		if start > 0:
			previous_id = seq.block_table[start - 1]
			prefix = self._block_keys.get(previous_id)

			if prefix is None:
				raise RuntimeError("Previous complete block has no cache key")

		for index in range(start, end):
			block_id = seq.block_table[index]
			block = self.pool.blocks[block_id]

			if block.ref_count <= 0:
				raise RuntimeError("Cannot publish cache for a free block")

			tokens = seq.block(index)

			if len(tokens) != self.block_size:
				raise RuntimeError("Only complete blocks can be published")

			key = self.compute_hash(tokens, prefix)
			token_tuple = tuple(tokens)

			old_key = self._block_keys.get(block_id)
			old_tokens = self._block_tokens.get(block_id)

			if block.ref_count > 1 and (old_key != key or old_tokens != token_tuple):
				raise RuntimeError("Cannot change shared cache content")

			self._forget_block(block_id)

			self._block_keys[block_id] = key
			self._block_tokens[block_id] = token_tuple
			self.hash_to_block_id[key] = block_id

			prefix = key

	@staticmethod
	def compute_hash(token_ids: list[int], prefix: bytes | None = None,) -> bytes:
		h = hashlib.sha256()

		if prefix is None:
			h.update(b"\x00")
		else:
			h.update(b"\x01")
			h.update(prefix)

		h.update(struct.pack("<Q", len(token_ids)))

		for token_id in token_ids:
			h.update(struct.pack("<q", token_id))

		return h.digest()

	def _forget_block(self, block_id: int):
		old_key = self._block_keys.pop(block_id, None)
		self._block_tokens.pop(block_id, None)

		if (old_key is not None and self.hash_to_block_id.get(old_key) == block_id):
			del self.hash_to_block_id[old_key]

	def _acquire_block(self) -> int:
		block_id = self.pool.acquire()

		self._forget_block(block_id)
		return block_id

	def _find_cached_blocks(self, seq: Sequence) -> list[int]:
		matched = []
		prefix = None

		for index in range(seq.num_blocks - 1):
			tokens = seq.block(index)
			key = self.compute_hash(tokens, prefix)

			block_id = self.hash_to_block_id.get(key)
			if block_id is None:
				break

			if self._block_keys.get(block_id) != key:
				raise RuntimeError("Stale prefix cache index")

			if self._block_tokens.get(block_id) != tuple(tokens):
				break

			matched.append(block_id)
			prefix = key

		return matched
