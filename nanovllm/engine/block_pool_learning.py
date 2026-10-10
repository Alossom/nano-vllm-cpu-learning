from collections import deque
from dataclasses import dataclass

@dataclass
class Block:
	block_id: int
	ref_count: int = 0

class BlockPool:
	def __init__(self, num_blocks: int):
		if num_blocks <= 0:
			raise ValueError("num_blocks must be positive")

		self.blocks = [Block(block_id = i) for i in range(num_blocks)]
		self._free_block_ids = deque(range(num_blocks))
		self._used_block_ids: set[int] = set()

	@property
	def num_free_blocks(self) -> int:
		return len(self._free_block_ids)

	@property
	def num_used_blocks(self) -> int:
		return len(self._used_block_ids)

	def _get_block(self, block_id: int) -> Block:
		if not 0 <= block_id < len(self.blocks):
			raise IndexError(f"Invalid block id: {block_id}")

		return self.blocks[block_id]

	def acquire(self) -> int:
		if not self._free_block_ids:
			raise RuntimeError("No free KV blocks available")

		block_id = self._free_block_ids[0]
		block = self._get_block(block_id)

		if block.ref_count != 0 or block_id in self._used_block_ids:
			raise RuntimeError("Inconsistent free block state")

		self._free_block_ids.popleft()
		block.ref_count = 1
		self._used_block_ids.add(block_id)

		return block_id

	def retain(self, block_id: int):
		block = self._get_block(block_id)

		if block.ref_count <= 0 or block_id not in self._used_block_ids:
			raise RuntimeError("Cannot retain a free block")

		block.ref_count += 1

	def reactivate(self, block_id: int) -> int:
		block = self._get_block(block_id)

		if block.ref_count != 0:
			raise RuntimeError("Cannot reactivate a referenced block")

		if block_id in self._used_block_ids:
			raise RuntimeError("Inconsistent block state")

		try:
			self._free_block_ids.remove(block_id)
		except ValueError as error:
			raise RuntimeError("Block is not present in the free queue") from error

		block.ref_count = 1
		self._used_block_ids.add(block_id)

		return block_id

	def release(self, block_id: int):
		block = self._get_block(block_id)

		if block.ref_count <= 0 or block_id not in self._used_block_ids:
			raise RuntimeError("Cannot release a free block")

		block.ref_count -= 1

		if block.ref_count == 0:
			self._used_block_ids.remove(block_id)
			self._free_block_ids.append(block_id)

	def check_invariants(self):
		free_ids = list(self._free_block_ids)
		free = set(free_ids)
		used = self._used_block_ids
		all_ids = set(range(len(self.blocks)))

		if len(free_ids) != len(free):
			raise RuntimeError("Duplicate ids in free block queue")

		if free & used:
			raise RuntimeError("A block is both free and used")

		if free | used != all_ids:
			raise RuntimeError("Missing or invalid block ids")

		for block in self.blocks:
			if block.block_id in free:
				if block.ref_count != 0:
					raise RuntimeError("Free block has nonzero references")

			else:
				if block.ref_count <= 0:
					raise RuntimeError("Used block has no references")
