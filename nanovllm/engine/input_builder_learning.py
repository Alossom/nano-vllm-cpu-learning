from dataclasses import dataclass

from nanovllm.engine.sequence_learning import Sequence

@dataclass(frozen=True)
class BatchInputs:
	is_prefill: bool
	seq_ids: tuple[int, ...]
	input_ids: tuple[int, ...]
	positions: tuple[int, ...]
	slot_mapping: tuple[int, ...]
	query_start_loc: tuple[int, ...]
	cached_lens: tuple[int, ...]
	context_lens: tuple[int, ...]
	block_tables: tuple[tuple[int, ...], ...]
	sample_indices: tuple[int | None, ...]

class InputBuilder:
	def __init__(self, block_size: int):
		if block_size <= 0:
			raise ValueError("block_size must be positive")

		if block_size != Sequence.block_size:
			raise ValueError("InputBuilder and Sequence block sizes differ")

		self.block_size = block_size

	def build(self, seqs: list[Sequence], is_prefill: bool) -> BatchInputs:
		if not seqs:
			raise ValueError("Cannot build an empty batch")

		if len({seq.seq_id for seq in seqs}) != len(seqs):
			raise ValueError("A sequence appears more than once")

		seq_ids = []
		input_ids = []
		positions = []
		slot_mapping = []
		query_start_loc = [0]
		cached_lens = []
		context_lens = []
		block_tables = []
		sample_indices = []

		for seq in seqs:
			if seq.is_finished:
				raise ValueError("Cannot execute a finished sequence")

			if seq.is_prefill != is_prefill:
				raise ValueError("Sequence phase does not match bath phase")

			start = seq.num_cached_tokens
			count = seq.num_scheduled_tokens
			end = start + count

			if count <= 0 or not 0 <= start < end <= seq.num_tokens:
				raise ValueError("Invalid scheduled token range")

			if not is_prefill and (count != 1 or end != seq.num_tokens):
				raise ValueError("Decode must process exactly the last uncached token")

			required_blocks = (end + self.block_size - 1) // self.block_size

			if len(seq.block_table) < required_blocks:
				raise ValueError("Insufficient block mappings")

			table = tuple(seq.block_table[:required_blocks])

			if any(block_id < 0 for block_id in table):
				raise ValueError("Invalid physical block id")

			seq_ids.append(seq.seq_id)
			input_ids.extend(seq.token_ids[start:end])
			positions.extend(range(start, end))

			for position in range(start, end):
				logical_block, offset = divmod(position, self.block_size)
				physical_block = table[logical_block]
				slot_mapping.append(physical_block * self.block_size + offset)

			query_start_loc.append(len(input_ids))
			cached_lens.append(start)
			context_lens.append(end)
			block_tables.append(table)

			sample_indices.append(len(input_ids) - 1 if end == seq.num_tokens else None)

		return BatchInputs(
			is_prefill = is_prefill,
			seq_ids = tuple(seq_ids),
			input_ids = tuple(input_ids),
			positions = tuple(positions),
			slot_mapping = tuple(slot_mapping),
			query_start_loc = tuple(query_start_loc),
			cached_lens = tuple(cached_lens),
			context_lens = tuple(context_lens),
			block_tables = tuple(block_tables),
			sample_indices = tuple(sample_indices),
		)
