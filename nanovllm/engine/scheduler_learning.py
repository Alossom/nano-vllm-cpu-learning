from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence_learning import Sequence, SequenceStatus
from nanovllm.engine.block_manager_learning import BlockManager

class Scheduler:
	def __init__(self, config: Config, policy: str = "alternating"):
		if policy not in {"prefill_first", "alternating"}:
			raise ValueError(f"Unkown scheduling policy: {policy}")

		if config.max_num_seqs <= 0:
			raise ValueError("max_num_seqs must be positive")

		if config.max_num_batched_tokens <= 0:
			raise ValueError("max_num_batched_tokens must be positive")

		if config.num_kvcache_blocks <= 0:
			raise ValueError("num_kvcache_blocks must be positive")

		if config.kvcache_block_size != Sequence.block_size:
			raise ValueError("Scheduler and Sequence must use the same block size")

		self.max_num_seqs = config.max_num_seqs
		self.max_num_batched_tokens = config.max_num_batched_tokens
		self.block_size = config.kvcache_block_size
		self.eos = config.eos

		self.block_manager = BlockManager(
			config.num_kvcache_blocks,
			self.block_size,
		)

		self.waiting: deque[Sequence] = deque()
		self.running: deque[Sequence] = deque()

		self.policy = policy
		self._prefer_decode = False

	def is_finished(self):
		return not self.waiting and not self.running

	def add(self, seq: Sequence):
		if seq.status != SequenceStatus.WAITING:
			raise ValueError("Only WAITING sequences can be added")

		if seq.num_tokens == 0:
			raise ValueError("An input sequence must contain at least one token")

		if seq.max_tokens <= 0:
			raise ValueError("This scheduler requires max_tokens > 0")

		if any(s.seq_id == seq.seq_id for s in self.waiting):
			raise ValueError("Sequence is already running")

		self.waiting.append(seq)

	def schedule(self) -> tuple[list[Sequence], bool]:
		if self.is_finished():
			return [], False

		decode_first = (self.policy == "alternating" and self._prefer_decode)

		if decode_first:
			attempts = (
				(self._schedule_decode, False),
				(self._schedule_prefill, True),
			)
		else:
			attempts = (
				(self._schedule_prefill, True),
				(self._schedule_decode, False),
			)

		for schedule_phase, is_prefill in attempts:
			scheduled = schedule_phase()
			if scheduled:
				self._prefer_decode = is_prefill
				return scheduled, is_prefill

		if self.waiting:
			scheduled = self._schedule_prefill()
			if scheduled:
				self._prefer_decode = True
				return scheduled, True

		raise RuntimeError(
			"Pending requests exist, but none can be scheduled."
			"Check KV block capacity and request lengths."
		)

	def _schedule_prefill(self) -> list[Sequence]:
		scheduled = []
		remaining = self.max_num_batched_tokens

		while (
			self.waiting
			and len(scheduled) < self.max_num_seqs
			and remaining > 0
		):
			seq = self.waiting[0]

			if not seq.block_table:
				cached_blocks = self.block_manager.can_allocate(seq)
				if cached_blocks == -1:
					break

				uncached_tokens = (seq.num_tokens - cached_blocks * self.block_size)
			else:
				uncached_tokens = seq.num_tokens - seq.num_cached_tokens

			if uncached_tokens <= 0:
				raise RuntimeError("A waiting sequence has no tokens left to process")

			if scheduled and uncached_tokens > remaining:
				break

			if not seq.block_table:
				self.block_manager.allocate(seq, cached_blocks)

			count = min(uncached_tokens, remaining)
			seq.num_scheduled_tokens = count
			seq.is_prefill = True

			scheduled.append(seq)
			remaining -= count

			if seq.num_cached_tokens + count == seq.num_tokens:
				self.waiting.popleft()
				seq.status = SequenceStatus.RUNNING
				self.running.append(seq)

		return scheduled

	def _schedule_decode(self) -> list[Sequence]:
		scheduled = []

		limit = min(self.max_num_seqs, self.max_num_batched_tokens,)

		while self.running and len(scheduled) < limit:
			seq = self.running.popleft()

			while not self.block_manager.can_append(seq):
				if self.running:
					self.preempt(self.running.pop())
				else:
					self.preempt(seq)
					break
			else:
				self.block_manager.may_append(seq)
				seq.num_scheduled_tokens = 1
				seq.is_prefill = False
				scheduled.append(seq)

		self.running.extend(scheduled)
		return scheduled

	def preempt(self, seq: Sequence):
		seq.status = SequenceStatus.WAITING
		seq.is_prefill = True
		seq.num_scheduled_tokens = 0

		self.block_manager.deallocate(seq)
		self.waiting.appendleft(seq)

	def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
		if len(seqs) != len(token_ids):
			raise ValueError("Each scheduled sequence must have one returned token entry")

		for seq, token_id in zip(seqs, token_ids):
			self.block_manager.hash_blocks(seq)

			seq.num_cached_tokens += seq.num_scheduled_tokens
			seq.num_scheduled_tokens = 0

			if is_prefill and seq.num_cached_tokens < seq.num_tokens:
				continue

			seq.append_token(token_id)

			reached_eos = (not seq.ignore_eos and token_id == self.eos)
			reached_limit = (seq.num_completion_tokens >= seq.max_tokens)

			if reached_eos or reached_limit:
				seq.status = SequenceStatus.FINISHED
				self.block_manager.deallocate(seq)
				self.running.remove(seq)
