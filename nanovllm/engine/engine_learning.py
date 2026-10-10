from dataclasses import dataclass
from typing import Protocol

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence_learning import Sequence
from nanovllm.engine.scheduler_learning import Scheduler
from nanovllm.engine.input_builder_learning import (InputBuilder, BatchInputs)

class Runner(Protocol):
	def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]: ...

class MockRunner:
	def __init__(self):
		self.input_builder = InputBuilder(Sequence.block_size)
		self.lash_batch: BatchInputs | None = None

	def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
		self.lash_batch = None

		batch = self.input_builder.build(seqs, is_prefill)
		self.last_batch = batch

		results = []

		for seq, sample_index in zip(seqs, batch.sample_indices):
			if sample_index is None:
				results.append(-1)
			else:
				results.append(10000 + seq.num_completion_tokens)
		return results

@dataclass(frozen=True)
class StepOutput:
	seq_id: int
	is_prefill: bool
	scheduled_tokens: int
	token_id: int | None
	finished: bool

class Engine:
	def __init__(self, config: Config, runner: Runner, policy: str = "alternating"):
		self.scheduler = Scheduler(config, policy = policy)
		self.runner = runner

		self._completed: dict[int, tuple[int, ...]] = {}

	def add_request(self, token_ids: list[int], sampling_params: SamplingParams | None = None) -> int:
		seq = Sequence(token_ids, sampling_params)
		self.scheduler.add(seq)
		return seq.seq_id

	def is_finished(self) -> bool:
		return self.scheduler.is_finished()

	def step(self) -> list[StepOutput]:
		if self.is_finished():
			return []

		seqs, is_prefill = self.scheduler.schedule()

		scheduled_counts = [seq.num_scheduled_tokens for seq in seqs]
		completion_counts = [seq.num_completion_tokens for seq in seqs]

		token_ids = self.runner.run(seqs, is_prefill)

		if len(token_ids) != len(seqs):
			raise RuntimeError("Runner must return one token entry per sequence")

		self.scheduler.postprocess(seqs, token_ids, is_prefill)

		outputs = []

		for seq, scheduled_count, old_count in zip(seqs, scheduled_counts, completion_counts):
			new_count = seq.num_completion_tokens

			if new_count not in (old_count, old_count + 1):
				raise RuntimeError("Unexpected output token count change")

			new_token = (seq.last_token if new_count == old_count + 1 else None)

			if seq.is_finished:
				self._completed[seq.seq_id] = tuple(seq.completion_token_ids)

			outputs.append(
				StepOutput(
					seq_id = seq.seq_id,
					is_prefill = is_prefill,
					scheduled_tokens = scheduled_count,
					token_id = new_token,
					finished = seq.is_finished,
				)
			)

		return outputs

	def take_result(self, seq_id: int) -> tuple[int, ...]:
		if seq_id not in self._completed:
			raise KeyError(f"No unclaimed completed result for sequence {seq_id}")

		return self._completed.pop(seq_id)
