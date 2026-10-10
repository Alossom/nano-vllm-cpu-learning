from dataclasses import dataclass

import torch

from nanovllm.engine.model_runner_learning import GreedyModelRunner
from nanovllm.engine.sequence_learning import Sequence
from nanovllm.layers.sampler_learning import Sampler, SamplingOptions

@dataclass
class RequestSamplingState:
	options: SamplingOptions
	generator: torch.Generator | None

class SamplingModelRunner(GreedyModelRunner):
	def __init__(self, model):
		super().__init__(model)

		self.sampler = Sampler()
		self._sampling_states: dict[int, RequestSamplingState] = {}

	@staticmethod
	def prepare_state(options: SamplingOptions, seed: int | None) -> RequestSamplingState:
		if not isinstance(options, SamplingOptions):
			raise ValueError("options must be SamplingOptions")

		if options.mode == "greedy":
			if seed is not None:
				raise ValueError("Greedy requests do not use a seed")

			return RequestSamplingState(options=options, generator=None)

		if(type(seed) is not int or not 0 <= seed < 2**63):
			raise ValueError("Random requests require an integer seed in [0, 2**63)")

		generator = torch.Generator(device="cpu")
		generator.manual_seed(seed)

		return RequestSamplingState(options=options, generator=generator)

	def register(self, seq_id: int, state: RequestSamplingState):
		if seq_id in self._sampling_states:
			raise ValueError("Sampling state already registered")

		self._sampling_states[seq_id] = state

	def release(self, seq_id: int):
		if seq_id not in self._sampling_states:
			raise RuntimeError("Sampling state not found")

		del self._sampling_states[seq_id]

	@property
	def num_sampling_states(self) -> int:
		return len(self._sampling_states)

	def rng_state(self, seq_id: int) -> torch.Tensor | None:
		state = self._sampling_states[seq_id]

		if state.generator is None:
			return None

		return state.generator.get_state().clone()

	@torch.no_grad()
	def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
		self.last_batch = None
		self.last_sample_rows = ()
		self.last_logits = None

		for seq in seqs:
			if seq.seq_id not in self._sampling_states:
				raise RuntimeError(f"Missing sampling state for seq_id={seq.seq_id}")

		batch = self.input_builder.build(seqs, is_prefill)
		self.last_batch = batch

		hidden_states = self.model.forward_hidden(batch)

		sample_pairs = [(request_index, row_index) for request_index, row_index in enumerate(batch.sample_indices) if row_index is not None]

		results = [-1] * len(seqs)

		if not sample_pairs:
			return results

		sample_rows = tuple(row_index for _, row_index in sample_pairs)
		self.last_sample_rows = sample_rows

		row_indices = torch.tensor(sample_rows, dtype=torch.long, device="cpu")

		selected_hidden = hidden_states.index_select(0, row_indices)
		logits = self.model.compute_logits(selected_hidden)

		if tuple(logits.shape) != (len(sample_pairs), self.model.vocab_size):
			raise RuntimeError("Unexpected logits shape")

		if not torch.isfinite(logits).all().item():
			raise RuntimeError("Model produced non-finite logits")

		self.last_logits = logits

		for logit_row, (request_index, _) in enumerate(sample_pairs):
			seq_id = seqs[request_index].seq_id
			state = self._sampling_states[seq_id]

			results[request_index] = self.sampler.sample(logits[logit_row], state.options, state.generator)

		return results
