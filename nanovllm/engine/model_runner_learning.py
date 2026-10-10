import torch
from nanovllm.engine.input_builder_learning import InputBuilder
from nanovllm.engine.sequence_learning import Sequence
from nanovllm.models.model_learning import TinyCausalLM

class GreedyModelRunner:
	def __init__(self, model: TinyCausalLM):
		if model.block_size != Sequence.block_size:
			raise ValueError("Model and Sequence must use the same block size")

		self.model = model
		self.model.eval()

		self.input_builder = InputBuilder(model.block_size)

		self.last_batch = None
		self.last_sample_rows = ()
		self.last_logits = None

	@torch.no_grad()
	def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
		self.last_batch = None
		self.last_sample_rows = ()
		self.last_logits = None

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

		expected_shape = (len(sample_pairs), self.model.vocab_size)
		if tuple(logits.shape) != expected_shape:
			raise RuntimeError("Unexpected logits shape")

		if not torch.isfinite(logits).all().item():
			raise RuntimeError("Model produced non-finite logits")

		self.last_logits = logits

		sampled_tokens = logits.argmax(dim=-1).tolist()

		for(request_index, _), token_id in zip(sample_pairs, sampled_tokens):
			results[request_index] = token_id

		return results
