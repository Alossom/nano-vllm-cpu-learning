from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.engine_learning import Engine
from nanovllm.engine.sampling_runner_learning import SamplingModelRunner
from nanovllm.layers.sampler_learning import SamplingOptions

class SamplingEngine(Engine):
	def __init__(self, config, runner: SamplingModelRunner, policy: str = "alternating"):
		if config.kvcache_block_size != runner.model.block_size:
			raise ValueError("Engine and model block sizes differ")

		if config.num_kvcache_blocks != runner.model.num_blocks:
			raise ValueError("Engine and model block counts differ")

		super().__init__(config, runner, policy=policy)

		self.sampling_runner = runner

	def add_request(self, token_ids: list[int], *, max_tokens: int = 64, ignore_eos: bool = False, options: SamplingOptions | None = None, seed: int | None = None) -> int:
		if options is None:
			options = SamplingOptions(mode="greedy")

		state = self.sampling_runner.prepare_state(options, seed)

		token_ids = list(token_ids)

		if not token_ids:
			raise ValueError("Request must contain at least one token")

		for token_id in token_ids:
			if(type(token_id) is not int or not 0 <= token_id < self.sampling_runner.model.vocab_size):
				raise ValueError(f"Invalid token id: {token_id}")

		sampling_params = SamplingParams(temperature=options.temperature, max_tokens=max_tokens, ignore_eos=ignore_eos)

		seq_id = super().add_request(token_ids, sampling_params)

		self.sampling_runner.register(seq_id, state)

		return seq_id

	def step(self):
		events = super().step()

		for event in events:
			if event.finished:
				self.sampling_runner.release(event.seq_id)

		return events
