import math
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class SamplingOptions:
	mode: str = "greedy"
	temperature: float = 1.0
	top_k: int = 0
	top_p: float = 1.0

	def __post_init__(self):
		if self.mode not in ("greedy", "sample"):
			raise ValueError("mode must be 'greedy' or 'sample'")
		if(not math.isfinite(self.temperature) or self.temperature <= 0):
			raise ValueError("temperature must be finite and positive")
		if type(self.top_k) is not int or self.top_k < 0:
			raise ValueError("top_k must be a nonnegative integer")
		if(not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
			raise ValueError("top_p must be in (0, 1]")

class Sampler:
	def _validate_logits(self, logits: torch.Tensor):
		if logits.ndim != 1 or logits.numel() == 0:
			raise ValueError("logits must be a nonempty vocabulary vector")
		if(logits.device.type != "cpu" or logits.dtype != torch.float32):
			raise ValueError("This reference implementation requires CPU float32 logits")
		if not torch.isfinite(logits).all().item():
			raise ValueError("logits must contain only finite values")

	@torch.no_grad()
	def probabilities(self, logits: torch.Tensor, options: SamplingOptions) -> torch.Tensor:
		self._validate_logits(logits)

		if options.mode == "greedy":
			probabilities = torch.zeros_like( logits, dtype=torch.float64)
			probabilities[int(logits.argmax().item())] = 1.0
			return probabilities

		scores = logits.to(torch.float64)
		scores = (scores - scores.max()) / options.temperature

		if 0 < options.top_k < scores.numel():
			_, indices = torch.topk(scores, k=options.top_k)

			filtered = torch.full_like(scores, -torch.inf)
			filtered[indices] = scores[indices]
			scores = filtered

		probabilities = torch.softmax(scores, dim=-1)

		if options.top_p < 1.0:
			sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

			cumulative = sorted_probs.cumsum(dim=-1)

			cumulative_before = torch.cat([torch.zeros_like(cumulative[:1]), cumulative[:-1]])

			keep = cumulative_before < options.top_p

			filtered_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))

			probabilities = torch.zeros_like(probabilities)
			probabilities[sorted_indices] = filtered_probs
			probabilities = probabilities / probabilities.sum()

		return probabilities

	@torch.no_grad()
	def sample(self, logits: torch.Tensor, options: SamplingOptions, generator: torch.Generator | None = None) -> int:
		if options.mode == "greedy":
			self._validate_logits(logits)
			return int(logits.argmax().item())

		if generator is None:
			raise ValueError("Random sampling requires an explicit generator")

		if torch.device(generator.device).type != "cpu":
			raise ValueError("Generator must be on CPU")

		probabilities = self.probabilities(logits, options)

		sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)

		return int(sampled.item())
