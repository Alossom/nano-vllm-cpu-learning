from importlib import import_module

_EXPORTS = {
    "TextGenerator": ("nanovllm.text_generator_learning", "TextGenerator"),
    "GenerationResult": ("nanovllm.text_generator_learning", "GenerationResult"),
    "CpuConfig": ("nanovllm.cpu_config_learning", "CpuConfig"),
    "SamplingOptions": ("nanovllm.layers.sampler_learning", "SamplingOptions"),
    "LLM": ("nanovllm.llm", "LLM"),
    "SamplingParams": ("nanovllm.sampling_params", "SamplingParams"),
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
