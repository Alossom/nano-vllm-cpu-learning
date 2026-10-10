import argparse
import sys

from nanovllm import SamplingOptions, TextGenerator


def main():
    parser = argparse.ArgumentParser(description="CPU Qwen3 text generation")
    parser.add_argument("text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    args = parser.parse_args()

    if args.greedy and any(value is not None for value in (
        args.seed, args.temperature, args.top_k, args.top_p,
    )):
        parser.error("--greedy cannot be combined with sampling parameters")
    if not args.text.strip():
        parser.error("text must not be empty")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")

    try:
        options = SamplingOptions(
            mode="greedy" if args.greedy else "sample",
            temperature=args.temperature if args.temperature is not None else 0.8,
            top_k=args.top_k if args.top_k is not None else 20,
            top_p=args.top_p if args.top_p is not None else 0.9,
        )
        generator = TextGenerator(args.model)
        kwargs = dict(
            max_tokens=args.max_tokens, options=options, seed=args.seed,
        )
        if args.raw:
            result = generator.generate(args.text, **kwargs)
        else:
            result = generator.generate_chat(
                [{"role": "user", "content": args.text}], **kwargs,
            )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(result.text)
    print(
        f"prompt_tokens={result.prompt_tokens} "
        f"generated_tokens={result.generated_tokens} seed={result.seed}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
