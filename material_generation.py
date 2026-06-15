"""Material generation script.

This script uses the language model to generate material structures starting
from the <MaterialS> tag. It generates samples in batches, calculates
perplexity (PPL) for each, and saves results to JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

from accelerate import Accelerator
from tqdm import tqdm

from utils import LMEvaluator


MATERIAL_E_TAG = "<MaterialE>"
PROMPT = "<MaterialS>"


def generate_batch(
    evaluator: LMEvaluator,
    batch_size: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    special_padding_token: bool,
    eos_token: str = None,
) -> dict:
    """Generate a batch of material samples and compute PPL for each."""

    prompts = [PROMPT] * batch_size

    generated_texts = evaluator.generate(
        prompts,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        special_padding_token=special_padding_token,
        eos_token=eos_token,
    )

    assert len(generated_texts) == batch_size, "Input and output lengths are inconsistent."

    ppls = []
    for text in generated_texts:
        ppl = evaluator.calculate_ppl(text)
        ppls.append(float(ppl.cpu()))

    return {
        "generated_raw": generated_texts,
        "ppls": ppls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Material generation.")
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Not used for material generation; kept for interface compatibility.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Checkpoint directory for LMEvaluator.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to store generation results.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Total number of material samples to generate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.85,
        help="Top-p value for nucleus sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty for generation.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="Model architecture type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    accelerator = Accelerator()
    evaluator = LMEvaluator(
        model_name=args.model_path,
        task_name="material_generation",
        data_manager="material_generation",
        device=accelerator.device,
    )

    output_path = os.path.join(
        args.results_dir,
        f"material_generation_{args.temperature}_{args.top_p}_{args.repetition_penalty}",
    )
    os.makedirs(output_path, exist_ok=True)

    num_samples = args.num_samples
    batch_size = args.batch_size
    total_batches = (num_samples + batch_size - 1) // batch_size

    start_time = time.time()

    output_file = os.path.join(output_path, "results.jsonl")
    generated_count = 0

    with open(output_file, "w", encoding="utf-8") as results_file:
        for i in tqdm(range(total_batches), desc="Generating materials"):
            current_batch_size = min(batch_size, num_samples - generated_count)

            res = generate_batch(
                evaluator,
                batch_size=current_batch_size,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                special_padding_token=(args.model_type == "llama"),
                eos_token=MATERIAL_E_TAG,
            )

            for seq, ppl in zip(res["generated_raw"], res["ppls"]):
                line = json.dumps(
                    {"text": seq, "ppl": ppl},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                results_file.write(line + "\n")

            generated_count += current_batch_size
            if (i + 1) % 10 == 0:
                print(f"Generated {generated_count}/{num_samples} samples")

    elapsed = time.time() - start_time

    print(f"Generation complete. Total: {generated_count} samples")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
