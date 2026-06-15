"""Protein-ligand interaction generation script.

This script reads protein-ligand interaction data (JSONL), generates interaction
sequences conditioned on protein pocket-ligand prompts, and saves generated
sequences with perplexity scores.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import torch
from typing import Iterable, List, Optional, Set
from datetime import datetime

from accelerate import Accelerator
from rdkit import Chem
from tqdm import tqdm

from utils import LMEvaluator


MOLECULE_E_TAG = "<MoleculeE>"


def read_jsonl_in_batches(file_path, batch_size=128):
    """Generator: yields batches of records from a JSONL file."""
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    batch.append(data)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")

    if batch:
        yield batch


def evaluate_sampleN(
    evaluator: LMEvaluator,
    inputs: list[dict],
    num_samples: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    special_padding_token: bool,
    eos_token: str = None,
) -> dict:
    """Run multiple generations for a batch of samples."""

    generated_texts = []
    prompts = []
    ids = []
    gts = []
    results = []
    ppl_results = []

    for data in inputs:
        prompts.append(data["protein_pocket_ligand"])
        ids.extend([data["id"]] * num_samples)
        gts.extend([data["groundtruth"]] * num_samples)

    for i in range(num_samples):
        generated = evaluator.generate(
            prompts,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            special_padding_token=special_padding_token,
            eos_token=eos_token,
        )
        assert len(generated) == len(prompts), "Input and output lengths are inconsistent."
        results.append(generated)

    for i in range(len(prompts)):
        for generated in results:
            gen_seq = generated[i]
            if isinstance(gen_seq, (list, tuple)):
                generated_texts.append(gen_seq[0] if generated else "")
            else:
                generated_texts.append(str(gen_seq))
            ppl = evaluator.calculate_ppl(generated_texts[-1])
            ppl_results.append(float(ppl.cpu()))

    return {
        "ids": ids,
        "groundtruths": gts,
        "generated_raw": generated_texts,
        "ppls": ppl_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protein-ligand interaction generation.")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to protein-ligand interaction JSONL dataset.",
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
        "--max_samples",
        type=int,
        default=100000000,
        help="Optional limit on number of samples to process.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
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
        "--num_samples",
        type=int,
        default=10,
        help="Number of generations per input sample.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="Model architecture type.",
    )
    return parser.parse_args()


def count_valid_lines(file_path):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size
    effective_num_samples = max(args.num_samples, 1)
    file_path = args.data_file
    data_name = file_path.split("/")[-2]
    print(f"===================data name is {data_name}======================")

    accelerator = Accelerator()
    evaluator = LMEvaluator(
        model_name=args.model_path,
        task_name="protein_ligand_interaction",
        data_manager="protein_ligand_interaction",
        device=accelerator.device,
    )

    output_path = os.path.join(
        args.results_dir,
        f"protein_ligand_interaction_{args.temperature}_{args.top_p}_{args.repetition_penalty}",
        data_name,
    )
    os.makedirs(output_path, exist_ok=True)

    total_lines = count_valid_lines(file_path)
    total_batches = (total_lines + batch_size - 1) // batch_size

    start_time = time.time()

    output_file = os.path.join(output_path, "results.jsonl")
    batch_generator = read_jsonl_in_batches(file_path, batch_size=batch_size)

    with open(output_file, "w", encoding="utf-8") as results_file:
        for i, batch in enumerate(
            tqdm(batch_generator, total=total_batches, desc="Processing batches")
        ):
            print(batch)
            res = evaluate_sampleN(
                evaluator,
                inputs=batch,
                num_samples=effective_num_samples,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                special_padding_token=(args.model_type == "llama"),
                eos_token=MOLECULE_E_TAG,
            )

            for seq, id_, gt, ppl in zip(
                res["generated_raw"], res["ids"], res["groundtruths"], res["ppls"]
            ):
                print(seq, id_, gt, ppl)
                line = json.dumps(
                    {"sequence": seq, "id": id_, "groundtruth": gt, "ppl": ppl},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                results_file.write(line + "\n")

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
