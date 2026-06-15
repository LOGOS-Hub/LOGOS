"""Reverse reaction generation script.

This script reads a reverse-reaction dataset (JSONL), runs the language model
to generate predictions for each `test_input`, and saves generated sequences
along with their perplexity scores.
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


REVERSE_TAG = "<ReverseReact>"
END_TOKENS = ("<|end_of_text|>", "</s>", "<eos>")
MOLECULE_PATTERN = re.compile(r"<MoleculeS>(.*?)<MoleculeE>", re.DOTALL)


def extract_reverse_region(text: str) -> str:
    """Return substring between the first `<ReverseReact>` and an end token."""
    start = text.find(REVERSE_TAG)
    if start == -1:
        return ""
    start += len(REVERSE_TAG)
    end_positions = [pos for tok in END_TOKENS if (pos := text.find(tok, start)) != -1]
    end = min(end_positions) if end_positions else len(text)
    return text[start:end]


def extract_smiles(section: str) -> List[str]:
    """Extract SMILES strings enclosed by <MoleculeS> ... <MoleculeE>."""
    return [match.strip() for match in MOLECULE_PATTERN.findall(section) if match.strip()]


def canonical_smiles_set(smiles_list: Iterable[str]) -> Set[str]:
    """Convert SMILES iterable to a canonicalized set using RDKit."""
    result: Set[str] = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        result.add(Chem.MolToSmiles(mol, canonical=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse reaction generation.")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to reverse reaction JSONL dataset.",
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
        default=0.7,
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
        default=32,
        help="Number of generations per input sample.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limit to 3 samples).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="Model architecture type.",
    )
    return parser.parse_args()


def evaluate_sampleN(
    evaluator: LMEvaluator,
    test_input: str,
    groundtruth: str,
    num_samples: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    special_padding_token: bool,
) -> dict:
    """Run multiple generations for one sample and collect results."""

    generated_texts: List[str] = []
    ppl_results = []
    batch_size = 16

    for start in range(0, num_samples, batch_size):
        batch_prompts = [test_input] * min(batch_size, num_samples - start)
        generated = evaluator.generate(
            batch_prompts,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            special_padding_token=special_padding_token,
        )
        for gen_seq in generated:
            if isinstance(gen_seq, (list, tuple)):
                generated_texts.append(gen_seq[0] if generated else "")
            else:
                generated_texts.append(str(gen_seq))
            ppl = evaluator.calculate_ppl(generated_texts[-1])
            ppl_results.append(float(ppl.cpu()))

    return {
        "groundtruth": groundtruth,
        "generated_raw": generated_texts,
        "ppls": ppl_results,
    }


def main() -> None:
    args = parse_args()
    if args.debug:
        args.max_samples = 3
    effective_num_samples = max(args.num_samples, 1)

    accelerator = Accelerator()
    evaluator = LMEvaluator(
        model_name=args.model_path,
        task_name="reverse_react_eval",
        data_manager="reverse_react_eval",
        device=accelerator.device,
    )

    output_path = os.path.join(
        args.results_dir,
        f"reversereact_eval_{args.temperature}_{args.top_p}_{args.repetition_penalty}",
    )
    os.makedirs(output_path, exist_ok=True)

    with open(args.data_file, "rb") as f:
        sample_total = sum(1 for line in f) if args.max_samples > 10000000 else args.max_samples

    start_time = time.time()
    idx = 0

    with open(args.data_file, "r", encoding="utf-8") as f:
        progress = tqdm(total=sample_total, desc="Generating", unit="sample")
        for line in f:
            if not line.strip():
                idx += 1
                continue
            record = json.loads(line)
            res = evaluate_sampleN(
                evaluator,
                test_input=record["test_input"],
                groundtruth=record["groundtruth"],
                num_samples=effective_num_samples,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                special_padding_token=(args.model_type == "llama"),
            )

            output_file = os.path.join(output_path, f"{idx}.jsonl")
            with open(output_file, "w", encoding="utf-8") as results_file:
                for seq, ppl in zip(res["generated_raw"], res["ppls"]):
                    gt = res["groundtruth"]
                    print(seq, ppl, gt)
                    line_out = json.dumps(
                        {"sequence": seq, "ppl": ppl, "groundtruth": gt},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    results_file.write(line_out + "\n")

            progress.update(1)
            idx += 1
            if args.max_samples and idx >= args.max_samples:
                break
        progress.close()

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
