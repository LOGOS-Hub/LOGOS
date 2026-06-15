"""Utility classes for LOGOS model inference."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski


class LMEvaluator:
    """Language Model Evaluator for generation and perplexity calculation."""

    def __init__(self, model_name, task_name, data_manager, device):
        """
        Initialize the language model evaluator.

        Args:
            model_name (str): Path to the model checkpoint.
            task_name (str): Name of the downstream task.
            data_manager (str): Data manager identifier.
            device: Compute device ('cuda'/'cpu').
        """
        print("=======Initializing LMEvaluator=========")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("pad_token is", self.tokenizer.pad_token, self.tokenizer.pad_token_id)
        print("eos_token is", self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        self.task_name = task_name
        self.data_manager = data_manager

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def calculate_ppl(self, text, max_length=4096):
        """
        Calculate perplexity (PPL) of a given text.

        Args:
            text (str): Input text.
            max_length (int): Maximum input length for the model.

        Returns:
            torch.Tensor: Perplexity value.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(self.device)

        stride = max_length
        seq_len = inputs["input_ids"].size(1)
        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = inputs["input_ids"][:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            shift_attention_mask = inputs["attention_mask"][:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,
                reduction="none",
            )
            each_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss = each_loss * shift_attention_mask.view(-1).float()
            neg_log_likelihood = each_loss.sum() / shift_attention_mask.sum().item()

            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens
        ppl = torch.exp(avg_nll)
        return ppl

    def generate(
        self,
        text,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        temperature=1.2,
        repetition_penalty=1.0,
        eos_token=None,
        special_padding_token=True,
    ):
        """
        Generate text from one or more prompts.

        Args:
            text (str or list[str]): Prompt(s) for generation.
            max_length (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling-based decoding.
            top_p (float): Nucleus sampling probability threshold.
            temperature (float): Sampling temperature.
            repetition_penalty (float): Repetition penalty factor.
            eos_token (str, optional): Custom end-of-sequence token string.
            special_padding_token (bool): Whether to use special padding token (LLaMA-style).

        Returns:
            list[str]: List of generated text sequences.
        """
        if special_padding_token:
            self.tokenizer.pad_token = "<|eot_id|>"
            self.tokenizer.pad_token_id = 128009
        else:
            print("pad_token is", self.tokenizer.pad_token, self.tokenizer.pad_token_id)

        eos_token_id = (
            self.tokenizer.eos_token_id
            if eos_token is None
            else self.tokenizer.convert_tokens_to_ids(eos_token)
        )

        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                attention_mask=attention_mask,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
            )

        generated_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        for i in range(len(generated_texts)):
            generated_texts[i] = generated_texts[i].replace(self.tokenizer.pad_token, "")

        print(generated_texts[0])
        return generated_texts

    @staticmethod
    def evaluate_smiles(smiles_list):
        """
        Evaluate validity and uniqueness of generated SMILES.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            tuple: (validity_ratio, uniqueness_ratio)
        """

        def is_valid_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None

        valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
        valid_count = len(valid_smiles)
        unique_valid_count = len(set(valid_smiles))

        validity_ratio = valid_count / len(smiles_list) if smiles_list else 0.0
        uniqueness_ratio = unique_valid_count / valid_count if valid_count > 0 else 0.0

        return validity_ratio, uniqueness_ratio

    @staticmethod
    def compute_QED(molecule_smiles):
        """Compute Quantitative Estimate of Drug-likeness (QED) score."""
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                return None
            return QED.qed(mol)
        except Exception as e:
            print(f"Error calculating QED for {molecule_smiles}: {e}")
            return None

    @staticmethod
    def compute_HBA_HBD(molecule_smiles):
        """Compute the number of hydrogen bond acceptors and donors."""
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                return None
            hba = Lipinski.NumHAcceptors(mol)
            hbd = Lipinski.NumHDonors(mol)
            return hba, hbd
        except Exception as e:
            print(f"Error calculating HBA/HBD for {molecule_smiles}: {e}")
            return None

    @staticmethod
    def compute_FSP3(molecule_smiles):
        """Compute the fraction of sp3-hybridized carbon atoms."""
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                return None
            num_carbons = Descriptors.HeavyAtomCount(mol)
            num_sp3_carbons = sum(
                1 for atom in mol.GetAtoms()
                if atom.GetSymbol() == "C" and atom.GetHybridization() == Chem.HybridizationType.SP3
            )
            return num_sp3_carbons / num_carbons if num_carbons > 0 else 0
        except Exception as e:
            print(f"Error calculating FSP3 for {molecule_smiles}: {e}")
            return None

    @staticmethod
    def compute_RotBonds(molecule_smiles):
        """Compute the number of rotatable bonds."""
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                return None
            return Lipinski.NumRotatableBonds(mol)
        except Exception as e:
            print(f"Error calculating RotBonds for {molecule_smiles}: {e}")
            return None

    @staticmethod
    def compute_TPSA(molecule_smiles):
        """Compute topological polar surface area (TPSA)."""
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                return None
            return Descriptors.TPSA(mol)
        except Exception as e:
            print(f"Error calculating TPSA for {molecule_smiles}: {e}")
            return None
