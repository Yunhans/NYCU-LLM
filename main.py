import argparse
import os
import random
import re
from collections import Counter
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
import wandb  # type: ignore[import-not-found]
from datasets import Dataset
from tqdm import tqdm
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, GenerationConfig, PreTrainedTokenizerBase, TrainerCallback
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


OPTION_LETTERS = ["A", "B", "C", "D"]

def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: List[str]) -> None:
	for path in paths:
		os.makedirs(path, exist_ok=True)


def get_model_dtype() -> torch.dtype:
	return torch.float16 if torch.cuda.is_available() else torch.float32


def load_dataset(path: str, with_answer: bool = True) -> pd.DataFrame:
	df = pd.read_csv(path)
	required = ["question_id", "question", "opa", "opb", "opc", "opd"]
	if with_answer:
		required.append("ans")

	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in {path}: {missing}")

	df = df.copy()
	for c in ["question", "opa", "opb", "opc", "opd"]:
		df[c] = df[c].astype(str).fillna("")

	if with_answer:
		df["ans"] = pd.to_numeric(df["ans"], errors="coerce").astype("Int64")
		df = df.dropna(subset=["ans"]).copy()
		df["ans"] = df["ans"].astype(int)
		bad = df[~df["ans"].isin([0, 1, 2, 3])]
		if len(bad) > 0:
			raise ValueError("Column 'ans' must contain only values 0/1/2/3.")

	return df

def build_chat_prompt(row: pd.Series, tokenizer: PreTrainedTokenizerBase) -> str:
	system_content = (
		"You are an expert pathologist with extensive knowledge of pathology. "
		"Carefully read the question and each option. Select the MOST CORRECT answer. "
		"Answer by choosing only one letter: A, B, C, or D. "
	)
	user_content = (
		f"{row['question']}\n"
		f"A. {row['opa']}\n"
		f"B. {row['opb']}\n"
		f"C. {row['opc']}\n"
		f"D. {row['opd']}"
	)
	
	messages = [
		{"role": "system", "content": system_content},
		{"role": "user", "content": user_content}
	]
	
	result = tokenizer.apply_chat_template(
		messages, 
		tokenize=False, 
		add_generation_prompt=True
	)
	return str(result)


def build_completion_text(row: pd.Series, tokenizer: PreTrainedTokenizerBase) -> str:
	ans_idx = int(row["ans"])
	ans_letter = OPTION_LETTERS[ans_idx]
	
	eos = str(tokenizer.eos_token) if tokenizer.eos_token else ""
	
	return f"Final Answer: {ans_letter}{eos}"


def build_train_example_from_fields(
	question: str,
	opa: str,
	opb: str,
	opc: str,
	opd: str,
	ans: int,
	tokenizer: PreTrainedTokenizerBase,
) -> dict:
	row = pd.Series(
		{
			"question": question,
			"opa": opa,
			"opb": opb,
			"opc": opc,
			"opd": opd,
			"ans": ans,
		}
	)
	prompt = build_chat_prompt(row, tokenizer)
	completion = build_completion_text(row, tokenizer)
	return {"prompt": prompt, "completion": completion}


def tokenize_prompt_completion(
	prompt: str,
	completion: str,
	tokenizer: PreTrainedTokenizerBase,
) -> dict:
	prompt_tokenized = cast(Any, tokenizer(text=prompt))
	full_tokenized = cast(Any, tokenizer(text=prompt + completion))

	prompt_ids_raw = cast(Any, prompt_tokenized["input_ids"])
	full_ids_raw = cast(Any, full_tokenized["input_ids"])

	# Some tokenizers return list[list[int]] for single input; normalize to list[int].
	if len(prompt_ids_raw) > 0 and isinstance(prompt_ids_raw[0], list):
		prompt_ids = cast(List[int], prompt_ids_raw[0])
	else:
		prompt_ids = cast(List[int], prompt_ids_raw)

	if len(full_ids_raw) > 0 and isinstance(full_ids_raw[0], list):
		full_ids = cast(List[int], full_ids_raw[0])
	else:
		full_ids = cast(List[int], full_ids_raw)

	completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
	return {"input_ids": full_ids, "completion_mask": completion_mask}


class EpochOptionShuffleCallback(TrainerCallback):
	def __init__(self, epoch_state: dict) -> None:
		super().__init__()
		self.epoch_state = epoch_state

	def on_epoch_begin(self, args, state, control, **kwargs):
		self.epoch_state["epoch"] = int(state.epoch or 0)
		return control


def build_train_example(
	row: pd.Series,
	tokenizer: PreTrainedTokenizerBase,
) -> dict:
	prompt = build_chat_prompt(row, tokenizer)
	completion = build_completion_text(row, tokenizer)
	return {"prompt": prompt, "completion": completion}

def split_data(
	df: pd.DataFrame,
	val_size: float,
	seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_df, val_df = train_test_split(
		df,
		test_size=val_size,
		random_state=seed,
		stratify=df["ans"],
	)
	return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def parse_pred_letter(text: str) -> Optional[str]:
	text = text.strip()

	match = re.search(r"(?:Final\s*answer|Answer)\s*:\s*([ABCD])\b", text, flags=re.IGNORECASE)
	if match:
		return match.group(1).upper()

	match = re.search(r"(?:Final\s*answer|Answer)\s*:\s*([0-3])\b", text, flags=re.IGNORECASE)
	if match:
		return OPTION_LETTERS[int(match.group(1))]

	letter_matches = re.findall(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
	if letter_matches:
		return letter_matches[-1].upper()

	number_matches = re.findall(r"\b([0-3])\b", text)
	if number_matches:
		return OPTION_LETTERS[int(number_matches[-1])]

	return None


def build_generation_config(model, tokenizer, max_new_tokens: int = 8) -> GenerationConfig:
	generation_config = GenerationConfig.from_model_config(model.config)
	generation_config.do_sample = False  # 關閉抽樣，每次取機率最高的 Token
	generation_config.pad_token_id = tokenizer.eos_token_id
	generation_config.max_new_tokens = max_new_tokens
	return generation_config


@torch.no_grad()
def run_generation(
	model,
	tokenizer,
	df: pd.DataFrame,
	max_new_tokens: int = 8,
	batch_size: int = 8,
) -> List[str]:
	model.eval()
	preds: List[str] = []
	generation_config = build_generation_config(model, tokenizer, max_new_tokens=max_new_tokens)

	n_rows = len(df)
	if n_rows == 0:
		return preds

	n_batches = (n_rows + batch_size - 1) // batch_size
	for start in tqdm(range(0, n_rows, batch_size), total=n_batches, desc="Inference"):
		batch_df = df.iloc[start : start + batch_size]

		all_prompts = [build_chat_prompt(row, tokenizer) for _, row in batch_df.iterrows()]
		inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
		inputs = {k: v.to(model.device) for k, v in inputs.items()}

		input_width = int(cast(torch.Tensor, inputs["input_ids"]).shape[1])

		output = model.generate(
			**inputs,
			generation_config=generation_config,
		)
		
		decoded_outputs = []
		for i in range(len(all_prompts)):
			generated_ids = output[i][input_width:]
			decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
			pred_letter = parse_pred_letter(decoded)
			if pred_letter is None:
				pred_letter = "A"
			decoded_outputs.append(pred_letter)

		preds.extend(decoded_outputs)

	return preds


def evaluate_accuracy(
	model,
	tokenizer,
	df: pd.DataFrame,
	generation_batch_size: int = 8,
	max_new_tokens: int = 8,
) -> float:
	pred_letters = run_generation(
		model,
		tokenizer,
		df,
		max_new_tokens=max_new_tokens,
		batch_size=generation_batch_size,
	)
	gold_letters = [OPTION_LETTERS[int(v)] for v in df["ans"].tolist()]
	return float(accuracy_score(gold_letters, pred_letters))


def save_split_csv(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	split_dir: str,
) -> None:
	ensure_dirs([split_dir])
	train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
	val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)


def build_model_and_tokenizer(model_name: str, lora_r: int = 16, lora_alpha: int = 32):
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		device_map="auto",
		dtype=get_model_dtype(),
	)

	lora_cfg = cast(Any, LoraConfig)(
		r=lora_r,
		lora_alpha=lora_alpha,
		lora_dropout=0.05,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, lora_cfg)
	return model, tokenizer


def init_wandb(args: argparse.Namespace) -> bool:
	if args.wandb_mode == "disabled":
		return False

	wandb_kwargs = {
		"project": args.wandb_project,
		"name": args.wandb_run_name,
		"mode": args.wandb_mode,
		"config": vars(args),
	}
	if args.wandb_entity:
		wandb_kwargs["entity"] = args.wandb_entity

	wandb.init(**wandb_kwargs)
	return True


def adapter_exists(save_dir: str) -> bool:
	return os.path.exists(os.path.join(save_dir, "adapter_config.json"))


def main():
	parser = argparse.ArgumentParser(description="PathoQA LoRA finetuning pipeline")
	parser.add_argument("--dataset-path", type=str, default="dataset/dataset.csv")
	parser.add_argument("--benchmark-path", type=str, default="dataset/benchmark.csv")
	parser.add_argument("--split-dir", type=str, default="dataset/splits")
	parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
	parser.add_argument("--save-dir", type=str, default="saved_models/llama32_1b_lora")
	parser.add_argument("--output-dir", type=str, default="outputs")
	parser.add_argument("--epochs", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--grad-accum", type=int, default=4)
	parser.add_argument("--max-seq-length", type=int, default=512)
	parser.add_argument("--early-stopping-patience", type=int, default=2)
	parser.add_argument("--val-size", type=float, default=0.1)       
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--mode", type=str, choices=["all", "train", "output"], default="all")
	parser.add_argument("--wandb-project", type=str, default="pathoqa-lora")
	parser.add_argument("--wandb-run-name", type=str, default=None)
	parser.add_argument("--wandb-entity", type=str, default=None)
	parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
	parser.add_argument("--generation-batch-size", type=int, default=8)
	parser.add_argument("--max-new-tokens", type=int, default=16)
	parser.add_argument("--weight-decay", type=float, default=0.15)
	parser.add_argument("--lora-r", type=int, default=32)
	parser.add_argument("--lora-alpha", type=int, default=64)
	args = parser.parse_args()

	ensure_dirs([args.save_dir, args.output_dir, args.split_dir])
	set_seed(args.seed)
	wandb_enabled = init_wandb(args) if args.mode in ["all", "train"] else False
	if wandb_enabled:
		wandb.config.update({"run_mode": args.mode}, allow_val_change=True)

	full_df = load_dataset(args.dataset_path, with_answer=True)
	benchmark_df = load_dataset(args.benchmark_path, with_answer=False)

	train_df, val_df = split_data(
		full_df,
		val_size=args.val_size,
		seed=args.seed,
	)
	
	print("\n=== Data Distribution Check ===")
	print(f"Train distribution:\n{train_df['ans'].value_counts().sort_index()}")
	print(f"Val distribution:\n{val_df['ans'].value_counts().sort_index()}\n")
	
	save_split_csv(train_df, val_df, args.split_dir)

	eval_model = None
	tokenizer = None

	if args.mode in ["all", "train"]:
		model, tokenizer = build_model_and_tokenizer(args.model_name, args.lora_r, args.lora_alpha)
		train_tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

		train_raw = train_df.copy().reset_index(drop=True)
		train_raw["row_id"] = np.arange(len(train_raw), dtype=np.int64)
		train_dataset = Dataset.from_pandas(train_raw, preserve_index=False)
		epoch_state = {"epoch": 0}

		def train_transform(batch):
			row_ids = batch["row_id"]
			is_single = not isinstance(row_ids, list)
			if is_single:
				row_ids = [row_ids]
				questions = [batch["question"]]
				opas = [batch["opa"]]
				opbs = [batch["opb"]]
				opcs = [batch["opc"]]
				opds = [batch["opd"]]
				ans_list = [batch["ans"]]
			else:
				questions = batch["question"]
				opas = batch["opa"]
				opbs = batch["opb"]
				opcs = batch["opc"]
				opds = batch["opd"]
				ans_list = batch["ans"]

			all_input_ids: List[List[int]] = []
			all_completion_masks: List[List[int]] = []
			for i, row_id in enumerate(row_ids):
				rng = random.Random(args.seed + epoch_state["epoch"] * 1_000_003 + int(row_id))
				perm = list(range(4))
				rng.shuffle(perm)

				orig_options = [str(opas[i]), str(opbs[i]), str(opcs[i]), str(opds[i])]
				new_options = [orig_options[old_idx] for old_idx in perm]
				new_ans = perm.index(int(ans_list[i]))

				example = build_train_example_from_fields(
					question=str(questions[i]),
					opa=new_options[0],
					opb=new_options[1],
					opc=new_options[2],
					opd=new_options[3],
					ans=new_ans,
					tokenizer=train_tokenizer,
				)
				tokenized = tokenize_prompt_completion(example["prompt"], example["completion"], train_tokenizer)
				all_input_ids.append(cast(List[int], tokenized["input_ids"]))
				all_completion_masks.append(cast(List[int], tokenized["completion_mask"]))

			if is_single:
				return {
					"input_ids": all_input_ids[0],
					"completion_mask": all_completion_masks[0],
				}
			return {
				"input_ids": all_input_ids,
				"completion_mask": all_completion_masks,
			}

		train_dataset.set_transform(train_transform)
		train_callbacks: List[Any] = [
			EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
			EpochOptionShuffleCallback(epoch_state),
		]

		val_examples = [build_train_example(r, train_tokenizer) for _, r in val_df.iterrows()]
		val_tokenized = [
			tokenize_prompt_completion(ex["prompt"], ex["completion"], train_tokenizer) for ex in val_examples
		]
		val_dataset = Dataset.from_pandas(pd.DataFrame(val_tokenized), preserve_index=False)

		model.print_trainable_parameters()
		cast(Any, model).config.use_cache = False

		training_args = cast(Any, SFTConfig)(
			output_dir=args.save_dir,
			num_train_epochs=args.epochs,
			per_device_train_batch_size=args.batch_size,
			per_device_eval_batch_size=args.batch_size,
			dataloader_num_workers=4,
			dataloader_pin_memory=True,
			gradient_accumulation_steps=args.grad_accum,
			learning_rate=args.learning_rate,
			logging_steps=100,
			eval_strategy="epoch",
			save_strategy="epoch",
			save_total_limit=2,
			warmup_ratio=0.1,
			lr_scheduler_type="cosine",
			weight_decay=args.weight_decay,
			max_grad_norm=1.0,
			fp16=False,
			bf16=torch.cuda.is_available(),
			gradient_checkpointing=True,
			report_to="wandb" if wandb_enabled else "none",
			remove_unused_columns=False,
			completion_only_loss=True,
			load_best_model_at_end=True,
			metric_for_best_model="eval_loss",
			greater_is_better=False,
			max_length=args.max_seq_length,
			dataset_kwargs={"skip_prepare_dataset": True},
			seed=args.seed,
		)

		trainer = SFTTrainer(
			model=cast(Any, model),
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=val_dataset,
			processing_class=train_tokenizer,
			callbacks=train_callbacks,
		)

		trainer.train()
		model.save_pretrained(args.save_dir)
		tokenizer.save_pretrained(args.save_dir)
		eval_model = model

		if args.mode == "train":
			if wandb_enabled:
				wandb.finish()
			print("=== Training Finished (train-only mode) ===")
			print(f"Saved model to: {args.save_dir}")
			return

	if not adapter_exists(args.save_dir):
		raise FileNotFoundError(
			f"No trained adapter found in {args.save_dir}. "
			"Run with --mode train or --mode all first."
		)

	tokenizer = AutoTokenizer.from_pretrained(args.save_dir, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"

	if eval_model is None:
		base_model = AutoModelForCausalLM.from_pretrained(
			args.model_name,
			device_map="auto",
			dtype=get_model_dtype(),
		)
		eval_model = PeftModel.from_pretrained(base_model, args.save_dir)

	train_acc = evaluate_accuracy(
		eval_model,
		tokenizer,
		train_df,
		generation_batch_size=args.generation_batch_size,
		max_new_tokens=args.max_new_tokens,
	)
	val_acc = evaluate_accuracy(
		eval_model,
		tokenizer,
		val_df,
		generation_batch_size=args.generation_batch_size,
		max_new_tokens=args.max_new_tokens,
	)

	metrics = {
		"train_accuracy": train_acc,
		"val_accuracy": val_acc,
		"n_train": len(train_df),
		"n_val": len(val_df),
		"max_new_tokens": args.max_new_tokens,
	}
	pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

	if wandb_enabled:
		wandb.log(
			{
				"eval/train_accuracy": train_acc,
				"eval/val_accuracy": val_acc,
				"data/n_train": len(train_df),
				"data/n_val": len(val_df),
			}
		)

	benchmark_preds = run_generation(
		eval_model,
		tokenizer,
		benchmark_df,
		max_new_tokens=args.max_new_tokens,
		batch_size=args.generation_batch_size,
	)
	
	print("\n=== Benchmark Prediction Distribution ===")
	print(Counter(benchmark_preds))
	
	pred_ans = [{"A": 0, "B": 1, "C": 2, "D": 3}[l] for l in benchmark_preds]
	benchmark_out = pd.DataFrame({"question_id": benchmark_df["question_id"], "pred": pred_ans})
	benchmark_out.to_csv(os.path.join(args.output_dir, "benchmark_predictions.csv"), index=False)

	if wandb_enabled:
		wandb.log({"benchmark/num_predictions": len(benchmark_out)})
		wandb.finish()

	print("\n=== Output Finished ===")
	print(f"Train accuracy: {train_acc:.4f}")
	print(f"Validation accuracy: {val_acc:.4f}")
	print(f"Metrics saved to: {os.path.join(args.output_dir, 'metrics.csv')}")
	print(f"Benchmark predictions saved to: {os.path.join(args.output_dir, 'benchmark_predictions.csv')}")


if __name__ == "__main__":
	main()